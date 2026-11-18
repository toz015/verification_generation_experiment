"""Collect one frozen generator reply per sampled math problem (no grading here).

prepare seals the sample, prompt version, runner identity and this file's hash.
collect sends each request at most once through BudgetedRequestCache. An uncertain
outcome is recorded as a generator service failure and the run continues (owner rule,
2026-10-19); HTTP 429 after the cache's own retries, rejections and budget stops halt
the run so nothing is resent blindly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, UncertainRequestError
from vgx.common.budget import BudgetLimitError, BudgetedRequestCache
from vgx.common.llm import Request
from vgx.common.storage import atomic_json, file_digest
from vgx.quantum_katas.load import sha256_text
from vgx.gpqa.sequential import seal, unseal
from vgx.math_trial import data, prompt
from vgx.quantum_katas import runners
from vgx.quantum_katas.run_pilot import AbandonedRequests


def spec(cfg):
    return runners.ModelSpec(**cfg['generator'])


def request(item, generation_id, version=prompt.PROMPT_VERSION):
    return Request(f'generator|{item["item_id"]}|{generation_id}', prompt.build_generator_prompt(item['problem'], version),
                   prompt.GENERATOR_SYSTEM, {'role': 'generator', 'item_id': item['item_id']})


def prepare(cfg_path, out):
    cfg = json.loads(Path(cfg_path).read_text()); out = Path(out)
    runner = spec(cfg).runner()
    sample, sources = {}, {}
    for name, n in cfg['sample'].items():
        items, meta = data.load(name)
        chosen, alloc = data.stratified_sample(items, n, cfg['seed'])
        sample[name] = [{k: it[k] for k in ('item_id', 'problem', 'stratum', 'topic')} for it in chosen]
        sources[name] = dict(meta, allocation=alloc)
    plan = seal({'config_sha256': file_digest(cfg_path), 'config': cfg, 'sources': sources,
                 'prompt_version': cfg.get('prompt_version', prompt.PROMPT_VERSION), 'runner': runner.identity,
                 'implementation_sha256': {m.__name__: file_digest(m.__file__) for m in (data, prompt)} |
                                          {'collect': file_digest(__file__)},
                 'request_keys': {it['item_id']: runner.cache_key(request(it, cfg['generation_id'], cfg.get('prompt_version', prompt.PROMPT_VERSION)))
                                  for rows in sample.values() for it in rows},
                 'sample': sample, 'answer_keys_read_for_sampling': False}, 'plan_id')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        old = json.loads((out/'plan.json').read_text())
        if old['plan_id'] != plan['plan_id']:
            raise ValueError('a different plan already exists here')
    else:
        atomic_json(out/'plan.json', plan)
    return {'plan_id': plan['plan_id'], 'n': {k: len(v) for k, v in sample.items()}, 'allocation':
            {k: v['allocation'] for k, v in sources.items()}}


def collect(out, *, allow_api=False, workers=4, limit=None):
    out = Path(out)
    plan = json.loads((out/'plan.json').read_text()); unseal(plan, 'plan_id')
    cfg = plan['config']; s = spec(cfg); runner = s.runner()
    if runner.identity != plan['runner']:
        raise ValueError('runner identity changed since prepare')
    if allow_api and not cfg.get('paid_collection_authorized'):
        raise ValueError('paid collection not authorized in config')
    b = cfg['budget']
    cache = BudgetedRequestCache(out/'cache', project=b['project'], pricing=b['pricing'], limit_usd=b['limit_usd'])
    items = [it for rows in plan['sample'].values() for it in rows][:limit]
    # Uncertain outcomes are settled as service failures (reservation counted as spent), as in
    # QuantumKatas, so the ledger does not block the remaining requests.
    abandoned = AbandonedRequests(SimpleNamespace(root=out))
    abandoned.settle_prior(cache, {plan['request_keys'][it['item_id']]: {'role': 'generator', 'item_id': it['item_id']}
                                   for it in items})
    stop = threading.Event(); rows, errors = {}, []
    lock = threading.Lock()

    def one(it):
        if stop.is_set():
            return
        req = request(it, cfg['generation_id'], plan['prompt_version'])
        if runner.cache_key(req) != plan['request_keys'][it['item_id']]:
            raise ValueError('request identity changed since prepare')
        key = plan['request_keys'][it['item_id']]
        if key in abandoned.keys():
            with lock:
                rows[it['item_id']] = {'item_id': it['item_id'], 'status': 'generator_service_failure', 'answer': None,
                                       'p_correct': None, 'failure': 'uncertain_request'}
            return
        try:
            call, reused = cache.get(runner, req, allow_api=allow_api)
        except OfflineCacheMiss:
            row = {'item_id': it['item_id'], 'status': 'not_sent'}
        except UncertainRequestError:
            with lock:
                abandoned.abandon(cache, key, {'role': 'generator', 'item_id': it['item_id']})
            row = {'item_id': it['item_id'], 'status': 'generator_service_failure', 'answer': None, 'p_correct': None,
                   'failure': 'uncertain_request'}
        except (ProviderRejectedError, BudgetLimitError) as exc:
            stop.set()
            with lock:
                errors.append({'item_id': it['item_id'], 'type': type(exc).__name__,
                               'http_status': getattr(exc, 'status', None), 'message': str(exc)})
            return
        else:
            finish = runners.finish_status(call)
            parsed = prompt.parse_generator_response(call.response, truncated=finish == 'truncated')
            usage = (call.meta or {}).get('usage') or ((call.meta or {}).get('provider_response') or {}).get('usage')
            row = {'item_id': it['item_id'], 'status': 'frozen', 'answer': parsed.answer, 'p_correct': parsed.p_correct,
                   'failure': parsed.failure, 'confidence_status': parsed.confidence_status, 'finish_status': finish,
                   'request_key': call.key, 'response_sha256': sha256_text(call.response), 'reused_cache': reused,
                   'latency_s': call.latency_s, 'usage': usage}
        with lock:
            rows[it['item_id']] = row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, items))
    ordered = [rows[it['item_id']] for it in items if it['item_id'] in rows]
    done = all(r['status'] != 'not_sent' for r in ordered) and len(ordered) == len(items) and not errors
    status = {'plan_id': plan['plan_id'], 'stage': 'complete' if done else ('blocked' if errors else 'partial'),
              'errors': errors, 'budget': cache.summary(),
              'counts': {k: sum(r['status'] == k for r in ordered) for k in ('frozen', 'generator_service_failure', 'not_sent')},
              'format_failures': sum(r.get('failure') not in (None, 'uncertain_request') for r in ordered)}
    atomic_json(out/'status.json', status)
    if done and limit is None:
        candidates = seal({'plan_id': plan['plan_id'], 'rows': ordered}, 'bundle_id')
        if not (out/'frozen_candidates.json').exists():
            atomic_json(out/'frozen_candidates.json', candidates)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'collect'])
    p.add_argument('--config', type=Path, default=Path('configs/math_trial_qwen3_20261020.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    v = prepare(a.config, a.out) if a.command == 'prepare' else collect(a.out, allow_api=a.allow_api, workers=a.workers, limit=a.limit)
    print(json.dumps(v, indent=2, default=str)[:4000])


if __name__ == '__main__':
    main()
