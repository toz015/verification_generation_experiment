"""Ask the five GPQA/QuantumKatas verifiers about every frozen math-trial answer (no grading here).

Each verifier sees the problem and the frozen final answer only (not the generator's reasoning
or p_correct). One request per (verifier, candidate); identical prompts share one cached
response. Uncertain outcomes are settled as verifier service failures and the run continues;
HTTP 429 after the cache's retries, other rejections and budget stops halt the run for resume.
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
from vgx.gpqa.sequential import seal, unseal
from vgx.math_trial import prompt
from vgx.quantum_katas import runners
from vgx.quantum_katas.prompt import parse_verifier_response
from vgx.quantum_katas.run_pilot import AbandonedRequests


def specs(cfg):
    return [runners.ModelSpec(**v) for v in cfg['verifiers']]


def candidates(cfg):
    out = []
    for name, path in cfg['generators'].items():
        plan = json.loads((Path(path)/'plan.json').read_text()); unseal(plan, 'plan_id')
        bundle = json.loads((Path(path)/'frozen_candidates.json').read_text()); unseal(bundle, 'bundle_id')
        problems = {it['item_id']: it['problem'] for rows in plan['sample'].values() for it in rows}
        for r in bundle['rows']:
            if r.get('answer') is not None:
                out.append({'candidate_id': f"{name}|{r['item_id']}", 'generator': name, 'item_id': r['item_id'],
                            'problem': problems[r['item_id']], 'answer': r['answer'], 'bundle_id': bundle['bundle_id']})
    return out


def request(spec, c):
    return Request(f"{spec.id}|{c['candidate_id']}", prompt.build_verifier_prompt(c['problem'], c['answer']),
                   prompt.VERIFIER_SYSTEM, {'role': 'verifier', 'verifier': spec.id, 'candidate_id': c['candidate_id']})


def run(cfg_path, out, *, allow_api=False, workers=8):
    cfg = json.loads(Path(cfg_path).read_text()); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if allow_api and not cfg.get('paid_collection_authorized'):
        raise ValueError('paid collection not authorized in config')
    b = cfg['budget']
    cache = BudgetedRequestCache(out/'cache', project=b['project'], pricing=b['pricing'], limit_usd=b['limit_usd'],
                                 gemini_output_reserve=b.get('gemini_output_reserve', 131072))
    vs = specs(cfg); rs = {s.id: s.runner() for s in vs}; cands = candidates(cfg)
    jobs = [(s, c) for c in cands for s in vs]
    abandoned = AbandonedRequests(SimpleNamespace(root=out))
    keyed = {rs[s.id].cache_key(request(s, c)): {'role': 'verifier', 'verifier': s.id, 'candidate_id': c['candidate_id']}
             for s, c in jobs}
    abandoned.settle_prior(cache, keyed)
    stop = threading.Event(); lock = threading.Lock(); signals, errors = {}, []

    def one(job):
        s, c = job
        if stop.is_set():
            return
        req = request(s, c); key = rs[s.id].cache_key(req)
        if key in abandoned.keys():
            row = {'score': None, 'failure': 'verifier_service_failure'}
        else:
            try:
                call, _ = cache.get(rs[s.id], req, allow_api=allow_api)
            except OfflineCacheMiss:
                return
            except UncertainRequestError:
                with lock:
                    abandoned.abandon(cache, key, keyed[key])
                row = {'score': None, 'failure': 'verifier_service_failure'}
            except (ProviderRejectedError, BudgetLimitError) as exc:
                stop.set()
                with lock:
                    errors.append({'job': req.key, 'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None),
                                   'message': str(exc)})
                return
            else:
                parsed = parse_verifier_response(call.response, truncated=runners.finish_status(call) == 'truncated')
                row = {'score': parsed.p_correct, 'failure': parsed.failure, 'request_key': call.key}
        with lock:
            signals.setdefault(c['candidate_id'], {})[s.id] = row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    done = sum(len(v) for v in signals.values()) == len(jobs) and not errors
    status = {'stage': 'complete' if done else ('blocked' if errors else 'partial'), 'jobs': len(jobs),
              'answered': sum(len(v) for v in signals.values()), 'errors': errors[:5], 'budget': cache.summary(),
              'failures': {s.id: sum(1 for v in signals.values() if v.get(s.id, {}).get('failure')) for s in vs}}
    atomic_json(out/'status.json', status)
    if done:
        value = seal({'config_sha256': file_digest(cfg_path), 'verify_sha256': file_digest(__file__),
                      'prompt_sha256': file_digest(prompt.__file__),
                      'bundles': sorted({c['bundle_id'] for c in cands}),
                      'rows': [dict({k: c[k] for k in ('candidate_id', 'generator', 'item_id')}, signals=signals[c['candidate_id']])
                               for c in cands]}, 'signals_id')
        atomic_json(out/'signals.json', value)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    print(json.dumps(run(a.config, a.out, allow_api=a.allow_api, workers=a.workers), indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
