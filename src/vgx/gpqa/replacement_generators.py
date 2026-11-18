"""Frozen 50-development-item screening of native reasoning generators.

Uses existing Vertex transport, exact-request caching and budget accounting.
Collection has no answer-key access. Existing experiment modules are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import re
import threading

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.billing import estimate_call
from vgx.common.budget import BudgetedRequestCache
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import generator_study as direct
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.prompt import GeneratorResponse, SYSTEM, build_verifier_prompt, parse_generator_response
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing


class NativeRunner(VertexBatchRunner):
    """Explicit gpt-oss effort, without changing historical runner identities."""

    @property
    def params(self):
        value = super().params
        if self.model.startswith('openai/gpt-oss-'):
            value['reasoning_effort'] = self.reasoning_effort
        return value

    @property
    def identity(self):
        return {**super().identity, 'native_reasoning_adapter': 1,
                'reasoning_effort': self.reasoning_effort if self.model.startswith('openai/gpt-oss-') else None}


class ScreeningCache(BudgetedRequestCache):
    def get(self, runner, request, *, allow_api=False):
        # First screen: no automatic HTTP retries or malformed-output repair.
        return self._get_once(runner, request, allow_api=allow_api)


def runner(model, project):
    return NativeRunner(model['model'], model['location'], project, **model['generation'])


def cache_for(cfg, output):
    return ScreeningCache(Path(output)/'cache', project=cfg['project'],
                          pricing=cfg['pricing'], limit_usd=cfg['stage_budget_usd'])


def parse_call(call):
    choices = call.meta.get('provider_response', {}).get('choices') or []
    finish = choices[0].get('finish_reason') if choices else None
    if finish != 'stop':
        return GeneratorResponse(None, None, False, 'finish_reason:'+str(finish))
    text = call.response.strip()
    if call.model.startswith('deepseek-ai/deepseek-r1-'):
        if '<think>' in text or '</think>' in text:
            match = re.fullmatch(r'<think>.*?</think>\s*(.*)', text, flags=re.DOTALL)
            if not match or text.count('<think>') != 1 or text.count('</think>') != 1:
                return GeneratorResponse(None, None, False, 'invalid_thinking_envelope')
            text = match.group(1).strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return GeneratorResponse(None, None, False, 'invalid_final_json')
    if not isinstance(payload, dict) or set(payload) != {'answer', 'p_correct'}:
        return GeneratorResponse(None, None, False, 'invalid_final_fields')
    return parse_generator_response(json.dumps(payload))


def sources(cfg):
    manifest, baseline, items, gemini = direct.source(cfg)
    qwen_path = Path(cfg['qwen_baseline'])
    qwen_bundle = json.loads(qwen_path.read_text())
    unseal(qwen_bundle, 'bundle_id')
    indexed = {c['item_id']: FrozenCandidate(**c) for c in qwen_bundle['groups']['qwen']}
    qwen = [indexed[i.item_id] for i in items]
    for c, i in zip(qwen, items):
        if (c.question, tuple(c.choices), c.subject, c.partition) != (i.question, tuple(i.choices), i.subject, 'calibration'):
            raise ValueError('historical Qwen input differs')
    identity = {'dataset_id': manifest['dataset_id'], 'gemini_bundle_id': baseline['bundle_id'],
                'qwen_sha256': file_digest(qwen_path),
                'inputs': [digest(asdict(i)) for i in items],
                'gemini': [digest(asdict(c)) for c in gemini],
                'qwen': [digest(asdict(c)) for c in qwen]}
    return items, {'gemini_baseline': gemini, 'qwen_baseline': qwen}, identity


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {**direct.implementation(), **{name: file_digest(root/name) for name in (
        'gpqa/replacement_generators.py', 'gpqa/score.py', 'gpqa/report.py', 'gpqa/sequential.py')}}


def immutable(path, value):
    with file_lock(str(path)+'.lock'):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('existing artifact differs: '+str(path))
        if not path.exists():
            atomic_json(path, value)


def prepare(cfg, output):
    if cfg['project'] != 'llm-applications-490420' or cfg['sample_size'] != 50:
        raise ValueError('project and original 50-item pilot required')
    if not 0 < cfg['stage_budget_usd'] <= 10:
        raise ValueError('screening budget must be within USD 10')
    expected = {'gpt_oss_120b': 'openai/gpt-oss-120b-maas',
                'deepseek_r1_0528': 'deepseek-ai/deepseek-r1-0528-maas'}
    if len(cfg['generators']) != 2 or {m['id']: m['model'] for m in cfg['generators']} != expected:
        raise ValueError('only the two declared replacement generators are allowed')
    items, _, identities = sources(cfg)
    cache = cache_for(cfg, output)
    arms = []
    for model in cfg['generators']:
        r = runner(model, cfg['project'])
        reqs = [generator_request(i, None) for i in items]
        arms.append({'id': model['id'], 'runner': r.identity,
                     'request_keys': [r.cache_key(q) for q in reqs],
                     'reservation_usd': sum(cache.reservation(r, q) for q in reqs)})
    value = seal({'config': cfg, 'sources': identities, 'implementation': implementation(),
                  'item_ids': [i.item_id for i in items], 'arms': arms,
                  'evaluation_requests': 0, 'verifier_requests': 0}, 'study_id')
    immutable(Path(output)/'plan.json', value)
    return {'stage': 'prepared', 'study_id': value['study_id'], 'questions_per_model': len(items),
            'total_reservation_usd': sum(a['reservation_usd'] for a in arms), 'budget_cap_usd': cfg['stage_budget_usd']}


def load(output):
    plan = json.loads((Path(output)/'plan.json').read_text())
    unseal(plan, 'study_id')
    items, baselines, identity = sources(plan['config'])
    if identity != plan['sources'] or implementation() != plan['implementation']:
        raise ValueError('frozen inputs or implementation changed')
    if plan['item_ids'] != [i.item_id for i in items]:
        raise ValueError('pilot membership changed')
    return plan, items, baselines


def collect(output, limit, allow_api):
    output = Path(output)
    plan, items, _ = load(output)
    cfg = plan['config']
    if limit not in (5, 50):
        raise ValueError('use the five-item format screen or full 50-item pilot')
    cache = cache_for(cfg, output)
    if limit == 50:
        # Gate solely on format, termination and accounting; no outcome labels.
        for model in cfg['generators']:
            r = runner(model, cfg['project'])
            for item in items[:5]:
                call, _ = cache.get(r, generator_request(item, plan['study_id']))
                if not parse_call(call).ok:
                    raise ValueError('five-item format gate failed for '+model['id'])
    results = {}; errors = []; stop = threading.Event(); mutex = threading.Lock()

    def snapshot(stage):
        with mutex:
            counts = {}
            for model in cfg['generators']:
                rows = [r for r in results.values() if r['model'] == model['id']]
                counts[model['id']] = {'responses': len(rows), 'valid': sum(r['ok'] for r in rows),
                    'invalid': dict(Counter(r['failure'] for r in rows if r['failure'])),
                    'reused': sum(r['reused'] for r in rows)}
            value = {'study_id': plan['study_id'], 'stage': stage, 'limit': limit,
                     'models': counts, 'errors': list(errors),
                     'new_calls_this_invocation': sum(not r['reused'] for r in results.values()),
                     'budget': cache.summary()}
            atomic_json(output/'status.json', value)
            return value

    def one_model(model):
        r = runner(model, cfg['project']); pacer = Pacer(model['interval_seconds'])
        for item in items[:limit]:
            if stop.is_set():
                return
            req = generator_request(item, plan['study_id'])
            try:
                try:
                    call, reused = cache.get(r, req)
                except OfflineCacheMiss:
                    if not allow_api:
                        raise
                    pacer.acquire()
                    if stop.is_set():
                        return
                    call, reused = cache.get(r, req, allow_api=True)
                parsed = parse_call(call)
                with mutex:
                    results[model['id'], item.item_id] = {'model': model['id'],
                        'ok': parsed.ok, 'failure': parsed.failure, 'reused': reused}
                snapshot('running')
            except Exception as exc:
                stop.set()
                with mutex:
                    errors.append({'model': model['id'], 'type': type(exc).__name__,
                                   'http_status': getattr(exc, 'status', None)})
                snapshot('blocked')
                return

    with file_lock(output/'collection.lock'):
        try:
            if allow_api:
                atomic_json(output/'billing_check.json', check_billing(cfg['project']))
            snapshot('running')
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(one_model, cfg['generators']))
        except Exception as exc:
            errors.append({'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)})
        stage = 'blocked' if errors else ('complete' if limit == 50 else 'format_complete')
        if not errors and limit == 5 and any(not r['ok'] for r in results.values()):
            stage = 'format_failed'
        value = snapshot(stage)
        immutable(output/f'collection_{limit}_{digest(value)[:12]}.json', value)
        return value


def report(output):
    # Explicit offline reporting boundary: only development labels are opened.
    import numpy as np
    from vgx.gpqa.report import forecast, rate_interval, risk_coverage
    from vgx.gpqa.score import correctness_outcome
    output = Path(output); plan, items, groups = load(output); cfg = plan['config']
    cache = cache_for(cfg, output); costs = {}; formats = {}
    for model in cfg['generators']:
        r = runner(model, cfg['project']); candidates = []; rows = []
        finishes = Counter(); reasoning_field_n = 0
        for item in items:
            req = generator_request(item, plan['study_id']); call, _ = cache.get(r, req)
            parsed = parse_call(call)
            candidates.append(FrozenCandidate(item.item_id, item.partition, item.subject, item.question,
                item.choices, parsed.answer, parsed.p_correct, parsed.ok, parsed.failure, r.model,
                call.key, direct.digest_text(req.prompt), direct.digest_text(call.response), SYSTEM,
                build_verifier_prompt(item, parsed.answer) if parsed.answer else None))
            choices = call.meta.get('provider_response', {}).get('choices') or [{}]
            finishes[str(choices[0].get('finish_reason'))] += 1
            reasoning_field_n += bool(choices[0].get('message', {}).get('reasoning_content'))
            rows.append(estimate_call(asdict(call), cfg['pricing']))
        if any(row['estimated_usd'] is None for row in rows):
            raise ValueError('unpriced new-model usage')
        costs[model['id']] = {'estimated_usd': sum(row['estimated_usd'] for row in rows),
            'input_tokens': sum(row['usage']['input_tokens'] for row in rows),
            'output_tokens_including_reasoning': sum(row['usage']['output_tokens'] for row in rows),
            'confirmed_billing_usd': None}
        formats[model['id']] = {'finish_reasons': dict(finishes), 'reasoning_content_present_n': reasoning_field_n}
        groups[model['id']] = candidates
    frozen = seal({'study_id': plan['study_id'], 'groups': {g: [asdict(c) for c in cs] for g, cs in groups.items()}}, 'bundle_id')
    immutable(output/'frozen_candidates.json', frozen)
    bundle = Path(cfg['baseline_bundle'])
    manifest = json.loads((bundle/'bundle_manifest.json').read_text())
    label_path = bundle/'calibration_records.jsonl'
    if file_digest(label_path) != manifest['files'][label_path.name]:
        raise ValueError('development labels changed')
    labels = {r['item_id']: r['correct_index'] for r in read_jsonl(label_path)}
    summaries = {}
    for group, cs in groups.items():
        y = [correctness_outcome(c.answer, labels[c.item_id]) for c in cs]
        pairs = [(c.p_correct, t) for c, t in zip(cs, y) if c.answer is not None and c.p_correct is not None]
        p = [p for p, t in pairs]; yp = [t for p, t in pairs]
        summaries[group] = {'n': len(cs), 'valid_answer_n': sum(c.answer is not None for c in cs),
            'confidence_n': len(pairs), 'correct': sum(y), 'accuracy_all': float(np.mean(y)),
            'accuracy_wilson95': rate_interval(y), 'mean_confidence': float(np.mean(p)) if p else None,
            'confidence_values': dict(Counter(str(v) for v in p)),
            'mean_confidence_if_correct': float(np.mean([p for p,t in pairs if t])) if any(yp) else None,
            'mean_confidence_if_incorrect': float(np.mean([p for p,t in pairs if not t])) if 0 in yp else None,
            'raw_metrics': forecast(yp, p, seed=cfg['seed']),
            'risk_coverage': risk_coverage(yp, p, len(cs)),
            'failures': dict(Counter(c.generator_failure for c in cs if c.generator_failure))}
    value = {'study_id': plan['study_id'], 'generators': summaries, 'new_model_costs': costs,
             'format': formats, 'budget': cache.summary(), 'confidence_calibrated': False,
             'limitations': cfg['limitations']}
    immutable(output/'comparison.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect', 'report'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(json.loads(args.config.read_text()), args.output)
    elif args.command == 'collect':
        value = collect(args.output, args.limit, args.allow_api)
    else:
        value = report(args.output)
    print(json.dumps(value, indent=2, allow_nan=False))
    if value.get('stage') in ('blocked', 'format_failed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
