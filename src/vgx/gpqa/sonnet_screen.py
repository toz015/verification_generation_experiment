"""Isolated Sonnet 5.5 development screen, preserving historical sealed studies."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import time

from vgx.common.api import OfflineCacheMiss
from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.common.vertex_partner import VertexPartnerRunner
from vgx.gpqa import generator_study as direct
from vgx.gpqa import replacement_generators as shared
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.prompt import GeneratorResponse, SYSTEM, build_verifier_prompt, parse_generator_response
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing


class SonnetRunner(VertexPartnerRunner):
    def __init__(self, cfg):
        super().__init__(cfg['model'], cfg['location'], cfg['project'],
                         max_tokens=cfg['generation']['max_tokens'])
        self.generation = cfg['generation']

    @property
    def params(self):
        return json.loads(json.dumps(self.generation))

    @property
    def identity(self):
        return {**super().identity, 'sonnet_screen_adapter': 1,
                'extended_thinking': True, 'sampling_parameters': 'omitted_provider_defaults'}


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {**shared.implementation(), **{p: file_digest(root/p) for p in (
        'gpqa/sonnet_screen.py', 'common/storage.py', 'common/llm.py')}}


def parse_call(call):
    body = call.meta.get('provider_response', {})
    if body.get('stop_reason') != 'end_turn':
        return GeneratorResponse(None, None, False, 'stop_reason:'+str(body.get('stop_reason')))
    text = call.response.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return GeneratorResponse(None, None, False, 'invalid_final_json')
    if not isinstance(payload, dict) or set(payload) != {'answer', 'p_correct'}:
        return GeneratorResponse(None, None, False, 'invalid_final_fields')
    if type(payload['p_correct']) not in (int, float):
        return GeneratorResponse(None, None, False, 'nonnumeric_confidence')
    return parse_generator_response(json.dumps(payload))


def priced_usage(call, cfg):
    body = call.meta.get('provider_response', {})
    if body.get('model') != 'claude-sonnet-5-5':
        raise ValueError('unexpected provider model identity')
    usage = body.get('usage') or {}
    if any(usage.get(k, 0) not in (0, None) for k in (
        'cache_read_input_tokens', 'cache_creation_input_tokens')):
        raise ValueError('unexpected prompt-cache usage; accounting review required')
    estimate = estimate_call(asdict(call), cfg['pricing'])
    if estimate['estimated_usd'] is None or estimate['issues']:
        raise ValueError('provider usage cannot be priced')
    return estimate


def budget_audit(cfg, output):
    """Conservative ledger sum; duplicates remain counted, no credentials read."""
    rows = []
    for path in sorted(Path('results').rglob('budget.json')):
        if path.resolve().is_relative_to(Path(output).resolve()):
            continue
        value = json.loads(path.read_text())
        if value.get('binding', {}).get('project') != cfg['project']:
            continue
        estimated, reserved = 0.0, 0.0
        for entry in value['requests'].values():
            status = entry['status']
            if status == 'rejected':
                continue
            amount = entry.get('estimated_usd') if status == 'settled' else entry.get('reserved_usd')
            if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
                raise ValueError('invalid historical ledger amount')
            if status == 'settled':
                estimated += amount
            else:
                reserved += amount
        rows.append({'path': str(path), 'sha256': file_digest(path),
                     'estimated_usd': estimated, 'reserved_usd': reserved})
    total = sum(r['estimated_usd'] + r['reserved_usd'] for r in rows)
    if total + cfg['stage_budget_usd'] > cfg['authorized_total_budget_usd']:
        raise ValueError('insufficient remaining authorized budget')
    return {'files': rows, 'prior_ledger_sum_including_reservations_usd': total,
            'new_stage_cap_usd': cfg['stage_budget_usd'],
            'authorized_total_budget_usd': cfg['authorized_total_budget_usd'],
            'confirmed_billing_usd': None,
            'note': 'Local Vertex ledgers only; duplicates are not removed. Unpriced old calls retain reservations. Not an invoice.'}


def prepare(cfg, output):
    if (cfg['schema'] != 1 or cfg['project'] != 'llm-applications-490420'
        or cfg['sample_size'] != 50 or cfg['model'] != 'anthropic/claude-sonnet-5-5'
        or cfg['location'] != 'global' or cfg['interval_seconds'] != 10
        or cfg['stage_budget_usd'] != 20 or cfg['authorized_total_budget_usd'] != 200
        or cfg['generation'] != {'max_tokens': 32768, 'thinking': {'type': 'adaptive'},
                                 'output_config': {'effort': 'high'}}):
        raise ValueError('configuration differs from declared Sonnet screen')
    items, _, identity = shared.sources(cfg)
    r = SonnetRunner(cfg)
    cache = shared.cache_for(cfg, output)
    reqs = [generator_request(i, None) for i in items]
    reservations = [cache.reservation(r, q) for q in reqs]
    if sum(reservations) > cfg['stage_budget_usd']:
        raise ValueError('whole-screen reservation exceeds budget')
    value = seal({'config': cfg, 'sources': identity, 'implementation': implementation(),
                  'item_ids': [i.item_id for i in items], 'runner': r.identity,
                  'request_keys': [r.cache_key(q) for q in reqs],
                  'total_reservation_usd': sum(reservations),
                  'first_five_reservation_usd': sum(reservations[:5]),
                  'budget_audit': budget_audit(cfg, output),
                  'evaluation_requests': 0, 'verifier_requests': 0,
                  'confidence_calibrated': False}, 'study_id')
    shared.immutable(Path(output)/'plan.json', value)
    return {'stage': 'prepared', 'study_id': value['study_id'], 'questions': len(items),
            'first_five_reservation_usd': sum(reservations[:5]),
            'total_reservation_usd': sum(reservations), 'budget_cap_usd': cfg['stage_budget_usd']}


def load(output):
    plan = json.loads((Path(output)/'plan.json').read_text())
    unseal(plan, 'study_id')
    items, baselines, identity = shared.sources(plan['config'])
    if identity != plan['sources'] or implementation() != plan['implementation']:
        raise ValueError('frozen inputs or implementation changed')
    r = SonnetRunner(plan['config'])
    if (plan['item_ids'] != [i.item_id for i in items] or r.identity != plan['runner']
        or plan['request_keys'] != [r.cache_key(generator_request(i, None)) for i in items]):
        raise ValueError('frozen requests changed')
    return plan, items, baselines


def collect(output, limit, allow_api=False):
    output = Path(output)
    plan, items, _ = load(output)
    cfg = plan['config']; r = SonnetRunner(cfg); cache = shared.cache_for(cfg, output)
    if limit not in (5, 50):
        raise ValueError('only five-item gate and fifty-item screen are allowed')
    rows, errors = [], []
    def snapshot(stage):
        value = {'study_id': plan['study_id'], 'stage': stage, 'limit': limit,
                 'updated_at_unix': time.time(), 'model': cfg['model'],
                 'responses': len(rows), 'valid': sum(x['ok'] for x in rows),
                 'invalid': dict(Counter(x['failure'] for x in rows if x['failure'])),
                 'new_calls_this_invocation': sum(not x['reused'] for x in rows),
                 'reused': sum(x['reused'] for x in rows), 'errors': errors,
                 'budget': cache.summary(), 'confidence_calibrated': False}
        atomic_json(output/'status.json', value)
        return value
    with file_lock(output/'collection.lock'):
        previous = output/'status.json'
        if previous.exists() and json.loads(previous.read_text()).get('stage') in ('blocked', 'format_failed'):
            raise ValueError('previous run is blocked; no automatic paid retry')
        try:
            if limit == 50:
                for item in items[:5]:
                    call, _ = cache.get(r, generator_request(item, plan['study_id']))
                    priced_usage(call, cfg)
                    if not parse_call(call).ok:
                        raise ValueError('five-item format gate failed')
            if allow_api:
                budget_audit(cfg, output)
                atomic_json(output/'billing_check.json', check_billing(cfg['project']))
            snapshot('running')
            pacer = Pacer(cfg['interval_seconds'])
            for item in items[:limit]:
                req = generator_request(item, plan['study_id'])
                try:
                    call, reused = cache.get(r, req)
                except OfflineCacheMiss:
                    if not allow_api:
                        raise
                    pacer.acquire()
                    call, reused = cache.get(r, req, allow_api=True)
                priced_usage(call, cfg)
                parsed = parse_call(call)
                rows.append({'ok': parsed.ok, 'failure': parsed.failure, 'reused': reused})
                snapshot('running')
        except Exception as exc:
            error = {'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)}
            if isinstance(exc, (ValueError, OfflineCacheMiss)):
                error['message'] = str(exc)
            # Only structured provider error details, never raw prompts or credentials.
            for p in cache.logs().values():
                for row in p.records():
                    if row.get('error') and row.get('meta', {}).get('provider_error'):
                        error['provider_error'] = row['meta']['provider_error']
            errors.append(error)
        stage = 'blocked' if errors else ('complete' if limit == 50 else 'format_complete')
        if not errors and limit == 5 and (len(rows) != 5 or any(not x['ok'] for x in rows)):
            stage = 'format_failed'
        value = snapshot(stage)
        shared.immutable(output/f'collection_{limit}_{digest(value)[:12]}.json', value)
        return value


def report(output):
    """Offline development scoring after all candidates have been frozen."""
    import numpy as np
    from vgx.gpqa.report import forecast, rate_interval, risk_coverage
    from vgx.gpqa.score import correctness_outcome
    output = Path(output); plan, items, groups = load(output); cfg = plan['config']
    cache = shared.cache_for(cfg, output); r = SonnetRunner(cfg)
    candidates, costs, finishes, block_types = [], [], Counter(), Counter()
    for item in items:
        req = generator_request(item, plan['study_id']); call, _ = cache.get(r, req)
        costs.append(priced_usage(call, cfg)); parsed = parse_call(call)
        body = call.meta['provider_response']; finishes[str(body.get('stop_reason'))] += 1
        block_types.update(b.get('type', 'unknown') for b in body.get('content', []))
        candidates.append(FrozenCandidate(item.item_id, item.partition, item.subject, item.question,
            item.choices, parsed.answer, parsed.p_correct, parsed.ok, parsed.failure, r.model,
            call.key, direct.digest_text(req.prompt), direct.digest_text(call.response), SYSTEM,
            build_verifier_prompt(item, parsed.answer) if parsed.answer else None))
    groups['sonnet_5_5'] = candidates
    shared.immutable(output/'frozen_candidates.json', seal({'study_id': plan['study_id'],
        'groups': {g: [asdict(c) for c in cs] for g, cs in groups.items()}}, 'bundle_id'))
    bundle = Path(cfg['baseline_bundle'])
    manifest = json.loads((bundle/'bundle_manifest.json').read_text())
    path = bundle/'calibration_records.jsonl'
    if file_digest(path) != manifest['files'][path.name]:
        raise ValueError('development labels changed')
    labels = {x['item_id']: x['correct_index'] for x in read_jsonl(path)}
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
    value = {'study_id': plan['study_id'], 'generators': summaries,
        'sonnet_usage': {'input_tokens': sum(x['usage']['input_tokens'] for x in costs),
            'output_tokens_including_thinking': sum(x['usage']['output_tokens'] for x in costs),
            'estimated_usd': sum(x['estimated_usd'] for x in costs), 'confirmed_billing_usd': None},
        'stop_reasons': dict(finishes), 'response_block_types': dict(block_types),
        'budget': cache.summary(), 'confidence_calibrated': False, 'limitations': cfg['limitations']}
    shared.immutable(output/'comparison.json', value)
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
