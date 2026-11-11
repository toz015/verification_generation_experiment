"""Extend the sealed Sonnet screen to the existing 445 inputs, without labels."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss
from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa import sonnet_screen as original, sonnet_screen_v2 as approved
from vgx.gpqa import replacement_generators as shared
from vgx.gpqa.artifacts import FrozenCandidate, load_candidates
from vgx.gpqa.fresh import generator_request, inputs
from vgx.gpqa.generator_study import digest_text
from vgx.gpqa.prompt import SYSTEM, build_verifier_prompt
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing


def output(root):
    return Path(root)/'expansion_445'


def source(root):
    parent, amendment, pilot, _ = approved.load(root)
    cfg = parent['config']
    manifest, all_items = inputs(Path(cfg['dataset_root']))
    baseline, gemini = load_candidates(Path(cfg['baseline_bundle']))
    qwen_path = Path(cfg['qwen_baseline'])
    qwen = json.loads(qwen_path.read_text())
    unseal(qwen, 'bundle_id')
    counts = dict(Counter(i.partition for i in all_items))
    if len(all_items) != 445 or counts != {'calibration': 249, 'evaluation': 196}:
        raise ValueError('expected the original 249 development + 196 evaluation inputs')
    if baseline['dataset_id'] != manifest['dataset_id']:
        raise ValueError('baseline dataset mismatch')
    by_id = {i.item_id: i for i in all_items}
    for name, candidates in [('gemini', [asdict(c) for c in gemini]),
                             ('qwen', qwen['groups']['qwen'])]:
        if len(candidates) != 445 or {c['item_id'] for c in candidates} != set(by_id):
            raise ValueError(name+' membership mismatch')
        for c in candidates:
            i = by_id[c['item_id']]
            if (c['question'], tuple(c['choices']), c['partition'], c['subject']) != (
                    i.question, tuple(i.choices), i.partition, i.subject):
                raise ValueError(name+' input content mismatch')
            if name == 'gemini' and c['generator_prompt_sha256'] != digest_text(generator_request(i, None).prompt):
                raise ValueError('original Gemini prompt differs')
    pilot_ids = {i.item_id for i in pilot}
    if len(pilot_ids) != 50 or any(by_id[i.item_id] != i for i in pilot):
        raise ValueError('pilot changed')
    # Reuse pilot first, then retain the original development/evaluation order.
    ordered = list(pilot)+[i for i in all_items if i.item_id not in pilot_ids]
    identity = {'dataset_id': manifest['dataset_id'], 'gemini_bundle_id': baseline['bundle_id'],
                'qwen_bundle_id': qwen['bundle_id'], 'qwen_sha256': file_digest(qwen_path),
                'input_hashes': [digest(asdict(i)) for i in ordered]}
    return parent, amendment, ordered, identity


def checked_call(cache, runner, req, cfg, allow_api=False):
    call, reused = cache.get(runner, req, allow_api=allow_api)
    if (call.model != runner.model or call.params != runner.params or call.prompt != req.prompt
            or call.meta.get('system') != req.system or call.meta.get('runner_identity') != runner.identity
            or any(call.meta.get(k) != req.meta[k] for k in ('role', 'item_id', 'partition'))
            or any(call.meta.get(k) != cfg['project'] for k in ('resource_project', 'quota_project'))):
        raise ValueError('cached request or billing project mismatch')
    original.priced_usage(call, cfg)
    return call, reused


def preserved(root, cache, items, runner):
    paths = [root/'plan.json', root/'status.json', root/'parser_v2/plan.json',
             root/'parser_v2/status.json', root/'parser_v2/frozen_candidates.json',
             root/'parser_v2/comparison.json', root/'parser_v2/completion_audit.json']
    paths += [p for p in Path('reports/gpqa_sonnet55_20261015').rglob('*') if p.is_file()]
    for i in items[:50]:
        path = cache.log(runner.cache_key(generator_request(i, None))).path
        paths += [path, Path(str(path)+'.attempts.jsonl')]
    return {str(p): file_digest(p) for p in paths}


def prepare(root):
    root = Path(root); out = output(root)
    parent, amendment, items, identity = source(root)
    cfg = parent['config']; runner = original.SonnetRunner(cfg); cache = shared.cache_for(cfg, root)
    if json.loads((root/'parser_v2/status.json').read_text())['stage'] != 'complete':
        raise ValueError('original fifty-item screen is not complete')
    for i in items[:50]:
        call, _ = checked_call(cache, runner, generator_request(i, None), cfg)
        if not approved.parse_call(call).ok:
            raise ValueError('original fifty-item format gate failed')
    missing = []
    for i in items[50:]:
        try:
            cache.get(runner, generator_request(i, None))
        except OfflineCacheMiss:
            missing.append(i)
    if len(missing) != 395:
        raise ValueError('expected exactly 395 new requests when preparing the extension')
    costs = cache.summary()
    value = seal({'parent_study_id': parent['study_id'], 'amendment_id': amendment['amendment_id'],
        'parent_plan_sha256': file_digest(root/'plan.json'),
        'amendment_plan_sha256': file_digest(root/'parser_v2/plan.json'),
        'implementation_sha256': file_digest(__file__), 'sources': identity,
        'runner': runner.identity, 'parser_version': approved.PARSER_VERSION,
        'item_ids': [i.item_id for i in items],
        'request_keys': [runner.cache_key(generator_request(i, None)) for i in items],
        'partition_counts': dict(Counter(i.partition for i in items)),
        'preserved_files_sha256': preserved(root, cache, items, runner),
        'prior_budget_audit': original.budget_audit(cfg, root),
        'shared_cache': str(root/'cache'), 'shared_budget_limit_usd': cfg['stage_budget_usd'],
        'initial_estimated_usd': costs['estimated_usd'],
        'additional_calls': 395, 'reused_calls': 50,
        'additional_cost_projection_usd': costs['estimated_usd']/50*395,
        'projection_basis': 'Pilot mean token usage, not a guarantee; reserve each request within the unchanged shared USD 20 cap.',
        'authorization': 'User requested all 445 existing Gemini/Qwen questions, reuse 50 Sonnet replies, same settings, raw confidence, preserve prior results.',
        'confidence_calibrated': False, 'answer_keys_read': False, 'verifier_requests': 0}, 'expansion_id')
    shared.immutable(out/'plan.json', value)
    return {'stage': 'prepared', 'expansion_id': value['expansion_id'],
        'partition_counts': value['partition_counts'], 'reused': 50, 'additional_calls': 395,
        'additional_cost_projection_usd': value['additional_cost_projection_usd'], 'budget': costs}


def load(root):
    root = Path(root)
    plan = json.loads((output(root)/'plan.json').read_text()); unseal(plan, 'expansion_id')
    parent, amendment, items, identity = source(root)
    runner = original.SonnetRunner(parent['config'])
    if (plan['parent_study_id'] != parent['study_id'] or plan['amendment_id'] != amendment['amendment_id']
        or plan['parent_plan_sha256'] != file_digest(root/'plan.json')
        or plan['amendment_plan_sha256'] != file_digest(root/'parser_v2/plan.json')
        or plan['implementation_sha256'] != file_digest(__file__)
        or plan['sources'] != identity or plan['runner'] != runner.identity
        or plan['item_ids'] != [i.item_id for i in items]
        or plan['request_keys'] != [runner.cache_key(generator_request(i, None)) for i in items]):
        raise ValueError('sealed source, settings or expansion changed')
    for path, sha in plan['preserved_files_sha256'].items():
        if file_digest(path) != sha:
            raise ValueError('preserved file changed: '+path)
    return plan, parent['config'], items


def finalize(root):
    """Freeze all candidates and report collection only. No answer-key reads."""
    root = Path(root); plan, cfg, items = load(root); out = output(root)
    runner = original.SonnetRunner(cfg); cache = shared.cache_for(cfg, root)
    candidates, rows, costs = [], [], []
    for i in items:
        req = generator_request(i, plan['expansion_id'])
        call, _ = checked_call(cache, runner, req, cfg)
        successes = [r for r in cache.log(call.key).records() if r.get('error') is None]
        if len(successes) != 1:
            raise ValueError('duplicate successful request')
        parsed = approved.parse_call(call); body = call.meta['provider_response']
        candidates.append(asdict(FrozenCandidate(i.item_id, i.partition, i.subject, i.question,
            i.choices, parsed.answer, parsed.p_correct, parsed.ok, parsed.failure, runner.model,
            call.key, digest_text(req.prompt), digest_text(call.response), SYSTEM,
            build_verifier_prompt(i, parsed.answer) if parsed.answer else None)))
        rows.append({'partition': i.partition, 'valid': parsed.ok, 'confidence': parsed.p_correct,
                     'failure': parsed.failure, 'stop_reason': body.get('stop_reason'),
                     'thinking_tokens': (body.get('usage', {}).get('output_tokens_details') or {}).get('thinking_tokens', 0)})
        costs.append(original.priced_usage(call, cfg))
    frozen = seal({'expansion_id': plan['expansion_id'], 'dataset_id': plan['sources']['dataset_id'],
        'parser_version': approved.PARSER_VERSION, 'groups': {'sonnet_5_5': candidates}}, 'bundle_id')
    shared.immutable(out/'frozen_candidates.json', frozen)
    summary = {}
    for partition in ('calibration', 'evaluation'):
        selected = [r for r in rows if r['partition'] == partition]
        probs = [r['confidence'] for r in selected if r['confidence'] is not None]
        summary[partition] = {'responses': len(selected), 'valid': sum(r['valid'] for r in selected),
            'confidence_n': len(probs), 'mean_raw_confidence': sum(probs)/len(probs) if probs else None,
            'confidence_values': dict(Counter(str(p) for p in probs)),
            'invalid': dict(Counter(r['failure'] for r in selected if r['failure'])),
            'stop_reasons': dict(Counter(str(r['stop_reason']) for r in selected))}
    spend = sum(c['estimated_usd'] for c in costs)
    if abs(spend-cache.summary()['estimated_usd']) > 1e-9:
        raise ValueError('token cost does not reconcile with shared ledger')
    value = {'expansion_id': plan['expansion_id'], 'stage': 'complete', 'target': 445,
        'reused_pilot_responses': 50, 'additional_responses': 395, 'partitions': summary,
        'original_fifty_and_reports_unchanged': True, 'duplicate_successful_requests': 0,
        'confidence_calibrated': False, 'answer_keys_read': False, 'accuracy_computed': False,
        'input_tokens': sum(c['usage']['input_tokens'] for c in costs),
        'output_tokens_including_thinking': sum(c['usage']['output_tokens'] for c in costs),
        'reported_thinking_tokens': sum(r['thinking_tokens'] or 0 for r in rows),
        'estimated_usd': spend, 'additional_estimated_usd': spend-plan['initial_estimated_usd'],
        'confirmed_billing_usd': None, 'budget': cache.summary(), 'bundle_id': frozen['bundle_id']}
    shared.immutable(out/'completion_report.json', value)
    return value


def collect(root, allow_api=False):
    root = Path(root); out = output(root)
    # Same nonblocking process lock as the screen, so no duplicate collector starts.
    with (root/'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, cfg, items = load(root)
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage'] == 'blocked':
            raise ValueError('saved blocked state requires explicit review; no automatic retry')
        runner = original.SonnetRunner(cfg); cache = shared.cache_for(cfg, root)
        rows, errors = [], []
        def snapshot(stage):
            value = {'expansion_id': plan['expansion_id'], 'stage': stage,
                'updated_at_unix': time.time(), 'target': 445, 'responses': len(rows),
                'valid': sum(r['valid'] for r in rows),
                'invalid': dict(Counter(r['failure'] for r in rows if r['failure'])),
                'partition_responses': dict(Counter(r['partition'] for r in rows)),
                'reused_this_invocation': sum(r['reused'] for r in rows),
                'new_calls_this_invocation': sum(not r['reused'] for r in rows),
                'budget': cache.summary(), 'errors': errors, 'confidence_calibrated': False,
                'answer_keys_read': False}
            atomic_json(out/'status.json', value)
            return value
        current_key = None; started = time.time()
        try:
            if allow_api:
                original.budget_audit(cfg, root)
                atomic_json(out/'billing_check.json', check_billing(cfg['project']))
            snapshot('running'); pacer = Pacer(cfg['interval_seconds'])
            for i in items:
                req = generator_request(i, plan['expansion_id']); current_key = runner.cache_key(req)
                try:
                    call, reused = checked_call(cache, runner, req, cfg)
                except OfflineCacheMiss:
                    if not allow_api:
                        raise
                    pacer.acquire()
                    call, reused = checked_call(cache, runner, req, cfg, allow_api=True)
                parsed = approved.parse_call(call)
                rows.append({'valid': parsed.ok, 'failure': parsed.failure,
                             'partition': i.partition, 'reused': reused})
                snapshot('running')
            finalize(root)
        except Exception as exc:
            error = {'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)}
            if isinstance(exc, (ValueError, OfflineCacheMiss)):
                error['message'] = str(exc)
            if current_key:
                for record in cache.log(current_key).records():
                    if record.get('error') and record.get('created', 0) >= started:
                        error['provider_error'] = record.get('meta', {}).get('provider_error')
            errors.append(error)
        result = snapshot('blocked' if errors else 'complete')
        shared.immutable(out/f'collection_{digest(result)[:12]}.json', result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'collect', 'report'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--allow-api', action='store_true')
    args = p.parse_args()
    value = (prepare(args.root) if args.command == 'prepare' else
             collect(args.root, args.allow_api) if args.command == 'collect' else finalize(args.root))
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)
    if value.get('stage') == 'blocked':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
