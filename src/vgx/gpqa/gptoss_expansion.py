"""Collect the remaining 395 GPT-OSS replies with the frozen pilot protocol.

No labels, scoring, calibration, or verifier requests. The original fifty replies
and two historical unknown-cost reservations remain immutable. New requests use
an isolated response cache and the existing USD 1 budget, with one process lock.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import time

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, RequestCache
from vgx.common.storage import atomic_json, digest, file_digest, file_lock
from vgx.gpqa import gptoss_finish50 as pilot
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa import gptoss_resume as transport
from vgx.gpqa import gptoss_retry22 as retry22
from vgx.gpqa.artifacts import FrozenCandidate, load_candidates
from vgx.gpqa.fresh import generator_request, inputs
from vgx.gpqa.generator_study import digest_text
from vgx.gpqa.prompt import SYSTEM, build_verifier_prompt
from vgx.gpqa.sequential import seal, unseal

ROOT = pilot.ROOT
DIRECTORY = 'expansion_445_20261018'
REPORT = Path('reports/gpqa_gptoss_expansion_20261018')
SONNET = Path('results/gpqa_sonnet55_20261015/expansion_445/frozen_candidates.json')


def source(root):
    completed, parent, first, _, runner, budget = pilot.load(root)
    cfg = parent['config']
    manifest, all_items = inputs(Path(cfg['dataset_root']))
    baseline, gemini = load_candidates(Path(cfg['baseline_bundle']))
    qwen_path = Path(cfg['qwen_baseline'])
    qwen = json.loads(qwen_path.read_text()); unseal(qwen, 'bundle_id')
    sonnet = json.loads(SONNET.read_text()); unseal(sonnet, 'bundle_id')
    if (len(all_items) != 445 or dict(Counter(i.partition for i in all_items)) !=
            {'calibration': 249, 'evaluation': 196}):
        raise ValueError('expected the original 249 development and 196 Diamond inputs')
    if baseline['dataset_id'] != manifest['dataset_id'] or sonnet['dataset_id'] != manifest['dataset_id']:
        raise ValueError('baseline dataset mismatch')
    by_id = {i.item_id: i for i in all_items}
    for name, rows in [('gemini', [asdict(c) for c in gemini]),
                       ('qwen', qwen['groups']['qwen']), ('sonnet', sonnet['groups']['sonnet_5_5'])]:
        if len(rows) != 445 or {r['item_id'] for r in rows} != set(by_id):
            raise ValueError(name + ' item membership mismatch')
        for row in rows:
            item = by_id[row['item_id']]
            if (row['question'], tuple(row['choices']), row['partition'], row['subject']) != (
                    item.question, tuple(item.choices), item.partition, item.subject):
                raise ValueError(name + ' question or choice order mismatch')
            if name == 'gemini' and row['generator_prompt_sha256'] != digest_text(generator_request(item, None).prompt):
                raise ValueError('original generator prompt differs')
    first_ids = {i.item_id for i in first}
    if len(first_ids) != 50 or any(by_id[i.item_id] != i for i in first):
        raise ValueError('pilot input membership changed')
    items = list(first) + [i for i in all_items if i.item_id not in first_ids]
    identity = {'dataset_id': manifest['dataset_id'], 'gemini_bundle_id': baseline['bundle_id'],
                'qwen_bundle_id': qwen['bundle_id'], 'sonnet_bundle_id': sonnet['bundle_id'],
                'qwen_sha256': file_digest(qwen_path), 'sonnet_sha256': file_digest(SONNET),
                'ordered_input_hashes': [digest(asdict(i)) for i in items]}
    return completed, parent, items, runner, budget, identity


def cache_for(root, position):
    return pilot.source_cache(root, position) if position <= 50 else RequestCache(root/DIRECTORY/'cache')


def prepare(root):
    out = root/DIRECTORY
    if (out/'plan.json').exists():
        return audit(root)
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = pilot.audit(root)
        if previous['stage'] != 'complete' or previous['responses'] != 50:
            raise ValueError('fifty received pilot replies required, including malformed replies')
        completed, parent, items, runner, budget, identity = source(root)
        state = budget._load(); keys = []; bodies = []; reserves = []; existing = {}
        for position, item in enumerate(items, 1):
            req = generator_request(item, parent['parent_study_id']); key = runner.cache_key(req)
            keys.append(key); bodies.append(digest(runner.request_body(req)))
            reserves.append(budget.reservation(runner, req))
            if position <= 50:
                call, _ = cache_for(root, position).get(runner, req)
                base.checked(call, runner, req, parent['config']); existing[key] = digest(asdict(call))
            else:
                for old_cache in (cache_for(root, position), RequestCache(root/'cache'),
                                  RequestCache(base.PARENT/'cache')):
                    rows, events, _, _ = transport.history(old_cache, key)
                    if rows or events or key in state['requests']:
                        raise ValueError('expansion request was already attempted')
        if len(set(keys)) != 445 or len(existing) != 50:
            raise ValueError('duplicate request identities')
        if budget.limit != 1.0 or retry22.committed(state) + max(reserves[50:]) > budget.limit:
            raise ValueError('unchanged USD1 budget cannot reserve the next request')
        inventory = base.budget_inventory()
        if inventory['committed_estimate_and_reservations_usd'] + budget.limit - retry22.committed(state) > 200:
            raise ValueError('overall authorized budget insufficient')
        protected = {str(p): file_digest(p) for p in root.rglob('*') if p.is_file()
                     and DIRECTORY not in p.parts and p != budget.path and not p.name.endswith('.lock')}
        protected.update({str(p): file_digest(p) for p in pilot.REPORT.rglob('*') if p.is_file()})
        plan = seal({'pilot_completion_id': completed['completion_id'], 'sources': identity,
            'implementation_sha256': file_digest(__file__), 'runner': runner.identity,
            'item_ids': [i.item_id for i in items], 'request_keys': keys, 'request_body_hashes': bodies,
            'reservations_usd': reserves, 'partition_counts': dict(Counter(i.partition for i in items)),
            'preserved_files_sha256': protected, 'existing_response_hashes': existing,
            'preserved_budget_entries': state['requests'], 'budget_binding': state['binding'],
            'budget_before': budget.summary(), 'pilot_received_estimated_usd': previous['all_received_gptoss_estimated_usd'],
            'initial_shared_committed_usd': retry22.committed(state),
            'additional_cost_projection_usd': previous['all_received_gptoss_estimated_usd']/50*395,
            'projection_basis': 'Pilot mean token usage; not guaranteed. Reserve and reconcile each request within the unchanged USD1 cap.',
            'reused_responses': 50, 'additional_requests': 395, 'target': 445,
            'minimum_interval_seconds': 30, 'spacing_basis': 'after previous attempt completes',
            'backoff_seconds': [60, 120], 'maximum_attempts_per_request': 3,
            'authorization': '2026-10-18 user requested all 445 existing Gemini/Qwen/Sonnet questions, reuse all 50 GPT-OSS replies, unchanged current settings and raw confidence, preserve prior results.',
            'confidence_calibrated': False, 'answer_keys_read': False, 'verifier_requests': 0,
            'confirmed_billing_usd': None}, 'expansion_id')
        base.immutable(out/'plan.json', plan)
    result = audit(root)
    base.immutable(out/'preflight.json', result)
    return result


def load(root):
    plan = json.loads((root/DIRECTORY/'plan.json').read_text()); unseal(plan, 'expansion_id')
    completed, parent, items, runner, budget, identity = source(root)
    if (plan['implementation_sha256'] != file_digest(__file__) or plan['sources'] != identity
            or plan['pilot_completion_id'] != completed['completion_id'] or plan['runner'] != runner.identity
            or plan['item_ids'] != [i.item_id for i in items]):
        raise ValueError('sealed source, model, protocol, or expansion changed')
    for path, sha in plan['preserved_files_sha256'].items():
        if file_digest(path) != sha:
            raise ValueError('historical artifact changed: ' + path)
    for index, item in enumerate(items):
        req = generator_request(item, parent['parent_study_id'])
        if (runner.cache_key(req) != plan['request_keys'][index]
                or digest(runner.request_body(req)) != plan['request_body_hashes'][index]
                or budget.reservation(runner, req) != plan['reservations_usd'][index]):
            raise ValueError('frozen request identity or pricing changed')
    return plan, parent, items, runner, budget


def check_ledger(plan, budget, state):
    if state['binding'] != plan['budget_binding']:
        raise ValueError('shared budget binding changed')
    for key, entry in plan['preserved_budget_entries'].items():
        if state['requests'].get(key) != entry:
            raise ValueError('historical budget entry changed')
    allowed = set(plan['preserved_budget_entries']) | set(plan['request_keys'][50:])
    if set(state['requests']) - allowed:
        raise ValueError('unplanned shared budget entry')
    if retry22.committed(state) > budget.limit:
        raise ValueError('shared budget exceeded')


def candidate(item, req, call):
    parsed = base.original.parse_call(call)
    return asdict(FrozenCandidate(item.item_id, item.partition, item.subject, item.question,
        item.choices, parsed.answer, parsed.p_correct, parsed.ok, parsed.failure, call.model,
        call.key, digest_text(req.prompt), digest_text(call.response), SYSTEM,
        build_verifier_prompt(item, parsed.answer) if parsed.answer else None))


def audit(root):
    plan, parent, items, runner, budget = load(root)
    state = budget._load(); check_ledger(plan, budget, state)
    rows = []; costs = []; records_all = []; attempts = 0; retry_gaps = []; rejections = Counter()
    frozen_pilot = json.loads((root/pilot.DIRECTORY/'frozen_candidates.json').read_text())
    unseal(frozen_pilot, 'bundle_id')
    original_rows = frozen_pilot['groups']['gpt_oss_120b']
    for position, item in enumerate(items, 1):
        req = generator_request(item, parent['parent_study_id']); key = runner.cache_key(req)
        cache = cache_for(root, position)
        if position > 50:
            records, _, starts, successes = transport.history(cache, key)
            attempts += len(starts); records_all.extend(records)
            rejections.update(str(r['meta']['http_status']) for r in records if r.get('error'))
            if len(starts) > 3:
                raise ValueError('HTTP429 retry allowance exceeded')
            by_execution = {r['meta']['execution_id']: r for r in records}
            for number, start in enumerate(starts[1:], 1):
                before = by_execution.get(starts[number-1]['execution_id'])
                if not before or before.get('error') != 'http_429':
                    raise ValueError('retry without an explicit HTTP429')
                gap = start['created'] - before['created'] - before['latency_s']; retry_gaps.append(gap)
                if gap < plan['backoff_seconds'][number-1] - .01:
                    raise ValueError('HTTP429 backoff violated')
            entry = state['requests'].get(key)
            if entry and entry['status'] not in ('settled', 'rejected'):
                raise ValueError('new unknown, unpriced, or in-flight request')
            if not successes:
                if bool(entry) != bool(starts) or (entry and (entry['status'] != 'rejected' or entry['estimated_usd'] != 0)):
                    raise ValueError('missing response does not reconcile with budget')
                continue
        call, _ = cache.get(runner, req)
        estimate = base.checked(call, runner, req, parent['config']); costs.append(estimate)
        if position <= 50:
            if digest(asdict(call)) != plan['existing_response_hashes'][key]:
                raise ValueError('pilot reply changed')
            if digest(candidate(item, req, call)) != digest(original_rows[position-1]):
                raise ValueError('pilot candidate or raw confidence changed')
        else:
            if (not entry or entry['status'] != 'settled' or entry['execution_id'] != call.meta['execution_id']
                    or abs(entry['estimated_usd'] - estimate['estimated_usd']) > 1e-12
                    or estimate['estimated_usd'] > plan['reservations_usd'][position-1]):
                raise ValueError('new provider usage does not reconcile')
        parsed = base.original.parse_call(call)
        choice = (call.meta['provider_response'].get('choices') or [{}])[0]
        rows.append({'partition': item.partition, 'valid': parsed.ok, 'failure': parsed.failure,
                     'confidence': parsed.p_correct, 'stop_reason': str(choice.get('finish_reason'))})
    ordered = sorted(records_all, key=lambda r: r['created'])
    gaps = [b['created']-a['created']-a['latency_s'] for a, b in zip(ordered, ordered[1:])]
    if gaps and min(gaps) < 29.99:
        raise ValueError('30-second spacing violated')
    summary = budget.summary(); spend = sum(c['estimated_usd'] for c in costs)
    if abs(spend-parent['original_gptoss_estimated_usd']-summary['estimated_usd']) > 1e-10:
        raise ValueError('total received usage does not reconcile with shared ledger')
    partitions = {}
    for partition in ('calibration', 'evaluation'):
        selected = [r for r in rows if r['partition'] == partition]
        probs = [r['confidence'] for r in selected if r['confidence'] is not None]
        partitions[partition] = {'responses': len(selected), 'valid': sum(r['valid'] for r in selected),
            'confidence_n': len(probs), 'mean_raw_confidence': sum(probs)/len(probs) if probs else None,
            'confidence_values': dict(Counter(str(p) for p in probs)),
            'invalid': dict(Counter(r['failure'] for r in selected if r['failure'])),
            'stop_reasons': dict(Counter(r['stop_reason'] for r in selected))}
    return {'stage': 'complete' if len(rows) == 445 else 'dry_run', 'expansion_id': plan['expansion_id'],
        'target': 445, 'responses': len(rows), 'valid': sum(r['valid'] for r in rows),
        'invalid': dict(Counter(r['failure'] for r in rows if r['failure'])), 'partitions': partitions,
        'original_50_responses_and_candidates_unchanged': True, 'old_unknown_reservations_preserved': True,
        'reused_pilot_responses': 50, 'additional_responses': len(rows)-50, 'remaining_requests': 445-len(rows),
        'additional_cost_projection_usd': plan['additional_cost_projection_usd'],
        'new_provider_attempts': attempts, 'new_rejections': dict(rejections),
        'minimum_seconds_after_previous_attempt': min(gaps) if gaps else None,
        'backoff_intervals_checked': len(retry_gaps), 'budget': summary,
        'additional_estimated_usd': summary['estimated_usd']-plan['budget_before']['estimated_usd'],
        'all_received_gptoss_estimated_usd': spend,
        'input_tokens': sum(c['usage']['input_tokens'] for c in costs),
        'output_tokens_including_reasoning': sum(c['usage']['output_tokens'] for c in costs),
        'historical_unknown_reserved_usd': plan['budget_before']['outstanding_reserved_usd'],
        'confirmed_billing_usd': None, 'confidence_calibrated': False,
        'answer_keys_read': False, 'accuracy_computed': False, 'verifier_requests': 0}


def collect(root, allow_api=False):
    if not allow_api:
        return audit(root)
    out = root/DIRECTORY
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage'] == 'blocked':
        raise ValueError('saved blocked expansion requires review; no automatic retry')
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, parent, items, runner, budget = load(root)
        status = audit(root); initial_count = status['responses']; active = None
        cache = RequestCache(out/'cache')

        def snapshot(stage, **extra):
            status.update(stage=stage, updated_at_unix=time.time(), budget=budget.summary(),
                          new_calls_this_invocation=status['responses']-initial_count, **extra)
            atomic_json(out/'status.json', status)

        try:
            inventory = base.budget_inventory()
            if inventory['committed_estimate_and_reservations_usd'] + budget.limit - retry22.committed(budget._load()) > 200:
                raise ValueError('overall authorized budget insufficient')
            atomic_json(out/'billing_check.json', base.check_billing(parent['config']['project']))
            snapshot('running')
            all_prior_records = [r for position, item in enumerate(items, 1)
                for r in cache_for(root, position).log(runner.cache_key(generator_request(item, parent['parent_study_id']))).records()]
            last_end = max((r['created']+r['latency_s'] for r in all_prior_records), default=0.)
            for position, item in enumerate(items[50:], 51):
                req = generator_request(item, plan['expansion_id']); active = runner.cache_key(req)
                try:
                    cache.get(runner, req)
                    continue
                except OfflineCacheMiss:
                    pass
                while True:
                    records, _, starts, _ = transport.history(cache, active)
                    if len(starts) >= 3:
                        raise ValueError('explicit HTTP429 retry allowance exhausted')
                    ready = max((r['created']+r['latency_s'] for r in records), default=0.)
                    if starts:
                        ready += plan['backoff_seconds'][len(starts)-1]
                    deadline = max(last_end+30, ready)
                    snapshot('waiting', current_position=position, current_partition=item.partition,
                             next_attempt_number=len(starts)+1, not_before_unix=deadline)
                    while time.time() < deadline:
                        time.sleep(max(0, min(30, deadline-time.time())))
                    with file_lock(str(budget.path)+'.lock'):
                        state = budget._load(); check_ledger(plan, budget, state)
                        if any(e['status'] not in ('settled', 'rejected') for k, e in state['requests'].items()
                               if k not in plan['preserved_budget_entries']):
                            raise ValueError('new unresolved or reserved execution')
                        reserve = plan['reservations_usd'][position-1]
                        if retry22.committed(state)+reserve > budget.limit:
                            raise ValueError('unchanged shared budget cannot reserve next request')
                        state['requests'][active] = {'status': 'reserved', 'reserved_usd': reserve,
                            'model': runner.model, 'role': 'generator', 'expansion_id': plan['expansion_id']}
                        atomic_json(budget.path, state)
                    snapshot('requesting')
                    try:
                        call, _ = cache.get(runner, req, allow_api=True); last_end = time.time()
                    except ProviderRejectedError as exc:
                        last_end = time.time()
                        with file_lock(str(budget.path)+'.lock'):
                            state = budget._load(); check_ledger(plan, budget, state)
                            state['requests'][active].update(status='rejected', estimated_usd=0.)
                            atomic_json(budget.path, state)
                        if exc.status != 429:
                            raise
                        status = audit(root); snapshot('backoff', current_position=position, last_http_status=429)
                        continue
                    estimate = base.checked(call, runner, req, parent['config'])
                    if estimate['estimated_usd'] > reserve:
                        raise ValueError('request exceeded reserved cost')
                    with file_lock(str(budget.path)+'.lock'):
                        state = budget._load(); check_ledger(plan, budget, state)
                        state['requests'][active].update(status='settled', estimated_usd=estimate['estimated_usd'],
                            issues=estimate['issues'], execution_id=call.meta['execution_id'])
                        atomic_json(budget.path, state)
                    status = audit(root); snapshot('collecting'); break
            status = audit(root)
            if status['responses'] != 445:
                raise ValueError('expansion incomplete')
            finalize(root)
            snapshot('complete')
            return status
        except Exception as exc:
            with file_lock(str(budget.path)+'.lock'):
                state = budget._load()
                if active in state['requests'] and state['requests'][active]['status'] == 'reserved':
                    state['requests'][active].update(status='unresolved', estimated_usd=None)
                    atomic_json(budget.path, state)
            # Provider payloads and exception messages may contain request data.
            detail = str(exc) if isinstance(exc, ValueError) else None
            snapshot('blocked', error_type=type(exc).__name__, error=detail,
                     http_status=getattr(exc, 'status', None))
            return status


def finalize(root):
    status = audit(root)
    if status['stage'] != 'complete':
        raise ValueError('all 445 received replies required for final collection report')
    plan, parent, items, runner, _ = load(root); out = root/DIRECTORY
    candidates = []
    for position, item in enumerate(items, 1):
        req = generator_request(item, parent['parent_study_id'])
        call, _ = cache_for(root, position).get(runner, req)
        candidates.append(candidate(item, req, call))
    frozen = seal({'expansion_id': plan['expansion_id'], 'dataset_id': plan['sources']['dataset_id'],
                   'groups': {'gpt_oss_120b': candidates}}, 'bundle_id')
    base.immutable(out/'frozen_candidates.json', frozen)
    result = {**status, 'bundle_id': frozen['bundle_id']}
    base.immutable(out/'completion_report.json', result)
    return result


def report(root):
    result = finalize(root); out = root/DIRECTORY
    status = json.loads((out/'status.json').read_text())
    if status['stage'] != 'complete':
        raise ValueError('saved expansion status is not complete')
    REPORT.mkdir(parents=True, exist_ok=True)
    names = ('plan.json', 'status.json', 'completion_report.json', 'billing_check.json')
    for name in names:
        dest = REPORT/name
        if dest.exists() and file_digest(dest) != file_digest(out/name):
            raise ValueError('existing exported report differs: '+name)
        if not dest.exists():
            shutil.copyfile(out/name, dest)
    base.immutable(REPORT/'audit.json', audit(root))
    readme = (
        'GPT-OSS-120B: expansion to the original 445 GPQA inputs (249 development, 196 Diamond).\n'
        'All 50 pilot replies, including one malformed final JSON, were reused unchanged; 395 new replies.\n'
        'Original model, prompt, medium reasoning, max_tokens 32768, parser and frozen prices.\n'
        '30 seconds after completion; explicit HTTP429 only: 60/120-second backoff, three attempts maximum.\n'
        'No answer-key reads, accuracy scoring, calibration, verifier requests or repairs in this stage.\n'
        'Usage estimates target llm-applications-490420. These are not confirmed billing.\n'
        'Shared USD1 cap unchanged. Both historical unknown reservations remain; actual total billing is unknown.\n'
        'Exports contain aggregate collection metadata, plans and audit hashes, without question text, answers or raw replies.\n'
    )
    target = REPORT/'README.txt'
    if target.exists() and target.read_text() != readme:
        raise ValueError('existing README differs')
    if not target.exists(): target.write_text(readme)
    manifest = ''.join(f'{file_digest(p)}  {p.name}\n' for p in sorted(REPORT.iterdir())
                       if p.is_file() and p.name != 'sha256.txt')
    path = REPORT/'sha256.txt'
    if path.exists() and path.read_text() != manifest:
        raise ValueError('existing export manifest differs')
    if not path.exists(): path.write_text(manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect', 'audit', 'report'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    result = collect(args.root, args.allow_api) if args.command == 'collect' else globals()[args.command](args.root)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if result.get('stage') == 'blocked': raise SystemExit(2)


if __name__ == '__main__':
    main()
