"""Authorized position179 re-execution, followed by the remaining266 requests.

Preserves all178 replies, the blocked expansion and all six unknown reserves.
Provider inputs, parser, pacing, pricing and shared USD1 cap are unchanged.
This separate module adapts the original sealed expansion without editing it.
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
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa import gptoss_retry176_expansion as original
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
DIRECTORY = 'retry179_resume_20261018'
REPORT = Path('reports/gpqa_gptoss_retry179_expansion_20261018')
SONNET = Path('results/gpqa_sonnet55_20261015/expansion_445/frozen_candidates.json')


def source(root):
    old, parent, items, runner, budget = original.load(root)
    return old, parent, items, runner, budget, old['sources']


def cache_for(root, position):
    return original.cache_for(root, position) if position <= 178 else RequestCache(root/DIRECTORY/'cache')


def entry_key(plan, position):
    return ('authorized_retry179:'+plan['expansion_id'] if position == 179
            else plan['request_keys'][position-1])


def max_new_attempts(position):
    # The original unknown execution of position179 counts toward the total3.
    return 2 if position == 179 else 3


def backoff_seconds(position, completed_new_attempts):
    historical = 1 if position == 179 else 0
    return (60, 120)[completed_new_attempts+historical-1]


def prepare(root):
    out = root/DIRECTORY
    if (out/'plan.json').exists(): return audit(root)
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old, parent, items, runner, budget, identity = source(root)
        stopped = json.loads((root/original.DIRECTORY/'status.json').read_text())
        if (stopped['stage'], stopped['responses'], stopped['current_position'], stopped['error_type']) != ('blocked', 178, 179, 'UncertainRequestError'):
            raise ValueError('expected178 received replies and an unknown position179 execution')
        state = budget._load(); original.check_ledger(old, budget, state)
        if budget.limit != 1.0: raise ValueError('shared budget cap changed')
        key179 = old['request_keys'][178]
        prior_cache = RequestCache(root/original.DIRECTORY/'cache')
        events = read_jsonl(str(prior_cache.log(key179).path)+'.attempts.jsonl')
        if ([e['event'] for e in events] != ['started', 'uncertain']
                or list(prior_cache.log(key179).records()) or state['requests'][key179]['status'] != 'unresolved'):
            raise ValueError('position179 original unknown execution changed')
        new_unknown = {k for k, e in state['requests'].items()
                       if k not in old['preserved_budget_entries'] and e['status'] not in ('settled', 'rejected')}
        if new_unknown != {key179}: raise ValueError('unexpected other unresolved expansion execution')
        existing = {}; candidates = {}; costs = []
        for position, item in enumerate(items, 1):
            req = generator_request(item, parent['parent_study_id']); key = runner.cache_key(req)
            if position <= 178:
                call, _ = cache_for(root, position).get(runner, req)
                estimate = base.checked(call, runner, req, parent['config']); costs.append(estimate)
                existing[key] = digest(asdict(call)); candidates[key] = digest(candidate(item, req, call))
                if key in old['existing_response_hashes'] and existing[key] != old['existing_response_hashes'][key]:
                    raise ValueError('previously sealed reply changed')
                if position > 50:
                    records, _, starts, successes = transport.history(original.cache_for(root, position), key)
                    if len(successes) != 1 or len(starts) != 1: raise ValueError('unexpected prior expansion attempts')
                    # Earlier authorized retries have separate ledger keys. Match
                    # the successful execution, preserving its original unknown entry.
                    matches = [(ledger_key, entry) for ledger_key, entry in state['requests'].items()
                               if entry.get('execution_id') == call.meta['execution_id']]
                    if len(matches) != 1:
                        raise ValueError('prior successful execution lacks a unique budget entry')
                    ledger_key, e = matches[0]
                    if (e['status'] != 'settled' or e.get('model') != runner.model
                            or e.get('role') != 'generator'
                            or e.get('canonical_request_key', ledger_key) != key
                            or abs(e['estimated_usd']-estimate['estimated_usd']) > 1e-12):
                        raise ValueError('prior178 usage mismatch')
            elif position > 179:
                records, attempts, _, _ = transport.history(prior_cache, key)
                if records or attempts or key in state['requests']: raise ValueError('remaining item already attempted')
            if position > 178:
                records, attempts, _, _ = transport.history(cache_for(root, position), key)
                if records or attempts: raise ValueError('new continuation cache is not empty')
        total = sum(c['estimated_usd'] for c in costs)
        if abs(total-parent['original_gptoss_estimated_usd']-budget.summary()['estimated_usd']) > 1e-10:
            raise ValueError('prior178 total usage does not reconcile')
        if retry22.committed(state)+max(old['reservations_usd'][178:]) > budget.limit:
            raise ValueError('unchanged budget cannot reserve next execution')
        protected = {str(p): file_digest(p) for p in root.rglob('*') if p.is_file()
                     and DIRECTORY not in p.parts and p != budget.path and not p.name.endswith('.lock')}
        protected.update({str(p): file_digest(p) for p in pilot.REPORT.rglob('*') if p.is_file()})
        plan = seal({'parent_expansion_id': old['expansion_id'], 'sources': identity,
            'implementation_sha256': file_digest(__file__), 'runner': runner.identity,
            'item_ids': old['item_ids'], 'request_keys': old['request_keys'],
            'request_body_hashes': old['request_body_hashes'], 'reservations_usd': old['reservations_usd'],
            'partition_counts': old['partition_counts'], 'preserved_files_sha256': protected,
            'existing_response_hashes': existing, 'existing_candidate_hashes': candidates,
            'preserved_budget_entries': state['requests'], 'budget_binding': state['binding'],
            'budget_before': budget.summary(), 'received_before_estimated_usd': total,
            'initial_shared_committed_usd': retry22.committed(state),
            'additional_cost_projection_usd': total/178*267,
            'projection_basis': 'Observed178 mean token use; not guaranteed. Per-request reservation under unchanged USD1 cap.',
            'reused_responses': 178, 'additional_requests': 267, 'target': 445,
            'authorized_unknown_retry_position': 179, 'original179_attempts': 1,
            'maximum_new_attempts_position179': 2, 'previous_execution_finished_at': events[-1]['created'],
            'minimum_interval_seconds': 30, 'spacing_basis': 'after previous attempt completes',
            'backoff_seconds': [60, 120], 'maximum_attempts_per_request': 3,
            'authorization': '2026-10-18 user requested continuing the remaining questions after the reported position179 unknown execution: retry179 then continue remaining items; preserve178 replies, settings and budget; raw confidence; compare four generators after445 complete.',
            'confidence_calibrated': False, 'answer_keys_read': False, 'verifier_requests': 0,
            'confirmed_billing_usd': None}, 'expansion_id')
        base.immutable(out/'plan.json', plan)
    result = audit(root); base.immutable(out/'preflight.json', result)
    return result


def load(root):
    plan = json.loads((root/DIRECTORY/'plan.json').read_text()); unseal(plan, 'expansion_id')
    old, parent, items, runner, budget, identity = source(root)
    if (plan['implementation_sha256'] != file_digest(__file__) or plan['sources'] != identity
            or plan['parent_expansion_id'] != old['expansion_id'] or plan['runner'] != runner.identity
            or plan['item_ids'] != [i.item_id for i in items]):
        raise ValueError('sealed source or authorized continuation changed')
    for field in ('request_keys', 'request_body_hashes', 'reservations_usd', 'partition_counts'):
        if plan[field] != old[field]: raise ValueError('original request settings changed: '+field)
    for path, sha in plan['preserved_files_sha256'].items():
        if file_digest(path) != sha: raise ValueError('historical artifact changed: '+path)
    return plan, parent, items, runner, budget


def check_ledger(plan, budget, state):
    if state['binding'] != plan['budget_binding']:
        raise ValueError('shared budget binding changed')
    for key, entry in plan['preserved_budget_entries'].items():
        if state['requests'].get(key) != entry:
            raise ValueError('historical budget entry changed')
    allowed = set(plan['preserved_budget_entries']) | {entry_key(plan, p) for p in range(179, 446)}
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
    for position, item in enumerate(items, 1):
        req = generator_request(item, parent['parent_study_id']); key = runner.cache_key(req)
        cache = cache_for(root, position)
        if position > 178:
            records, _, starts, successes = transport.history(cache, key)
            attempts += len(starts); records_all.extend(records)
            rejections.update(str(r['meta']['http_status']) for r in records if r.get('error'))
            if len(starts) > max_new_attempts(position):
                raise ValueError('HTTP429 retry allowance exceeded')
            by_execution = {r['meta']['execution_id']: r for r in records}
            for number, start in enumerate(starts[1:], 1):
                before = by_execution.get(starts[number-1]['execution_id'])
                if not before or before.get('error') != 'http_429':
                    raise ValueError('retry without an explicit HTTP429')
                gap = start['created'] - before['created'] - before['latency_s']; retry_gaps.append(gap)
                if gap < backoff_seconds(position, number) - .01:
                    raise ValueError('HTTP429 backoff violated')
            entry = state['requests'].get(entry_key(plan, position))
            if entry and entry['status'] not in ('settled', 'rejected'):
                raise ValueError('new unknown, unpriced, or in-flight request')
            if not successes:
                if bool(entry) != bool(starts) or (entry and (entry['status'] != 'rejected' or entry['estimated_usd'] != 0)):
                    raise ValueError('missing response does not reconcile with budget')
                continue
        call, _ = cache.get(runner, req)
        estimate = base.checked(call, runner, req, parent['config']); costs.append(estimate)
        if position <= 178:
            if digest(asdict(call)) != plan['existing_response_hashes'][key]:
                raise ValueError('preserved178 reply changed')
            if digest(candidate(item, req, call)) != plan['existing_candidate_hashes'][key]:
                raise ValueError('preserved178 candidate or raw confidence changed')
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
    if ordered and ordered[0]['created']-plan['previous_execution_finished_at'] < 29.99:
        raise ValueError('first resumed attempt was too early')
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
        'parent_expansion_id': plan['parent_expansion_id'], 'target': 445, 'responses': len(rows), 'valid': sum(r['valid'] for r in rows),
        'invalid': dict(Counter(r['failure'] for r in rows if r['failure'])), 'partitions': partitions,
        'original_178_responses_and_candidates_unchanged': True, 'old_unknown_reservations_preserved': True,
        'reused_pilot_responses': 50, 'reused_at_resume': 178, 'additional_responses': len(rows)-178,
        'responses_since_original_pilot': len(rows)-50, 'remaining_requests': 445-len(rows),
        'additional_cost_projection_usd': plan['additional_cost_projection_usd'],
        'new_provider_attempts': attempts, 'original179_unknown_attempts': 1, 'new_rejections': dict(rejections),
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
            last_end = max(last_end, plan['previous_execution_finished_at'])
            for position, item in enumerate(items[178:], 179):
                req = generator_request(item, plan['expansion_id']); canonical_key = runner.cache_key(req)
                active = entry_key(plan, position)
                try:
                    cache.get(runner, req)
                    continue
                except OfflineCacheMiss:
                    pass
                while True:
                    records, _, starts, _ = transport.history(cache, canonical_key)
                    if len(starts) >= max_new_attempts(position):
                        raise ValueError('explicit HTTP429 retry allowance exhausted')
                    ready = max((r['created']+r['latency_s'] for r in records), default=0.)
                    if starts:
                        ready += backoff_seconds(position, len(starts))
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
                            'model': runner.model, 'role': 'generator', 'expansion_id': plan['expansion_id'],
                            'canonical_request_key': canonical_key, 'position': position}
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
    frozen = seal({'expansion_id': plan['expansion_id'], 'parent_expansion_id': plan['parent_expansion_id'], 'dataset_id': plan['sources']['dataset_id'],
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
        'All178 received replies, including two malformed final JSON and two abort replies, were reused unchanged; 267 new replies.\n'
        'Position179 is an explicitly authorized re-execution; original unknown attempt/reserve retained.\n'
        'Original model, prompt, medium reasoning, max_tokens 32768, parser and frozen prices.\n'
        '30 seconds after completion; explicit HTTP429 only: 60/120-second backoff, three attempts maximum.\n'
        'No answer-key reads, accuracy scoring, calibration, verifier requests or repairs in this stage.\n'
        'Usage estimates target llm-applications-490420. These are not confirmed billing.\n'
        'Shared USD1 cap unchanged. All six historical unknown reserves remain; actual total billing is unknown.\n'
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
