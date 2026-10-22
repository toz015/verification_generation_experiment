"""Finish the GPT-OSS-120B 445-input GPQA generator collection after the position292 block.

Continues retry179_ledgerfix without editing it. All 291 received replies (including
malformed ones) are reused unchanged; positions 292-445 are collected with the same
runner, prompt, parser, prices, 30-second spacing and explicit-HTTP429 backoff.

Unknown provider outcomes (client timeout or connection loss) follow the owner's
2026-10-19 instruction to continue instead of stopping: the unknown attempt keeps its
reservation forever, and the same input is re-executed once in a fresh cache
directory (as every earlier GPT-OSS unknown was, by explicit authorization). A second
unknown on the same input is recorded as a generator service failure (no answer,
invalid candidate) and collection continues. Nothing is repaired or regenerated.
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

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, RequestCache, UncertainRequestError
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa import gptoss_retry22 as retry22
from vgx.gpqa import gptoss_retry179_ledgerfix as original
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.generator_study import digest_text
from vgx.gpqa.prompt import SYSTEM
from vgx.gpqa.sequential import seal, unseal

ROOT = original.ROOT
DIRECTORY = 'finish445_20261020'
REPORT = Path('reports/gpqa_gptoss_finish445_20261020')
FIRST = 292            # first position collected here; 1..291 are reused
HISTORICAL_UNKNOWN = {292: 1}  # unknown attempts recorded before this module
MAX_UNKNOWN = 2        # second unknown on one input -> service failure
MAX_ATTEMPTS_PER_TRY = 3
SPACING = 30
BACKOFF = (60, 120)
UNKNOWN_WAIT = 120
AUTHORIZATION = ('2026-10-20 owner asked to finish the GPT-OSS-120B GPQA experiment; 2026-10-19 owner '
                 'instruction for similar situations: continue instead of stopping. Unknown outcomes are '
                 're-executed once, a second unknown is recorded as a service failure.')


def try_cache(root, position, k):
    if position < FIRST:
        return original.cache_for(root, position)
    if k < HISTORICAL_UNKNOWN.get(position, 0):
        return original.cache_for(root, position)
    return RequestCache(root/DIRECTORY/'cache'/f'try{k}')


def entry_key(plan, position, k, attempt):
    return f'finish445:{position}:{k}:{attempt}:'+plan['expansion_id']


def try_history(cache, key):
    """(rows, starts, success, outcome) where outcome is success / unknown / rejected429 / none."""
    path = cache.log(key).path
    rows = read_jsonl(path) if path.exists() else []
    attempts = read_jsonl(str(path)+'.attempts.jsonl', tolerate_tail=True)
    if any(r.get('key') != key for r in rows+attempts):
        raise ValueError('request log identity changed')
    starts = [a for a in attempts if a['event'] == 'started']
    successes = [r for r in rows if r.get('error') is None]
    if len(successes) > 1:
        raise ValueError('multiple successful responses')
    closed = {a['execution_id'] for a in attempts if a['event'] in ('rejected', 'completed')}
    unknown = [a for a in starts if a['execution_id'] not in closed]
    errors = [r for r in rows if r.get('error')]
    if any(r.get('error') != 'http_429' for r in errors):
        raise ValueError('non-429 provider error requires review')
    outcome = 'success' if successes else ('unknown' if unknown else ('rejected429' if starts else 'none'))
    return rows, starts, (successes[0] if successes else None), outcome


def prepare(root):
    out = root/DIRECTORY
    if (out/'plan.json').exists():
        return audit(root)
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old, parent, items, runner, budget = original.load(root)
        stopped = json.loads((root/original.DIRECTORY/'status.json').read_text())
        if (stopped['stage'], stopped['responses'], stopped['error_type'], stopped['current_position']) != (
                'blocked', FIRST-1, 'UncertainRequestError', FIRST):
            raise ValueError('expected the position292 unknown-outcome block with 291 replies')
        state = budget._load(); original.check_ledger(old, budget, state)
        if budget.limit != 1.0:
            raise ValueError('shared budget cap changed')
        key292 = old['request_keys'][FIRST-1]
        _, starts, success, outcome = try_history(original.cache_for(root, FIRST), key292)
        if outcome != 'unknown' or success or len(starts) != 1:
            raise ValueError('position292 history differs from one unknown attempt')
        existing = {}
        for position, item in enumerate(items, 1):
            req = generator_request(item, parent['parent_study_id']); key = runner.cache_key(req)
            if key != old['request_keys'][position-1]:
                raise ValueError('request identity changed')
            if position < FIRST:
                call, _ = original.cache_for(root, position).get(runner, req)
                base.checked(call, runner, req, parent['config'])
                existing[key] = digest(asdict(call))
            elif position > FIRST:
                rows, starts, _, _ = try_history(original.cache_for(root, position), key)
                if rows or starts or key in state['requests']:
                    raise ValueError('remaining item already attempted')
        inventory = original.budget_inventory()
        if inventory['committed_estimate_and_reservations_usd'] + budget.limit - retry22.committed(state) > 200:
            raise ValueError('overall authorized budget insufficient')
        protected = {str(p): file_digest(p) for p in root.rglob('*') if p.is_file()
                     and DIRECTORY not in p.parts and p != budget.path and not p.name.endswith('.lock')}
        last_end = max(r['created']+r['latency_s'] for position in range(1, FIRST+1)
                       for r in original.cache_for(root, position).log(old['request_keys'][position-1]).records())
        plan = seal({'parent_expansion_id': old['expansion_id'], 'sources': old['sources'],
            'implementation_sha256': file_digest(__file__), 'runner': runner.identity,
            'item_ids': old['item_ids'], 'request_keys': old['request_keys'],
            'request_body_hashes': old['request_body_hashes'], 'reservations_usd': old['reservations_usd'],
            'partition_counts': old['partition_counts'], 'preserved_files_sha256': protected,
            'existing_response_hashes': existing, 'preserved_budget_entries': state['requests'],
            'budget_binding': state['binding'], 'budget_before': budget.summary(),
            'global_budget_before': inventory, 'reused_responses': FIRST-1, 'target': 445,
            'first_new_position': FIRST, 'historical_unknown_attempts': {str(k): v for k, v in HISTORICAL_UNKNOWN.items()},
            'unknown_policy': {'reexecute_once': True, 'max_unknown_per_input': MAX_UNKNOWN,
                               'after_max': 'generator service failure, counted as invalid', 'wait_seconds': UNKNOWN_WAIT},
            'minimum_interval_seconds': SPACING, 'backoff_seconds': list(BACKOFF),
            'maximum_attempts_per_try': MAX_ATTEMPTS_PER_TRY, 'previous_execution_finished_at': last_end,
            'authorization': AUTHORIZATION, 'confidence_calibrated': False, 'answer_keys_read': False,
            'verifier_requests': 0, 'confirmed_billing_usd': None}, 'expansion_id')
        base.immutable(out/'plan.json', plan)
    result = audit(root); base.immutable(out/'preflight.json', result)
    return result


def load(root):
    plan = json.loads((root/DIRECTORY/'plan.json').read_text()); unseal(plan, 'expansion_id')
    old, parent, items, runner, budget = original.load(root)
    if (plan['implementation_sha256'] != file_digest(__file__) or plan['parent_expansion_id'] != old['expansion_id']
            or plan['runner'] != runner.identity or plan['item_ids'] != [i.item_id for i in items]
            or plan['request_keys'] != old['request_keys']):
        raise ValueError('sealed source or continuation changed')
    for path, sha in plan['preserved_files_sha256'].items():
        if file_digest(path) != sha:
            raise ValueError('historical artifact changed: '+path)
    return plan, parent, items, runner, budget


def check_ledger(plan, budget, state):
    if state['binding'] != plan['budget_binding']:
        raise ValueError('shared budget binding changed')
    for key, entry in plan['preserved_budget_entries'].items():
        if state['requests'].get(key) != entry:
            raise ValueError('historical budget entry changed')
    for key in set(state['requests']) - set(plan['preserved_budget_entries']):
        if not key.startswith('finish445:') or not key.endswith(plan['expansion_id']):
            raise ValueError('unplanned shared budget entry')
    if retry22.committed(state) > budget.limit:
        raise ValueError('shared budget exceeded')


def position_outcome(root, plan, position):
    """Walk the tries of one input: ('success', call-row, tries) / ('service_failure', None, tries) / ('pending', None, tries)."""
    key = plan['request_keys'][position-1]
    tries, unknown = [], 0
    for k in range(MAX_UNKNOWN+1):
        rows, starts, success, outcome = try_history(try_cache(root, position, k), key)
        tries.append({'try': k, 'attempts': len(starts), 'outcome': outcome,
                      'historical': position >= FIRST and k < HISTORICAL_UNKNOWN.get(position, 0)})
        if outcome == 'success':
            return 'success', success, tries
        if outcome == 'unknown':
            unknown += 1
            if unknown >= MAX_UNKNOWN:
                return 'service_failure', None, tries
            continue
        return 'pending', None, tries
    return 'service_failure', None, tries


def candidate(item, req, call):
    if call is None:
        return asdict(FrozenCandidate(item.item_id, item.partition, item.subject, item.question, item.choices,
            None, None, False, 'service_failure_unknown_outcome', 'openai/gpt-oss-120b-maas', req.key,
            digest_text(req.prompt), digest_text(''), SYSTEM, None))
    return original.candidate(item, req, call)


def audit(root):
    plan, parent, items, runner, budget = load(root)
    state = budget._load(); check_ledger(plan, budget, state)
    rows, costs, failures, attempts = [], [], [], 0
    for position, item in enumerate(items, 1):
        req = generator_request(item, parent['parent_study_id']); key = runner.cache_key(req)
        if position < FIRST:
            call, _ = original.cache_for(root, position).get(runner, req)
            if digest(asdict(call)) != plan['existing_response_hashes'][key]:
                raise ValueError('preserved reply changed')
        else:
            outcome, row, tries = position_outcome(root, plan, position)
            attempts += sum(t['attempts'] for t in tries if not t['historical'])
            if outcome == 'pending':
                continue
            if outcome == 'service_failure':
                failures.append(position)
                rows.append({'partition': item.partition, 'valid': False, 'failure': 'service_failure_unknown_outcome',
                             'confidence': None, 'stop_reason': 'unknown'})
                continue
            call, _ = try_cache(root, position, len(tries)-1).get(runner, req)
        estimate = base.checked(call, runner, req, parent['config']); costs.append(estimate)
        parsed = base.original.parse_call(call)
        choice = (call.meta['provider_response'].get('choices') or [{}])[0]
        rows.append({'partition': item.partition, 'valid': parsed.ok, 'failure': parsed.failure,
                     'confidence': parsed.p_correct, 'stop_reason': str(choice.get('finish_reason'))})
    partitions = {}
    for partition in ('calibration', 'evaluation'):
        sel = [r for r in rows if r['partition'] == partition]
        probs = [r['confidence'] for r in sel if r['confidence'] is not None]
        partitions[partition] = {'responses': len(sel), 'valid': sum(r['valid'] for r in sel),
            'mean_raw_confidence': sum(probs)/len(probs) if probs else None,
            'invalid': dict(Counter(r['failure'] for r in sel if r['failure']))}
    summary = budget.summary()
    return {'stage': 'complete' if len(rows) == 445 else 'dry_run', 'expansion_id': plan['expansion_id'],
            'target': 445, 'responses': len(rows), 'valid': sum(r['valid'] for r in rows),
            'invalid': dict(Counter(r['failure'] for r in rows if r['failure'])), 'partitions': partitions,
            'service_failure_positions': failures, 'new_provider_attempts': attempts,
            'reused_responses': FIRST-1, 'remaining_requests': 445-len(rows), 'budget': summary,
            'all_received_gptoss_estimated_usd': sum(c['estimated_usd'] for c in costs),
            'input_tokens': sum(c['usage']['input_tokens'] for c in costs),
            'output_tokens_including_reasoning': sum(c['usage']['output_tokens'] for c in costs),
            'confirmed_billing_usd': None, 'confidence_calibrated': False, 'answer_keys_read': False,
            'accuracy_computed': False, 'verifier_requests': 0}


def collect(root, allow_api=False):
    if not allow_api:
        return audit(root)
    out = root/DIRECTORY
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, parent, items, runner, budget = load(root)
        status = audit(root); active = None

        def snapshot(stage, **extra):
            status.update(stage=stage, updated_at_unix=time.time(), budget=budget.summary(), **extra)
            atomic_json(out/'status.json', status)

        def ledger(key, **fields):
            with file_lock(str(budget.path)+'.lock'):
                state = budget._load(); check_ledger(plan, budget, state)
                if fields.get('status') == 'reserved':
                    if retry22.committed(state)+fields['reserved_usd'] > budget.limit:
                        raise ValueError('unchanged shared budget cannot reserve next request')
                    state['requests'][key] = fields
                else:
                    state['requests'][key].update(fields)
                atomic_json(budget.path, state)
        try:
            inventory = original.budget_inventory(); atomic_json(out/'budget_inventory.json', inventory)
            if inventory['committed_estimate_and_reservations_usd'] + budget.limit - retry22.committed(budget._load()) > 200:
                raise ValueError('overall authorized budget insufficient')
            atomic_json(out/'billing_check.json', base.check_billing(parent['config']['project']))
            last_end = plan['previous_execution_finished_at']
            for position, item in enumerate(items[FIRST-1:], FIRST):
                req = generator_request(item, plan['expansion_id']); key = runner.cache_key(req)
                while True:
                    outcome, _, tries = position_outcome(root, plan, position)
                    if outcome != 'pending':
                        break
                    k = len(tries)-1; cur = tries[-1]
                    cache = try_cache(root, position, k)
                    rows, starts, _, _ = try_history(cache, key)
                    if len(starts) >= MAX_ATTEMPTS_PER_TRY:
                        raise ValueError('explicit HTTP429 retry allowance exhausted')
                    ready = max((r['created']+r['latency_s'] for r in rows), default=0.)
                    if starts:
                        ready += BACKOFF[len(starts)-1]
                    if k > 0 and not starts:
                        ready = max(ready, last_end+UNKNOWN_WAIT)
                    deadline = max(last_end+SPACING, ready)
                    snapshot('waiting', current_position=position, current_try=k, not_before_unix=deadline)
                    while time.time() < deadline:
                        time.sleep(max(0, min(30, deadline-time.time())))
                    active = entry_key(plan, position, k, len(starts))
                    ledger(active, status='reserved', reserved_usd=plan['reservations_usd'][position-1],
                           model=runner.model, role='generator', expansion_id=plan['expansion_id'],
                           canonical_request_key=key, position=position, try_index=k)
                    snapshot('requesting', current_position=position, current_try=k)
                    try:
                        call, _ = cache.get(runner, req, allow_api=True)
                    except ProviderRejectedError as exc:
                        last_end = time.time()
                        ledger(active, status='rejected', estimated_usd=0.)
                        if exc.status != 429:
                            raise
                        continue
                    except UncertainRequestError:
                        last_end = time.time()
                        ledger(active, status='unresolved', estimated_usd=None)  # reservation kept
                        status = audit(root); snapshot('unknown_outcome_continuing', current_position=position)
                        continue
                    last_end = time.time()
                    estimate = base.checked(call, runner, req, parent['config'])
                    if estimate['estimated_usd'] > plan['reservations_usd'][position-1]:
                        raise ValueError('request exceeded reserved cost')
                    ledger(active, status='settled', estimated_usd=estimate['estimated_usd'],
                           issues=estimate['issues'], execution_id=call.meta['execution_id'])
                    active = None
                status = audit(root); snapshot('collecting')
            status = audit(root)
            if status['responses'] != 445:
                raise ValueError('collection incomplete')
            finalize(root); snapshot('complete')
            return status
        except Exception as exc:
            if active:
                with file_lock(str(budget.path)+'.lock'):
                    state = budget._load()
                    if state['requests'].get(active, {}).get('status') == 'reserved':
                        state['requests'][active].update(status='unresolved', estimated_usd=None)
                        atomic_json(budget.path, state)
            snapshot('blocked', error_type=type(exc).__name__, error=str(exc) if isinstance(exc, ValueError) else None,
                     http_status=getattr(exc, 'status', None))
            return status


def finalize(root):
    status = audit(root)
    if status['stage'] != 'complete':
        raise ValueError('all 445 inputs need a reply or a recorded service failure')
    plan, parent, items, runner, _ = load(root); out = root/DIRECTORY
    candidates = []
    for position, item in enumerate(items, 1):
        req = generator_request(item, parent['parent_study_id'])
        if position < FIRST:
            call, _ = original.cache_for(root, position).get(runner, req)
        else:
            outcome, _, tries = position_outcome(root, plan, position)
            call = try_cache(root, position, len(tries)-1).get(runner, req)[0] if outcome == 'success' else None
        candidates.append(candidate(item, req, call))
    frozen = seal({'expansion_id': plan['expansion_id'], 'parent_expansion_id': plan['parent_expansion_id'],
                   'dataset_id': plan['sources']['dataset_id'], 'groups': {'gpt_oss_120b': candidates}}, 'bundle_id')
    base.immutable(out/'frozen_candidates.json', frozen)
    result = {**status, 'bundle_id': frozen['bundle_id']}
    base.immutable(out/'completion_report.json', result)
    return result


def report(root):
    result = finalize(root); out = root/DIRECTORY
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in ('plan.json', 'status.json', 'completion_report.json', 'billing_check.json', 'budget_inventory.json'):
        if (out/name).exists():
            shutil.copyfile(out/name, REPORT/name)
    (REPORT/'README.txt').write_text(
        'GPT-OSS-120B GPQA generator collection completed for all 445 inputs (249 development, 196 Diamond).\n'
        'Positions 1-291 reused unchanged; 292-445 collected with the same runner, prompt, parser and prices.\n'
        'Unknown outcomes were re-executed once; a second unknown counts as a service failure. Usage estimates only.\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect', 'audit', 'report'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    result = collect(args.root, args.allow_api) if args.command == 'collect' else globals()[args.command](args.root)
    print(json.dumps({k: v for k, v in result.items() if k != 'partitions'}, indent=2, allow_nan=False), flush=True)
    if result.get('stage') == 'blocked':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
