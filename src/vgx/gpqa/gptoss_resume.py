"""User-authorized 30-second resume amendment; reuse the sealed collector/cache.

Only absent development requests may be sent. At most three attempts in total
per request, including the historical explicit 429. Backoffs are 60/120 seconds
from rejection completion. No retry for any other error or malformed reply.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, RequestCache
from vgx.common.storage import atomic_json, digest, file_digest, read_jsonl
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.sequential import seal, unseal

ROOT = Path('results/gpqa_gptoss_continuation_20261017')
AMENDMENT = 'resume_30s_20261017'
INTERVAL = 30.0
BACKOFF = (60.0, 120.0)


def history(cache, key):
    path = cache.log(key).path
    rows = read_jsonl(path) if path.exists() else []
    attempts = read_jsonl(str(path)+'.attempts.jsonl')
    if any(r.get('key') != key for r in rows + attempts):
        raise ValueError('request log identity changed')
    starts = [a for a in attempts if a['event'] == 'started']
    successes = [r for r in rows if r.get('error') is None]
    if len(successes) > 1:
        raise ValueError('multiple successful responses for one request')
    if not successes:
        rejected = {a['execution_id']: a for a in attempts if a['event'] == 'rejected'}
        if any(a['execution_id'] not in rejected for a in starts):
            raise ValueError('unknown provider outcome; no retry')
        if any(a['http_status'] != 429 for a in rejected.values()):
            raise ValueError('only explicit HTTP 429 is eligible for retry')
        if len(rows) != len(starts) or len(rejected) != len(starts):
            raise ValueError('incomplete rejection record; no retry')
        if any(r.get('error') != 'http_429' or r.get('meta', {}).get('http_status') != 429 for r in rows):
            raise ValueError('unexpected non-429 error')
        if {r['meta']['execution_id'] for r in rows} != set(rejected):
            raise ValueError('rejection record mismatch')
    return rows, attempts, starts, successes


def prepare(root):
    parent, items, _, runner = base.load(root)
    cache = RequestCache(root/'cache')
    out = root/AMENDMENT
    if (out/'plan.json').exists():
        return audit(root)
    status = json.loads((root/'status.json').read_text())
    if (status['stage'], status['http_status'], status['responses']) != ('blocked', 429, 18):
        raise ValueError('unexpected previous stop; review required')
    existing = {}; missing = []; preserved = {}; prefixes = {}
    started_before = {}; reservation = 0.0
    budget = base.cache_for(root, parent['config'])
    for item, key in zip(items, parent['request_keys']):
        rows, attempts, starts, successes = history(cache, key)
        started_before[key] = len(starts)
        if successes:
            call, _ = cache.get(runner, generator_request(item, parent['parent_study_id']))
            base.checked(call, runner, generator_request(item, parent['parent_study_id']), parent['config'])
            existing[key] = digest(asdict(call))
            for p in (cache.log(key).path, Path(str(cache.log(key).path)+'.attempts.jsonl')):
                if p.exists(): preserved[str(p)] = file_digest(p)
        else:
            if len(starts) >= 3: raise ValueError('retry allowance exhausted')
            missing.append(key)
            reservation += budget.reservation(runner, generator_request(item, parent['parent_study_id']))
            for p in (cache.log(key).path, Path(str(cache.log(key).path)+'.attempts.jsonl')):
                if p.exists(): prefixes[str(p)] = {'bytes':p.stat().st_size, 'sha256':file_digest(p)}
    before = budget.summary()
    if len(existing) != 18 or len(missing) != 32 or before['limit_usd'] != 1.0 or before['outstanding_reserved_usd']:
        raise ValueError('unexpected response count or budget state')
    if before['estimated_usd'] + reservation > before['limit_usd']:
        raise ValueError('remaining allowance insufficient')
    for p in Path('reports/gpqa_gptoss_continuation_20261017').rglob('*'):
        if p.is_file(): preserved[str(p)] = file_digest(p)
    # Snapshot previous status/ledger without creating an additional budget.json.
    base.immutable(out/'previous_status.json', status)
    base.immutable(out/'budget_before.json', json.loads(budget.path.read_text()))
    for name in ('previous_status.json', 'budget_before.json'):
        preserved[str(out/name)] = file_digest(out/name)
    plan = seal({'continuation_id':parent['continuation_id'],
        'parent_plan_sha256':file_digest(root/'plan.json'),
        'implementation_sha256':file_digest(__file__), 'existing_response_hashes':existing,
        'missing_request_keys':missing, 'attempt_counts_before':started_before,
        'preserved_files_sha256':preserved, 'append_only_prefixes':prefixes,
        'minimum_interval_seconds':INTERVAL, 'spacing_basis':'30 seconds after previous attempt completes',
        'backoff_seconds':list(BACKOFF), 'maximum_attempts_per_request':3,
        'historical_429_counts_toward_attempt_limit':True,
        'budget_before':before, 'budget_binding':budget.binding,
        'remaining_max_output_reservation_usd':reservation,
        'authorization':'2026-10-17: resume only remaining 32, preserve 18 responses; 30-second interval; explicit 429 only 60/120-second limited retry; unchanged budget; compare original 50 development questions offline after completion.',
        'confidence_calibrated':False, 'evaluation_requests':0, 'verifier_requests':0}, 'resume_id')
    base.immutable(out/'plan.json', plan)
    return audit(root)


def load(root):
    parent, items, groups, runner = base.load(root)
    out = root/AMENDMENT
    plan = json.loads((out/'plan.json').read_text()); unseal(plan, 'resume_id')
    if (plan['continuation_id'] != parent['continuation_id']
        or plan['parent_plan_sha256'] != file_digest(root/'plan.json')
        or plan['implementation_sha256'] != file_digest(__file__)
        or plan['minimum_interval_seconds'] != INTERVAL or plan['backoff_seconds'] != list(BACKOFF)):
        raise ValueError('sealed resume amendment changed')
    for path, expected in plan['preserved_files_sha256'].items():
        if file_digest(path) != expected: raise ValueError('preserved artifact changed: '+path)
    import hashlib
    for path, expected in plan['append_only_prefixes'].items():
        with Path(path).open('rb') as fh: prefix = fh.read(expected['bytes'])
        if len(prefix) != expected['bytes'] or hashlib.sha256(prefix).hexdigest() != expected['sha256']:
            raise ValueError('historical rejection record changed')
    cache = base.cache_for(root, parent['config'])
    if cache.binding != plan['budget_binding']:
        raise ValueError('budget binding changed')
    return plan, parent, items, runner, cache


def audit(root):
    plan, parent, items, runner, cache = load(root)
    rows = []; estimates = []; attempt_counts = []; new_starts = []; all_records = []; retry_gaps = []
    for item, key in zip(items, parent['request_keys']):
        records, attempts, starts, successes = history(cache, key)
        all_records.extend(records)
        if len(starts) > 3: raise ValueError('attempt limit exceeded')
        attempt_counts.append(len(starts))
        new_starts.extend(a for a in starts[plan['attempt_counts_before'][key]:])
        by_execution = {r.get('meta', {}).get('execution_id'):r for r in records}
        for index, start in enumerate(starts):
            if index == 0 or index < plan['attempt_counts_before'][key]: continue
            previous = by_execution.get(starts[index-1]['execution_id'])
            if previous is None or previous.get('error') != 'http_429':
                raise ValueError('retry without a confirmed 429 rejection')
            gap = start['created'] - previous['created'] - previous['latency_s']
            retry_gaps.append(gap)
            if gap < BACKOFF[index-1] - 0.01: raise ValueError('429 backoff violated')
        if key not in plan['existing_response_hashes'] and key not in plan['missing_request_keys']:
            raise ValueError('unplanned request')
        if not successes: continue
        call, _ = RequestCache.get(cache, runner, generator_request(item, parent['parent_study_id']))
        if key in plan['existing_response_hashes'] and digest(asdict(call)) != plan['existing_response_hashes'][key]:
            raise ValueError('existing response changed')
        estimates.append(base.checked(call, runner, generator_request(item, parent['parent_study_id']), parent['config']))
        parsed = base.original.parse_call(call)
        rows.append({'valid':parsed.ok, 'failure':parsed.failure})
    budget = cache.summary()
    if budget['outstanding_reserved_usd'] or budget['estimated_usd'] > budget['limit_usd']:
        raise ValueError('unknown usage or exceeded budget')
    if abs(sum(e['estimated_usd'] for e in estimates) - parent['original_gptoss_estimated_usd'] - budget['estimated_usd']) > 1e-10:
        raise ValueError('cost reconciliation failed')
    ordered = sorted(all_records, key=lambda r:r['created'])
    new_ids = {a['execution_id'] for a in new_starts}
    gaps = []
    for i, record in enumerate(ordered):
        if i and record.get('meta', {}).get('execution_id') in new_ids:
            previous = ordered[i-1]
            gap = record['created'] - previous['created'] - previous['latency_s']
            gaps.append(gap)
            if gap < INTERVAL - 0.01: raise ValueError('resume request interval violated')
    return {'stage':'complete' if len(rows)==50 else 'dry_run', 'resume_id':plan['resume_id'],
        'continuation_id':parent['continuation_id'], 'responses':len(rows), 'target':50,
        'valid':sum(r['valid'] for r in rows), 'invalid':dict(Counter(r['failure'] for r in rows if r['failure'])),
        'reused_at_resume':18, 'new_responses_since_resume':len(rows)-18,
        'remaining_requests':50-len(rows), 'new_provider_attempts_since_resume':len(new_starts),
        'maximum_attempts_observed':max(attempt_counts),
        'resume_minimum_seconds_after_previous_attempt':min(gaps) if gaps else None,
        'retry_backoffs_checked_n':len(retry_gaps),
        'retry_backoff_checks_passed':True,
        'rejected_http_statuses':dict(Counter(str(r['meta']['http_status']) for r in all_records if r.get('error'))),
        'budget':budget, 'resume_increment_estimated_usd':budget['estimated_usd']-plan['budget_before']['estimated_usd'],
        'all_gptoss_response_estimate_usd':sum(e['estimated_usd'] for e in estimates),
        'existing_18_responses_unchanged':True, 'source_and_request_identity_checks_passed':True,
        'confidence_calibrated':False, 'answer_keys_read':False, 'evaluation_requests':0, 'verifier_requests':0}


def collect(root, allow_api=False):
    plan, parent, items, runner, cache = load(root); out = root/AMENDMENT
    if not allow_api: return audit(root)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage']=='blocked':
        raise ValueError('blocked resume requires review')
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = audit(root); initial_count = state['responses']
        def snapshot(stage, **extra):
            state.update(stage=stage, updated_at_unix=time.time(), new_calls_this_invocation=state['responses']-initial_count, **extra)
            atomic_json(out/'status.json',state)
        try:
            if base.budget_inventory()['committed_estimate_and_reservations_usd'] + base.CAP - cache.summary()['estimated_usd'] > 200:
                raise ValueError('overall budget allowance insufficient')
            atomic_json(out/'billing_check.json',base.check_billing(parent['config']['project']))
            snapshot('running')
            completed_times = [r['created']+r['latency_s'] for key in parent['request_keys'] for r in history(cache,key)[0]]
            last_finished = max(completed_times, default=0.0)
            for position, (item,key) in enumerate(zip(items,parent['request_keys']),1):
                request=generator_request(item,parent['parent_study_id'])
                try:
                    RequestCache.get(cache,runner,request)
                    continue
                except OfflineCacheMiss:
                    if key not in plan['missing_request_keys']: raise ValueError('original cached response missing')
                while True:
                    records, _, starts, successes = history(cache,key)
                    if successes: break
                    if len(starts)>=3: raise ValueError('explicit 429 retry allowance exhausted')
                    retry_ready = max((r['created']+r['latency_s'] for r in records),default=0.0)
                    if starts: retry_ready += BACKOFF[len(starts)-1]
                    deadline = max(last_finished+INTERVAL, retry_ready)
                    snapshot('waiting',current_position=position,next_attempt_number=len(starts)+1,not_before_unix=deadline)
                    while time.time()<deadline: time.sleep(max(0.0,min(30.0,deadline-time.time())))
                    snapshot('requesting')
                    try:
                        call,_=cache.get(runner,request,allow_api=True)
                        last_finished=time.time()
                        base.checked(call,runner,request,parent['config'])
                        state=audit(root); snapshot('collecting')
                        break
                    except ProviderRejectedError as exc:
                        last_finished=time.time()
                        if exc.status!=429: raise
                        state=audit(root)
                        snapshot('backoff',last_http_status=429,current_position=position)
                        # Durable attempt records, including the old rejection,
                        # determine whether another iteration is permitted.
            state=audit(root)
            if state['responses']!=50: raise ValueError('incomplete collection')
            snapshot('complete')
            # Original blocked status is retained in previous_status.json and
            # the old report. Let the unchanged offline scorer read completion.
            atomic_json(root/'status.json',state)
            return state
        except Exception as exc:
            state['budget']=cache.summary()
            snapshot('blocked',error_type=type(exc).__name__,http_status=getattr(exc,'status',None))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','collect','audit'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare': value=prepare(args.root)
    elif args.command=='audit': value=audit(args.root)
    else: value=collect(args.root,args.allow_api)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__': main()
