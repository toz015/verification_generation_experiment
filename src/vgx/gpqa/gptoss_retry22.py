"""One explicitly authorized re-execution of pilot position 22.

Preserve all previous request/attempt files and ledger entries. Record this
execution separately while charging its reservation to the same USD1 budget.
The canonical model/prompt request identity is unchanged. No automatic retry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa import gptoss_resume as previous
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.sequential import seal, unseal

ROOT=previous.ROOT
DIRECTORY='retry22_20261017'


def committed(state):
    return sum(e['estimated_usd'] if e['status'] in ('settled','rejected') else e['reserved_usd']
               for e in state['requests'].values())


def prepare(root):
    out=root/DIRECTORY
    if (out/'plan.json').exists():
        p,*_=load(root)
        return {'stage':'prepared','retry_id':p['retry_id'],'existing_plan':True}
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,parent,items,runner,cache=previous.load(root)
        item=items[21]; request=generator_request(item,parent['parent_study_id']); key=runner.cache_key(request)
        if item.partition!='calibration' or key!=parent['request_keys'][21]: raise ValueError('unexpected item')
        events=read_jsonl(str(cache.log(key).path)+'.attempts.jsonl')
        if list(cache.log(key).records()) or [e['event'] for e in events]!=['started','uncertain']:
            raise ValueError('original position22 outcome/history changed')
        state=cache._load()
        if state['requests'][key]['status']!='unresolved': raise ValueError('original reserve must remain unresolved')
        status=json.loads((root/'skip22_20261017/status.json').read_text())
        if (status['stage'],status['responses'],status['current_position'])!=('blocked',47,49):
            raise ValueError('unexpected latest collection state')
        reserve=cache.reservation(runner,request)
        if committed(state)+reserve>base.CAP: raise ValueError('same USD1 budget cannot reserve this attempt')
        protected={str(p):file_digest(p) for p in root.rglob('*') if p.is_file()
            and DIRECTORY not in p.parts and p!=cache.path and not p.name.endswith('.lock')}
        plan=seal({'continuation_id':parent['continuation_id'],'position':22,'request_key':key,
            'item_id':item.item_id,'implementation_sha256':file_digest(__file__),
            'runner_identity':runner.identity,'request_body_digest':digest(runner.request_body(request)),
            'protected_files_sha256':protected,'preserved_budget_entries':state['requests'],
            'budget_binding':state['binding'],'budget_before':cache.summary(),
            'request_reserved_usd':reserve,'maximum_new_provider_attempts':1,
            'original_received_responses':47,'original_received_estimated_usd':status['all_received_gptoss_estimated_usd'],
            'unknown_usage_reserved_before_usd':cache.summary()['outstanding_reserved_usd'],
            'authorization':'2026-10-17 user: retry original question22. One new explicitly authorized execution; retain original unknown request and reservation. No request for49/50 or other items.',
            'confidence_calibrated':False,'answer_keys_read':False,'evaluation_requests':0,'verifier_requests':0},'retry_id')
        base.immutable(out/'plan.json',plan)
        return {'stage':'prepared','retry_id':plan['retry_id'],'position':22,'new_requests':1,
            'request_reserved_usd':reserve,'shared_budget_limit_usd':base.CAP,
            'shared_committed_after_reservation_usd':committed(state)+reserve,
            'original_unknown_reservations_preserved':True}


def load(root):
    _,parent,items,runner,budget=previous.load(root)
    plan=json.loads((root/DIRECTORY/'plan.json').read_text()); unseal(plan,'retry_id')
    if plan['implementation_sha256']!=file_digest(__file__) or plan['continuation_id']!=parent['continuation_id']:
        raise ValueError('retry plan/source changed')
    for path,sha in plan['protected_files_sha256'].items():
        if file_digest(path)!=sha: raise ValueError('prior experiment artifact changed: '+path)
    request=generator_request(items[21],parent['parent_study_id'])
    if (runner.cache_key(request)!=plan['request_key'] or runner.identity!=plan['runner_identity']
        or digest(runner.request_body(request))!=plan['request_body_digest']):
        raise ValueError('provider request changed')
    return plan,parent,runner,request,budget


def ledger_check(plan,budget,state):
    if state['binding']!=plan['budget_binding']: raise ValueError('budget binding changed')
    for key,entry in plan['preserved_budget_entries'].items():
        if state['requests'].get(key)!=entry: raise ValueError('historical ledger entry changed')
    execution_key='authorized_retry:'+plan['retry_id']
    if set(state['requests'])-set(plan['preserved_budget_entries'])-{execution_key}:
        raise ValueError('unexpected additional budget entry')
    if committed(state)>budget.limit: raise ValueError('budget exceeded')
    return execution_key


def collect(root,allow_api=False):
    plan,parent,runner,request,budget=load(root); out=root/DIRECTORY
    cache=RequestCache(out/'cache'); key=plan['request_key']
    attempts_path=Path(str(cache.log(key).path)+'.attempts.jsonl')
    # Metadata links the re-execution; it changes neither prompt nor API body.
    request=replace(request,meta={**request.meta,'authorized_retry_id':plan['retry_id']})
    def snapshot(stage,**extra):
        value={'stage':stage,'retry_id':plan['retry_id'],'position':22,
            'original_47_responses_preserved':True,'confidence_calibrated':False,'answer_keys_read':False,
            'evaluation_requests':0,'verifier_requests':0,'budget':budget.summary(),
            'updated_at_unix':time.time(),**extra}
        atomic_json(out/'status.json',value); return value
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with file_lock(str(budget.path)+'.lock'):
            state=budget._load(); entry_key=ledger_check(plan,budget,state)
        try:
            try: call,_=cache.get(runner,request)
            except OfflineCacheMiss:
                if read_jsonl(attempts_path): raise ValueError('authorized attempt already sent; no further retry')
                if not allow_api: return snapshot('dry_run',planned_new_requests=1)
                atomic_json(out/'billing_check.json',base.check_billing(parent['config']['project']))
                with file_lock(str(budget.path)+'.lock'):
                    state=budget._load(); entry_key=ledger_check(plan,budget,state)
                    if entry_key in state['requests']: raise ValueError('retry already reserved; review before restarting')
                    if committed(state)+plan['request_reserved_usd']>budget.limit: raise ValueError('insufficient shared budget')
                    state['requests'][entry_key]={'status':'reserved','reserved_usd':plan['request_reserved_usd'],
                        'model':runner.model,'role':request.meta.get('role'),
                        'canonical_request_key':key,'authorized_retry_id':plan['retry_id']}
                    atomic_json(budget.path,state)
                snapshot('requesting',new_provider_attempts=1)
                call,_=cache.get(runner,request,allow_api=True)
            estimate=base.checked(call,runner,request,parent['config'])
            if estimate['estimated_usd']>plan['request_reserved_usd']: raise ValueError('retry exceeded reserved cost')
            events=read_jsonl(attempts_path)
            if [e['event'] for e in events]!=['started','completed']:
                raise ValueError('expected exactly one new completed attempt')
            with file_lock(str(budget.path)+'.lock'):
                state=budget._load(); entry_key=ledger_check(plan,budget,state)
                state['requests'][entry_key].update(status='settled',estimated_usd=estimate['estimated_usd'],
                    issues=estimate['issues'],execution_id=call.meta['execution_id'])
                ledger_check(plan,budget,state); atomic_json(budget.path,state)
            parsed=base.original.parse_call(call)
            base.immutable(out/'candidate.json',seal({'retry_id':plan['retry_id'],
                'item_id':plan['item_id'],'request_key':key,'answer':parsed.answer,'p_correct':parsed.p_correct,
                'valid':parsed.ok,'failure':parsed.failure,'response_digest':digest(asdict(call))},'candidate_id'))
            load(root)
            return snapshot('complete',received_reply=True,valid=parsed.ok,failure=parsed.failure,
                raw_confidence=parsed.p_correct,new_provider_attempts=1,
                finish_reason=(call.meta['provider_response'].get('choices') or [{}])[0].get('finish_reason'),
                input_tokens=estimate['usage']['input_tokens'],output_tokens_including_reasoning=estimate['usage']['output_tokens'],
                new_estimated_usd=estimate['estimated_usd'],
                all_received_gptoss_estimated_usd=plan['original_received_estimated_usd']+estimate['estimated_usd'],
                total_received_responses=48,unknown_usage_reserved_usd=plan['unknown_usage_reserved_before_usd'],
                old_unknown_attempt_still_unresolved=True,original_47_files_and_budget_entries_unchanged=True)
        except Exception as exc:
            # Never release a historical reserve. Settle only this execution.
            events=read_jsonl(attempts_path)
            starts=[e for e in events if e['event']=='started']
            rejected={e['execution_id'] for e in events if e['event']=='rejected'}
            with file_lock(str(budget.path)+'.lock'):
                state=budget._load(); entry_key=ledger_check(plan,budget,state)
                entry=state['requests'].get(entry_key)
                if entry is not None and entry['status']!='settled':
                    entry.update(status='rejected' if starts and all(e['execution_id'] in rejected for e in starts) else 'unresolved',
                                 estimated_usd=0.0 if starts and all(e['execution_id'] in rejected for e in starts) else None)
                    atomic_json(budget.path,state)
            snapshot('blocked',error_type=type(exc).__name__,http_status=getattr(exc,'status',None),
                     new_provider_attempts=len(starts),
                     received_reply=any(r.get('error') is None for r in cache.log(key).records()))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['prepare','collect'])
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--allow-api',action='store_true')
    a=p.parse_args(); value=prepare(a.root) if a.command=='prepare' else collect(a.root,a.allow_api)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__': main()
