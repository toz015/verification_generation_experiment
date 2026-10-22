"""Authorized gpt-oss-only continuation of the frozen 50-item generator pilot.

Reuse the five successful original requests. Preserve the blocked joint study;
inherit its request, parser and transport unchanged, with a separate USD 1 cap.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_digest, read_jsonl
from vgx.gpqa import replacement_generators as original
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.prompt import SYSTEM, build_verifier_prompt
from vgx.gpqa.qwen_signal_completion import budget_inventory
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing
from vgx.gpqa.sonnet_offline_score import summarize

PARENT = Path('results/gpqa_replacement_generators_20261014')
CAP = 1.0


def immutable(path, value):
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError('immutable artifact differs: '+str(path))
    else:
        atomic_json(path, value)


def implementation():
    base = Path(__file__).resolve().parents[1]
    return {**original.implementation(), **{p: file_digest(base/p) for p in (
        'gpqa/gptoss_continuation.py', 'gpqa/sonnet_offline_score.py',
        'gpqa/qwen_signal_completion.py', 'common/storage.py', 'common/llm.py')}}


def checked(call, runner, req, cfg):
    if (call.key != runner.cache_key(req) or call.model != runner.model
        or call.prompt != req.prompt or call.params != runner.params
        or call.meta.get('system') != req.system
        or call.meta.get('runner_identity') != runner.identity
        or any(call.meta.get(k) != req.meta[k] for k in ('role', 'item_id', 'partition'))
        or any(call.meta.get(k) != cfg['project'] for k in ('resource_project', 'quota_project'))
        or call.meta.get('provider_response', {}).get('model') != runner.model):
        raise ValueError('cached request/model/project identity differs')
    e = estimate_call(asdict(call), cfg['pricing'])
    if e['estimated_usd'] is None or e['issues']:
        raise ValueError('unpriced or inconsistent provider token usage')
    return e


def cache_for(out, cfg):
    return original.ScreeningCache(out/'cache', project=cfg['project'],
                                   pricing=cfg['pricing'], limit_usd=CAP)


def prepare(out):
    parent, items, _ = original.load(PARENT); cfg = parent['config']
    model = next(m for m in cfg['generators'] if m['id'] == 'gpt_oss_120b')
    if (cfg['project'] != 'llm-applications-490420' or len(items) != 50
        or any(i.partition != 'calibration' for i in items)
        or model != {'id':'gpt_oss_120b', 'model':'openai/gpt-oss-120b-maas',
                     'location':'us-central1', 'generation':{'max_tokens':32768,
                     'temperature':0.0,'top_p':1.0,'reasoning_effort':'medium'},
                     'interval_seconds':10}):
        raise ValueError('unexpected frozen single-model pilot')
    runner = original.runner(model, cfg['project']); old = RequestCache(PARENT/'cache')
    cache = cache_for(out, cfg); keys=[]; historical={}; reservation=0.; initial_cost=0.
    for n,item in enumerate(items):
        req=generator_request(item,parent['study_id']); key=runner.cache_key(req); keys.append(key)
        records=list(old.log(key).records())
        successes=[r for r in records if r.get('error') is None and r.get('key')==key]
        if n<5:
            if len(successes)!=1: raise ValueError('expected one original successful reply')
            call,_=old.get(runner,req); e=checked(call,runner,req,cfg)
            if not original.parse_call(call).ok: raise ValueError('original model-specific format gate failed')
            historical[key]=digest(asdict(call)); initial_cost+=e['estimated_usd']
        else:
            if records or read_jsonl(str(old.log(key).path)+'.attempts.jsonl',tolerate_tail=True):
                raise ValueError('unexpected prior request; do not regenerate')
            reservation+=cache.reservation(runner,req)
    arm=next(a for a in parent['arms'] if a['id']=='gpt_oss_120b')
    if keys!=arm['request_keys'] or runner.identity!=arm['runner']:
        raise ValueError('request identity differs from the original pilot')
    inventory=budget_inventory()
    original_cost=original.cache_for(cfg,PARENT).summary()['estimated_usd']
    if reservation>CAP or inventory['committed_estimate_and_reservations_usd']+CAP>200 or original_cost+CAP>cfg['stage_budget_usd']:
        raise ValueError('insufficient existing budget authorization')
    preserved={str(p):file_digest(p) for root in (PARENT,Path('reports/gpqa_replacement_generators_20261014'))
               for p in root.rglob('*') if p.is_file() and not p.name.endswith('.lock')}
    sonnet=Path('results/gpqa_sonnet55_20261015/parser_v2/frozen_candidates.json')
    value=seal({'parent_study_id':parent['study_id'],'parent_plan_sha256':file_digest(PARENT/'plan.json'),
        'config':cfg,'selected_model':model,'runner':runner.identity,'request_keys':keys,
        'sources':parent['sources'],'item_ids':[i.item_id for i in items],
        'implementation':implementation(),'preserved_files_sha256':preserved,
        'original_response_hashes':historical,'original_gptoss_estimated_usd':initial_cost,
        'new_requests':45,'reused_requests':5,'new_budget_cap_usd':CAP,
        'remaining_max_output_reservation_usd':reservation,'budget_before':inventory,
        'sonnet_pilot_bundle_sha256':file_digest(sonnet),
        'authorization':'2026-10-17 user approved continuing gpt-oss to the original 50 development questions, reusing five replies.',
        'amendment':'Run gpt-oss independently of the blocked joint DeepSeek gate. No request or parser changes.',
        'confidence_calibrated':False,'evaluation_requests':0,'verifier_requests':0},'continuation_id')
    immutable(out/'plan.json',value)
    return {'stage':'prepared','continuation_id':value['continuation_id'],'questions':50,
            'reuse':5,'new_requests':45,'max_output_reservation_usd':reservation,'new_cap_usd':CAP,
            'project':cfg['project'],'prior_budget_committed_usd':inventory['committed_estimate_and_reservations_usd']}


def load(out):
    plan=json.loads((out/'plan.json').read_text()); unseal(plan,'continuation_id')
    parent,items,groups=original.load(PARENT)
    if (plan['parent_study_id']!=parent['study_id'] or plan['implementation']!=implementation()
        or plan['parent_plan_sha256']!=file_digest(PARENT/'plan.json')
        or plan['sources']!=parent['sources'] or plan['config']!=parent['config']
        or plan['item_ids']!=[i.item_id for i in items]):
        raise ValueError('frozen continuation or sources changed')
    for path,h in plan['preserved_files_sha256'].items():
        if file_digest(path)!=h: raise ValueError('historical artifact changed: '+path)
    runner=original.runner(plan['selected_model'],plan['config']['project'])
    if runner.identity!=plan['runner'] or [runner.cache_key(generator_request(i,parent['study_id'])) for i in items]!=plan['request_keys']:
        raise ValueError('frozen request identity changed')
    return plan,items,groups,runner


def collect(out,allow_api=False):
    plan,items,_,runner=load(out); cfg=plan['config']; cache=cache_for(out,cfg)
    old=RequestCache(PARENT/'cache'); rows=[]; estimates=[]
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage']=='blocked':
        raise ValueError('blocked continuation requires review; no automatic retry')
    def snapshot(stage,**extra):
        value={'continuation_id':plan['continuation_id'],'stage':stage,'target':50,'responses':len(rows),
            'valid':sum(x['valid'] for x in rows),'invalid':dict(Counter(x['failure'] for x in rows if x['failure'])),
            'reused_original':sum(x['original'] for x in rows),'new_calls_this_invocation':sum(x['new'] for x in rows),
            'budget':cache.summary(),'all_gptoss_response_estimate_usd':sum(x['estimated_usd'] for x in estimates),
            'confidence_calibrated':False,'answer_keys_read':False,'evaluation_requests':0,'verifier_requests':0,
            'updated_at_unix':time.time(),**extra}
        atomic_json(out/'status.json',value); return value
    with (out/'collection.lock').open('a') as lock, (PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if allow_api:
                inventory=budget_inventory()
                if inventory['committed_estimate_and_reservations_usd']+CAP-cache.summary()['estimated_usd']>200:
                    raise ValueError('overall authorized budget exceeded')
                atomic_json(out/'billing_check.json',check_billing(cfg['project']))
            pacer=Pacer(plan['selected_model']['interval_seconds']); missing=0
            snapshot('running')
            for item,key in zip(items,plan['request_keys']):
                req=generator_request(item,plan['parent_study_id']); is_original=key in plan['original_response_hashes']
                if is_original:
                    call,_=old.get(runner,req); checked(call,runner,req,cfg)
                    if digest(asdict(call))!=plan['original_response_hashes'][key]: raise ValueError('original reply changed')
                    cache.import_call(call)
                try:
                    call,reused=cache.get(runner,req)
                except OfflineCacheMiss:
                    if not allow_api: missing+=1; continue
                    # Never resend an unresolved or rejected attempt, even on resume.
                    if read_jsonl(str(cache.log(key).path)+'.attempts.jsonl',tolerate_tail=True):
                        raise ValueError('prior attempt exists without a reply; review required')
                    pacer.acquire(); call,reused=cache.get(runner,req,allow_api=True)
                estimates.append(checked(call,runner,req,cfg)); parsed=original.parse_call(call)
                rows.append({'valid':parsed.ok,'failure':parsed.failure,'original':is_original,'new':not reused})
                snapshot('collecting')
            load(out)
            return snapshot('dry_run' if missing else 'complete',missing_requests=missing)
        except Exception as exc:
            snapshot('blocked',error_type=type(exc).__name__,http_status=getattr(exc,'status',None))
            raise


def report(out):
    plan,items,groups,runner=load(out); cfg=plan['config']; cache=cache_for(out,cfg)
    if json.loads((out/'status.json').read_text())['stage']!='complete': raise ValueError('collection is incomplete')
    cs=[]; es=[]; finishes=Counter(); reasoning=0
    for item in items:
        req=generator_request(item,plan['parent_study_id']); call,_=cache.get(runner,req)
        es.append(checked(call,runner,req,cfg)); p=original.parse_call(call)
        cs.append(FrozenCandidate(item.item_id,item.partition,item.subject,item.question,item.choices,
            p.answer,p.p_correct,p.ok,p.failure,runner.model,call.key,original.direct.digest_text(req.prompt),
            original.direct.digest_text(call.response),SYSTEM,build_verifier_prompt(item,p.answer) if p.answer else None))
        choice=call.meta['provider_response']['choices'][0]
        finishes[str(choice.get('finish_reason'))]+=1; reasoning+=bool(choice.get('message',{}).get('reasoning_content'))
    groups['gpt_oss_120b']=cs
    sonnet_path=Path('results/gpqa_sonnet55_20261015/parser_v2/frozen_candidates.json')
    if file_digest(sonnet_path)!=plan['sonnet_pilot_bundle_sha256']: raise ValueError('Sonnet pilot changed')
    sonnet=json.loads(sonnet_path.read_text()); unseal(sonnet,'bundle_id')
    sonnet_cs=[FrozenCandidate(**c) for c in sonnet['groups']['sonnet_5_5']]
    indexed={c.item_id:c for c in sonnet_cs}; groups['sonnet_5_5']=[indexed[i.item_id] for i in items]
    for candidates in groups.values():
        if len(candidates)!=50: raise ValueError('comparison size differs')
        for c,i in zip(candidates,items):
            if (c.item_id,c.partition,c.question,tuple(c.choices))!=(i.item_id,'calibration',i.question,tuple(i.choices)):
                raise ValueError('comparison questions or choice order differs')
    frozen=seal({'continuation_id':plan['continuation_id'],'groups':{g:[asdict(c) for c in rows] for g,rows in groups.items()}},'bundle_id')
    immutable(out/'frozen_candidates.json',frozen)
    # First label access: development only, after candidates are frozen.
    bundle=Path(cfg['baseline_bundle']); manifest=json.loads((bundle/'bundle_manifest.json').read_text()); unseal(manifest,'bundle_id')
    label_path=bundle/'calibration_records.jsonl'
    if file_digest(label_path)!=manifest['files'][label_path.name]: raise ValueError('development key changed')
    records=read_jsonl(label_path)
    if any(r['partition']!='calibration' for r in records): raise ValueError('non-development label')
    labels={r['item_id']:r['correct_index'] for r in records}
    result={g:{**summarize([asdict(c) for c in candidates],labels),
               'confidence_values':dict(Counter(str(c.p_correct) for c in candidates if c.answer and c.p_correct is not None))}
            for g,candidates in groups.items()}
    budget=cache.summary(); total=sum(e['estimated_usd'] for e in es)
    if abs(total-plan['original_gptoss_estimated_usd']-budget['estimated_usd'])>1e-10: raise ValueError('budget does not reconcile')
    value=seal({'continuation_id':plan['continuation_id'],'bundle_id':frozen['bundle_id'],
        'development_label_sha256':file_digest(label_path),'generators':result,'confidence_calibrated':False,
        'evaluation_labels_read':False,'new_api_requests_in_report':0,'budget':budget,
        'original_gptoss_estimated_usd':plan['original_gptoss_estimated_usd'],'total_gptoss_estimated_usd':total,
        'input_tokens':sum(e['usage']['input_tokens'] for e in es),
        'output_tokens_including_reasoning':sum(e['usage']['output_tokens'] for e in es),
        'finish_reasons':dict(finishes),'reasoning_content_present_n':reasoning,'confirmed_billing_usd':None,
        'limitations':['Original exploratory 50-item pilot; not independent confirmatory evidence.',
            'Gemini, two-step Qwen, Sonnet and gpt-oss use different protocols and serving dates.',
            'Raw subjective confidence, not calibrated correctness probability.',
            'No verifier likelihood or stopping policy is fitted in this screen.'],
        'unit_tests_run':False},'result_id')
    immutable(out/'comparison.json',value); return value


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['prepare','collect','report'])
    p.add_argument('--output',type=Path,default=Path('results/gpqa_gptoss_continuation_20261017'))
    p.add_argument('--allow-api',action='store_true'); args=p.parse_args()
    if args.command=='prepare': value=prepare(args.output)
    elif args.command=='collect': value=collect(args.output,args.allow_api)
    else: value=report(args.output)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__': main()
