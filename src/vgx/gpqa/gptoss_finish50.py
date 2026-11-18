"""Authorized position49 re-execution and position50 completion, then scoring.

Preserve all 48 received replies and both historical unknown reservations.
Use an explicit execution overlay, the same request identities and USD1 ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil
import time

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, RequestCache
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa import gptoss_resume as previous
from vgx.gpqa import gptoss_retry22 as retry22
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.prompt import SYSTEM, build_verifier_prompt
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.sonnet_offline_score import summarize

ROOT=previous.ROOT
DIRECTORY='finish50_20261017'
REPORT=Path('reports/gpqa_gptoss_completed50_20261017')


def source_cache(root,position):
    if position==22: return RequestCache(root/retry22.DIRECTORY/'cache')
    if position in (49,50): return RequestCache(root/DIRECTORY/'cache')
    return RequestCache(root/'cache')


def prepare(root):
    if (root/DIRECTORY/'plan.json').exists(): return audit(root)
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior,parent,runner,_,budget=retry22.load(root)
        _,items,_,_=base.load(root)
        prior_status=json.loads((root/retry22.DIRECTORY/'status.json').read_text())
        if prior_status['stage']!='complete' or prior_status['total_received_responses']!=48:
            raise ValueError('expected 48 received replies including position22 retry')
        state=budget._load(); existing={}; jobs={}
        for position,item in enumerate(items,1):
            req=generator_request(item,parent['parent_study_id']); key=runner.cache_key(req)
            if position in (49,50):
                events=read_jsonl(str(budget.log(key).path)+'.attempts.jsonl')
                if list(budget.log(key).records()): raise ValueError('unexpected original reply at final position')
                if position==49 and ([e['event'] for e in events]!=['started','uncertain'] or state['requests'][key]['status']!='unresolved'):
                    raise ValueError('position49 unknown history differs')
                if position==50 and (events or key in state['requests']): raise ValueError('position50 already attempted')
                jobs[str(position)]={'request_key':key,'item_id':item.item_id,
                    'request_body_digest':digest(runner.request_body(req)),
                    'historical_attempts':len([e for e in events if e['event']=='started']),
                    'maximum_new_attempts':2 if position==49 else 3,
                    'reserved_usd':budget.reservation(runner,req)}
            else:
                call,_=source_cache(root,position).get(runner,req)
                base.checked(call,runner,req,parent['config']); existing[key]=digest(asdict(call))
        if len(existing)!=48: raise ValueError('received request count differs')
        reserve=sum(j['reserved_usd'] for j in jobs.values())
        if retry22.committed(state)+reserve>budget.limit: raise ValueError('unchanged shared budget insufficient')
        preserved={str(p):file_digest(p) for p in root.rglob('*') if p.is_file()
            and DIRECTORY not in p.parts and p!=budget.path and not p.name.endswith('.lock')}
        plan=seal({'continuation_id':parent['continuation_id'],'retry22_id':prior['retry_id'],
            'implementation_sha256':file_digest(__file__),'protected_files_sha256':preserved,
            'existing_response_hashes':existing,'jobs':jobs,'preserved_budget_entries':state['requests'],
            'budget_binding':state['binding'],'budget_before':budget.summary(),
            'remaining_reserved_usd':reserve,'original_received_responses':48,'target_responses':50,
            'minimum_interval_seconds':30,'backoff_seconds':[60,120],'maximum_total_attempts_per_item':3,
            'authorization':'2026-10-17 user: retry position49 and finish position50; preserve replies, raw confidence, budget; then compare four models on the same questions.',
            'confidence_calibrated':False,'answer_keys_read':False,'evaluation_requests':0,'verifier_requests':0},'completion_id')
        base.immutable(root/DIRECTORY/'plan.json',plan)
    return audit(root)


def load(root):
    prior,parent,runner,_,budget=retry22.load(root)
    _,items,groups,_=base.load(root)
    plan=json.loads((root/DIRECTORY/'plan.json').read_text()); unseal(plan,'completion_id')
    if (plan['implementation_sha256']!=file_digest(__file__) or plan['retry22_id']!=prior['retry_id']
        or plan['continuation_id']!=parent['continuation_id']): raise ValueError('completion source/plan changed')
    for path,sha in plan['protected_files_sha256'].items():
        if file_digest(path)!=sha: raise ValueError('historical artifact changed: '+path)
    for position in (49,50):
        req=generator_request(items[position-1],parent['parent_study_id']); job=plan['jobs'][str(position)]
        if runner.cache_key(req)!=job['request_key'] or digest(runner.request_body(req))!=job['request_body_digest']:
            raise ValueError('provider request identity changed')
    return plan,parent,items,groups,runner,budget


def entry_key(plan,position):
    return 'authorized_completion:'+plan['completion_id']+':'+str(position)


def check_ledger(plan,budget,state):
    if state['binding']!=plan['budget_binding']: raise ValueError('budget binding changed')
    for key,entry in plan['preserved_budget_entries'].items():
        if state['requests'].get(key)!=entry: raise ValueError('historical ledger entry changed')
    allowed=set(plan['preserved_budget_entries'])|{entry_key(plan,p) for p in (49,50)}
    if set(state['requests'])-allowed: raise ValueError('unexpected extra execution')
    if retry22.committed(state)>budget.limit: raise ValueError('shared budget exceeded')


def audit(root):
    plan,parent,items,_,runner,budget=load(root)
    state=budget._load(); check_ledger(plan,budget,state)
    parsed=[]; costs=[]; new_records=[]; new_attempts=0; rejections=Counter()
    for position,item in enumerate(items,1):
        req=generator_request(item,parent['parent_study_id']); key=runner.cache_key(req); cache=source_cache(root,position)
        if position in (49,50):
            records,events,starts,successes=previous.history(cache,key)
            if len(starts)>plan['jobs'][str(position)]['maximum_new_attempts']: raise ValueError('retry cap exceeded')
            new_attempts+=len(starts); new_records.extend(records)
            rejections.update(str(r['meta']['http_status']) for r in records if r.get('error'))
            by_exec={r['meta']['execution_id']:r for r in records}
            for n,start in enumerate(starts[1:],1):
                previous_record=by_exec[starts[n-1]['execution_id']]
                if previous_record.get('error')!='http_429': raise ValueError('retry without explicit429')
                if start['created']-previous_record['created']-previous_record['latency_s']<plan['backoff_seconds'][n-1]-0.01:
                    raise ValueError('backoff interval violated')
            if not successes: continue
        call,_=cache.get(runner,req); e=base.checked(call,runner,req,parent['config']); costs.append(e)
        if position not in (49,50):
            if digest(asdict(call))!=plan['existing_response_hashes'][key]: raise ValueError('existing reply changed')
        else:
            entry=state['requests'][entry_key(plan,position)]
            if entry['status']!='settled' or abs(entry['estimated_usd']-e['estimated_usd'])>1e-12:
                raise ValueError('new usage does not reconcile')
        parsed.append(base.original.parse_call(call))
    for position in (49,50):
        entry=state['requests'].get(entry_key(plan,position))
        if entry and entry['status'] not in ('settled','rejected'): raise ValueError('new unknown or in-flight execution')
    ordered=sorted(new_records,key=lambda r:r['created'])
    gaps=[b['created']-a['created']-a['latency_s'] for a,b in zip(ordered,ordered[1:])]
    if gaps and min(gaps)<29.99: raise ValueError('request interval violated')
    summary=budget.summary(); total=sum(e['estimated_usd'] for e in costs)
    if abs(total-parent['original_gptoss_estimated_usd']-summary['estimated_usd'])>1e-10:
        raise ValueError('total costs do not reconcile')
    return {'stage':'complete' if len(parsed)==50 else 'dry_run','completion_id':plan['completion_id'],
        'target':50,'responses':len(parsed),'valid':sum(p.ok for p in parsed),
        'invalid':dict(Counter(p.failure for p in parsed if p.failure)),
        'original_48_responses_unchanged':True,'old_unknown_reservations_preserved':True,
        'new_responses':len(parsed)-48,'remaining_requests':50-len(parsed),'new_provider_attempts':new_attempts,
        'new_rejections':dict(rejections),'minimum_seconds_after_previous_attempt':min(gaps) if gaps else None,
        'budget':summary,'new_estimated_usd':summary['estimated_usd']-plan['budget_before']['estimated_usd'],
        'all_received_gptoss_estimated_usd':total,
        'input_tokens':sum(e['usage']['input_tokens'] for e in costs),
        'output_tokens_including_reasoning':sum(e['usage']['output_tokens'] for e in costs),
        'unknown_usage_reserved_usd':plan['budget_before']['outstanding_reserved_usd'],
        'confidence_calibrated':False,'answer_keys_read':False,'evaluation_requests':0,'verifier_requests':0}


def collect(root,allow_api=False):
    if not allow_api: return audit(root)
    out=root/DIRECTORY
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage']=='blocked':
        raise ValueError('blocked execution requires review')
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,parent,items,_,runner,budget=load(root); status=audit(root)
        def snapshot(stage,**extra):
            status.update(stage=stage,updated_at_unix=time.time(),**extra); atomic_json(out/'status.json',status)
        active=None; cache=RequestCache(out/'cache')
        try:
            atomic_json(out/'billing_check.json',base.check_billing(parent['config']['project']))
            snapshot('running'); last_end=0.0
            for position in (49,50):
                job=plan['jobs'][str(position)]; active=entry_key(plan,position)
                req=generator_request(items[position-1],parent['parent_study_id'])
                req=replace(req,meta={**req.meta,'authorized_completion_id':plan['completion_id']})
                try:
                    call,_=cache.get(runner,req); last_end=max(last_end,call.created+call.latency_s); continue
                except OfflineCacheMiss: pass
                while True:
                    records,_,starts,_=previous.history(cache,job['request_key'])
                    if len(starts)>=job['maximum_new_attempts']: raise ValueError('explicit429 retry allowance exhausted')
                    ready=max((r['created']+r['latency_s'] for r in records),default=0.0)
                    if starts: ready+=plan['backoff_seconds'][len(starts)-1]
                    deadline=max(last_end+30,ready)
                    snapshot('waiting',current_position=position,next_attempt_number=len(starts)+1,not_before_unix=deadline)
                    while time.time()<deadline: time.sleep(max(0,min(30,deadline-time.time())))
                    with file_lock(str(budget.path)+'.lock'):
                        state=budget._load(); check_ledger(plan,budget,state)
                        for other in (49,50):
                            e=state['requests'].get(entry_key(plan,other))
                            if e and e['status'] not in ('settled','rejected'):
                                raise ValueError('new unresolved or reserved execution')
                        if retry22.committed(state)+job['reserved_usd']>budget.limit: raise ValueError('insufficient shared budget')
                        state['requests'][active]={'status':'reserved','reserved_usd':job['reserved_usd'],
                            'model':runner.model,'role':req.meta.get('role'),'canonical_request_key':job['request_key'],
                            'authorized_completion_id':plan['completion_id']}
                        atomic_json(budget.path,state)
                    snapshot('requesting')
                    try:
                        call,_=cache.get(runner,req,allow_api=True); last_end=time.time()
                    except ProviderRejectedError as exc:
                        last_end=time.time()
                        with file_lock(str(budget.path)+'.lock'):
                            state=budget._load(); check_ledger(plan,budget,state)
                            state['requests'][active].update(status='rejected',estimated_usd=0.0)
                            atomic_json(budget.path,state)
                        if exc.status!=429: raise
                        status=audit(root); snapshot('backoff',current_position=position,last_http_status=429); continue
                    estimate=base.checked(call,runner,req,parent['config'])
                    if estimate['estimated_usd']>job['reserved_usd']: raise ValueError('request exceeded reserve')
                    with file_lock(str(budget.path)+'.lock'):
                        state=budget._load(); check_ledger(plan,budget,state)
                        state['requests'][active].update(status='settled',estimated_usd=estimate['estimated_usd'],
                            issues=estimate['issues'],execution_id=call.meta['execution_id'])
                        atomic_json(budget.path,state)
                    status=audit(root); snapshot('collecting'); break
            status=audit(root)
            if status['responses']!=50: raise ValueError('collection incomplete')
            snapshot('complete'); return status
        except Exception as exc:
            with file_lock(str(budget.path)+'.lock'):
                state=budget._load(); check_ledger(plan,budget,state)
                if active in state['requests'] and state['requests'][active]['status']=='reserved':
                    state['requests'][active].update(status='unresolved',estimated_usd=None)
                    atomic_json(budget.path,state)
            status['budget']=budget.summary()
            snapshot('blocked',error_type=type(exc).__name__,http_status=getattr(exc,'status',None)); raise


def report(root):
    status=audit(root)
    if status['stage']!='complete': raise ValueError('all 50 replies required for this comparison')
    plan,parent,items,groups,runner,budget=load(root); out=root/DIRECTORY
    sonnet_path=Path('results/gpqa_sonnet55_20261015/parser_v2/frozen_candidates.json')
    if file_digest(sonnet_path)!=parent['sonnet_pilot_bundle_sha256']: raise ValueError('Sonnet pilot changed')
    sonnet=json.loads(sonnet_path.read_text()); unseal(sonnet,'bundle_id')
    indexed={c['item_id']:FrozenCandidate(**c) for c in sonnet['groups']['sonnet_5_5']}
    groups['sonnet_5_5']=[indexed[i.item_id] for i in items]
    candidates=[]; finishes=Counter(); reasoning=0
    for position,item in enumerate(items,1):
        req=generator_request(item,parent['parent_study_id']); call,_=source_cache(root,position).get(runner,req)
        base.checked(call,runner,req,parent['config']); p=base.original.parse_call(call)
        candidates.append(FrozenCandidate(item.item_id,item.partition,item.subject,item.question,item.choices,
            p.answer,p.p_correct,p.ok,p.failure,runner.model,call.key,base.original.direct.digest_text(req.prompt),
            base.original.direct.digest_text(call.response),SYSTEM,build_verifier_prompt(item,p.answer) if p.answer else None))
        choice=(call.meta['provider_response'].get('choices') or [{}])[0]
        finishes[str(choice.get('finish_reason'))]+=1; reasoning+=bool(choice.get('message',{}).get('reasoning_content'))
    groups['gpt_oss_120b']=candidates
    for rows in groups.values():
        if len(rows)!=50: raise ValueError('comparison count mismatch')
        for c,i in zip(rows,items):
            if (c.item_id,c.partition,c.question,tuple(c.choices))!=(i.item_id,'calibration',i.question,tuple(i.choices)):
                raise ValueError('comparison membership or choice order differs')
    frozen=seal({'completion_id':plan['completion_id'],'groups':{g:[asdict(c) for c in rows] for g,rows in groups.items()}},'bundle_id')
    base.immutable(out/'frozen_candidates.json',frozen)
    common=set.intersection(*[{c.item_id for c in rows if c.answer in ('A','B','C','D') and c.p_correct is not None} for rows in groups.values()])
    cohort=seal({'completion_id':plan['completion_id'],'received_item_ids':[i.item_id for i in items],
        'common_valid_item_ids':sorted(common),'labels_used':False},'cohort_id')
    base.immutable(out/'scoring_cohort.json',cohort)
    # Only after freezing predictions and both cohorts, read development labels.
    bundle=Path(parent['config']['baseline_bundle']); manifest=json.loads((bundle/'bundle_manifest.json').read_text()); unseal(manifest,'bundle_id')
    label_path=bundle/'calibration_records.jsonl'
    if file_digest(label_path)!=manifest['files'][label_path.name]: raise ValueError('development key changed')
    records=read_jsonl(label_path)
    if any(r['partition']!='calibration' for r in records): raise ValueError('non-development labels')
    labels={r['item_id']:r['correct_index'] for r in records}
    full={g:{**summarize([asdict(c) for c in rows],labels),
        'confidence_values':dict(Counter(str(c.p_correct) for c in rows if c.answer and c.p_correct is not None))} for g,rows in groups.items()}
    matched={g:summarize([asdict(c) for c in rows if c.item_id in common],labels) for g,rows in groups.items()}
    for selection,metrics in ((None,full),(common,matched)):
        for g,all_rows in groups.items():
            rows=[c for c in all_rows if selection is None or c.item_id in selection]
            pairs=[(c.p_correct,int(c.answer=='ABCD'[labels[c.item_id]]))
                   for c in rows if c.answer in ('A','B','C','D') and c.p_correct is not None]
            m=metrics[g]; correct=sum(c.answer=='ABCD'[labels[c.item_id]] for c in rows)
            if correct!=m['correct'] or len(pairs)!=m['brier_n']: raise ValueError('independent metric count mismatch')
            if pairs:
                for actual,expected in ((sum(p for p,y in pairs)/len(pairs),m['mean_confidence']),
                    (sum((p-y)**2 for p,y in pairs)/len(pairs),m['raw_brier'])):
                    if abs(actual-expected)>1e-12: raise ValueError('independent metric arithmetic mismatch')
            wrong=[p for p,y in pairs if not y]
            if len(wrong)!=m['wrong_with_confidence_n'] or (wrong and abs(sum(wrong)/len(wrong)-m['mean_confidence_if_wrong'])>1e-12):
                raise ValueError('independent wrong-confidence mismatch')
    value=seal({'completion_id':plan['completion_id'],'bundle_id':frozen['bundle_id'],'cohort_id':cohort['cohort_id'],
        'development_label_sha256':file_digest(label_path),'cohort_n':50,'generators':full,
        'common_valid_n':len(common),'common_valid_generators':matched,
        'confidence_calibrated':False,'evaluation_labels_read':False,'new_api_requests_in_report':0,
        'collection_audit':status,'independent_metric_arithmetic_matches':True,
        'finish_reasons':dict(finishes),'reasoning_content_present_n':reasoning,
        'confirmed_billing_usd':None,'actual_total_cost_unknown':True,
        'limitations':['Original exploratory 50 development questions, not Diamond or independent confirmatory evidence.',
            'Raw confidence unchanged; malformed responses retained. Accuracy uses all50; probability metrics use valid pairs only.',
            'Generation protocols differ: Qwen two-step, Sonnet adaptive high effort, GPT-OSS medium effort.',
            'Position22 and49 successful replies are user-authorized re-executions; historical unknown attempts remain unresolved and reserved.'],
        'unit_tests_run':False},'result_id')
    base.immutable(out/'result.json',value)
    if REPORT.exists():
        for line in (REPORT/'sha256.txt').read_text().splitlines():
            sha,name=line.split('  ',1)
            if file_digest(REPORT/name)!=sha: raise ValueError('existing report checksum mismatch')
        if json.loads((REPORT/'result.json').read_text())!=value: raise ValueError('existing report differs')
        return value
    REPORT.mkdir(parents=True)
    for name in ('plan.json','status.json','billing_check.json','result.json'): shutil.copyfile(out/name,REPORT/name)
    base.immutable(REPORT/'audit.json',status)
    names={'gpt_oss_120b':'GPT-OSS-120B','gemini_baseline':'Gemini','qwen_baseline':'Qwen（两步）','sonnet_5_5':'Sonnet 5.5'}
    lines=['四个generator在同一50道开发题上的原始confidence比较','',
        '原48条回复保留；新收到原第49题的授权重试回复，以及原第50题的首次回复。',
        '全部confidence未校准。无效回复原样保留，不进行解析修复或答案再生成。',
        '模型 | 正确/50 | 有效答案/概率 | 平均confidence | 答错confidence（n） | raw Brier（n）']
    for g,label in names.items():
        m=full[g]; wrong=m['mean_confidence_if_wrong']
        lines.append(f"{label} | {m['correct']}/50 ({m['accuracy_all']:.2%}) | {m['confidence_n']}/50 | {m['mean_confidence']:.2%} | {('未定义' if wrong is None else f'{wrong:.2%}')} ({m['wrong_with_confidence_n']}) | {m['raw_brier']:.6f} ({m['brier_n']})")
    lines.extend(['',f"四模型共同有效子集为{len(common)}题；同一子集的指标见result.json/common_valid_generators。",
        '正确率分母为50，解析失败算未答对；confidence及Brier只使用有效答案/概率，不插补。Brier=mean((p-是否正确)^2)。',
        f"本轮49/50新增用量估算USD {status['new_estimated_usd']:.8f}；GPT-OSS全部50条已收到回复累计估算USD {status['all_received_gptoss_estimated_usd']:.8f}。",
        f"历史两次未知请求继续预留USD {status['unknown_usage_reserved_usd']:.8f}，实际总收费仍未知；共享USD1上限未变。",
        '计费项目llm-applications-490420。美元数为provider token用量按冻结价格估算，不是确认账单。',
        f"finish_reason统计{dict(finishes)}；解析失败{status['invalid']}；含reasoning_content回复{reasoning}/50。",
        '原48条回复/历史尝试/预算条目哈希未变；先冻结预测和共有有效子集，再只读开发标签；独立重算所有汇总指标一致。',
        '未校准confidence，未拟合likelihood，未调用verifier，未读取Diamond标签。',
        '50题已用于多轮探索；模型/协议/服务日期不同，尤其Qwen使用两步生成，不能将差异全部归因于模型。',
        '本目录只有汇总结果、计划、状态、审计和哈希，不含题干、答案键、原始回复或逐题对错。未提交推送；未新增或运行单元测试。'])
    (REPORT/'README.txt').write_text('\n'.join(lines)+'\n')
    banned={'question','choices','answer','correct_index','response','provider_response','prompt','system','access_token','refresh_token','private_key','client_secret'}
    def safe(v):
        if isinstance(v,dict):
            if banned.intersection(v): raise ValueError('unsafe report field')
            for x in v.values(): safe(x)
        elif isinstance(v,list):
            for x in v: safe(x)
    for p in REPORT.glob('*.json'): safe(json.loads(p.read_text()))
    load(root)
    (REPORT/'sha256.txt').write_text(''.join(f'{file_digest(p)}  {p.name}\n' for p in sorted(REPORT.iterdir()) if p.is_file()))
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['prepare','collect','audit','report'])
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--allow-api',action='store_true'); a=p.parse_args()
    if a.command=='prepare': value=prepare(a.root)
    elif a.command=='collect': value=collect(a.root,a.allow_api)
    elif a.command=='audit': value=audit(a.root)
    else: value=report(a.root)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__': main()
