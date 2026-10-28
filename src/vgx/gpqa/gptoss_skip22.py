"""Explicitly skip the one authorized unknown request; keep its budget reserve.

Reuse 21 replies, request only the untouched 28, and compare the same 49
development items offline. All previous sealed modules/artifacts are unchanged.
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
from vgx.common.budget import BudgetLimitError
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa import gptoss_resume as previous
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.prompt import SYSTEM, build_verifier_prompt
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.sonnet_offline_score import summarize

ROOT = previous.ROOT
AMENDMENT = 'skip22_20261017'
REPORT = Path('reports/gpqa_gptoss_skip22_20261017')


class ScopedCache(base.original.ScreeningCache):
    """Same ledger/cap; permit only the exact preserved unknown reservation.

The generic budget cache blocks every unresolved request. This local amendment
narrows that guard for the user's named exclusion only, without modifying its
state, releasing its reserve, or treating its unknown usage as zero.
    """
    def __init__(self, root, parent, plan):
        super().__init__(root/'cache',project=parent['config']['project'],
                         pricing=parent['config']['pricing'],limit_usd=base.CAP)
        self.plan=plan

    def check_state(self,state):
        if state['binding']!=self.plan['budget_binding']:
            raise BudgetLimitError('budget binding changed')
        for key, entry in self.plan['preserved_budget_entries'].items():
            if state['requests'].get(key)!=entry:
                raise BudgetLimitError('historical budget entry changed')
        allowed=set(self.plan['preserved_budget_entries'])|set(self.plan['new_request_keys'])
        if set(state['requests'])-allowed:
            raise BudgetLimitError('unplanned request in shared budget')
        for key,entry in state['requests'].items():
            if key==self.plan['skipped_request_key']: continue
            if entry['status'] in ('unpriced','unresolved'):
                raise BudgetLimitError('new unknown or unpriced usage; stop')
        committed=sum(e['estimated_usd'] if e['status'] in ('settled','rejected') else e['reserved_usd']
                      for e in state['requests'].values())
        if committed>self.limit: raise BudgetLimitError('shared budget exceeded')
        return committed

    def get(self,runner,request,*,allow_api=False):
        key=runner.cache_key(request)
        if key==self.plan['skipped_request_key']:
            raise ValueError('excluded unknown request must never be queried or recovered automatically')
        reserve=self.reservation(runner,request)
        try:
            result=RequestCache.get(self,runner,request)
            self._settle(key)
            return result
        except OfflineCacheMiss:
            if not allow_api: raise
        if key not in self.plan['new_request_keys']:
            raise ValueError('paid request is outside the 28 authorized keys')
        with file_lock(self.root/'budget-request-locks'/f'{digest(key)}.lock'):
            try:
                result=RequestCache.get(self,runner,request)
                self._settle(key)
                return result
            except OfflineCacheMiss: pass
            _,_,starts,_=previous.history(self,key)
            if len(starts)>=3: raise ValueError('429 retry allowance exhausted')
            with file_lock(str(self.path)+'.lock'):
                state=self._load(); committed=self.check_state(state)
                entry=state['requests'].get(key)
                if entry is None or entry['status']=='rejected':
                    if committed+reserve>self.limit:
                        raise BudgetLimitError('insufficient remaining shared budget')
                    state['requests'][key]={'status':'reserved','reserved_usd':reserve,
                                           'model':runner.model,'role':request.meta.get('role')}
                    atomic_json(self.path,state)
                elif entry['status']!='reserved':
                    raise BudgetLimitError('unexpected ledger state without a reply')
            try:
                return RequestCache.get(self,runner,request,allow_api=True)
            finally:
                self._settle(key)


def prepare(root):
    out=root/AMENDMENT
    if (out/'plan.json').exists(): return audit(root)
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old,parent,items,runner,cache=previous.load(root)
        old_status=json.loads((root/previous.AMENDMENT/'status.json').read_text())
        if (old_status['stage'],old_status['error_type'],old_status['responses'],old_status['current_position']) != ('blocked','UncertainRequestError',21,22):
            raise ValueError('previous stop differs from authorized exclusion')
        excluded=parent['request_keys'][21]
        ledger=cache._load()
        if ledger['requests'][excluded]['status']!='unresolved':
            raise ValueError('excluded unknown request state changed')
        log=cache.log(excluded).path; events_path=Path(str(log)+'.attempts.jsonl')
        events=read_jsonl(events_path)
        if log.exists() or [e['event'] for e in events]!=['started','uncertain']:
            raise ValueError('excluded request history changed')
        if events[0]['execution_id']!=events[1]['execution_id'] or any(e['key']!=excluded for e in events):
            raise ValueError('excluded request attempt identity mismatch')
        preserved={str(events_path):file_digest(events_path)}; responses={}; new=[]; reservation=0.0
        for i,(item,key) in enumerate(zip(items,parent['request_keys'])):
            if key==excluded: continue
            rows,attempts,_,successes=previous.history(cache,key)
            req=generator_request(item,parent['parent_study_id'])
            if successes:
                call,_=RequestCache.get(cache,runner,req)
                base.checked(call,runner,req,parent['config']); responses[key]=digest(asdict(call))
                for p in (cache.log(key).path,Path(str(cache.log(key).path)+'.attempts.jsonl')):
                    if p.exists(): preserved[str(p)]=file_digest(p)
            else:
                if rows or attempts or i<22: raise ValueError('expected only untouched positions 23 through 50')
                new.append(key); reservation+=cache.reservation(runner,req)
        if len(responses)!=21 or len(new)!=28: raise ValueError('unexpected remaining request count')
        before=cache.summary()
        if before['estimated_usd']+before['outstanding_reserved_usd']+reservation>base.CAP:
            raise ValueError('insufficient unchanged budget including unknown reserve')
        for p in (root/previous.AMENDMENT).rglob('*'):
            if p.is_file() and not p.name.endswith('.lock'): preserved[str(p)]=file_digest(p)
        for name in ('plan.json','status.json','billing_check.json','blocked_audit.json'):
            p=root/name
            if p.exists(): preserved[str(p)]=file_digest(p)
        plan=seal({'previous_resume_id':old['resume_id'],'continuation_id':parent['continuation_id'],
            'implementation_sha256':file_digest(__file__), 'preserved_files_sha256':preserved,
            'existing_response_hashes':responses, 'new_request_keys':new,
            'skipped_position':22,'skipped_item_id':items[21].item_id,'skipped_request_key':excluded,
            'skipped_log_must_remain_absent':str(log),'skipped_attempt_finished_unix':events[-1]['created'],
            'preserved_budget_entries':ledger['requests'],'budget_binding':ledger['binding'],
            'budget_before':before,'remaining_max_output_reservation_usd':reservation,
            'original_questions':50,'target_received_responses':49,'reused_responses':21,'new_requests':28,
            'minimum_interval_seconds':30,'spacing_basis':'after previous attempt completes',
            'backoff_seconds':[60,120],'maximum_attempts_per_request':3,
            'comparison_item_ids':[i.item_id for i in items if i.item_id!=items[21].item_id],
            'comparison_membership_basis':'exclude only position 22 from every generator; no labels inspected',
            'authorization':'2026-10-17 user: skip unknown position 22 without retry; preserve 21 replies; finish untouched 28; unchanged budget and raw confidence; compare all four models on the same completed questions.',
            'confidence_calibrated':False,'evaluation_requests':0,'verifier_requests':0},'skip_id')
        base.immutable(out/'plan.json',plan)
    return audit(root)


def load(root):
    old,parent,items,runner,_=previous.load(root)
    plan=json.loads((root/AMENDMENT/'plan.json').read_text()); unseal(plan,'skip_id')
    if (plan['implementation_sha256']!=file_digest(__file__) or plan['previous_resume_id']!=old['resume_id']
        or plan['continuation_id']!=parent['continuation_id'] or plan['skipped_request_key']!=parent['request_keys'][21]
        or plan['comparison_item_ids']!=[i.item_id for n,i in enumerate(items) if n!=21]):
        raise ValueError('frozen skip amendment changed')
    for path,sha in plan['preserved_files_sha256'].items():
        if file_digest(path)!=sha: raise ValueError('preserved file changed: '+path)
    if Path(plan['skipped_log_must_remain_absent']).exists():
        raise ValueError('excluded response log appeared; review required')
    cache=ScopedCache(root,parent,plan)
    with file_lock(str(cache.path)+'.lock'): cache.check_state(cache._load())
    return plan,parent,items,runner,cache


def audit(root):
    plan,parent,items,runner,cache=load(root)
    estimates=[]; parsed=[]; all_records=[]; new_attempts=[]; retry_gaps=[]; costs={}
    for item,key in zip(items,parent['request_keys']):
        if key==plan['skipped_request_key']: continue
        records,events,starts,successes=previous.history(cache,key)
        all_records.extend(records)
        if key in plan['new_request_keys']:
            if len(starts)>3: raise ValueError('retry allowance exceeded')
            new_attempts.extend(starts)
            by_exec={r['meta']['execution_id']:r for r in records}
            for n,start in enumerate(starts[1:],1):
                preceding=by_exec[starts[n-1]['execution_id']]
                if preceding.get('error')!='http_429': raise ValueError('retry without explicit 429')
                gap=start['created']-preceding['created']-preceding['latency_s']
                if gap<plan['backoff_seconds'][n-1]-0.01: raise ValueError('retry backoff violated')
                retry_gaps.append(gap)
        if not successes: continue
        call,_=RequestCache.get(cache,runner,generator_request(item,parent['parent_study_id']))
        if key in plan['existing_response_hashes'] and digest(asdict(call))!=plan['existing_response_hashes'][key]:
            raise ValueError('existing reply changed')
        e=base.checked(call,runner,generator_request(item,parent['parent_study_id']),parent['config'])
        estimates.append(e); costs[key]=e['estimated_usd']; parsed.append(base.original.parse_call(call))
    state=cache._load(); cache.check_state(state)
    for key,entry in state['requests'].items():
        if key==plan['skipped_request_key']: continue
        if entry['status'] not in ('settled','rejected'): raise ValueError('additional outstanding request')
        if entry['status']=='settled' and (key not in costs or abs(entry['estimated_usd']-costs[key])>1e-12 or entry.get('issues')):
            raise ValueError('response usage and ledger do not reconcile')
    summary=cache.summary(); total=sum(e['estimated_usd'] for e in estimates)
    if abs(total-parent['original_gptoss_estimated_usd']-summary['estimated_usd'])>1e-10:
        raise ValueError('total costs do not reconcile')
    new_ids={e['execution_id'] for e in new_attempts}; gaps=[]
    last_end=plan['skipped_attempt_finished_unix']
    for record in sorted(all_records,key=lambda r:r['created']):
        if record['meta'].get('execution_id') in new_ids:
            gap=record['created']-last_end
            if gap<plan['minimum_interval_seconds']-0.01: raise ValueError('request interval violated')
            gaps.append(gap)
        last_end=max(last_end,record['created']+record['latency_s'])
    return {'stage':'complete' if len(parsed)==49 else 'dry_run','skip_id':plan['skip_id'],
        'original_questions':50,'target_received_responses':49,'responses':len(parsed),
        'valid':sum(p.ok for p in parsed),'invalid':dict(Counter(p.failure for p in parsed if p.failure)),
        'skipped_unknown_positions':[22],'reused_responses':21,'new_responses_since_amendment':len(parsed)-21,
        'remaining_requests':49-len(parsed),'new_attempts_since_amendment':len(new_attempts),
        'minimum_seconds_after_previous_attempt':min(gaps) if gaps else None,
        'retry_backoffs_checked':len(retry_gaps),'retry_backoff_checks_passed':True,
        'new_explicit_rejections':dict(Counter(str(r['meta']['http_status']) for r in all_records
                                             if r.get('error') and r['key'] in plan['new_request_keys'])),
        'budget':summary,'new_estimated_usd':summary['estimated_usd']-plan['budget_before']['estimated_usd'],
        'all_received_gptoss_estimated_usd':total,
        'unknown_usage_reserved_usd':plan['preserved_budget_entries'][plan['skipped_request_key']]['reserved_usd'],
        'original_21_replies_unchanged':True,'excluded_request_and_budget_entry_unchanged':True,
        'confidence_calibrated':False,'answer_keys_read':False,'evaluation_requests':0,'verifier_requests':0}


def collect(root,allow_api=False):
    if not allow_api: return audit(root)
    out=root/AMENDMENT
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['stage']=='blocked':
        raise ValueError('new blocked run needs review; no automatic restart')
    with (root/'collection.lock').open('a') as lock, (base.PARENT/'collection.lock').open('a') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,parent,items,runner,cache=load(root); status=audit(root); initial=status['responses']
        def snapshot(stage,**extra):
            status.update(stage=stage,updated_at_unix=time.time(),new_calls_this_invocation=status['responses']-initial,**extra)
            atomic_json(out/'status.json',status)
        try:
            summary=cache.summary()
            unallocated=base.CAP-summary['estimated_usd']-summary['outstanding_reserved_usd']
            if base.budget_inventory()['committed_estimate_and_reservations_usd']+unallocated>200:
                raise BudgetLimitError('overall authorization exceeded')
            atomic_json(out/'billing_check.json',base.check_billing(parent['config']['project']))
            snapshot('running')
            last_end=max([plan['skipped_attempt_finished_unix']]+[r['created']+r['latency_s']
                for key in parent['request_keys'] if key!=plan['skipped_request_key'] for r in previous.history(cache,key)[0]])
            for position,(item,key) in enumerate(zip(items,parent['request_keys']),1):
                if key not in plan['new_request_keys']: continue
                req=generator_request(item,parent['parent_study_id'])
                try:
                    cache.get(runner,req); continue
                except OfflineCacheMiss: pass
                while True:
                    records,_,starts,successes=previous.history(cache,key)
                    if successes: break
                    if len(starts)>=3: raise ValueError('explicit 429 retry allowance exhausted')
                    ready=max((r['created']+r['latency_s'] for r in records),default=0.0)
                    if starts: ready+=plan['backoff_seconds'][len(starts)-1]
                    deadline=max(last_end+plan['minimum_interval_seconds'],ready)
                    snapshot('waiting',current_position=position,next_attempt_number=len(starts)+1,not_before_unix=deadline)
                    while time.time()<deadline: time.sleep(max(0,min(30,deadline-time.time())))
                    snapshot('requesting')
                    try:
                        call,_=cache.get(runner,req,allow_api=True); last_end=time.time()
                        base.checked(call,runner,req,parent['config'])
                        status=audit(root); snapshot('collecting'); break
                    except ProviderRejectedError as exc:
                        last_end=time.time()
                        if exc.status!=429: raise
                        status=audit(root); snapshot('backoff',current_position=position,last_http_status=429)
            status=audit(root)
            if status['responses']!=49: raise ValueError('expected 49 received replies')
            snapshot('complete'); return status
        except Exception as exc:
            status['budget']=cache.summary()
            snapshot('blocked',error_type=type(exc).__name__,http_status=getattr(exc,'status',None))
            raise


def report(root):
    status=audit(root)
    if status['stage']!='complete': raise ValueError('finish all 28 authorized requests before comparison')
    plan,parent,items,runner,cache=load(root); out=root/AMENDMENT
    _,_,groups,_=base.load(root)
    selected=[i for i in items if i.item_id!=plan['skipped_item_id']]
    allowed=set(plan['comparison_item_ids'])
    groups={g:[c for c in rows if c.item_id in allowed] for g,rows in groups.items()}
    sonnet_path=Path('results/gpqa_sonnet55_20261015/parser_v2/frozen_candidates.json')
    if file_digest(sonnet_path)!=parent['sonnet_pilot_bundle_sha256']: raise ValueError('Sonnet pilot changed')
    sonnet=json.loads(sonnet_path.read_text()); unseal(sonnet,'bundle_id')
    indexed={c['item_id']:FrozenCandidate(**c) for c in sonnet['groups']['sonnet_5_5']}
    groups['sonnet_5_5']=[indexed[i.item_id] for i in selected]
    candidates=[]; estimates=[]; finishes=Counter(); reasoning=0
    for item in selected:
        req=generator_request(item,parent['parent_study_id']); call,_=RequestCache.get(cache,runner,req)
        estimates.append(base.checked(call,runner,req,parent['config'])); parsed=base.original.parse_call(call)
        candidates.append(FrozenCandidate(item.item_id,item.partition,item.subject,item.question,item.choices,
            parsed.answer,parsed.p_correct,parsed.ok,parsed.failure,runner.model,call.key,
            base.original.direct.digest_text(req.prompt),base.original.direct.digest_text(call.response),SYSTEM,
            build_verifier_prompt(item,parsed.answer) if parsed.answer else None))
        choice=call.meta['provider_response']['choices'][0]
        finishes[str(choice.get('finish_reason'))]+=1; reasoning+=bool(choice.get('message',{}).get('reasoning_content'))
    groups['gpt_oss_120b']=candidates
    for rows in groups.values():
        if len(rows)!=49: raise ValueError('comparison cohort size differs')
        for c,i in zip(rows,selected):
            if (c.item_id,c.partition,c.question,tuple(c.choices))!=(i.item_id,'calibration',i.question,tuple(i.choices)):
                raise ValueError('comparison item or choices differ')
    frozen=seal({'skip_id':plan['skip_id'],'groups':{g:[asdict(c) for c in rows] for g,rows in groups.items()}},'bundle_id')
    base.immutable(out/'frozen_candidates.json',frozen)
    common=set.intersection(*[{c.item_id for c in rows if c.answer in ('A','B','C','D') and c.p_correct is not None} for rows in groups.values()])
    cohort=seal({'skip_id':plan['skip_id'],'received_item_ids':plan['comparison_item_ids'],
                 'common_valid_item_ids':sorted(common),'labels_used':False},'cohort_id')
    base.immutable(out/'scoring_cohort.json',cohort)
    # Only development labels, after all predictions and membership are frozen.
    bundle=Path(parent['config']['baseline_bundle'])
    manifest=json.loads((bundle/'bundle_manifest.json').read_text()); unseal(manifest,'bundle_id')
    labels_path=bundle/'calibration_records.jsonl'
    if file_digest(labels_path)!=manifest['files'][labels_path.name]: raise ValueError('development key changed')
    records=read_jsonl(labels_path)
    if any(r['partition']!='calibration' for r in records): raise ValueError('non-development label')
    labels={r['item_id']:r['correct_index'] for r in records}
    full={g:{**summarize([asdict(c) for c in rows],labels),
             'confidence_values':dict(Counter(str(c.p_correct) for c in rows if c.answer and c.p_correct is not None))} for g,rows in groups.items()}
    matched={g:summarize([asdict(c) for c in rows if c.item_id in common],labels) for g,rows in groups.items()}
    for g,rows in groups.items():
        pairs=[(c.p_correct,int(c.answer=='ABCD'[labels[c.item_id]])) for c in rows if c.answer in ('A','B','C','D') and c.p_correct is not None]
        m=full[g]; correct=sum(c.answer=='ABCD'[labels[c.item_id]] for c in rows)
        if correct!=m['correct'] or len(pairs)!=m['brier_n']: raise ValueError('independent count mismatch')
        if pairs:
            if abs(sum((p-y)**2 for p,y in pairs)/len(pairs)-m['raw_brier'])>1e-12: raise ValueError('independent Brier mismatch')
            if abs(sum(p for p,y in pairs)/len(pairs)-m['mean_confidence'])>1e-12: raise ValueError('independent confidence mean mismatch')
    value=seal({'skip_id':plan['skip_id'],'bundle_id':frozen['bundle_id'],'cohort_id':cohort['cohort_id'],
        'development_label_sha256':file_digest(labels_path),'cohort_n':49,'original_pilot_n':50,
        'skipped_unknown_n':1,'skipped_positions':[22],'generators':full,
        'common_valid_n':len(common),'common_valid_generators':matched,
        'confidence_calibrated':False,'evaluation_labels_read':False,'new_api_requests_in_report':0,
        'collection_audit':status,'independent_metric_arithmetic_matches':True,
        'input_tokens':sum(e['usage']['input_tokens'] for e in estimates),
        'output_tokens_including_reasoning':sum(e['usage']['output_tokens'] for e in estimates),
        'finish_reasons':dict(finishes),'reasoning_content_present_n':reasoning,
        'confirmed_billing_usd':None,'actual_total_cost_unknown':True,
        'limitations':['Same 49 received-request items for all models; excluded position 22 has an unknown outcome and unknown actual cost.',
            'Received malformed replies are retained; accuracy uses all 49, confidence/Brier use valid pairs only.',
            'Raw subjective confidence without calibration. This is an exploratory development subset.',
            'Protocol/model/serving dates differ; Qwen uses two-step generation.',
            'Completion-based selection may bias results; not a full 50-item or Diamond evaluation.'],
        'unit_tests_run':False},'result_id')
    base.immutable(out/'result.json',value)
    if REPORT.exists():
        for line in (REPORT/'sha256.txt').read_text().splitlines():
            sha,name=line.split('  ',1)
            if file_digest(REPORT/name)!=sha: raise ValueError('existing report checksum mismatch')
        if json.loads((REPORT/'result.json').read_text())!=value: raise ValueError('existing report differs')
        return value
    REPORT.mkdir(parents=True)
    for name in ('plan.json','status.json','billing_check.json','result.json'):
        shutil.copyfile(out/name,REPORT/name)
    base.immutable(REPORT/'audit.json',status)
    names={'gemini_baseline':'Gemini','qwen_baseline':'Qwen（两步）','sonnet_5_5':'Sonnet 5.5','gpt_oss_120b':'gpt-oss-120b'}
    lines=['GPT-OSS跳过第22题：同一49题开发集离线比较','',
        '原21条回复原样保留，新增28条；第22题结果未知且不重试，其预算预留继续保留。',
        '四模型统一排除第22题。confidence未校准；格式无效回复没有修复。',
        '模型 | 正确/49 | 完整答案及confidence | 平均confidence | 答错confidence（n） | raw Brier（n）']
    for name,label in names.items():
        m=full[name]; wrong=m['mean_confidence_if_wrong']
        lines.append(f"{label} | {m['correct']}/49 ({m['accuracy_all']:.2%}) | {m['confidence_n']}/49 | {m['mean_confidence']:.2%} | {('未定义' if wrong is None else f'{wrong:.2%}')} ({m['wrong_with_confidence_n']}) | {m['raw_brier']:.6f} ({m['brier_n']})")
    lines.extend(['',f'四模型共同有效子集为{len(common)}题，汇总另见result.json的common_valid_generators；成员在读标签前冻结。',
        f"本次新增费用估算USD {status['new_estimated_usd']:.8f}；已收到49条回复的GPT-OSS累计估算USD {status['all_received_gptoss_estimated_usd']:.8f}。",
        f"未知请求仍保留USD {status['unknown_usage_reserved_usd']:.8f}预算；该数不是已确认支出或实际收费。",
        '共享预算USD1未变；计费项目llm-applications-490420。费用为provider token用量估算，不是确认账单。',
        f"截断/终止统计{dict(finishes)}；解析失败{status['invalid']}。",
        '请求/缓存/原21条回复及未知请求记录核对通过。沿用30秒间隔和明确429的60/120秒有限重试。',
        '先冻结候选及共同样本，再只读开发标签评分。未读取Diamond标签，未拟合likelihood，未调用verifier。',
        '49题是收到回复后的子集，可能有完成情况造成的选择偏差；不是完整50题结果。Qwen两步等协议差异不能归因于模型本身。',
        '本目录仅含汇总结果、计划、状态、审计及哈希，不含题干、答案键、原始回复、逐题对错或凭证。',
        '未提交或推送；本次未新增或运行单元测试。'])
    (REPORT/'README.txt').write_text('\n'.join(lines)+'\n')
    banned={'question','choices','answer','correct_index','response','provider_response','prompt','system','access_token','refresh_token','private_key','client_secret'}
    def safe(v):
        if isinstance(v,dict):
            if banned.intersection(v): raise ValueError('unsafe export field')
            for x in v.values(): safe(x)
        elif isinstance(v,list):
            for x in v: safe(x)
    for path in REPORT.glob('*.json'): safe(json.loads(path.read_text()))
    load(root)
    (REPORT/'sha256.txt').write_text(''.join(f'{file_digest(p)}  {p.name}\n' for p in sorted(REPORT.iterdir()) if p.is_file()))
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','collect','audit','report'])
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--allow-api',action='store_true')
    args=p.parse_args()
    if args.command=='prepare': value=prepare(args.root)
    elif args.command=='collect': value=collect(args.root,args.allow_api)
    elif args.command=='audit': value=audit(args.root)
    else: value=report(args.root)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__': main()
