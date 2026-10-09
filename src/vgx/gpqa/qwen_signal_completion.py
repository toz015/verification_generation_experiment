"""Complete only absent verifier requests for frozen development Qwen answers.

Uses unchanged screening requests, preserves returned failures, blocks uncertain
attempts, and never sends generator/evaluation requests. Explicit API opt-in.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import threading
import time

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.budget import BudgetedRequestCache
from vgx.common.concurrency import parallel_map
from vgx.common.storage import atomic_json, digest, file_digest, read_jsonl
from vgx.gpqa.generator_calibration import source, index_cache, audit_coverage
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import request_for, runner_for, parse_signal, check_billing, implementation_hashes


def budget_inventory():
    entries=[]
    for p in sorted(Path('results').glob('**/budget.json')):
        d=json.loads(p.read_text())
        if d['binding']['project']!='llm-applications-490420':
            raise ValueError('unexpected project ledger')
        values=list(d['requests'].values())
        entries.append({'path':str(p),'sha256':file_digest(p),
            'estimated_usd':sum(e['estimated_usd'] for e in values if e['status']=='settled'),
            'outstanding_reserved_usd':sum(e['reserved_usd'] for e in values if e['status'] not in ('settled','rejected')),
            'unsettled_states':dict(Counter(e['status'] for e in values if e['status'] not in ('settled','rejected')))})
    return {'ledgers':entries,'committed_estimate_and_reservations_usd':sum(
        e['estimated_usd']+e['outstanding_reserved_usd'] for e in entries),
        'confirmed_billing_usd':None,'note':'Local experiment only; outstanding historical usage is reserved, not treated as free.'}


def inputs(config):
    plan,groups,labels,pilot,label_hash,bundle=source(Path(config['generator_results']))
    screen=json.loads(Path(config['screen_config']).read_text())
    index=index_cache(config['cache_roots'])
    audit=audit_coverage(groups,labels,screen['models'],screen['project'],index)
    return groups,screen,index,audit,bundle


def prepare(config_path,output):
    cfg=json.loads(Path(config_path).read_text());groups,screen,index,audit,bundle=inputs(cfg)
    if screen['project']!='llm-applications-490420':raise ValueError('unauthorized project')
    missing=[r for r in audit['missing_requests'] if r['generator']=='qwen']
    if len({r['request_key'] for r in missing})!=len(missing):raise ValueError('duplicate requests')
    # Historical uncertain outcomes are not eligible for automatic resubmission.
    for root in cfg['cache_roots']:
        cache=RequestCache(root)
        for job in missing:
            attempts=read_jsonl(str(cache.log(job['request_key']).path)+'.attempts.jsonl',tolerate_tail=True)
            started={r['execution_id'] for r in attempts if r['event']=='started'}
            rejected={r['execution_id'] for r in attempts if r['event']=='rejected'}
            if started-rejected:raise ValueError('unresolved historical attempt')
    inventory=budget_inventory();cap=5.
    if inventory['committed_estimate_and_reservations_usd']+cap>200:
        raise ValueError('insufficient authorized total budget')
    # Also retain the pre-existing USD20 signal-study allowance.
    signal_committed=sum(e['estimated_usd']+e['outstanding_reserved_usd'] for e in inventory['ledgers']
        if '/gpqa_signal_study' in e['path'])
    if signal_committed+cap>20:raise ValueError('insufficient signal-study allowance')
    qwen={c.item_id:c for c in groups['qwen']};models={m['id']:m for m in screen['models']}
    planning=0.
    for job in missing:
        m=models[job['verifier']];runner=runner_for(m,screen['project']);req=request_for(qwen[job['item_id']],m,job['signal'])
        if runner.cache_key(req)!=job['request_key']:raise ValueError('request identity mismatch')
        rate=screen['pricing']['usd_per_million_tokens'][runner.model]
        planning+=(len((req.prompt+req.system).encode())+1024)*rate['input']/1e6+1024*rate['output']/1e6
    hashes=implementation_hashes();hashes['gpqa/qwen_signal_completion.py']=file_digest(Path(__file__))
    hashes['gpqa/generator_calibration.py']=file_digest(Path(__file__).with_name('generator_calibration.py'))
    plan=seal({'config':cfg,'screen':screen,'source_bundle_id':bundle,
        'candidate_hashes':{g:digest([asdict(c) for c in cs]) for g,cs in groups.items()},
        'existing_response_hashes':audit['used_cache_record_hashes'],'missing_requests':missing,
        'implementation_hashes':hashes,'project':screen['project'],'stage_limit_usd':cap,
        'budget_before':inventory,'planning_at_1024_output_tokens_usd':planning,
        'generator_calls':0,'evaluation_calls':0,'prior_calibration':False,
        'workers':5,'minimum_interval_seconds':3.,
        'note':'Unknown historical usage remains reserved. Current requests block on unknown outcome or unpriced usage.'},'completion_id')
    atomic_json(Path(output)/'plan.json',plan)
    return plan


def run(config_path,output,allow_api=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'plan.json'
        plan=json.loads(path.read_text()) if path.exists() else prepare(config_path,output)
        unseal(plan,'completion_id')
        if json.loads(Path(config_path).read_text())!=plan['config']:raise ValueError('config changed')
        hashes=implementation_hashes();hashes['gpqa/qwen_signal_completion.py']=file_digest(Path(__file__))
        hashes['gpqa/generator_calibration.py']=file_digest(Path(__file__).with_name('generator_calibration.py'))
        if hashes!=plan['implementation_hashes']:raise ValueError('implementation changed')
        groups,screen,index,audit,bundle=inputs(plan['config'])
        if bundle!=plan['source_bundle_id'] or any(digest([asdict(c) for c in groups[g]])!=h for g,h in plan['candidate_hashes'].items()):
            raise ValueError('frozen candidates changed')
        if any(digest(index[k])!=h for k,h in plan['existing_response_hashes'].items()):raise ValueError('existing responses changed')
        cache=BudgetedRequestCache(output/'cache',project=plan['project'],pricing=screen['pricing'],limit_usd=plan['stage_limit_usd'])
        candidates={c.item_id:c for c in groups['qwen']}
        results=[];mutex=threading.Lock();stop=threading.Event()
        def snapshot(stage,**extra):
            with mutex:
                value={'completion_id':plan['completion_id'],'stage':stage,'target':len(plan['missing_requests']),
                    'completed':len(results),'new_calls_this_invocation':sum(not r['reused'] for r in results),
                    'by_arm':dict(Counter(r['arm'] for r in results)),
                    'invalid_signals':dict(Counter(r['arm']+':'+r['failure'] for r in results if r['failure'])),
                    'budget':cache.summary(),'updated_at':time.time(),**extra}
                atomic_json(output/'status.json',value)
                return value
        def model_job(model):
            runner=runner_for(model,plan['project']);pacer=Pacer(plan['minimum_interval_seconds'])
            try:
                for job in plan['missing_requests']:
                    if job['verifier']!=model['id']:continue
                    if stop.is_set():return
                    req=request_for(candidates[job['item_id']],model,job['signal'],plan['completion_id'])
                    if runner.cache_key(req)!=job['request_key']:raise ValueError('unplanned request')
                    # A response that arrived elsewhere since preparation is reused, even if malformed.
                    if job['request_key'] in index:
                        from vgx.common.llm import Call
                        cache.import_call(Call(**index[job['request_key']]))
                    try:call,reused=cache.get(runner,req)
                    except OfflineCacheMiss:
                        if not allow_api:raise
                        pacer.acquire()
                        if stop.is_set():return
                        call,reused=cache.get(runner,req,allow_api=True)
                    score,failure=parse_signal(call.response,job['signal'])
                    with mutex:results.append({'arm':model['id']+':'+job['signal'],'failure':failure,'reused':reused})
                    snapshot('collecting')
            except Exception:
                stop.set();raise
        try:
            snapshot('preflight')
            if not allow_api:
                return snapshot('prepared_offline',planning_estimated_usd=plan['planning_at_1024_output_tokens_usd'])
            atomic_json(output/'billing_check.json',check_billing(plan['project']))
            # Recheck local aggregate immediately before sending. The stage limit is a sublimit of USD200.
            inventory=budget_inventory();remaining=plan['stage_limit_usd']-cache.summary()['estimated_usd']
            if inventory['committed_estimate_and_reservations_usd']+remaining>200:raise ValueError('total budget blocked')
            parallel_map(model_job,screen['models'],plan['workers'])
            if len(results)!=len(plan['missing_requests']):raise ValueError('incomplete collection')
            end_groups,_,end_index,_,end_bundle=inputs(plan['config'])
            if end_bundle!=bundle or any(digest([asdict(c) for c in end_groups[g]])!=h for g,h in plan['candidate_hashes'].items()):
                raise ValueError('source candidates changed')
            if any(digest(end_index[k])!=h for k,h in plan['existing_response_hashes'].items()):raise ValueError('source responses changed')
            return snapshot('complete',preserved_sources=True,evaluation_labels_read=False)
        except Exception as exc:
            snapshot('blocked',error_type=type(exc).__name__,http_status=getattr(exc,'status',None))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/gpqa_raw_prior_analysis.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-api',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.config,a.output,a.allow_api)))
