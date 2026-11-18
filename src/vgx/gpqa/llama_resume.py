"""Resume only missing Llama requests, retaining the original study and budget.

No Qwen API path. Existing HTTP-200 replies, including malformed ones, are
immutable. Only explicit HTTP 429 rejections receive bounded slow retries.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError
from vgx.common.storage import atomic_json, digest, file_digest, file_lock
from vgx.gpqa import staged_generator as study
from vgx.gpqa.expansion_analysis import analyze
from vgx.gpqa.sequential import seal, unseal


def cached_calls(plan, items, cache):
    """Require all Qwen drafts/finals offline; record every existing Llama reply."""
    project=plan['config']['project'];calls={};missing=[]
    qwen=study.runner_for(plan['models']['qwen'],project)
    final=study.runner_for(plan['models']['qwen_final'],project)
    llama=study.runner_for(plan['models']['llama'],project)
    for item in items:
        draft,_=cache.get(qwen,study.draft_request(item))
        answer,_=cache.get(final,study.final_request(item,draft))
        for call in (draft,answer):calls[call.key]=digest(asdict(call))
        req=study.previous.generator_request(item)
        try:
            call,_=cache.get(llama,req);calls[call.key]=digest(asdict(call))
        except OfflineCacheMiss:missing.append(item.item_id)
    return calls,missing


def verify_preserved(cache, hashes):
    for key, expected in hashes.items():
        rows=[r for r in cache.log(key).records() if r.get('key')==key and r.get('error') is None]
        if len(rows)!=1 or digest(rows[0])!=expected:
            raise ValueError('an existing provider reply changed or became ambiguous')


def run(root, output, *, allow_api=False, interval=10.):
    if interval<10:raise ValueError('resume requires at least 10 seconds between requests')
    root,output=Path(root),Path(output);plan,items=study.load(root);cfg=plan['config']
    cache=study.direct.budget(cfg,root)
    state={'study_id':plan['study_id'],'root':str(root),'updated_at':time.time()}
    def snapshot(stage,**extra):
        state.update(stage=stage,updated_at=time.time(),budget=cache.summary(),**extra)
        atomic_json(output/'status.json',state)
        return state
    with file_lock(output/'resume.lock'):
        try:
            with file_lock(root/'collection.lock'):
                calls,missing=cached_calls(plan,items,cache)
                binding={'study_id':plan['study_id'],'implementation_sha256':file_digest(Path(__file__)),
                         'interval_seconds':interval,'retry_backoff_seconds':[60,120],
                         'max_attempts':3,'budget_limit_usd':cfg['stage_budget_usd']}
                path=output/'manifest.json'
                if path.exists():
                    manifest=json.loads(path.read_text());unseal(manifest,'resume_id')
                    if manifest['binding']!=binding:raise ValueError('resume configuration changed')
                    verify_preserved(cache,manifest['existing_call_hashes'])
                else:
                    manifest=seal({'binding':binding,'existing_call_hashes':calls,
                                   'initial_missing_llama_ids':missing,'before_budget':cache.summary()},'resume_id')
                    atomic_json(path,manifest)
                    atomic_json(output/'previous_collection.json',json.loads((root/'collection.json').read_text()))
                snapshot('collecting_llama',initial_missing=len(manifest['initial_missing_llama_ids']),
                         completed_llama=len(items)-len(missing),target_llama=len(items),new_calls_this_run=0)
                if allow_api:atomic_json(output/'billing_check.json',study.check_billing(cfg['project']))
                runner=study.runner_for(plan['models']['llama'],cfg['project'])
                pacer=study.Pacer(interval);new_calls=0;completed=0
                for item in items:
                    req=study.previous.generator_request(item,plan['study_id'])
                    study.direct.import_exact(cache,runner,req,cfg['reuse_caches'])
                    try:call,_=cache.get(runner,req)
                    except OfflineCacheMiss:
                        if not allow_api:raise
                        # Use the same model lock as BudgetedRequestCache.get,
                        # replacing its short backoff with 60/120-second waits.
                        with file_lock(cache.root/'provider-locks'/f'{digest(runner.model)}.lock'):
                            for attempt in range(3):
                                pacer.acquire()
                                try:
                                    call,_=cache._get_once(runner,req,allow_api=True)
                                    new_calls+=1;break
                                except ProviderRejectedError as exc:
                                    if exc.status!=429 or attempt==2:raise
                                    snapshot('backing_off',http_status=429,retry_number=attempt+1,
                                             retry_wait_seconds=60*(attempt+1))
                                    time.sleep(60*(attempt+1))
                    completed+=1
                    snapshot('collecting_llama',completed_llama=completed,new_calls_this_run=new_calls,
                             last_failure=study.previous.parse_final(call.response).failure)
                verify_preserved(cache,manifest['existing_call_hashes'])
                snapshot('freezing_offline',preserved_existing_calls=len(manifest['existing_call_hashes']))
            # All provider calls are now cached. This only builds the original
            # study's frozen bundle and cannot send a paid request.
            study.collect(root,allow_api=False)
            report=analyze(root,staged=True)
            current=cache.summary()
            return snapshot('complete',calibration_summary=str(root/'calibration_summary.json'),
                summaries=report['summaries'],evaluation_labels_read=False,
                incremental_estimated_usd=current['estimated_usd']-manifest['before_budget']['estimated_usd'])
        except Exception as exc:
            snapshot('blocked',error_type=type(exc).__name__,http_status=getattr(exc,'status',None),error=str(exc))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-api',action='store_true')
    p.add_argument('--interval',type=float,default=10.);a=p.parse_args()
    print(json.dumps(run(a.root,a.output,allow_api=a.allow_api,interval=a.interval)))
