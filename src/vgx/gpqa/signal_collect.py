"""Bounded parallel collection for a prepared, calibration-only signal study.

Requests for different candidates/arms are independent. Per-model pacing avoids
bursts. A transport/accounting error stops submission; cached invalid replies
are retained and never automatically replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import threading
import time

from vgx.common.api import OfflineCacheMiss, RequestCache
from vgx.common.budget import BudgetedRequestCache
from vgx.common.concurrency import parallel_map
from vgx.common.storage import atomic_json, digest, file_digest, file_lock
from vgx.gpqa.sequential import seal
from vgx.gpqa.signal_study import check_billing, load_plan, parse_signal, request_for, runner_for


class Pacer:
    def __init__(self, interval):
        if isinstance(interval,bool) or not isinstance(interval,(int,float)) or not math.isfinite(interval) or interval<0:
            raise ValueError('nonnegative finite model request interval required')
        self.interval=interval;self.lock=threading.Lock();self.next_time=0.

    def acquire(self):
        with self.lock:
            delay=self.next_time-time.monotonic()
            if delay>0:time.sleep(delay)
            self.next_time=time.monotonic()+self.interval


def collect(output, *, limit=10, allow_api=False):
    output=Path(output);plan,candidates=load_plan(output);cfg=plan['config']
    if type(limit) is not int or not 1<=limit<=len(candidates):
        raise ValueError('limit must be within calibration cohort')
    settings=cfg['collection'];workers=settings['workers']
    if type(workers) is not int or not 1<=workers<=20:
        raise ValueError('workers must be in 1..20')
    candidates=candidates[:limit]
    cache=BudgetedRequestCache(output/'cache',project=cfg['project'],pricing=cfg['pricing'],limit_usd=cfg['stage_budget_usd'])
    cached_only=RequestCache(output/'cache')
    runners={m['id']:runner_for(m,cfg['project']) for m in cfg['models']}
    pacers={m['id']:Pacer(settings['minimum_request_interval_seconds'][m['id']]) for m in cfg['models']}
    binding=seal({'study_id':plan['study_id'],'settings':settings,
                  'implementation':{name:file_digest(Path(__file__).resolve().parents[1]/name)
                                    for name in ('gpqa/signal_collect.py','common/concurrency.py')}},'collector_id')
    progress_lock=threading.Lock();results=[]
    def snapshot(error=None):
        with progress_lock:
            counts=Counter(r['arm'] for r in results)
            failures=Counter(r['arm']+':'+r['failure'] for r in results if r['failure'])
            value={'study_id':plan['study_id'],'collector_id':binding['collector_id'],
                   'candidate_limit':limit,'target_requests':limit*len(cfg['models'])*len(cfg['signals']),
                   'completed':len(results),'reused':sum(r['reused'] for r in results),
                   'by_arm':dict(counts),'parse_failures':dict(failures),
                   'error':error,'updated_at':time.time(),'budget':cache.summary()}
            atomic_json(output/f'parallel_collection_{limit}.json',value)
            return value
    def one(job):
        candidate,model,signal=job;runner=runners[model['id']]
        request=request_for(candidate,model,signal,plan['study_id'])
        try:
            # Cached replies need no pacing, and still pass budget reconciliation.
            cached_only.get(runner,request)
        except OfflineCacheMiss:
            if allow_api:pacers[model['id']].acquire()
        call,reused=cache.get(runner,request,allow_api=allow_api)
        _,failure=parse_signal(call.response,signal)
        with progress_lock:
            results.append({'arm':model['id']+':'+signal,'failure':failure,'reused':reused})
        snapshot()
    jobs=[]
    for c in candidates:
        for model in cfg['models']:
            for signal in sorted(cfg['signals'],key=lambda s:digest([cfg['seed'],c.item_id,s])):
                jobs.append((c,model,signal))
    with file_lock(output/'collection.lock'):
        path=output/'parallel_collector.json'
        if path.exists() and json.loads(path.read_text())!=binding:
            raise ValueError('collector changed; preserve the old study and prepare a new one')
        atomic_json(path,binding)
        if allow_api:atomic_json(output/'billing_check.json',check_billing(cfg['project']))
        try:
            parallel_map(one,jobs,workers)
        except Exception as exc:
            snapshot({'type':type(exc).__name__,'http_status':getattr(exc,'status',None)})
            raise
        return snapshot()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    print(json.dumps(collect(args.output,limit=args.limit,allow_api=args.allow_api)))


if __name__=='__main__':main()
