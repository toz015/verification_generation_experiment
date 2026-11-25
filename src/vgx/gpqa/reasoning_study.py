"""Prospective rationale-before-answer condition on a frozen calibration pilot.

Uses the same non-thinking endpoints and generation parameters as the direct
condition. This changes the prompt, not the provider's native reasoning mode.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import threading

from vgx.common.concurrency import parallel_map
from vgx.common.llm import Request
from vgx.common.storage import atomic_json, digest, file_digest, file_lock
from vgx.gpqa.artifacts import FrozenCandidate
from vgx.gpqa import generator_study as direct
from vgx.gpqa.prompt import SYSTEM, GeneratorResponse, parse_generator_response, build_verifier_prompt
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_collect import Pacer
from vgx.gpqa.signal_study import check_billing, parse_signal, request_for, runner_for

PROMPT_VERSION='rationale_then_final_v1'
INSTRUCTION=(
    'Solve the problem before choosing your final answer. Explain the reasoning and calculations '
    'needed to distinguish the options, checking your conclusion for mistakes. '
    'Then report your probability that your selected answer is correct.\n'
    'End your response with exactly one final block, and write nothing after it:\n'
    '<final>{"answer": "A"|"B"|"C"|"D", "p_correct": 0.0}</final>\n'
    'Replace the answer alternatives and example probability with your actual answer and probability. '
    'Put all explanation before the final block. Do not place any other JSON answer object in the explanation.'
)


def generator_request(item, operation=None):
    options='\n'.join(f'{letter}. {choice}' for letter,choice in zip('ABCD',item.choices))
    return Request(f'generator|{item.item_id}',
        f'Subject: {item.subject}\n\nQuestion: {item.question}\n\n{options}\n\n{INSTRUCTION}', SYSTEM,
        {'role':'generator','partition':item.partition,'item_id':item.item_id,
         'prompt_version':PROMPT_VERSION,'operation_id':operation})


def parse_final(text):
    # Never mistake an intermediate JSON object for the final decision.
    if text.count('<final>')!=1 or text.count('</final>')!=1:
        return GeneratorResponse(None,None,False,'missing_or_multiple_final_blocks')
    match=re.search(r'<final>(.*?)</final>\s*$',text,flags=re.DOTALL)
    if match is None:
        return GeneratorResponse(None,None,False,'invalid_final_block')
    try:
        value=json.loads(match.group(1))
    except (ValueError,TypeError):
        return GeneratorResponse(None,None,False,'invalid_final_json')
    if not isinstance(value,dict):
        return GeneratorResponse(None,None,False,'invalid_final_json')
    return parse_generator_response(json.dumps(value))


def candidate_for(item, runner, cache, operation, *, allow_api=False):
    request=generator_request(item,operation)
    call,_=cache.get(runner,request,allow_api=allow_api)
    parsed=parse_final(call.response)
    return FrozenCandidate(item.item_id,item.partition,item.subject,item.question,item.choices,
        parsed.answer,parsed.p_correct,parsed.ok,parsed.failure,runner.model,call.key,
        direct.digest_text(request.prompt),direct.digest_text(call.response),SYSTEM,
        build_verifier_prompt(item,parsed.answer) if parsed.answer else None)


def sources(config):
    old_plan,items,baseline=direct.load(Path(config['direct_results']))
    cfg=old_plan['config']
    for key in ('dataset_root','baseline_bundle','sample_size','generators','verifiers'):
        if config[key]!=cfg[key]:
            raise ValueError('prompt-only condition requires unchanged '+key)
    frozen=Path(config['direct_results'])/'frozen_candidates.json'
    bundle=json.loads(frozen.read_text());unseal(bundle,'bundle_id')
    if bundle['study_id']!=old_plan['study_id']:
        raise ValueError('direct candidates belong to another study')
    groups={name:[FrozenCandidate(**r) for r in rows] for name,rows in bundle['groups'].items()}
    for candidates in groups.values():
        if [c.item_id for c in candidates]!=[i.item_id for i in items]:
            raise ValueError('direct candidate sample differs')
        for c,i in zip(candidates,items):
            if (c.question,tuple(c.choices),c.subject,c.partition)!=(i.question,tuple(i.choices),i.subject,'calibration'):
                raise ValueError('direct input content differs')
    return old_plan,items,groups,file_digest(frozen)


def implementation():
    return {**direct.implementation(),'gpqa/reasoning_study.py':file_digest(Path(__file__))}


def prepare(config,output):
    if config['project']!='llm-applications-490420':raise ValueError('unauthorized project')
    old,items,_,source_hash=sources(config)
    cache=direct.budget(config,output);arms=[]
    for m in config['generators']:
        runner=runner_for(m,config['project']);requests=[generator_request(i) for i in items]
        arms.append({'id':m['id'],'identity':runner.identity,
                     'keys':[runner.cache_key(r) for r in requests],
                     'reservation_usd':sum(cache.reservation(runner,r) for r in requests)})
    value=seal({'config':config,'direct_study_id':old['study_id'],'direct_candidates_sha256':source_hash,
        'candidate_ids':[i.item_id for i in items],'implementation':implementation(),
        'prompt_version':PROMPT_VERSION,'instruction':INSTRUCTION,'system':SYSTEM,'arms':arms,
        'evaluation_requests':0},'study_id')
    path=Path(output)/'plan.json'
    with file_lock(str(path)+'.lock'):
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('changed study; use a new directory')
        atomic_json(path,value)
    return value


def load(output):
    plan=json.loads((Path(output)/'plan.json').read_text());unseal(plan,'study_id')
    old,items,groups,source_hash=sources(plan['config'])
    if (plan['implementation']!=implementation() or source_hash!=plan['direct_candidates_sha256']
        or old['study_id']!=plan['direct_study_id'] or plan['candidate_ids']!=[i.item_id for i in items]):
        raise ValueError('source or implementation changed')
    return plan,items,groups


def groups_for(plan,items,old,cache):
    groups={'gemini_baseline':old['gemini_baseline']}
    for model in plan['config']['generators']:
        name=model['id'];groups[name+':direct']=old[name]
        runner=runner_for(model,plan['config']['project'])
        groups[name+':reasoned']=[candidate_for(i,runner,cache,plan['study_id']) for i in items]
    return groups


def collect(output,stage,*,limit=None,allow_api=False):
    output=Path(output);plan,items,old=load(output);cfg=plan['config'];cache=direct.budget(cfg,output)
    limit=len(items) if limit is None else limit
    if type(limit)is not int or not 1<=limit<=len(items):raise ValueError('invalid limit')
    items=items[:limit];old={g:cs[:limit] for g,cs in old.items()}
    if stage not in ('generators','verifiers'):raise ValueError('invalid stage')
    models=cfg['generators'] if stage=='generators' else cfg['verifiers']
    pacers={m['id']:Pacer(m['interval_seconds']) for m in models}
    jobs=[(m['id'],i,m) for i in items for m in models] if stage=='generators' else [
        (group,c,m) for group,cs in groups_for(plan,items,old,cache).items()
        for c in cs if c.answer is not None for m in models]
    results=[];lock=threading.Lock()
    def snapshot(error=None):
        with lock:
            value={'study_id':plan['study_id'],'stage':stage,'limit':limit,'target_logical_requests':len(jobs),
                   'completed':len(results),'reused':sum(r['reused'] for r in results),
                   'failures':dict(Counter(r['failure'] for r in results if r['failure'])),
                   'error':error,'budget':cache.summary(),'results':list(results)}
            atomic_json(output/f'{stage}_{limit}.json',value);return value
    def one(job):
        group,item,model=job;runner=runner_for(model,cfg['project'])
        req=(generator_request(item,plan['study_id']) if stage=='generators' else
             request_for(item,model,'probability',plan['study_id']))
        reused=direct.import_exact(cache,runner,req,cfg['reuse_caches'])
        if allow_api and not reused:pacers[model['id']].acquire()
        call,cached=cache.get(runner,req,allow_api=allow_api)
        failure=parse_final(call.response).failure if stage=='generators' else parse_signal(call.response,'probability')[1]
        with lock:
            results.append({'group':group,'item_id':item.item_id,'model':model['id'],
                            'key':call.key,'failure':failure,'reused':cached})
        snapshot()
    with file_lock(output/'collection.lock'):
        if allow_api:atomic_json(output/'billing_check.json',check_billing(cfg['project']))
        try:parallel_map(one,jobs,cfg['workers'])
        except Exception as exc:
            snapshot({'type':type(exc).__name__,'http_status':getattr(exc,'status',None)});raise
        value=snapshot()
        if stage=='generators' and limit==cfg['sample_size']:
            groups=groups_for(plan,items,old,cache)
            bundle=seal({'study_id':plan['study_id'],'groups':{g:[asdict(c) for c in cs] for g,cs in groups.items()}},'bundle_id')
            path=output/'frozen_candidates.json'
            if path.exists() and digest(json.loads(path.read_text()))!=digest(bundle):raise ValueError('frozen answers changed')
            atomic_json(path,bundle)
        return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','generators','verifiers'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path)
    parser.add_argument('--limit',type=int);parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        v=prepare(json.loads(args.config.read_text()),args.output)
        print(json.dumps({'study_id':v['study_id'],'generator_reservation_usd':sum(a['reservation_usd'] for a in v['arms'])}))
    else:
        v=collect(args.output,args.command,limit=args.limit,allow_api=args.allow_api)
        print(json.dumps({k:x for k,x in v.items() if k!='results'}))
