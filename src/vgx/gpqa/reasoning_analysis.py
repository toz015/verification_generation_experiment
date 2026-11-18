"""Offline paired analysis of direct versus rationale-before-answer prompts."""
from collections import Counter
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from vgx.common.api import RequestCache
from vgx.common.billing import estimate_call, summarize_vertex_usage
from vgx.common.storage import atomic_json, file_digest, read_jsonl
from vgx.gpqa.generator_study import budget
from vgx.gpqa.reasoning_study import load, groups_for, generator_request
from vgx.gpqa.report import rate_interval
from vgx.gpqa.score import binary_forecast_metrics, correctness_outcome
from vgx.gpqa.signal_analysis import cross_validate, tail_diagnostics, dependence_summary, likelihood_stability
from vgx.gpqa.signal_study import request_for, runner_for, parse_signal


def paired_summary(direct,reasoned,*,seed=20261005,repeats=2000):
    """Both conditions use their own correctness label; resample paired questions."""
    if [r['item_id'] for r in direct]!=[r['item_id'] for r in reasoned]:
        raise ValueError('paired question order differs')
    yd=np.array([r['outcome'] for r in direct]);yr=np.array([r['outcome'] for r in reasoned])
    vd=np.array([r['generator_answer'] is not None for r in direct])
    vr=np.array([r['generator_answer'] is not None for r in reasoned])
    rng=np.random.default_rng(seed)
    def estimate(delta):
        if not len(delta):return {'n':0,'difference':None,'bootstrap95':None}
        ix=rng.integers(0,len(delta),(repeats,len(delta)))
        return {'n':len(delta),'difference':float(np.mean(delta)),
                'bootstrap95':np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist()}
    paired=[(d,r) for d,r in zip(direct,reasoned)
            if d['generator_p_correct'] is not None and r['generator_p_correct'] is not None
            and d['generator_answer'] is not None and r['generator_answer'] is not None]
    brier=np.array([(r['generator_p_correct']-r['outcome'])**2-(d['generator_p_correct']-d['outcome'])**2 for d,r in paired])
    return {'direction':'reasoned minus direct','accuracy':estimate(yr-yd),
            'wrong_to_correct':int(((yd==0)&vd&(yr==1)).sum()),
            'correct_to_wrong':int(((yd==1)&(yr==0)&vr).sum()),
            'correct_to_incomplete':int(((yd==1)&(~vr)).sum()),
            'incomplete_to_correct':int(((~vd)&(yr==1)).sum()),
            'raw_brier_on_matched_confidence':estimate(brier),
            'matched_direct_brier':float(np.mean([(d['generator_p_correct']-d['outcome'])**2 for d,r in paired])) if paired else None,
            'matched_reasoned_brier':float(np.mean([(r['generator_p_correct']-r['outcome'])**2 for d,r in paired])) if paired else None,
            'note':'Question bootstrap on this exploratory pilot; no model-selection or repeated-call uncertainty.'}


def analyze(root):
    root=Path(root);plan,items,old=load(root);cfg=plan['config'];cache=budget(cfg,root)
    groups=groups_for(plan,items,old,cache)
    label_path=Path(cfg['baseline_bundle'])/'calibration_records.jsonl'
    manifest=json.loads((Path(cfg['baseline_bundle'])/'bundle_manifest.json').read_text())
    if file_digest(label_path)!=manifest['files'][label_path.name]:raise ValueError('changed calibration labels')
    labels={r['item_id']:r['correct_index'] for r in read_jsonl(label_path)}
    all_rows={};summaries={};tags=[m['id'] for m in cfg['verifiers']]
    for group,candidates in groups.items():
        rows=[];failures=Counter()
        for c in candidates:
            row={'item_id':c.item_id,'partition':'calibration','generator_answer':c.answer,
                 'generator_p_correct':c.p_correct,'correct_index':labels[c.item_id],
                 'outcome':correctness_outcome(c.answer,labels[c.item_id]),'verifiers':{}}
            if c.generator_failure:failures['generator:'+c.generator_failure]+=1
            if c.answer is not None:
                for model in cfg['verifiers']:
                    call,_=cache.get(runner_for(model,cfg['project']),request_for(c,model,'probability',plan['study_id']))
                    p,failure=parse_signal(call.response,'probability')
                    if failure:failures[model['id']+':'+failure]+=1
                    else:row['verifiers'][model['id']]={'p_correct':p}
            rows.append(row)
        all_rows[group]=rows
        valid=[r for r in rows if r['generator_answer'] is not None and r['generator_p_correct'] is not None]
        y=[r['outcome'] for r in valid];p=[r['generator_p_correct'] for r in valid]
        complete=[r for r in valid if all(t in r['verifiers'] for t in tags)]
        joint=cross_validate(complete,tags,{t:'probability' for t in tags},seed=cfg['seed'])
        sensitivity=[]
        for seed in cfg['protocol']['fold_seeds']:
            fit=joint if seed==cfg['seed'] else cross_validate(complete,tags,{t:'probability' for t in tags},seed=seed)
            sensitivity.append({'seed':seed,'status':fit['status'],'metrics':fit.get('metrics'),
                                'differences_vs_calibrated_generator':fit.get('differences_vs_calibrated_generator')})
        summaries[group]={'n':len(rows),'correct':sum(r['outcome'] for r in rows),
             'valid_answers':sum(c.answer is not None for c in candidates),
             'incomplete_answers':sum(c.answer is None for c in candidates),
             'incorrect_completed_answers':sum(c.answer is not None and r['outcome']==0 for c,r in zip(candidates,rows)),
             'accuracy_wilson95':rate_interval([r['outcome'] for r in rows]),
             'valid_confidence':len(valid),'failures':dict(failures),
             'mean_confidence':float(np.mean(p)) if p else None,
             'mean_confidence_if_wrong':float(np.mean([v for v,t in zip(p,y) if not t])) if any(t==0 for t in y) else None,
             'raw_metrics':binary_forecast_metrics(y,p).to_dict(),'raw_tails':tail_diagnostics(y,{'raw':p}),
             'initial_calibration':cross_validate(valid,[],{},seed=cfg['seed']),
             'verifier_comparison':joint,'fold_sensitivity':sensitivity,
             'dependence':dependence_summary(complete,tags),
             'likelihoods':{t:likelihood_stability(rows,t,'probability',seed=cfg['seed']) for t in tags},
             'verifier_rejections_at_0_5':{t:{
                 'valid':sum(t in r['verifiers'] for r in rows),
                 'wrong_rejected':sum(r['outcome']==0 and r['verifiers'][t]['p_correct']<.5 for r in rows if t in r['verifiers']),
                 'correct_rejected':sum(r['outcome']==1 and r['verifiers'][t]['p_correct']<.5 for r in rows if t in r['verifiers'])} for t in tags}}
    paired={m['id']:paired_summary(all_rows[m['id']+':direct'],all_rows[m['id']+':reasoned'],seed=cfg['seed'],
                                   repeats=cfg['protocol']['bootstrap_repeats']) for m in cfg['generators']}
    generation_usage={}
    for model in cfg['generators']:
        rows=[];runner=runner_for(model,cfg['project'])
        for item in items:
            call,_=cache.get(runner,generator_request(item,plan['study_id']))
            rows.append({'rationale_characters':len(call.response.split('<final>')[0]),
                         'output_tokens':estimate_call(vars(call),cfg['pricing'])['usage']['output_tokens'],
                         'estimated_usd':estimate_call(vars(call),cfg['pricing'])['estimated_usd'],
                         'provider_error_envelope':bool(call.meta['provider_response'].get('error')),
                         'finish_reasons':[c.get('finish_reason') for c in call.meta['provider_response'].get('choices',[])]})
        priced=[r for r in rows if r['estimated_usd'] is not None]
        tokenized=[r for r in rows if r['output_tokens'] is not None]
        generation_usage[model['id']]={'mean_rationale_characters':float(np.mean([r['rationale_characters'] for r in rows])),
            'mean_output_tokens':float(np.mean([r['output_tokens'] for r in tokenized])) if tokenized else None,
            'output_usage_n':len(tokenized),'usage_missing_n':len(rows)-len(tokenized),
            'estimated_usd_for_priced_calls':sum(r['estimated_usd'] for r in priced),
            'provider_error_envelopes':sum(r['provider_error_envelope'] for r in rows),
            'length_finish_count':sum('length' in r['finish_reasons'] for r in rows)}
        historical=[]
        old_cache=RequestCache(Path(cfg['direct_results'])/'cache')
        for candidate in old[model['id']]:
            matches=[r for r in old_cache.log(candidate.generator_call_key).records()
                     if r.get('key')==candidate.generator_call_key and r.get('error') is None]
            if len(matches)!=1:raise ValueError('missing or ambiguous direct generator response')
            historical.append(estimate_call(matches[0],cfg['pricing']))
        generation_usage[model['id']]['direct_condition']={
            'mean_output_tokens':float(np.mean([r['usage']['output_tokens'] for r in historical])),
            'estimated_usd':sum(r['estimated_usd'] for r in historical),
            'note':'Historical costs only; excluded from new spend.'}
    usage=summarize_vertex_usage(cache.logs(),cfg['pricing'])
    calls=[r for log in cache.logs().values() for r in log.records() if r.get('error') is None]
    audit={'study_id':plan['study_id'],'all_projects_match':all(r['meta'].get('resource_project')==cfg['project']
            and r['meta'].get('quota_project')==cfg['project'] for r in calls),
           'evaluation_calls':sum(r['meta'].get('partition')=='evaluation' for r in calls),
           'unique_received_http200_records':len({r['meta']['execution_id'] for r in calls}),
           'http200_error_envelopes':sum(bool(r['meta'].get('provider_response',{}).get('error')) for r in calls),
           'cache_valuation_including_historical_imports':usage,
           'new_spend':cache.summary(),'generator_usage':generation_usage}
    recovery=cfg.get('recovery',{})
    costs={'current_ledger_known_usd':cache.summary()['estimated_usd'],
           'recovered_known_usd':recovery.get('known_usage_estimate_usd',0.),
           'unknown_usage_reserved_usd':recovery.get('unknown_usage_reserved_usd',0.),
           'unknown_usage_executions':recovery.get('unknown_usage_executions',0),
           'confirmed_billing_usd':None,
           'note':'Known usage estimates plus a conservative reservation for missing usage; historical verifier imports excluded.'}
    costs['total_known_usage_estimate_usd']=costs['current_ledger_known_usd']+costs['recovered_known_usd']
    costs['total_with_unknown_reservation_usd']=costs['total_known_usage_estimate_usd']+costs['unknown_usage_reserved_usd']
    audit['combined_stage_accounting']=costs
    value={'study_id':plan['study_id'],'analysis_sha256':file_digest(Path(__file__)),
           'analysis_dependencies':{name:file_digest(Path(__file__).with_name(name))
                for name in ('signal_analysis.py','offline_validation.py','score.py','report.py')},
           'package_versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn')},
           'generators':summaries,'paired_conditions':paired,'new_spend':cache.summary(),
           'combined_stage_accounting':costs,'generator_usage':generation_usage,'limitations':cfg['limitations']}
    atomic_json(root/'comparison.json',value);atomic_json(root/'analysis_records.json',all_rows);atomic_json(root/'audit.json',audit)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    result=analyze(parser.parse_args().root)
    print(json.dumps({'generators':{g:{k:s[k] for k in ('n','correct','valid_confidence','mean_confidence','failures')}
                                  for g,s in result['generators'].items()},'paired':result['paired_conditions'],'new_spend':result['new_spend']}))
