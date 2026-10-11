"""Offline initial-belief calibration and exact verifier-cache coverage audit.

No API collection, evaluation scoring, live-policy change, or verifier selection.
Use the existing fixed logistic specification and all prespecified fold seeds.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from vgx.common.api import RequestCache
from vgx.common.storage import atomic_json, digest, file_digest, read_jsonl
from vgx.gpqa import staged_generator as study
from vgx.gpqa.artifacts import FrozenCandidate, load_candidates
from vgx.gpqa.offline_validation import CorrectnessForecast
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.signal_analysis import cross_validate
from vgx.gpqa.signal_study import request_for, runner_for, parse_signal

SEEDS = [20261005,20261006,20261007,20261008,20261009]


def source(root):
    plan,items=study.load(root)
    old,pilot,_=study.previous.load(Path(plan['config']['previous_results']))
    baseline_root=Path(old['config']['baseline_bundle'])
    manifest,baseline=load_candidates(baseline_root)
    label_path=baseline_root/'calibration_records.jsonl'
    if file_digest(label_path)!=manifest['files'][label_path.name]:raise ValueError('calibration labels changed')
    labels={r['item_id']:r['correct_index'] for r in read_jsonl(label_path)}
    bundle=json.loads((Path(root)/'frozen_candidates.json').read_text());unseal(bundle,'bundle_id')
    if bundle['study_id']!=plan['study_id']:raise ValueError('candidate study changed')
    groups={'gemini':baseline,'qwen':[FrozenCandidate(**c) for c in bundle['groups']['qwen']]}
    expected={i.item_id for i in items if i.partition=='calibration'}
    public={i.item_id:i for i in items}
    for name,cs in groups.items():
        cs=[c for c in cs if c.partition=='calibration']
        if {c.item_id for c in cs}!=expected or len(cs)!=len(expected):raise ValueError('candidate membership differs')
        for c in cs:
            i=public[c.item_id]
            if (c.question,tuple(c.choices),c.subject)!=(i.question,i.choices,i.subject):raise ValueError('input content changed')
        groups[name]=sorted(cs,key=lambda c:c.item_id)
    return plan,groups,labels,{i.item_id for i in pilot},file_digest(label_path),bundle['bundle_id']


def fit_prior(rows):
    if any(r['partition']!='calibration' for r in rows):raise ValueError('calibration rows required')
    fits=[cross_validate(rows,[],{},folds=3,seed=seed,C=1.,clip=1e-4) for seed in SEEDS]
    fitted=CorrectnessForecast(C=1.,clip=1e-4).fit(rows)
    grid=[.25,.5,.8,.85,.9,.92,.95,.98,.99]
    mapping=fitted.predict([{'generator_p_correct':p,'verifiers':{}} for p in grid])
    return {'primary_oof':fits[0], 'fold_seed_sensitivity':[
        {'seed':seed,'metrics':fit.get('metrics'),'status':fit['status']} for seed,fit in zip(SEEDS,fits)],
        'review_only_full_calibration_fit':{'C':1.,'clip':1e-4,'feature':'standardized logit(raw confidence)',
            'coefficient':fitted.model.coef_.tolist(),'intercept':fitted.model.intercept_.tolist(),
            'scaler_mean':fitted.scaler.mean_.tolist(),'scaler_scale':fitted.scaler.scale_.tolist(),
            'mapping':dict(zip(map(str,grid),map(float,mapping))),
            'note':'Fitted on all rows for review; mapping is not held-out evidence and is not installed in a policy.'}}


def index_cache(roots):
    calls={}
    for root in roots:
        for log in RequestCache(root).logs().values():
            for row in log.records():
                if row.get('error') is not None or row.get('meta',{}).get('role')!='verifier':continue
                if row['key'] in calls:
                    a=calls[row['key']]
                    if a['response']!=row['response'] or a['meta'].get('execution_id')!=row['meta'].get('execution_id'):
                        raise ValueError('conflicting verifier cache executions')
                calls[row['key']]=row
    return calls


def audit_coverage(groups,labels,models,project,index):
    coverage={};missing=[];used={}
    for name,cs in groups.items():
        eligible=[c for c in cs if c.answer is not None and c.p_correct is not None]
        coverage[name]={}
        for model in models:
            runner=runner_for(model,project)
            for signal in ('binary','probability'):
                key=model['id']+':'+signal
                counts={'eligible':len(eligible),'cached_valid':0,'cached_invalid':0,'missing':0,
                        'cached_valid_correct':0,'cached_valid_wrong':0}
                for c in eligible:
                    req=request_for(c,model,signal);call_key=runner.cache_key(req);call=index.get(call_key)
                    if call is None:
                        counts['missing']+=1
                        missing.append({'generator':name,'item_id':c.item_id,'verifier':model['id'],
                                        'signal':signal,'request_key':call_key})
                        continue
                    if (call['prompt']!=req.prompt or call['meta'].get('system')!=req.system
                        or call['meta'].get('runner_identity')!=runner.identity or call['params']!=runner.params
                        or call['meta'].get('resource_project')!=project or call['meta'].get('quota_project')!=project):
                        raise ValueError('cached verifier identity/provenance mismatch')
                    used[call_key]=digest(call)
                    score,failure=parse_signal(call['response'],signal)
                    if failure:counts['cached_invalid']+=1
                    else:
                        counts['cached_valid']+=1
                        y=int('ABCD'.index(c.answer)==labels[c.item_id])
                        counts['cached_valid_correct' if y else 'cached_valid_wrong']+=1
                coverage[name][key]=counts
    return {'coverage':coverage,'missing_requests':missing,'unique_missing_requests':len({m['request_key'] for m in missing}),
            'used_cache_record_hashes':used,
            'note':'Incomplete cached subsets may favor shared answers. No likelihoods fitted on these subsets; no collection authorized by this audit.'}


def analyze(root,output,screen_config,cache_roots):
    root,output=Path(root),Path(output)
    plan,groups,labels,pilot_ids,label_hash,bundle_id=source(root)
    screen=json.loads(Path(screen_config).read_text())
    protocol=seal({'source_study_id':plan['study_id'],'frozen_bundle_id':bundle_id,'calibration_label_sha256':label_hash,
        'folds':3,'seeds':SEEDS,'C':1.,'clip':1e-4,'screen_config_sha256':file_digest(Path(screen_config)),
        'cache_roots':list(map(str,cache_roots)), 'cohorts':['all_calibration','remaining_calibration'],
        'implementation':{name:file_digest(Path(__file__).with_name(name)) for name in
            ['generator_calibration.py','signal_analysis.py','offline_validation.py','score.py','report.py']},
        'package_versions':{name:importlib.metadata.version(name) for name in ('numpy','scikit-learn','scipy')},
        'api_requests':0,'evaluation_labels_read':False},'analysis_id')
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('analysis changed; use a new directory')
    atomic_json(path,protocol)
    result={'analysis_id':protocol['analysis_id'],'generators':{}}
    for name,cs in groups.items():
        valid=[c for c in cs if c.answer is not None and c.p_correct is not None]
        rows=[{'item_id':c.item_id,'partition':'calibration','generator_answer':c.answer,'generator_p_correct':c.p_correct,
               'correct_index':labels[c.item_id],'outcome':int('ABCD'.index(c.answer)==labels[c.item_id]),'verifiers':{}} for c in valid]
        result['generators'][name]={'requested_n':len(cs),'valid_n':len(valid),'missing_n':len(cs)-len(valid),
            'all_calibration':fit_prior(rows),
            'remaining_calibration':fit_prior([r for r in rows if r['item_id'] not in pilot_ids])}
    result['verifier_audit']=audit_coverage(groups,labels,screen['models'],plan['config']['project'],index_cache(cache_roots))
    result['limitations']=['Exploratory calibration cross-validation after inspecting model results; no independent evaluation claim.',
        'Fixed OOF bootstrap does not include fitting/selection uncertainty.',
        'Full-data fit saved for review only; no live prior, likelihood, verifier order, cost or stopping table changed.',
        'Calibration improvement does not establish a 95% or 99% release guarantee.']
    atomic_json(output/'report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--screen-config',required=True,type=Path)
    p.add_argument('--cache',action='append',required=True);a=p.parse_args()
    r=analyze(a.root,a.output,a.screen_config,a.cache)
    print(json.dumps({'analysis_id':r['analysis_id'],'metrics':{g:d['all_calibration']['primary_oof']['metrics'] for g,d in r['generators'].items()},
                     'coverage':r['verifier_audit']['coverage'],'unique_missing_requests':r['verifier_audit']['unique_missing_requests']}))
