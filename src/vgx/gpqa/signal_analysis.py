"""Exploratory paired calibration-fold forecasts for the five-verifier screen.

No model requests. Does not select a winning model, prompt, or live policy.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from vgx.common.storage import atomic_json, file_digest, read_jsonl
from vgx.gpqa.offline_validation import CorrectnessForecast
from vgx.gpqa.report import _paired_forecasts, rate_interval
from vgx.gpqa.score import binary_forecast_metrics, correctness_outcome, fit_verifier_likelihood
from vgx.gpqa.signal_study import load_plan


def dependence_summary(rows, tags):
    """Descriptive correlations only; few errors cannot establish independence."""
    result={}
    for label in (0,1):
        subset=[r for r in rows if r['outcome']==label]
        correlations=[]
        names=['generator_confidence',*tags]
        columns=[[r['generator_p_correct'] for r in subset]]+[
            [r['verifiers'][tag]['p_correct'] for r in subset] for tag in tags]
        for i,left in enumerate(names):
            for j in range(i+1,len(names)):
                x,y=columns[i],columns[j]
                rho=float(np.corrcoef(x,y)[0,1]) if len(subset)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None
                correlations.append({'left':left,'right':names[j],'pearson':rho})
        result[str(label)]={'n':len(subset),'pairs':correlations}
    return result


def tail_diagnostics(y, predictions):
    result={}
    y=np.asarray(y)
    for name,p in predictions.items():
        values=[]
        for threshold in (.95,.99):
            chosen=np.asarray(p)>=threshold
            outcomes=y[chosen]
            values.append({'threshold':threshold,'n':int(chosen.sum()),
                           'wrong':int((1-outcomes).sum()),
                           'accuracy':float(outcomes.mean()) if len(outcomes) else None,
                           'accuracy_wilson95':rate_interval(outcomes),
                           'note':'Descriptive OOF tail, not a certified release guarantee.'})
        result[name]=values
    return result


def likelihood_stability(rows, tag, signal, *, repeats=200, seed=20261005):
    usable=[r for r in rows if tag in r['verifiers']]
    outcomes=[r['outcome'] for r in usable]
    scores=[r['verifiers'][tag]['p_correct'] for r in usable]
    if len(set(outcomes))<2:
        return {'status':'insufficient_correctness_classes','n':len(usable)}
    bins=2 if signal=='binary' else 3
    fit=fit_verifier_likelihood(outcomes,scores,bins=bins,laplace=1.)
    y=np.array(outcomes);p=np.array(scores);rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(y==label) for label in (0,1)]
    tables=[]
    for _ in range(repeats):
        ids=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups])
        sample=fit_verifier_likelihood(y[ids].tolist(),p[ids].tolist(),bins=bins,laplace=1.)
        tables.append([sample.p_bin_if_correct,sample.p_bin_if_incorrect])
    tables=np.asarray(tables);ratios=tables[:,0,:]/tables[:,1,:]
    point=np.array(fit.p_bin_if_correct)/np.array(fit.p_bin_if_incorrect)
    interval=lambda values:{'low':np.quantile(values,.025,axis=0).tolist(),'high':np.quantile(values,.975,axis=0).tolist()}
    return {'status':'descriptive_calibration_bootstrap','n':len(usable),'wrong':int((y==0).sum()),
            'edges':list(fit.edges),'p_bin_if_correct':list(fit.p_bin_if_correct),
            'p_bin_if_incorrect':list(fit.p_bin_if_incorrect),'likelihood_ratio':point.tolist(),
            'counts_by_correctness':{str(label):np.bincount([fit.bin_index(float(s)) for s in p[y==label]],minlength=bins).tolist()
                                     for label in (0,1)},
            'p_correct_interval':interval(tables[:,0,:]),'p_incorrect_interval':interval(tables[:,1,:]),
            'likelihood_ratio_interval':interval(ratios),'resamples':repeats,
            'ratio_nondecreasing_in_score':bool(np.all(np.diff(point)>=0)),
            'note':'Resample within correctness classes; conditional on class counts, bins and Laplace smoothing. '
                   'Not a test of independence or a bound on deployment error.'}


def cross_validate(rows, tags, formats, *, folds=3, seed=20261005, C=1., clip=1e-4, allow_missing=False):
    """Every transform/likelihood is fitted on training folds of the same cohort."""
    y=np.array([r['outcome'] for r in rows]);raw=np.array([r['generator_p_correct'] for r in rows])
    if len(rows)<folds or min(int((y==0).sum()),int((y==1).sum()))<folds:
        return {'n':len(rows),'wrong':int((y==0).sum()),'status':'insufficient_correctness_classes_for_folds'}
    names=['base_rate','calibrated_generator']+[f'joint:{tag}' for tag in tags]
    if not allow_missing:names += [f'bayes:{tag}' for tag in tags]
    predictions={name:np.zeros(len(rows)) for name in names}
    predictions['raw_generator']=raw
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    fold_membership=[]
    for tr,va in cv.split(rows,y):
        train=[rows[i] for i in tr];val=[rows[i] for i in va]
        fold_membership.append({'train':[r['item_id'] for r in train],'validation':[r['item_id'] for r in val]})
        prior_model=CorrectnessForecast(C=C,clip=clip).fit(train)
        prior=prior_model.predict(val)
        predictions['base_rate'][va]=float(y[tr].mean())
        predictions['calibrated_generator'][va]=prior
        for tag in tags:
            model=CorrectnessForecast((tag,),C=C,clip=clip).fit(train)
            predictions['joint:'+tag][va]=model.predict(val)
            if allow_missing:
                continue  # Missing indicators and medians fitted by training fold; no Bayes imputation.
            channel=fit_verifier_likelihood(y[tr].tolist(),[r['verifiers'][tag]['p_correct'] for r in train],
                                           bins=2 if formats[tag]=='binary' else 3,laplace=1.)
            predictions['bayes:'+tag][va]=[channel.posterior(float(b),r['verifiers'][tag]['p_correct']) for b,r in zip(prior,val)]
    metrics={name:binary_forecast_metrics(y.tolist(),p.tolist()).to_dict() for name,p in predictions.items()}
    baseline=metrics['calibrated_generator']
    return {'status':'exploratory_oof','n':len(rows),'wrong':int((y==0).sum()),'metrics':metrics,
            'missing_signal_handling':'training_fold_median_and_indicator' if allow_missing else 'complete_cohort',
            'differences_vs_calibrated_generator':{name:{key:metrics[name][key]-baseline[key] for key in ('brier','log_loss')}
                                                  for name in metrics if name!='calibrated_generator'},
            'folds':fold_membership,
            'high_confidence_tails':tail_diagnostics(y,predictions),
            'paired_prediction_resampling':{
                name:_paired_forecasts(y,predictions['calibrated_generator'],p,1000,seed)
                for name,p in predictions.items() if name!='calibrated_generator'},
            'resampling_scope':'Fixed OOF predictions; descriptive only. Overlapping training folds, '
                               'model fitting and selection uncertainty are not included.',
            'predictions':{name:p.tolist() for name,p in predictions.items()},
            'item_ids':[r['item_id'] for r in rows]}


def analyze(output):
    output=Path(output);plan,candidates=load_plan(output);config=plan['config'];protocol=config['protocol']
    bundle=Path(config['bundle']);manifest=json.loads((bundle/'bundle_manifest.json').read_text())
    label_path=bundle/'calibration_records.jsonl'
    if file_digest(label_path)!=manifest['files'][label_path.name]:
        raise ValueError('calibration records changed')
    labels={r['item_id']:r for r in read_jsonl(label_path)}
    if set(labels)!=set(manifest['calibration_ids']) or any(r['partition']!='calibration' for r in labels.values()):
        raise ValueError('invalid calibration labels')
    rows={c.item_id:{'item_id':c.item_id,'partition':'calibration','generator_answer':c.answer,
           'generator_p_correct':c.p_correct,'correct_index':labels[c.item_id]['correct_index'],
           'outcome':correctness_outcome(c.answer,labels[c.item_id]['correct_index']),'verifiers':{}} for c in candidates}
    records_path=output/'signal_records.json';records=json.loads(records_path.read_text())
    formats={m['id']+':'+s:s for m in config['models'] for s in config['signals']}
    allowed_keys={a['id']:dict(zip(plan['candidate_ids'],a['request_keys'])) for a in plan['arms']}
    seen=set()
    for r in records:
        tag=r['model']+':'+r['signal'];key=(r['item_id'],tag)
        if key in seen or r['item_id'] not in rows or tag not in formats:
            raise ValueError('duplicate or out-of-study record')
        seen.add(key)
        if r['request_key']!=allowed_keys[tag][r['item_id']] or r['correct']!=rows[r['item_id']]['outcome']:
            raise ValueError('record provenance mismatch')
        if r['failure'] is None and r['score'] is not None:
            rows[r['item_id']]['verifiers'][tag]={'p_correct':r['score']}
    kwargs={'folds':protocol['folds'],'seed':config['seed'],'C':protocol['prior_logistic_C'],
            'clip':protocol['prior_logit_clip']}
    prior=cross_validate(list(rows.values()),[],{},**kwargs)
    pairs={}
    for model in config['models']:
        tags=[model['id']+':'+s for s in config['signals']]
        usable=[r for r in rows.values() if all(t in r['verifiers'] for t in tags)]
        fit=cross_validate(usable,tags,formats,**kwargs)
        if fit['status']=='exploratory_oof':
            fit['probability_minus_binary']={
                kind:_paired_forecasts([r['outcome'] for r in usable],
                                       fit['predictions'][kind+':'+model['id']+':binary'],
                                       fit['predictions'][kind+':'+model['id']+':probability'],1000,config['seed'])
                for kind in ('joint','bayes')}
        pairs[model['id']]=fit
    tags=list(formats)
    common=[r for r in rows.values() if all(t in r['verifiers'] for t in tags)]
    # Prespecified nearby seeds diagnose fold sensitivity, without selecting one.
    common_fit=cross_validate(common,tags,formats,**kwargs)
    stability=[]
    for offset in range(5):
        fit=common_fit if offset==0 else cross_validate(common,tags,formats,**{**kwargs,'seed':config['seed']+offset})
        stability.append({'seed':config['seed']+offset,'status':fit['status'],
                          'differences_vs_calibrated_generator':fit.get('differences_vs_calibrated_generator')})
    result={'study_id':plan['study_id'],'signal_records_sha256':file_digest(records_path),
            'analysis_implementation_sha256':file_digest(Path(__file__)),
            'analysis_dependencies':{name:file_digest(Path(__file__).with_name(name))
                                     for name in ('offline_validation.py','score.py','report.py')},
            'package_versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn')},
            'initial_confidence':prior,'within_model_paired_formats':pairs,
            'all_models_common_cohort':common_fit,
            'all_candidates_missingness_diagnostic':cross_validate(list(rows.values()),tags,formats,allow_missing=True,**kwargs),
            'fold_seed_sensitivity':stability,
            'dependence_given_candidate_correctness':dependence_summary(common,tags),
            'likelihood_stability':{tag:likelihood_stability(list(rows.values()),tag,formats[tag],seed=config['seed'])
                                    for tag in tags},
            'limitations':['Exploratory calibration-fold estimates; no evaluation labels read.',
                          'No automatic model/prompt selection or live-policy replacement.',
                          'Complete-pair comparisons exclude parsing/missing failures; consult signal_report for all failures.',
                          'All-candidate joint forecasts use training-fold imputation plus missingness indicators. '
                          'They do not change live abstention on missing signals and may exploit failure patterns.',
                          'Global Brier does not establish calibration at 99 percent or under distribution shift.',
                          'Joint forecasts are diagnostics, not independent likelihoods or a replacement planner.',
                          'The Bayes comparator assumes verifier independence from generator confidence given correctness.',
                          'Small screening samples and selection across ten arms need independent confirmation.']}
    atomic_json(output/'calibration_comparison.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.output)
    print(json.dumps({'initial_confidence':result['initial_confidence']['status'],
                      'common_cohort':{k:result['all_models_common_cohort'][k] for k in ('status','n','wrong')}}))


if __name__=='__main__':main()
