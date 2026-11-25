"""Offline raw-prior verifier experiment; no generator calibration or API transport.

Fit observation likelihoods and expected costs in training folds. Compare frozen
raw probabilities against their Bayes updates, then replay one-verifier value
tables on unseen folds. Incomplete collection blocks the affected generator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.generator_calibration import source, index_cache, audit_coverage
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.report import mean_interval, rate_interval
from vgx.gpqa.score import binary_forecast_metrics, fit_verifier_likelihood
from vgx.gpqa.sequential import seal
from vgx.gpqa.signal_study import request_for, runner_for, parse_signal


def candidate_valid(row):
    p = row['prior']
    return (row['answer'] in 'ABCD' if isinstance(row['answer'], str) and len(row['answer']) == 1 else False) and (
        isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1)


def baseline_decision(prior, valid, loss, reward):
    return {'action': 'assert' if valid and prior >= loss/(reward+loss) else 'abstain',
            'posterior': prior, 'verifiers_used': 0, 'expected_cost_usd': 0.,
            'usage_estimated_cost_usd': 0., 'failure': None if valid else 'invalid_generator',
            'signal_reads': 0}


def decide_public(prior, valid, planner, query, *, expected_cost_usd, force=False):
    """Decision receives no labels or future observations; query is lazy and local."""
    if not valid:
        return baseline_decision(prior, False, planner.loss, planner.reward)
    decision = planner.decide(0, prior)
    if not force and decision['action'] != 'query':
        result = baseline_decision(prior, True, planner.loss, planner.reward)
        result['decision'] = decision
        return result
    observation = query()
    score = observation['score']
    if score is None:
        action, posterior, failure = 'abstain', prior, observation['failure'] or 'invalid_signal'
    else:
        posterior = planner.update(0, prior, score)
        action, failure = planner.decide(1, posterior)['action'], None
    return {'action': action, 'posterior': posterior, 'verifiers_used': 1,
            'expected_cost_usd': expected_cost_usd,
            'usage_estimated_cost_usd': observation['estimated_usd'],
            'failure': failure, 'signal_reads': 1, 'decision': decision}


def summarize_policy(rows, predictions, config, loss):
    n = len(rows)
    released = np.array([r['action'] == 'assert' for r in predictions])
    y = np.array([r['outcome'] for r in rows])
    exp = np.array([r['expected_cost_usd'] for r in predictions])
    observed = np.array([r['usage_estimated_cost_usd'] for r in predictions])
    gross = released * np.where(y == 1, config['correct_reward'], -loss)
    utility = gross - config['utility_per_usd']*exp
    return {'n': n, 'released': int(released.sum()), 'correct_released': int(y[released].sum()),
            'wrong_released': int((1-y[released]).sum()), 'coverage': float(released.mean()),
            'released_accuracy': float(y[released].mean()) if released.any() else None,
            'released_accuracy_wilson95': rate_interval(y[released]),
            'queries': sum(r['verifiers_used'] for r in predictions),
            'signal_failures_queried': sum(r['verifiers_used'] > 0 and r['failure'] is not None for r in predictions),
            'expected_cost_usd': float(exp.sum()), 'usage_estimated_cost_usd': float(observed.sum()),
            'mean_utility': float(utility.mean()),
            'mean_utility_at_observed_usage': float((gross-config['utility_per_usd']*observed).mean()),
            'utility_per_item': utility.tolist(), 'confirmed_billing_usd': None}


def forecast_comparison(rows, posterior, indices, config, seed):
    y = np.array([rows[i]['outcome'] for i in indices])
    before = np.array([rows[i]['prior'] for i in indices])
    after = np.array([posterior[i] for i in indices])
    delta = (after-y)**2-(before-y)**2
    return {'n': len(indices), 'raw': binary_forecast_metrics(y.tolist(), before.tolist()).to_dict(),
            'updated': binary_forecast_metrics(y.tolist(), after.tolist()).to_dict(),
            'brier_difference': mean_interval(delta, config['bootstrap_repeats'], seed)}


def evaluate(rows, tags, formats, config, seed):
    if any(r['partition'] != 'calibration' for r in rows):
        raise ValueError('calibration rows only')
    ids = [r['item_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate question')
    y = np.array([r['outcome'] for r in rows])
    if min(int((y == 0).sum()), int((y == 1).sum())) < config['folds']:
        return {'status': 'insufficient_class_counts'}
    valid = [candidate_valid(r) for r in rows]
    common = [i for i,r in enumerate(rows) if valid[i] and all(r['signals'][t]['score'] is not None for t in tags)]
    losses = [config['primary_incorrect_loss'], *config['sensitivity_incorrect_losses']]
    predictions = {str(loss): {'raw': [baseline_decision(r['prior'],v,loss,config['correct_reward'])
                                     for r,v in zip(rows,valid)]} for loss in losses}
    posteriors = {tag: [None]*len(rows) for tag in tags}
    for loss in losses:
        for tag in tags:
            for mode in ('always_query','sequential'):
                predictions[str(loss)][mode+':'+tag] = [None]*len(rows)
    folds = []
    assignment = [-1]*len(rows)
    replay_agreements = 0
    for fold_id,(tr,va) in enumerate(StratifiedKFold(n_splits=config['folds'],shuffle=True,random_state=seed).split(ids,y)):
        fold = {'train_ids': [ids[i] for i in tr], 'validation_ids': [ids[i] for i in va], 'arms': {}}
        if set(fold['train_ids']) & set(fold['validation_ids']):
            raise AssertionError('fold leakage')
        for i in va:
            if assignment[i] != -1: raise AssertionError('repeated validation item')
            assignment[i] = fold_id
        for tag in tags:
            fitting = [int(i) for i in tr if valid[i] and rows[i]['signals'][tag]['score'] is not None]
            # Costs include paid parse failures. Validation usage never enters this mean.
            cost_ids = [int(i) for i in tr if valid[i]]
            if not cost_ids or len({rows[i]['outcome'] for i in fitting}) < 2:
                raise ValueError('insufficient training observations')
            channel = fit_verifier_likelihood([rows[i]['outcome'] for i in fitting],
                [rows[i]['signals'][tag]['score'] for i in fitting],
                bins=config['binary_bins'] if formats[tag]=='binary' else config['probability_bins'],
                laplace=config['laplace'])
            cost = float(np.mean([rows[i]['signals'][tag]['estimated_usd'] for i in cost_ids]))
            fold['arms'][tag] = {'likelihood': asdict(channel), 'expected_cost_usd': cost,
                'likelihood_fit_ids': [ids[i] for i in fitting], 'cost_fit_ids': [ids[i] for i in cost_ids],
                'fit_correct': sum(rows[i]['outcome'] for i in fitting),
                'fit_wrong': sum(1-rows[i]['outcome'] for i in fitting)}
            for i in va:
                if valid[i] and rows[i]['signals'][tag]['score'] is not None:
                    posteriors[tag][i] = channel.posterior(rows[i]['prior'],rows[i]['signals'][tag]['score'])
            for loss in losses:
                planner = NestedPlanner([channel],[config['utility_per_usd']*cost],config['correct_reward'],loss,
                                        grid_size=config['grid_size'])
                for i in va:
                    row = rows[i]
                    for mode in ('always_query','sequential'):
                        reads = []
                        def query():
                            reads.append(tag)
                            return row['signals'][tag]
                        decision = decide_public(row['prior'],valid[i],planner,query,expected_cost_usd=cost,
                                                 force=mode=='always_query')
                        if len(reads) != decision['verifiers_used']:
                            raise AssertionError('signal accessed outside a selected query')
                        predictions[str(loss)][mode+':'+tag][i] = decision
                        if mode == 'sequential' and valid[i]:
                            # Replay gets a lazy sequence too: unselected signals cannot be read.
                            class LazyScore:
                                def __len__(self): return 1
                                def __getitem__(self,index):
                                    if index != 0: raise AssertionError('future signal')
                                    return query()['score']
                            reads.clear()
                            replay = planner.replay(row['prior'],LazyScore())
                            if any(replay[k] != decision[k] for k in ('action','posterior','verifiers_used')):
                                raise AssertionError('replay mismatch')
                            if len(reads) != replay['verifiers_used']:
                                raise AssertionError('replay accessed unselected signal')
                            replay_agreements += 1
        folds.append(fold)
    if any(a < 0 for a in assignment): raise AssertionError('missing OOF item')
    forecasts = {}
    for tag in tags:
        matched = [i for i,p in enumerate(posteriors[tag]) if p is not None]
        forecasts[tag] = {'matched_available': forecast_comparison(rows,posteriors[tag],matched,config,seed),
                          'common_all_arms': forecast_comparison(rows,posteriors[tag],common,config,seed)}
    policies = {}
    for loss in losses:
        entries = predictions[str(loss)]
        summaries = {name:summarize_policy(rows,values,config,loss) for name,values in entries.items()}
        base = np.array(summaries['raw']['utility_per_item'])
        for name,s in summaries.items():
            if name != 'raw':
                s['paired_utility_difference'] = mean_interval(np.array(s['utility_per_item'])-base,
                                                               config['bootstrap_repeats'],seed)
            del s['utility_per_item']
        policies[str(loss)] = summaries
    pairs = {}
    for model in sorted({t.split(':')[0] for t in tags}):
        bt,pt=model+':binary',model+':probability'
        matched=[i for i in range(len(rows)) if posteriors[bt][i] is not None and posteriors[pt][i] is not None]
        delta=[(posteriors[pt][i]-y[i])**2-(posteriors[bt][i]-y[i])**2 for i in matched]
        pairs[model]={'n':len(matched),'probability_minus_binary_brier':mean_interval(delta,config['bootstrap_repeats'],seed)}
    return {'status':'complete_exploratory_oof','n':len(rows),'valid_generator_n':sum(valid),'common_forecast_n':len(common),
            'seed':seed,'folds':folds,'item_ids':ids,'fold_assignment':assignment,
            'raw_priors':[r['prior'] for r in rows], 'outcomes':y.tolist(),
            'posteriors':posteriors,'forecast_comparisons':forecasts,'paired_formats':pairs,
            'policies':policies,'decisions':predictions,'replay_agreements':replay_agreements}


def analyze(config_path, output):
    config_path,output=Path(config_path),Path(output)
    config=json.loads(config_path.read_text())
    if config['prior']!='unmodified_generator_p_correct_including_endpoints' or config['new_api_requests']!=0:
        raise ValueError('raw-prior offline protocol required')
    plan,groups,labels,pilot,label_hash,bundle_id=source(Path(config['generator_results']))
    screen=json.loads(Path(config['screen_config']).read_text())
    if screen['project']!=plan['config']['project']:raise ValueError('project mismatch')
    index=index_cache(config['cache_roots'])
    coverage=audit_coverage(groups,labels,screen['models'],screen['project'],index)
    protocol=seal({'config':config,'config_sha256':file_digest(config_path),
        'source_study_id':plan['study_id'],'bundle_id':bundle_id,'calibration_label_sha256':label_hash,
        'screen_config_sha256':file_digest(Path(config['screen_config'])),
        'source_candidate_hashes':{g:digest([asdict(c) for c in cs]) for g,cs in groups.items()},
        'used_cache_record_hashes':coverage['used_cache_record_hashes'],
        'implementation_sha256':{name:file_digest(Path(__file__).with_name(name)) for name in
            ('raw_prior_analysis.py','generator_calibration.py','planner.py','score.py','report.py','signal_study.py')},
        'package_versions':{n:importlib.metadata.version(n) for n in ('numpy','scikit-learn','scipy')},
        'evaluation_labels_read':False,'api_requests':0},'analysis_id')
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('protocol changed; use a new output directory')
    atomic_json(path,protocol)  # Written before looking at any experiment comparison.
    atomic_json(output/'coverage.json',coverage)
    result={'analysis_id':protocol['analysis_id'],'generators':{},'new_api_spend_usd':0.,'confirmed_billing_usd':None,
            'project':screen['project'],'evaluation_labels_read':False,
            'limitations':config['notes']+['Development-set OOF analysis after exploratory model screening; not independent confirmation.',
                'Bayes likelihood update assumes signals are independent of raw generator confidence given correctness.',
                'Invalid-response abstention is enforced; Q tables model valid signal bins and do not model failure probabilities.',
                'No three-verifier order selected and no live policy installed.']}
    for name,candidates in groups.items():
        if any(c['missing'] for c in coverage['coverage'][name].values()):
            result['generators'][name]={'status':'blocked_incomplete_collection',
                'missing_unique_requests':len({m['request_key'] for m in coverage['missing_requests'] if m['generator']==name}),
                'reason':'Cached subset is selection biased; no likelihoods or policy ranking fitted on it.'}
            continue
        rows=[]
        for c in candidates:
            row={'item_id':c.item_id,'partition':c.partition,'answer':c.answer,'prior':c.p_correct,
                 'outcome':int(c.answer is not None and 'ABCD'.index(c.answer)==labels[c.item_id]),'signals':{}}
            if candidate_valid(row):
                for model in screen['models']:
                    runner=runner_for(model,screen['project'])
                    for signal in screen['signals']:
                        key=runner.cache_key(request_for(c,model,signal));call=index[key]
                        estimate=estimate_call(call,screen['pricing'])
                        if estimate['estimated_usd'] is None:raise ValueError('unpriced verifier request: '+key)
                        score,failure=parse_signal(call['response'],signal)
                        row['signals'][model['id']+':'+signal]={'score':score,'failure':failure,
                            'estimated_usd':estimate['estimated_usd'],'request_key':key}
            rows.append(row)
        formats={m['id']+':'+s:s for m in screen['models'] for s in screen['signals']};tags=list(formats)
        generator={'status':'complete','cohorts':{}}
        for cohort in config['cohorts']:
            subset=rows if cohort=='all_calibration' else [r for r in rows if r['item_id'] not in pilot]
            fits=[]
            for seed in config['seeds']:
                fit=evaluate(subset,tags,formats,config,seed)
                print(json.dumps({'generator':name,'cohort':cohort,'seed':seed,'status':fit['status']}),flush=True)
                atomic_json(output/name/cohort/f'seed_{seed}.json',fit)
                fits.append(fit)
            primary=fits[0]
            generator['cohorts'][cohort]={'n':primary['n'],'valid_generator_n':primary['valid_generator_n'],
                'common_forecast_n':primary['common_forecast_n'],'forecast_comparisons':primary['forecast_comparisons'],
                'paired_formats':primary['paired_formats'],'policies':primary['policies'],
                'seed_sensitivity':[{'seed':f['seed'],'forecasts':{t:{scope:d['brier_difference']['estimate'] for scope,d in v.items()}
                                      for t,v in f['forecast_comparisons'].items()},
                                     'policy_utility_differences':{loss:{k:v['paired_utility_difference']['estimate'] for k,v in ps.items() if k!='raw'}
                                                                   for loss,ps in f['policies'].items()}} for f in fits],
                'replay_agreements':sum(f['replay_agreements'] for f in fits)}
        result['generators'][name]=generator
    # Read-only provenance audit after analysis: no source candidates or responses changed.
    _,end_groups,_,_,end_label,end_bundle=source(Path(config['generator_results']))
    end_index=index_cache(config['cache_roots'])
    if end_label!=label_hash or end_bundle!=bundle_id or any(
        digest([asdict(c) for c in end_groups[g]])!=protocol['source_candidate_hashes'][g] for g in groups):
        raise ValueError('source changed during analysis')
    if any(digest(end_index[k])!=v for k,v in protocol['used_cache_record_hashes'].items()):
        raise ValueError('cached response changed during analysis')
    result['source_integrity_unchanged']=True
    atomic_json(output/'report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/gpqa_raw_prior_analysis.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=analyze(args.config,args.output)
    print(json.dumps({'analysis_id':value['analysis_id'],'status':{g:v['status'] for g,v in value['generators'].items()}}))
