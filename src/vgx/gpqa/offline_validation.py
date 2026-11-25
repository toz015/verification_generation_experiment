"""Offline forecast and stopping validation; no inference or credential imports."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.planner import ImpossibleObservation, NestedPlanner
from vgx.gpqa.report import _paired_forecasts, _interval, forecast, mean_interval, rate_interval
from vgx.gpqa.score import correctness_outcome, fit_verifier_likelihood


def outcome(row):
    return correctness_outcome(row['generator_answer'],row['correct_index'])


def valid(row):
    return row['generator_answer'] is not None and row['generator_p_correct'] is not None


def logit(p, clip):
    p=np.clip(p,clip,1-clip)
    return np.log(p/(1-p))


class CorrectnessForecast:
    """Fixed regularization; all feature transforms are fitted on calibration only."""
    def __init__(self,tags=(),*,C=1.,clip=1e-4):
        self.tags=tuple(tags);self.C=C;self.clip=clip

    def _features(self,rows):
        columns=[logit(np.array([r['generator_p_correct'] for r in rows]),self.clip)]
        for i,tag in enumerate(self.tags):
            x=np.array([r['verifiers'].get(tag,{}).get('p_correct') for r in rows],dtype=float)
            columns.extend([np.where(np.isnan(x),self.medians[i],x),np.isnan(x).astype(float)])
        return np.column_stack(columns)

    def fit(self,rows):
        if not rows or any(r['partition']!='calibration' for r in rows):
            raise ValueError('forecasts must be fitted on calibration rows only')
        rows=[r for r in rows if valid(r)]
        self.medians=[]
        for tag in self.tags:
            values=[r['verifiers'].get(tag,{}).get('p_correct') for r in rows]
            values=[p for p in values if p is not None]
            self.medians.append(float(np.median(values)) if values else .5)
        y=np.array([outcome(r) for r in rows])
        if len(set(y))!=2:
            raise ValueError('both correctness classes are required')
        self.scaler=StandardScaler().fit(self._features(rows))
        self.model=LogisticRegression(C=self.C,solver='lbfgs',max_iter=1000).fit(self.scaler.transform(self._features(rows)),y)
        return self

    def predict(self,rows):
        # Caller supplies valid public candidates; no labels are accessed here.
        return self.model.predict_proba(self.scaler.transform(self._features(rows)))[:,1]


def fit_channels(rows,tags):
    if not rows or any(r['partition']!='calibration' for r in rows):
        raise ValueError('likelihoods must be fitted on calibration rows only')
    fits={}
    for tag in tags:
        usable=[r for r in rows if valid(r) and r['verifiers'].get(tag,{}).get('p_correct') is not None]
        fits[tag]=fit_verifier_likelihood([outcome(r) for r in usable],[r['verifiers'][tag]['p_correct'] for r in usable])
    return fits


def stop_action(planner,b):
    return 'assert' if b>=planner.threshold else 'abstain'


def myopic_query_value(planner,stage,b):
    model=planner.likelihoods[stage]
    value=-planner.costs[stage]
    for p1,p0 in zip(model.p_bin_if_correct,model.p_bin_if_incorrect):
        mass=b*p1+(1-b)*p0
        if mass:
            post=b*p1/mass
            value+=mass*max(0.,(planner.reward+planner.loss)*post-planner.loss)
    return value


def decide_policy(prior,scores,planner,kind,*,width=0.):
    """Sequentially access only selected signals, independent of outcome labels."""
    if kind not in ('none','always','gate','myopic','nested'):
        raise ValueError('unknown stopping policy')
    if len(scores)!=len(planner.costs):
        raise ValueError('one potential score required per layer')
    if not math.isfinite(width) or width<0:
        raise ValueError('gate width must be finite and nonnegative')
    if prior is None:
        return {'action':'abstain','posterior':None,'used':0,'failure':'invalid_candidate','query_indices':[]}
    planner.decide(0,prior)  # Common validation of the public prior.
    if kind=='nested':
        result=planner.replay(prior,scores)
        return {'action':result['action'],'posterior':result['posterior'],'used':result['verifiers_used'],
                'failure':result['failure'],'query_indices':list(range(result['verifiers_used']))}
    b=prior;used=0
    gate=kind=='gate' and abs(prior-planner.threshold)<=width+1e-12
    for stage in range(len(planner.costs)):
        query=kind=='always' or gate
        if kind=='myopic':
            query=myopic_query_value(planner,stage,b)>max(0.,(planner.reward+planner.loss)*b-planner.loss)
        if not query:
            break
        p=scores[stage]
        used+=1
        if p is None:
            return {'action':'abstain','posterior':b,'used':used,'failure':'missing_verifier_score','query_indices':list(range(used))}
        try:
            b=planner.update(stage,b,p)
        except ImpossibleObservation:
            return {'action':'abstain','posterior':b,'used':used,'failure':'impossible_verifier_observation','query_indices':list(range(used))}
    return {'action':stop_action(planner,b),'posterior':b,'used':used,'failure':None,'query_indices':list(range(used))}


class ReplayScores:
    """Expose each cached signal only when the policy indexes that stage."""
    def __init__(self,row,tags):
        self.row,self.tags=row,tuple(tags)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self,index):
        return self.row['verifiers'].get(self.tags[index],{}).get('p_correct')


def run_policy(rows,priors,planner,tags,kind,width=0.):
    if len(rows)!=len(priors) or len(tags)!=len(planner.costs):
        raise ValueError('row/prior or verifier/cost length mismatch')
    result=[]
    for row,b in zip(rows,priors):
        scores=ReplayScores(row,tags)
        decision=decide_policy(float(b) if valid(row) else None,scores,planner,kind,width=width)
        cost=sum(planner.costs[:decision['used']])
        result.append({**decision,'item_id':row['item_id'],'cost':cost})
    return result


def policy_metrics(rows,decisions,reward,loss,repeats,seed):
    if not rows or len(rows)!=len(decisions):
        raise ValueError('one decision per evaluation row is required')
    release=np.array([d['action']=='assert' for d in decisions])
    y=np.array([outcome(r) for r in rows])
    costs=np.array([d['cost'] for d in decisions])
    utilities=release*np.where(y,reward,-loss)-costs
    return {'n':len(rows),'released':int(release.sum()),'correct_released':int(y[release].sum()),
            'wrong_released':int((1-y[release]).sum()),'coverage':float(release.mean()),
            'accuracy':float(y[release].mean()) if release.any() else None,
            'accuracy_ci95':rate_interval(y[release]),'queries':sum(d['used'] for d in decisions),
            'mean_cost_utility':float(costs.mean()),'mean_utility':mean_interval(utilities,repeats,seed),
            'failure_counts':dict(Counter(d['failure'] for d in decisions if d['failure']))}


def matched_coverage(y,p,k):
    """Expected errors when a score tie at the boundary is sampled uniformly."""
    y,p=np.asarray(y),np.asarray(p,dtype=float)
    available=np.isfinite(p)
    y,p=y[available],p[available]
    if k<1 or k>len(y):
        raise ValueError('coverage count exceeds available forecasts')
    threshold=float(np.sort(p)[::-1][k-1])
    above=p>threshold;tie=p==threshold
    take=k-int(above.sum());bad=int((1-y[tie]).sum());n=int(tie.sum())
    fixed_wrong=int((1-y[above]).sum())
    expected=fixed_wrong+take*bad/n
    return {'released':k,'expected_wrong':float(expected),'expected_accuracy':float(1-expected/k),
            'threshold':threshold,'strictly_above_n':int(above.sum()),'strictly_above_wrong':fixed_wrong,
            'boundary_tie_n':n,'boundary_wrong_n':bad,'boundary_selected_n':take,
            'tie_randomization_error_interval95':{
                'low':float(fixed_wrong+hypergeom.ppf(.025,n,bad,take)),
                'high':float(fixed_wrong+hypergeom.ppf(.975,n,bad,take))},
            'note':'Random-tie interval conditional on these observed items, not a population confidence interval.'}


def _priors(rows,model,mode):
    result=np.full(len(rows),np.nan)
    ids=[i for i,r in enumerate(rows) if valid(r)]
    if ids:
        result[ids]=model.predict([rows[i] for i in ids]) if mode=='calibrated' else [rows[i]['generator_p_correct'] for i in ids]
    return result


def choose_rules(cal,tags,costs,config):
    """Order and gate width selected by out-of-fold calibration utility only."""
    eligible=[r for r in cal if valid(r)]
    folds=StratifiedKFold(n_splits=config['folds'],shuffle=True,random_state=config['seed'])
    orders=list(itertools.permutations(tags));scores={}
    for fold,(train_ids,val_ids) in enumerate(folds.split(eligible,[outcome(r) for r in eligible])):
        train=[eligible[i] for i in train_ids];val=[eligible[i] for i in val_ids]
        prior_model=CorrectnessForecast(C=config['logistic_C'],clip=config['logit_clip']).fit(train)
        channels=fit_channels(train,tags)
        for mode in ('raw','calibrated'):
            priors=_priors(val,prior_model,mode)
            for order in orders:
                planner=NestedPlanner([channels[t] for t in order],[costs[t] for t in order],config['correct_reward'],config['incorrect_loss'])
                key=('order',mode,tuple(order))
                dec=run_policy(val,priors,planner,order,'nested')
                scores.setdefault(key,[]).extend((config['correct_reward'] if outcome(r) else -config['incorrect_loss'])*(d['action']=='assert')-d['cost'] for r,d in zip(val,dec))
            planner=NestedPlanner([channels[t] for t in tags],[costs[t] for t in tags],config['correct_reward'],config['incorrect_loss'])
            for width in config['gate_widths']:
                dec=run_policy(val,priors,planner,tags,'gate',width)
                scores.setdefault(('width',mode,width),[]).extend((config['correct_reward'] if outcome(r) else -config['incorrect_loss'])*(d['action']=='assert')-d['cost'] for r,d in zip(val,dec))
    selection={}
    for mode in ('raw','calibrated'):
        best_order=max(orders,key=lambda order:np.mean(scores[('order',mode,order)]))
        width=max(config['gate_widths'],key=lambda w:np.mean(scores[('width',mode,w)]))
        selection[mode]={'selected_order':list(best_order),'gate_width':width,
                         'order_oof_utilities':[{'order':list(o),'utility':float(np.mean(scores[('order',mode,o)]))} for o in orders],
                         'gate_oof_utilities':[{'width':w,'utility':float(np.mean(scores[('width',mode,w)]))} for w in config['gate_widths']]}
    return selection


def forecast_study(cal,ev,config):
    cal_valid=[r for r in cal if valid(r)];ev_valid=[r for r in ev if valid(r)]
    tags=config['diagnostic_verifiers']
    groups={'calibrated_generator':(),**{f'joint_{t}':(t,) for t in tags},
            'joint_three_verifiers':tuple(config['primary_order'])}
    predictions={'calibration_base_rate':np.full(len(ev_valid),np.mean([outcome(r) for r in cal_valid]))}
    oof={'raw_generator':np.array([r['generator_p_correct'] for r in cal_valid]),
         'calibration_base_rate':np.zeros(len(cal_valid))}
    for name,features in groups.items():
        model=CorrectnessForecast(features,C=config['logistic_C'],clip=config['logit_clip']).fit(cal_valid)
        predictions[name]=model.predict(ev_valid)
        oof[name]=np.zeros(len(cal_valid))
    cv=StratifiedKFold(n_splits=config['folds'],shuffle=True,random_state=config['seed'])
    for tr,va in cv.split(cal_valid,[outcome(r) for r in cal_valid]):
        train=[cal_valid[i] for i in tr];val=[cal_valid[i] for i in va]
        oof['calibration_base_rate'][va]=np.mean([outcome(r) for r in train])
        for name,features in groups.items():
            model=CorrectnessForecast(features,C=config['logistic_C'],clip=config['logit_clip']).fit(train)
            oof[name][va]=model.predict(val)
    predictions['raw_generator']=np.array([r['generator_p_correct'] for r in ev_valid])
    prior=predictions['calibrated_generator'];channels=fit_channels(cal,tags)
    for tag in tags:
        for mode,base in [('raw',predictions['raw_generator']),('calibrated',prior)]:
            predictions[f'bayes_{mode}_{tag}']=np.array([
                channels[tag].posterior(float(b),r['verifiers'][tag]['p_correct'])
                if r['verifiers'][tag]['p_correct'] is not None else np.nan for b,r in zip(base,ev_valid)])
    output={'calibration_out_of_fold':{},'evaluation':{},'paired_evaluation':{},'matched_coverage':{},
            'note':'All fitting is calibration-only. Joint logistic forecasts are diagnostic baselines, not plugged into the independent-channel planner.'}
    for split,rows,preds in [('calibration_out_of_fold',cal_valid,oof),('evaluation',ev_valid,predictions)]:
        y=np.array([outcome(r) for r in rows])
        for name,p in preds.items():
            keep=np.isfinite(p)
            output[split][name]=forecast(y[keep].tolist(),p[keep].tolist(),200,config['seed'])
            output[split][name]['missing_forecast_n']=int((~keep).sum())
    y=np.array([outcome(r) for r in ev_valid])
    for name,p in predictions.items():
        if name=='raw_generator':continue
        mask=np.isfinite(p)
        output['paired_evaluation'][name+'_minus_raw_generator']=_paired_forecasts(y[mask],predictions['raw_generator'][mask],p[mask],config['bootstrap_evaluation'],config['seed'])
        if name.startswith('joint_'):
            output['paired_evaluation'][name+'_minus_calibrated_generator']=_paired_forecasts(y[mask],prior[mask],p[mask],config['bootstrap_evaluation'],config['seed'])
    for name,p in predictions.items():
        output['matched_coverage'][name]=[{**matched_coverage(y,p,k),'coverage':k/len(ev)} for k in range(1,int(np.isfinite(p).sum())+1)]
    return output


def refit_bootstrap(cal,ev,costs,config):
    """Refit primary models per calibration resample; resample evaluation separately."""
    rng=np.random.default_rng(config['seed']);tags=config['primary_order'];statistics={};failed=0
    reward,loss=config['correct_reward'],config['incorrect_loss']
    for _ in range(config['bootstrap_refit']):
        train=[cal[i] for i in rng.integers(0,len(cal),len(cal))]
        test=[ev[i] for i in rng.integers(0,len(ev),len(ev))]
        try:
            prior_model=CorrectnessForecast(C=config['logistic_C'],clip=config['logit_clip']).fit(train)
            joint=CorrectnessForecast(tags,C=config['logistic_C'],clip=config['logit_clip']).fit(train)
            channels=fit_channels(train,tags)
        except ValueError:
            failed+=1;continue
        planner=NestedPlanner([channels[t] for t in tags],[costs[t] for t in tags],reward,loss)
        usable=[r for r in test if valid(r)];y=np.array([outcome(r) for r in usable])
        p0=prior_model.predict(usable);p1=joint.predict(usable)
        statistics.setdefault('joint_minus_calibrated_generator_brier',[]).append(float(np.mean((p1-y)**2-(p0-y)**2)))
        for mode in ('raw','calibrated'):
            priors=_priors(test,prior_model,mode);values={}
            for kind in ('none','myopic','nested'):
                dec=run_policy(test,priors,planner,tags,kind)
                values[kind]=float(np.mean([(reward if outcome(r) else -loss)*(d['action']=='assert')-d['cost'] for r,d in zip(test,dec)]))
            for base in ('none','myopic'):
                statistics.setdefault(f'{mode}_nested_minus_{base}_utility',[]).append(values['nested']-values[base])
    return {'requested':config['bootstrap_refit'],'successful':config['bootstrap_refit']-failed,'failed':failed,
            'intervals':{k:{'bootstrap_mean':float(np.mean(v)),'ci95':_interval(v)} for k,v in statistics.items()},
            'scope':'Calibration and evaluation resampled independently; prior and likelihoods refitted. Fixed primary order, C, binning, costs and utility. Does not propagate tuning-choice or model-training uncertainty.'}


def policy_study(cal,ev,costs,config,selection):
    tags=config['primary_order'];reward,loss=config['correct_reward'],config['incorrect_loss']
    channels=fit_channels(cal,tags);prior_model=CorrectnessForecast(C=config['logistic_C'],clip=config['logit_clip']).fit(cal)
    all_results={};all_decisions={};sensitivity=[]
    for mode in ('raw','calibrated'):
        priors=_priors(ev,prior_model,mode)
        planner=NestedPlanner([channels[t] for t in tags],[costs[t] for t in tags],reward,loss)
        for kind in ('none','always','gate','myopic','nested'):
            dec=run_policy(ev,priors,planner,tags,kind,selection[mode]['gate_width'])
            key=f'{mode}_{kind}';all_decisions[key]=dec
            all_results[key]=policy_metrics(ev,dec,reward,loss,config['bootstrap_evaluation'],config['seed'])
        order=selection[mode]['selected_order']
        selected=NestedPlanner([channels[t] for t in order],[costs[t] for t in order],reward,loss)
        dec=run_policy(ev,priors,selected,order,'nested')
        key=f'{mode}_nested_selected_order';all_decisions[key]=dec
        all_results[key]=policy_metrics(ev,dec,reward,loss,config['bootstrap_evaluation'],config['seed'])
        for multiplier in config['cost_multipliers']:
            for scenario_loss in config['loss_sensitivity']:
                pp=NestedPlanner([channels[t] for t in tags],[costs[t]*multiplier for t in tags],reward,scenario_loss)
                for kind in ('none','myopic','nested'):
                    dd=run_policy(ev,priors,pp,tags,kind)
                    sensitivity.append({'prior':mode,'cost_multiplier':multiplier,'loss':scenario_loss,'kind':kind,
                        **policy_metrics(ev,dd,reward,scenario_loss,100,config['seed'])})
    paired={}
    for mode in ('raw','calibrated'):
        a=all_decisions[f'{mode}_nested']
        for base in ('none','always','gate','myopic'):
            b=all_decisions[f'{mode}_{base}']
            delta=[(reward if outcome(r) else -loss)*((x['action']=='assert')-(z['action']=='assert'))-x['cost']+z['cost'] for r,x,z in zip(ev,a,b)]
            paired[f'{mode}_nested_minus_{base}']={'utility':mean_interval(delta,config['bootstrap_evaluation'],config['seed']),
                'action_disagreements':sum(x['action']!=z['action'] for x,z in zip(a,b)),
                'query_count_disagreements':sum(x['used']!=z['used'] for x,z in zip(a,b))}
    y=np.array([outcome(r) for r in ev if valid(r)]);p=np.array([r['generator_p_correct'] for r in ev if valid(r)])
    matches={}
    for name,metric in all_results.items():
        if metric['released']:
            m=matched_coverage(y,p,metric['released'])
            m['policy_wrong']=metric['wrong_released']
            matches[name]=m
    return {'policies':all_results,'paired_differences':paired,'matched_raw_confidence':matches,
            'cost_loss_sensitivity':sensitivity,'note':'Utility intervals here condition on fitted models; separate refit bootstrap includes calibration uncertainty.'},all_decisions


def run(config,output):
    if not math.isfinite(config['utility_per_usd']) or config['utility_per_usd']<=0:
        raise ValueError('utility_per_usd must be finite and positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sources={key:file_digest(config[key]) for key in ('records','vertex_policy','jev_policy')}
    code_sources={name:file_digest(Path(__file__).with_name(name)) for name in
                  ('offline_validation.py','simulation_validation.py','planner.py','score.py','report.py')}
    protocol={'config':config,'source_sha256':sources,'code_sha256':code_sources,
              'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn')},
              'status':'exploratory_offline_validation'}
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=protocol:
        raise ValueError('output already belongs to a different analysis protocol')
    atomic_json(output/'protocol.json',protocol)
    records=json.loads(Path(config['records']).read_text())
    if len({r['item_id'] for r in records})!=len(records):raise ValueError('duplicate question IDs')
    cal=[r for r in records if r['partition']=='calibration'];ev=[r for r in records if r['partition']=='evaluation']
    if len(cal)+len(ev)!=len(records):raise ValueError('unknown partition')
    costs={}
    for path in (config['vertex_policy'],config['jev_policy']):
        pol=json.loads(Path(path).read_text())
        original_conversion=pol['cost_mapping']['utility_per_usd']
        if not math.isfinite(original_conversion) or original_conversion<=0:
            raise ValueError('source utility conversion must be positive')
        costs.update({s['id']:c/original_conversion*config['utility_per_usd'] for s,c in zip(pol['verifiers'],pol['planner']['costs'])})
    print('Selecting order and gate width using calibration folds only.',flush=True)
    selection=choose_rules(cal,config['primary_order'],costs,config)
    atomic_json(output/'calibration_selection.json',selection)
    print('Scoring forecasts and fixed policy comparisons offline.',flush=True)
    forecasts=forecast_study(cal,ev,config)
    policies,decisions=policy_study(cal,ev,costs,config,selection)
    atomic_json(output/'policy_decisions.json',decisions)
    print('Refitting calibration in the two-sample bootstrap.',flush=True)
    bootstrap=refit_bootstrap(cal,ev,costs,config)
    from vgx.gpqa.simulation_validation import simulation_study
    print('Evaluating exact finite-tree simulations.',flush=True)
    simulations=simulation_study(config)
    result={'schema':1,'protocol':protocol,'calibration_n':len(cal),'evaluation_n':len(ev),
            'calibration_valid_n':sum(valid(r) for r in cal),'evaluation_valid_n':sum(valid(r) for r in ev),
            'calibration_incorrect_valid_n':sum(1-outcome(r) for r in cal if valid(r)),
            'evaluation_incorrect_valid_n':sum(1-outcome(r) for r in ev if valid(r)),
            'costs_utility':costs,'selection':selection,'forecasts':forecasts,'policy_study':policies,
            'refit_bootstrap':bootstrap,'simulations':simulations,
            'source_artifacts_unchanged':all(file_digest(config[k])==v for k,v in sources.items()),'api_calls':0}
    if not result['source_artifacts_unchanged']:
        raise ValueError('source artifacts changed during offline analysis')
    atomic_json(output/'report.json',result)
    print(json.dumps({'output':str(output/'report.json'),'api_calls':0,'source_artifacts_unchanged':result['source_artifacts_unchanged']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/gpqa_offline_validation.json'))
    parser.add_argument('--output',type=Path,default=Path('results/gpqa_offline_validation_20261004'))
    args=parser.parse_args();run(json.loads(args.config.read_text()),args.output)


if __name__=='__main__':main()
