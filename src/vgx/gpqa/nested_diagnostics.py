"""Read-only post-hoc diagnosis of frozen nested policies on development folds.

Counterfactual continuations use previously cached signals only AFTER the frozen
policy has been evaluated. They never enter the original policy or its selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.planner import NestedPlanner, ImpossibleObservation
from vgx.gpqa.report import mean_interval
from vgx.gpqa.score import VerifierLikelihood
from vgx.gpqa.sequential import seal


def exact_decision(planner, stage, belief):
    """Enumerate finite remaining signal outcomes without any grid interpolation."""
    assertion = (planner.reward+planner.loss)*belief-planner.loss
    stop = max(0., assertion)
    q = None
    if stage < len(planner.costs):
        q = -planner.costs[stage]
        channel = planner.likelihoods[stage]
        for p1,p0 in zip(channel.p_bin_if_correct,channel.p_bin_if_incorrect):
            mass = belief*p1+(1-belief)*p0
            if mass:
                q += mass*exact_decision(planner,stage+1,belief*p1/mass)['value']
    action = 'query' if q is not None and q>stop else ('assert' if belief>=planner.threshold else 'abstain')
    return {'action':action,'value':max(stop,q) if q is not None else stop,'q':q}


def suffix(planner, belief, stage, observations, *, force_first=False, exact=False):
    """Counterfactual suffix; intentionally label-free, using fixed full-cache path."""
    start = stage
    failure = None
    while True:
        decision = exact_decision(planner,stage,belief) if exact else planner.decide(stage,belief)
        if stage == len(planner.costs) or (decision['action']!='query' and not (force_first and stage==start)):
            break
        score = observations[stage]['score']
        stage += 1
        if score is None:
            failure = 'missing_verifier_score'
            break
        try:
            belief = planner.update(stage-1,belief,score)
        except ImpossibleObservation:
            failure = 'impossible_verifier_observation'
            break
    return {'action':'abstain' if failure else decision['action'],'posterior':belief,
            'end_stage':stage,'future_utility_cost':sum(planner.costs[start:stage]),'failure':failure}


def gross(action, outcome, loss, reward):
    return (reward if outcome else -loss) if action=='assert' else 0.


def average(values):
    return float(np.mean(values)) if values else None


def analyze(root, output):
    root, output = Path(root), Path(output)
    source_protocol=json.loads((root/'protocol.json').read_text())
    source_root=Path(source_protocol['config']['source_results'])
    cfg=json.loads(Path(source_protocol['config']['source_config']).read_text())
    source_report=json.loads((root/'report.json').read_text())
    paths=[root/'protocol.json', root/'report.json']
    for key in source_report['results']:
        paths += [source_root/(key+'.json'),root/key/'results.json']
        paths += list((root/key).glob('decisions_loss_*.json'))
        paths += list((root/key).glob('loss_*_fold_*.*'))
    protocol=seal({'source_analysis_id':source_report['analysis_id'],
        'source_files':{str(p):file_digest(p) for p in sorted(paths)},
        'implementation_sha256':file_digest(Path(__file__)),
        'new_api_requests':0,'evaluation_labels_read':False,'refit_or_policy_changes':False,
        'scope':'Post-hoc development-set diagnosis; cached future outputs used only for explicit counterfactuals.',
        'primary':'all_calibration seed_20261005 loss_19.0; other existing seeds/cohort/loss retained as sensitivity.'},'diagnostic_id')
    p=output/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol: raise ValueError('new output required')
    atomic_json(p,protocol)
    results={}; cases=[]; state_rows=[]
    for key in sorted(source_report['results']):
        fit=json.loads((source_root/(key+'.json')).read_text())
        selections=json.loads((root/key/'results.json').read_text())
        if selections['item_ids']!=fit['item_ids']: raise ValueError('item alignment')
        results[key]={}
        for loss_key in selections['policies']:
            loss=float(loss_key); reward=cfg['correct_reward']; alpha=cfg['utility_per_usd']
            ds=json.loads((root/key/f'decisions_loss_{loss_key}.json').read_text())
            planners={}; orders={}; table_checks=0
            for fold in range(cfg['folds']):
                stem=root/key/f'loss_{loss_key}_fold_{fold}'
                meta=json.loads(Path(str(stem)+'.json').read_text())
                npz=Path(str(stem)+'.npz')
                if file_digest(npz)!=meta['tables_sha256']: raise ValueError('table hash changed')
                planner=NestedPlanner([VerifierLikelihood(**x) for x in meta['likelihoods']],
                    [alpha*x for x in meta['expected_cost_usd']],reward,loss,grid_size=meta['grid_size'])
                with np.load(npz) as tables:
                    for name,actual in [('grid',planner.grid),('J',planner.j),('Q',planner.q)]:
                        if not np.array_equal(tables[name],np.array(actual)): raise ValueError('saved table differs from reconstruction')
                table_checks+=1; planners[fold]=planner; orders[fold]=meta['order']
            items=[]; trace_errors=[]; stage_disagreements=0; exact_path_changes=0
            for i,(d,full,raw) in enumerate(zip(ds['selected_three'],ds['always_selected_three'],ds['raw'])):
                y=fit['outcomes'][i]; prior=fit['raw_priors'][i]; fold=fit['fold_assignment'][i]
                planner=planners[fold]; order=orders[fold]; n=d['verifiers_used']
                valid=raw['failure']!='invalid_generator'
                actual=gross(d['action'],y,loss,reward)-alpha*d['expected_cost_usd']
                raw_actual=gross(raw['action'],y,loss,reward)
                predicted=d['trace'][0]['value'] if valid else 0.
                raw_predicted=max(0.,(reward+loss)*prior-loss) if valid else 0.
                item={'item_id':fit['item_ids'][i],'fold':fold,'valid':valid,'outcome':y,'prior':prior,
                    'action':d['action'],'posterior':d['posterior'],'queries':n,'failure':d['failure'],
                    'predicted_utility':predicted,'actual_utility':actual,'raw_predicted_utility':raw_predicted,
                    'raw_actual_utility':raw_actual,'full_action':full['action'],'full_posterior':full['posterior'],
                    'full_utility':gross(full['action'],y,loss,reward)-alpha*full['expected_cost_usd'],
                    'order':order,'scores':[o['score'] for o in full.get('observations',[])],
                    'early_stop':valid and d['failure'] is None and n<3}
                if valid:
                    # Reconstruct observed prefix and verify action without labels.
                    replay=suffix(planner,prior,0,full['observations'])
                    if any(replay[a]!=d[b] for a,b in [('action','action'),('posterior','posterior'),('end_stage','verifiers_used'),('failure','failure')]):
                        raise ValueError('frozen replay mismatch')
                    exact=suffix(planner,prior,0,full['observations'],exact=True)
                    changed=exact['action']!=d['action'] or exact['end_stage']!=n
                    exact_path_changes+=changed
                    item.update({'exact_action':exact['action'],'exact_queries':exact['end_stage'],
                        'exact_posterior':exact['posterior'],'exact_trajectory_changed':changed,
                        'exact_utility':gross(exact['action'],y,loss,reward)-exact['future_utility_cost']})
                    for state in d['trace']:
                        st=state['stage']; b=state['belief']; ex=exact_decision(planner,st,b)
                        stage_disagreements+=ex['action']!=state['action']
                        error=None if ex['q'] is None else state['continuation_value']-ex['q']
                        if error is not None: trace_errors.append(error)
                        realized_remaining=gross(d['action'],y,loss,reward)-sum(planner.costs[st:n])
                        state_rows.append({'run':key,'loss':loss,'item_id':item['item_id'],'stage':st,
                            'outcome':y,'belief':b,'action':state['action'],'stop_value':state['stop_value'],
                            'grid_Q':state['continuation_value'],'exact_Q':ex['q'],'exact_action':ex['action'],
                            'predicted_remaining_value':state['value'],'actual_remaining_utility':realized_remaining})
                    item['trace']=d['trace']
                    if item['early_stop']:
                        after=suffix(planner,d['posterior'],n,full['observations'],force_first=True)
                        stop_gross=gross(d['action'],y,loss,reward)
                        realized=gross(after['action'],y,loss,reward)-after['future_utility_cost']
                        terminal=d['trace'][-1]
                        item.update({'forced_next_action':after['action'],'forced_next_end_stage':after['end_stage'],
                            'forced_next_posterior':after['posterior'],'forced_next_failure':after['failure'],
                            'forced_next_actual_gain':realized-stop_gross,
                            'forced_next_predicted_gain':terminal['continuation_value']-terminal['stop_value']})
                items.append(item)
            released=[r for r in items if r['action']=='assert']
            early=[r for r in items if r['early_stop']]
            early_wrong=[r for r in early if r['action']=='assert' and not r['outcome']]
            counter={'early_stop_n':len(early),'early_wrong_released':len(early_wrong),
                'early_wrong_rejected_if_force_next':sum(r['forced_next_action']=='abstain' for r in early_wrong),
                'early_wrong_rejected_if_force_all':sum(r['full_action']=='abstain' for r in early_wrong),
                'early_correct_rejected_if_force_next':sum(r['action']=='assert' and r['outcome'] and r['forced_next_action']=='abstain' for r in early),
                'early_correct_rejected_if_force_all':sum(r['action']=='assert' and r['outcome'] and r['full_action']=='abstain' for r in early),
                'forced_next_mean_predicted_gain':average([r['forced_next_predicted_gain'] for r in early]),
                'forced_next_mean_actual_gain':average([r['forced_next_actual_gain'] for r in early]),
                'forced_next_gain_ci':mean_interval(np.array([r['forced_next_actual_gain'] for r in early]),cfg['bootstrap_repeats'],fit['seed']) if early else None}
            stages=[]
            for stage in range(4):
                released_stage=[r for r in released if r['queries']==stage]
                stages.append({'queries':stage,'released':len(released_stage),
                    'wrong_released':sum(1-r['outcome'] for r in released_stage),
                    'posterior_predicted_wrong':sum(1-r['posterior'] for r in released_stage),
                    'mean_raw_confidence':average([r['prior'] for r in released_stage]),
                    'mean_posterior':average([r['posterior'] for r in released_stage])})
            # Descriptive joint-positive mismatch, conditional on candidate correctness.
            joint=[]
            for outcome in (0,1):
                subset=[r for r in items if r['valid'] and r['outcome']==outcome and all(s is not None for s in r['scores'])]
                observed=[]; expected=[]
                for r in subset:
                    planner=planners[r['fold']]; positive=[]; probability=1.
                    for channel,score in zip(planner.likelihoods,r['scores']):
                        b=channel.bin_index(score)
                        positive.append(channel.p_bin_if_correct[b]>channel.p_bin_if_incorrect[b])
                        dist=channel.p_bin_if_correct if outcome else channel.p_bin_if_incorrect
                        probability*=sum(p for p,p1,p0 in zip(dist,channel.p_bin_if_correct,channel.p_bin_if_incorrect) if p1>p0)
                    observed.append(all(positive)); expected.append(probability)
                joint.append({'outcome':outcome,'n_complete_signals':len(subset),'all_positive_observed':sum(observed),
                    'all_positive_expected_under_independence':sum(expected),
                    'note':'Descriptive class-conditional mismatch, not a test of independence conditional on (Y,w). Fitted marginal mismatch is confounded.'})
            mean=lambda field: average([r[field] for r in items])
            selected_summary=selections['policies'][loss_key]['selected_three']
            if not np.isclose(mean('actual_utility'),selected_summary['mean_utility'],atol=1e-12): raise ValueError('utility reconciliation')
            predicted_gain=mean('predicted_utility')-mean('raw_predicted_utility')
            actual_gain=mean('actual_utility')-mean('raw_actual_utility')
            summary={'n':len(items),'valid_n':sum(r['valid'] for r in items),'released':len(released),
                'wrong_released':sum(1-r['outcome'] for r in released),
                'posterior_predicted_wrong':sum(1-r['posterior'] for r in released),
                'mean_released_posterior':average([r['posterior'] for r in released]),
                'released_accuracy':average([r['outcome'] for r in released]),
                'predicted_mean_utility':mean('predicted_utility'),'actual_mean_utility':mean('actual_utility'),
                'raw_predicted_mean_utility':mean('raw_predicted_utility'),'raw_actual_mean_utility':mean('raw_actual_utility'),
                'predicted_verification_gain':predicted_gain,'actual_verification_gain':actual_gain,
                'verification_gain_optimism':predicted_gain-actual_gain,
                'raw_stop_value_optimism':mean('raw_predicted_utility')-mean('raw_actual_utility'),
                'total_value_optimism':mean('predicted_utility')-mean('actual_utility'),
                'stop_stages':stages,'counterfactual':counter,'joint_positive_diagnostic':joint,
                'table_reconstruction_checks':table_checks,'max_grid_Q_error':max(map(abs,trace_errors),default=0.),
                'exact_vs_grid_stage_action_disagreements':stage_disagreements,
                'exact_vs_grid_trajectory_changes':int(exact_path_changes),
                'exact_vs_grid_terminal_action_changes':sum(r['exact_action']!=r['action'] for r in items if r['valid']),
                'exact_mean_utility':average([r.get('exact_utility',0.) for r in items]),
                'exact_wrong_released':sum(r['valid'] and r['exact_action']=='assert' and not r['outcome'] for r in items),
                'exact_queries':sum(r.get('exact_queries',0) for r in items)}
            if not np.isclose(summary['total_value_optimism'],summary['raw_stop_value_optimism']+summary['verification_gain_optimism']): raise ValueError('gap decomposition')
            results[key][loss_key]=summary
            for r in items:
                if r['action']=='assert' and not r['outcome']:
                    cases.append({'run':key,'loss':loss,**r})
            atomic_json(output/key/f'items_loss_{loss_key}.json',items)
        print(json.dumps({'completed':key}),flush=True)
    if any(file_digest(p)!=h for p,h in protocol['source_files'].items()): raise ValueError('source mutated')
    report={'diagnostic_id':protocol['diagnostic_id'],'results':results,'source_integrity_unchanged':True,
        'new_api_spend_usd':0,'confirmed_billing_usd':None,'evaluation_labels_read':False,
        'limitations':['All findings are post-hoc development diagnosis, not independent causal identification.',
            'Counterfactual suffixes reuse cached future signals for analysis only, never original decisions.',
            'No refitting, calibration, ranking changes or new policy deployment.',
            'Predicted wrong counts are sums of subjective posteriors, not externally validated risk guarantees.',
            'Uncertainty intervals condition on already fitted and selected policies.',
            'Repeated folds/cohorts reuse questions; they are not independent replications.']}
    atomic_json(output/'report.json',report)
    atomic_json(output/'wrong_release_cases.json',cases)
    with (output/'state_value_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(state_rows[0])); writer.writeheader(); writer.writerows(state_rows)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/gpqa_nested_combinations_20261008')
    p.add_argument('--output',required=True)
    args=p.parse_args(); analyze(args.input,args.output)
