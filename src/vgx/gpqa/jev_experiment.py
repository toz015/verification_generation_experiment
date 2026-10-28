"""Controlled Jev comparison on the existing frozen GPQA cohort; offline by default."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from vgx.common.api import RequestCache
from vgx.common.billing import summarize_vertex_usage
from vgx.common.jev_budget import JevBudgetCache, load_jev_key
from vgx.common.storage import atomic_json, digest, file_digest, read_jsonl
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.experiment import collect_verifiers, progress, specs_from_config
from vgx.gpqa.fresh import _write_immutable
from vgx.gpqa.report import build_report, mean_interval, rate_interval, _paired_forecasts
from vgx.gpqa.score import correctness_outcome
from vgx.gpqa.sequential import execute_bundle, policy_parts, seal
from vgx.gpqa.verifiers import load_specs
from vgx.gpqa.workflow import prepare_policy, score_execution


class ComparisonCache:
    """New calls are Jev-only; Vertex access always requires an existing response."""
    def __init__(self, jev, vertex):
        self.jev, self.vertex = jev, vertex

    def get(self, runner, request, *, allow_api=False):
        if runner.identity.get('provider') == 'typesafe_systemone':
            return self.jev.get(runner, request, allow_api=allow_api)
        return self.vertex.get(runner, request, allow_api=False)

    def logs(self, keys=None):
        if keys is None:
            return {**{'jev:'+k:v for k,v in self.jev.logs().items()},
                    **{'vertex:'+k:v for k,v in self.vertex.logs().items()}}
        return {k:self.jev.log(k) if self.jev.log(k).path.exists() else self.vertex.log(k) for k in set(keys)}


def direct_policy(rows, tag, cost, config):
    """Secondary baseline using Jev's raw probability as the final forecast."""
    threshold = config['incorrect_loss']/(config['correct_reward']+config['incorrect_loss'])
    releases, outcomes, utilities, queries = [], [], [], []
    for row in rows:
        candidate = row['generator_answer'] is not None
        p = row['verifiers'][tag]['p_correct']
        release = candidate and p is not None and p >= threshold
        y = correctness_outcome(row['generator_answer'], row['correct_index'])
        releases.append(release)
        if release:
            outcomes.append(y)
        queries.append(int(candidate))
        utilities.append((config['correct_reward'] if y else -config['incorrect_loss']) * release - cost*candidate)
    return {'n':len(rows), 'released_n':sum(releases), 'correct_released_n':sum(outcomes),
            'accuracy_among_released':sum(outcomes)/len(outcomes) if outcomes else None,
            'accuracy_among_released_ci95':rate_interval(outcomes),
            'release_coverage':sum(releases)/len(rows), 'mean_queries':sum(queries)/len(rows),
            'utility':mean_interval(utilities, config['bootstrap_repeats'], config['seed']),
            'note':'Queries every valid answer, then thresholds raw Jev probability; no generator-prior update.'}


def score_comparison(root, source, config, cache, policies, decisions, specs, pricing):
    """Offline boundary after every live arm and extra baseline collection finishes."""
    manifest, candidates = load_candidates(source/'frozen')
    keys = []
    labels = {}
    for name in ('calibration_records.jsonl','evaluation_labels.jsonl'):
        path = source/'frozen'/name
        if file_digest(path) != manifest['files'][name]:
            raise ValueError('frozen labels changed')
        labels.update({r['item_id']:r['correct_index'] for r in read_jsonl(path)})
    records = []
    for c in candidates:
        signals = {}
        for spec in specs.values():
            score, failure = None, 'invalid_generator_answer'
            if c.answer is not None:
                call, _ = cache.get(spec.runner(),spec.request(c),allow_api=False)
                keys.append(call.key)
                score, failure = spec.signal(call.response,c)
            signals[spec.id] = {'p_correct':score,'ok':score is not None,'failure':failure}
        records.append({'item_id':c.item_id,'partition':c.partition,'subject':c.subject,
                       'correct_index':labels[c.item_id],'generator_answer':c.answer,
                       'generator_p_correct':c.p_correct,'generator_ok':c.generator_ok,
                       'generator_failure':c.generator_failure,'verifiers':signals})
    by_id = {r['item_id']:r for r in records}
    arm_reports = {}
    for name, policy in policies.items():
        ordered, planner = policy_parts(policy)
        for d in decisions[name]['results']:
            row = by_id[d['candidate_id']]
            if row['generator_answer'] is None or row['generator_p_correct'] is None:
                if d['action']!='abstain' or d['verifiers_used']!=0:
                    raise ValueError('invalid candidate live/replay mismatch')
            else:
                replay = planner.replay(row['generator_p_correct'],[row['verifiers'][s.id]['p_correct'] for s in ordered])
                if any(replay[k]!=d[k] for k in ('action','verifiers_used','failure')) or not math.isclose(replay['posterior'],d['posterior'],abs_tol=1e-12):
                    raise ValueError('live/replay mismatch')
        mapped = [{**r,'verifiers':{f'verifier_{i}':r['verifiers'][s.id] for i,s in enumerate(ordered,1)}} for r in records]
        analysis = {'sample_size':len(candidates),'calibration_size':len(manifest['calibration_ids']),
                    'models':{'verifiers':[s.model for s in ordered]},
                    'routing_scenarios':[{'name':'primary_95','correct_reward':planner.reward,
                                         'incorrect_loss':planner.loss,'verifier_costs':list(planner.costs)}]}
        report = build_report(mapped,analysis,bootstrap_repeats=config['bootstrap_repeats'],seed=config['seed'])
        report['verifier_mapping'] = {f'verifier_{i}':s.id for i,s in enumerate(ordered,1)}
        evaluation = [r for r in mapped if r['partition']=='evaluation']
        report['direct_jev_policy'] = direct_policy(evaluation,'verifier_1',planner.costs[0],config)
        eligible = [r for r in evaluation if r['generator_answer'] is not None and r['generator_p_correct'] is not None
                    and r['verifiers']['verifier_1']['p_correct'] is not None]
        report['raw_jev_minus_generator'] = _paired_forecasts(
            [correctness_outcome(r['generator_answer'],r['correct_index']) for r in eligible],
            [r['generator_p_correct'] for r in eligible],
            [r['verifiers']['verifier_1']['p_correct'] for r in eligible],config['bootstrap_repeats'],config['seed'])
        report['live'] = score_execution(source/'frozen',policy,decisions[name])
        report['live_replay_matched_n'] = len(decisions[name]['results'])
        selected_keys = [k for d in decisions[name]['results'] for k in d['observed_request_keys']]
        report['selected_request_usage_estimate'] = summarize_vertex_usage(cache.logs(selected_keys),pricing)
        report['incremental_operation_usage_estimate'] = summarize_vertex_usage(cache.logs(selected_keys),pricing,operation_id=decisions[name]['operation_id'])
        atomic_json(root/'arm_reports'/f'{name}.json',report)
        # Keep the top-level report concise and credential/question-free.
        arm_reports[name] = {k:report[k] for k in ('verifier_mapping','direct_jev_policy','raw_jev_minus_generator','live','live_replay_matched_n')}
        arm_reports[name]['forecast'] = report['evaluation_metrics']['verifiers']['verifier_1']
        arm_reports[name]['policies'] = report['evaluation_metrics']['routing_scenarios']['primary_95']
        arm_reports[name]['calibration'] = report['calibration']
        arm_reports[name]['selected_request_estimated_usd'] = report['selected_request_usage_estimate']['estimated_usd_for_priced_calls']
    usage = summarize_vertex_usage(cache.jev.logs(),config['api_pricing'])
    atomic_json(root/'usage_estimate.json',usage)
    result = {'schema':1,'bundle_id':manifest['bundle_id'],'primary_arm':config['primary_arm'],
              'calibration_n':len(manifest['calibration_ids']),'evaluation_n':len(manifest['evaluation_ids']),
              'evaluation_status':'exploratory: this Diamond cohort was already examined in the Vertex experiment',
              'arms':arm_reports,'budget':cache.jev.summary(),'new_vertex_calls':0,
              'generator_calls':0,'confirmed_billing_usd':None,
              'baseline_report_sha256':file_digest(source/'report.json'),
              'analysis_plan_sha256':file_digest(root/'analysis_plan.json')}
    atomic_json(root/'report.json',result)
    atomic_json(root/'analysis_records.json',records)
    return result


def run(root, config, *, allow_api=False, phase='pilot'):
    root, source = Path(root), Path(config['source_root'])
    manifest, candidates = load_candidates(source/'frozen')
    if manifest['bundle_id'] != config['bundle_id']:
        raise ValueError('candidate bundle differs from the predeclared comparison')
    _write_immutable(root/'config.json',json.dumps(config,sort_keys=True,indent=2)+'\n')
    jev = JevBudgetCache(root/'cache',pricing=config['api_pricing'],limit_usd=config['budget_usd'],
                         model=config['model'],account_scope=config['account_scope'])
    cache = ComparisonCache(jev,RequestCache(source/'cache'))
    jev_specs = load_specs(config)
    originals = specs_from_config(manifest['collection_config'])
    specs = {s.id:s for s in (*originals,*jev_specs)}
    for arm in config['arms']:
        if (not arm['order'] or len(set(arm['order']))!=len(arm['order'])
                or set(arm['order'])-specs.keys()
                or sum(specs[s].provider=='typesafe' for s in arm['order'])!=1):
            raise ValueError('each arm requires one Jev mode; Choice and Noul cannot be chained')
    pricing = {'sources':[config['api_pricing'],manifest['collection_config']['api_pricing']],
               'usd_per_million_tokens':{**manifest['collection_config']['api_pricing']['usd_per_million_tokens'],
                                         **config['api_pricing']['usd_per_million_tokens']}}
    plan = seal({'config_sha256':digest(config),'bundle_id':manifest['bundle_id'],
                 'primary':config['primary_arm'],'arms':config['arms'],
                 'baseline_report_sha256':file_digest(source/'report.json'),
                 'forecast_comparisons':['raw Jev','likelihood-updated generator prior','calibration-only score logistic'],
                 'direct_probability_threshold':config['incorrect_loss']/(config['correct_reward']+config['incorrect_loss']),
                 'evaluation_status':'exploratory; original evaluation results already observed'},'plan_id')
    _write_immutable(root/'analysis_plan.json',json.dumps(plan,sort_keys=True,indent=2)+'\n')
    if allow_api:
        load_jev_key()
    calibration = [c for c in candidates if c.partition=='calibration']
    pilot = [c for c in calibration if c.answer is not None][:config['pilot_calibration_size']]
    progress(root,'pilot',budget=jev.summary())
    observed = collect_verifiers(pilot,jev_specs,cache,'jev-pilot',allow_api=allow_api,workers=1)
    pilot_result = {'candidate_n':len(pilot),'signals':len(observed),'valid_signals':sum(o['score'] is not None for o in observed),
                    'budget':jev.summary()}
    atomic_json(root/'pilot.json',pilot_result)
    if any(o['score'] is None for o in observed):
        raise ValueError('pilot response parsing failed; inspect saved responses before expanding')
    if phase=='pilot':
        progress(root,'pilot_complete',**pilot_result)
        return pilot_result
    progress(root,'calibration',budget=jev.summary())
    atomic_json(root/'calibration_collection.json',collect_verifiers(calibration,jev_specs,cache,'jev-calibration',allow_api=allow_api,workers=1))
    policies = {}
    for arm in config['arms']:
        policy = prepare_policy(source/'frozen',[specs[tag] for tag in arm['order']],cache,
            correct_reward=config['correct_reward'],incorrect_loss=config['incorrect_loss'],
            pricing=pricing,utility_per_usd=config['utility_per_usd'])
        _write_immutable(root/'policies'/f"{arm['name']}.json",json.dumps(policy,sort_keys=True,indent=2)+'\n')
        policies[arm['name']] = policy
    decisions = {}
    for name in [config['primary_arm'],*[n for n in policies if n!=config['primary_arm']]]:
        progress(root,'live_'+name,budget=jev.summary())
        decisions[name] = execute_bundle(source/'frozen',policies[name],cache,root/'live'/name,allow_api=allow_api)
    progress(root,'extra_baseline_signals',budget=jev.summary())
    evaluation = [c for c in candidates if c.partition=='evaluation']
    atomic_json(root/'evaluation_collection.json',collect_verifiers(evaluation,jev_specs,cache,'jev-extra-baselines',allow_api=allow_api,workers=1))
    progress(root,'offline_scoring',budget=jev.summary())
    report = score_comparison(root,source,config,cache,policies,decisions,specs,pricing)
    progress(root,'complete',budget=jev.summary(),report=str(root/'report.json'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/gpqa_jev_20261004'))
    parser.add_argument('--config',type=Path,default=Path('configs/gpqa_jev_comparison.json'))
    parser.add_argument('--phase',choices=['pilot','complete'],default='pilot')
    parser.add_argument('--allow-api',action='store_true')
    args = parser.parse_args()
    run(args.root,json.loads(args.config.read_text()),allow_api=args.allow_api,phase=args.phase)


if __name__=='__main__':
    main()
