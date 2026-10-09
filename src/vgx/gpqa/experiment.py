"""Run the predeclared Vertex-only experiment on a fresh frozen GPQA cohort.

No Jev calls. The primary live evaluation completes before counterfactual
baseline signals are collected or evaluation answer keys are opened for scoring.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from vgx.common.budget import BudgetedRequestCache
from vgx.common.concurrency import parallel_map
from vgx.common.billing import summarize_vertex_usage
from vgx.common.storage import append_jsonl, atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.fresh import _write_immutable, collect_generators, freeze, inputs
from vgx.gpqa.prompt import parse_verifier_response
from vgx.gpqa.sequential import execute_bundle, policy_parts, seal
from vgx.gpqa.verifiers import VerifierSpec
from vgx.gpqa.workflow import prepare_policy, score_execution


def specs_from_config(config):
    return tuple(VerifierSpec(id=f'verifier_{i}',provider='vertex_ai',model=model,
        location=config['inference']['model_locations'][model],project=config['inference']['project_id'],
        generation=config['generation']) for i,model in enumerate(config['models']['verifiers'],1))


def progress(root, phase, **details):
    event={'phase':phase,'updated_unix':time.time(),**details}
    atomic_json(Path(root)/'progress.json',event)
    append_jsonl(Path(root)/'progress_events.jsonl',event)
    print(json.dumps(event),flush=True)


def collect_verifiers(candidates, specs, cache, operation, *, allow_api=False, workers=4):
    def collect(candidate):
        rows=[]
        for spec in specs:
            if candidate.answer is None:
                continue
            call,cached=cache.get(spec.runner(),spec.request(candidate,operation),allow_api=allow_api)
            score,failure=spec.signal(call.response,candidate)
            rows.append({'item_id':candidate.item_id,'verifier_id':spec.id,'score':score,
                         'failure':failure,'request_key':call.key,'used_cached_response':cached})
        return rows
    return [row for batch in parallel_map(collect,candidates,workers) for row in batch]


def smoke(root, config, cache, *, allow_api=False, workers=4):
    candidates=collect_generators(root,config,cache,subset='pilot',allow_api=allow_api,workers=workers)
    observations=collect_verifiers(candidates,specs_from_config(config),cache,'pilot-calibration',
                                  allow_api=allow_api,workers=workers)
    result={'n':len(candidates),'generator_parse_ok_n':sum(c.generator_ok for c in candidates),
            'verifier_observations':len(observations),
            'verifier_parse_ok_n':sum(o['score'] is not None for o in observations),
            'budget':cache.summary()}
    atomic_json(Path(root)/'pilot_collection_check.json',result)
    # This gate is engineering-only and does not inspect candidate correctness.
    if any(not c.generator_ok for c in candidates) or any(o['score'] is None for o in observations):
        raise ValueError('pilot has parsing failures; inspect frozen responses before expanding collection')
    return result


def prepare_policies(root, config, cache):
    specs=specs_from_config(config)
    policies={}
    for scenario in config['policy_scenarios']:
        policy=prepare_policy(Path(root)/'frozen',specs,cache,
            correct_reward=scenario['correct_reward'],incorrect_loss=scenario['incorrect_loss'],
            pricing=config['api_pricing'],utility_per_usd=scenario['utility_per_usd'])
        _write_immutable(Path(root)/'policies'/f"{scenario['name']}.json",json.dumps(policy,sort_keys=True,indent=2)+'\n')
        policies[scenario['name']]=policy
    plan=seal({'dataset_id':inputs(root)[0]['dataset_id'],
               'policies':{name:p['policy_id'] for name,p in policies.items()},
               'primary':config['protocol']['primary_scenario'],
               'primary_order':list(config['protocol']['primary_order']),
               'baselines':['always_answer','always_abstain','confidence_only',
                            'query_one_verifier','query_second_verifier','query_all_verifiers'],
               'baseline_collection':'after primary live decisions are sealed',
               'config_sha256':digest(config)},'analysis_plan_id')
    _write_immutable(Path(root)/'analysis_plan.json',json.dumps(plan,sort_keys=True,indent=2)+'\n')
    return policies


def baseline_records(root, cache, specs):
    """Offline scoring boundary; called only after live decisions are sealed."""
    root=Path(root)
    manifest,candidates=load_candidates(root/'frozen')
    labels_path=root/'frozen'/'evaluation_labels.jsonl'
    if file_digest(labels_path)!=manifest['files']['evaluation_labels.jsonl']:
        raise ValueError('evaluation answer key changed')
    calibration=read_jsonl(root/'frozen'/'calibration_records.jsonl')
    labels={r['item_id']:r['correct_index'] for r in [*calibration,*read_jsonl(labels_path)]}
    records=[]
    for c in candidates:
        verifiers={}
        for spec in specs:
            if c.answer is None:
                verifiers[spec.id]={'p_correct':None,'ok':False,'failure':'invalid_generator_answer'}
            else:
                call,_=cache.get(spec.runner(),spec.request(c),allow_api=False)
                verifiers[spec.id]=asdict(parse_verifier_response(call.response))
        records.append({'item_id':c.item_id,'partition':c.partition,'subject':c.subject,
            'correct_index':labels[c.item_id],'generator_answer':c.answer,'generator_p_correct':c.p_correct,
            'generator_ok':c.generator_ok,'generator_failure':c.generator_failure,'verifiers':verifiers})
    return records


def report(root,config,cache,policies,primary_decisions):
    from vgx.gpqa.report import build_report, _policy_report
    root=Path(root)
    records=baseline_records(root,cache,specs_from_config(config))
    rows={r['item_id']:r for r in records}
    primary_name=config['protocol']['primary_scenario']
    specs,planner=policy_parts(policies[primary_name])
    checked=0
    for decision in primary_decisions['results']:
        row=rows[decision['candidate_id']]
        if row['generator_answer'] is None or row['generator_p_correct'] is None:
            if decision['action']!='abstain' or decision['verifiers_used']!=0:
                raise ValueError('invalid candidate live/replay mismatch')
        else:
            replay=planner.replay(row['generator_p_correct'],[row['verifiers'][s.id]['p_correct'] for s in specs])
            if any(replay[k]!=decision[k] for k in ('action','verifiers_used','failure')) or not math.isclose(replay['posterior'],decision['posterior'],abs_tol=1e-12):
                raise ValueError('live/replay decision mismatch')
        checked+=1
    analysis_config={**config,'routing_scenarios':[
        {'name':name,'correct_reward':p['planner']['correct_reward'],
         'incorrect_loss':p['planner']['incorrect_loss'],'verifier_costs':p['planner']['costs']}
        for name,p in policies.items()]}
    result=build_report(records,analysis_config,seed=config['seed'])
    # Both individual verifiers are needed to interpret the cost/value of the
    # pair. All baseline signals are collected after the primary live run.
    for name,policy in policies.items():
        fitted_specs,fitted_planner=policy_parts(policy)
        if len(fitted_specs)==2:
            tags=[s.id for s in fitted_specs]
            fits=dict(zip(tags,fitted_planner.likelihoods))
            reverse={'correct_reward':fitted_planner.reward,'incorrect_loss':fitted_planner.loss,
                     'verifier_costs':list(reversed(fitted_planner.costs))}
            second=_policy_report([r for r in records if r['partition']=='evaluation'],
                                  list(reversed(tags)),fits,reverse,500,config['seed'])
            section=result['evaluation_metrics']['routing_scenarios'][name]
            section['policies']['query_second_verifier']=second['policies']['query_one_verifier']
            section['paired_differences']['query_second_verifier_minus_confidence_only']=second['paired_differences']['query_one_verifier_minus_confidence_only']
    result['fresh_protocol']={'dataset':inputs(root)[0],
        'analysis_plan_sha256':file_digest(root/'analysis_plan.json'),
        'primary_live':score_execution(root/'frozen',policies[primary_name],primary_decisions),
        'live_replay_matched_n':checked,
        'assumptions':config['protocol']['assumptions'],
        'note':'Diamond evaluation follows calibration on Main-minus-Diamond. This is a transfer feasibility study.'}
    ledger=summarize_vertex_usage(cache.logs(),config['api_pricing'])
    atomic_json(root/'usage_estimate.json',ledger)
    selected_keys=[k for r in primary_decisions['results'] for k in r['observed_request_keys']]
    atomic_json(root/'primary_live_usage.json',summarize_vertex_usage(cache.logs(selected_keys),config['api_pricing'],operation_id=primary_decisions['operation_id']))
    atomic_json(root/'report.json',result)
    (root/'summary.md').write_text(summary_markdown(result,ledger,config))
    _write_immutable(root/'analysis_records.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    return result


def summary_markdown(result,ledger,config):
    def pct(value):
        return 'N/A' if value is None else f'{100*value:.1f}%'
    def num(value):
        return 'N/A' if value is None else f'{value:.4f}'
    evaluation=result['evaluation_metrics']
    calibration=result['calibration_metrics']
    protocol=result['fresh_protocol']
    primary=config['protocol']['primary_scenario']
    scenario=next(s for s in config['policy_scenarios'] if s['name']==primary)
    excluded=protocol['dataset']['exclusions']
    policy_rows=evaluation['routing_scenarios'][primary]['policies']
    lines=['# Fresh GPQA Vertex experiment','',
        '## Design','',
        f"- Calibration: {calibration['n']} Main-minus-Diamond questions; evaluation: {evaluation['n']} disjoint Diamond questions.",
        f"- {len(excluded['main'])} Main questions with duplicate choices were excluded, including {len(excluded['diamond'])} Diamond questions. This is a filtered Diamond evaluation.",
        f"- Dataset revision: `{protocol['dataset']['dataset_revision']}`; split seed: `{config['seed']}`.",
        f"- Generator: `{config['models']['generator']}`.",
        '- Verifiers, in order: '+', '.join(f'`{m}`' for m in config['models']['verifiers'])+'.',
        '- New candidates generated once and shared by every policy; Jev excluded.',
        f"- Primary utility: correct release +{scenario['correct_reward']:g}, wrong release -{scenario['incorrect_loss']:g}, abstention 0; nominal assertion threshold {scenario['incorrect_loss']/(scenario['correct_reward']+scenario['incorrect_loss']):g}.",
        f"- USD costs converted at {scenario['utility_per_usd']:g} utility units/USD. Expected query costs were estimated on calibration and frozen before evaluation.",
        '', '## Primary results','',
        '| Policy | Released | Coverage | Accuracy among released | 95% Wilson interval | Mean verifier queries | Mean utility |',
        '|---|---:|---:|---:|---|---:|---:|']
    for name,row in policy_rows.items():
        ci=row['accuracy_among_released_ci95']
        lines.append(f"| {name} | {row['released_n']} | {pct(row['release_coverage'])} | {pct(row['accuracy_among_released'])} | {pct(ci['low'])}–{pct(ci['high'])} | {num(row['mean_queries'])} | {num(row['utility']['estimate'])} |")
    lines += ['',f"Generator accuracy across all evaluation questions: **{pct(evaluation['generator_answer_accuracy_all_items'])}**.",
        f"Live/replay decisions matched on **{protocol['live_replay_matched_n']}** evaluation questions.",
        '', '## Calibration and forecasts','',
        f"Calibration generator answers: {calibration['correct_n']} correct; {calibration['incorrect_or_invalid_answer_n']} incorrect or invalid.",
        '', '| Forecast on evaluation | Brier score | Log loss |','|---|---:|---:|',
        f"| Generator raw confidence | {num(evaluation['generator_confidence']['brier'])} | {num(evaluation['generator_confidence']['log_loss'])} |",
        f"| Calibration base rate | {num(evaluation['base_rate_forecast'].get('brier'))} | {num(evaluation['base_rate_forecast'].get('log_loss'))} |",
        f"| Posterior after both verifiers, independence assumption | {num(evaluation['sequential_independence_posterior']['brier'])} | {num(evaluation['sequential_independence_posterior']['log_loss'])} |",
        '', 'The posterior row queries both verifiers for an offline diagnostic. The live policy stops adaptively and has different query costs.',
        '', '## Cost and billing','',
        f"- Resource and quota project: `{config['inference']['project_id']}`.",
        f"- Successful provider responses: {ledger['successful_calls']}; priced responses: {ledger['priced_calls']}.",
        f"- Total token-based estimate: **USD {ledger['estimated_usd_for_priced_calls']:.6f}** against the USD {config['budget_usd']:.0f} cap.",
        f"- Explicit HTTP rejection records: {ledger['rejected_attempt_records']}; unresolved attempts: {len(ledger['unresolved_attempts'])}.",
        '- This is an estimate, not a confirmed bill. It includes generator, calibration, live verification and extra baseline collection.',
        '- Logical policy queries include cached responses. Actual incremental live charges are recorded separately in `primary_live_usage.json`.',
        '', '## Interpretation limits','',
        '- Main-minus-Diamond to Diamond changes the question distribution and subject mix. Likelihood transfer is an assumption.',
        '- Raw elicited confidence is not established as the model’s true belief or a calibrated correctness probability.',
        '- Conditional independence, including dependence on generator confidence, is not guaranteed. Diagnostics and sparse-class uncertainty are in `report.json`.',
        f"- Exactly 0/1 generator priors cannot be updated by this model; evaluation contains {evaluation['endpoint_confidence_n']} such forecasts.",
        '- The API model aliases do not guarantee immutable model weights. Provider model identifiers and usage are preserved where returned.',
        '- Confidence intervals are descriptive and conditional on the fitted calibration model; this experiment does not establish strategic truthfulness or incentive compatibility.',
        '', '## Sources','',
        '- [Official GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa)',
        '- [GPQA subset definitions](https://arxiv.org/html/2311.12022v1#S2.SS3)',
        '- [Google model pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing)', '']
    return '\n'.join(lines)


def run(root,config,*,allow_api=False,phase='complete',workers=4):
    root=Path(root)
    project=config['inference']['project_id']
    cache=BudgetedRequestCache(root/'cache',project=project,pricing=config['api_pricing'],limit_usd=config['budget_usd'])
    with file_lock(root/'experiment.lock'):
        _write_immutable(root/'experiment_config.json',json.dumps(config,sort_keys=True,indent=2)+'\n')
        manifest,_=inputs(root)
        expected_revision=config.get('dataset_source',{}).get('revision')
        if (expected_revision is not None and manifest['dataset_revision']!=expected_revision) or manifest['seed']!=config['seed']:
            raise ValueError('dataset revision or seed differs from reviewed configuration')
        if manifest['counts']!={'calibration':config['calibration_size'],'evaluation':config['sample_size']-config['calibration_size']}:
            raise ValueError('cohort counts differ from reviewed configuration')
        progress(root,'pilot_collection',api_enabled=allow_api)
        pilot=smoke(root,config,cache,allow_api=allow_api,workers=workers)
        progress(root,'pilot_complete',**pilot)
        if phase=='pilot':
            return pilot
        progress(root,'generator_collection')
        collect_generators(root,config,cache,allow_api=allow_api,workers=workers)
        frozen=freeze(root,config,cache)
        _,candidates=load_candidates(root/'frozen')
        progress(root,'calibration_collection',candidate_n=len(candidates),budget=cache.summary())
        observations=collect_verifiers([c for c in candidates if c.partition=='calibration'],specs_from_config(config),cache,
                                       'full-calibration',allow_api=allow_api,workers=workers)
        atomic_json(root/'calibration_collection.json',{'observations':observations})
        policies=prepare_policies(root,config,cache)
        progress(root,'policies_frozen',policies={k:v['policy_id'] for k,v in policies.items()},budget=cache.summary())
        if phase=='calibration':
            return {'policies':list(policies),'budget':cache.summary()}
        primary=config['protocol']['primary_scenario']
        decisions=execute_bundle(root/'frozen',policies[primary],cache,root/'live'/primary,
                                 allow_api=allow_api,workers=workers)
        progress(root,'primary_live_complete',logical_queries=decisions['logical_verifier_queries'],budget=cache.summary())
        if phase=='live':
            return {'decisions_id':decisions['decisions_id'],'budget':cache.summary()}
        # Additional responses support paired baselines. They cannot affect the
        # already completed live primary decisions and have a separate operation.
        observations=collect_verifiers([c for c in candidates if c.partition=='evaluation'],specs_from_config(config),cache,
                                       'evaluation-baselines',allow_api=allow_api,workers=workers)
        atomic_json(root/'baseline_collection.json',{'observations':observations})
        progress(root,'offline_reporting',budget=cache.summary())
        result=report(root,config,cache,policies,decisions)
        progress(root,'complete',report=str(root/'report.json'),budget=cache.summary())
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['pilot','calibration','live','complete'],default='complete')
    parser.add_argument('--allow-api',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    try:
        run(args.root,json.loads(args.config.read_text()),allow_api=args.allow_api,phase=args.phase,workers=args.workers)
    except Exception as error:
        progress(args.root,'paused_error',error_type=type(error).__name__,http_status=getattr(error,'status',None))
        raise


if __name__=='__main__':
    main()
