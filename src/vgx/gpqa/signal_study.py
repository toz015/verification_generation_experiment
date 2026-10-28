"""Phase-one calibration-only verifier screening. No generator requests or evaluation scoring.

prepare is offline; collect requires --allow-api; report reads durable caches only.
The candidate sample is ordered without consulting correctness labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, RequestCache, UncertainRequestError
from vgx.common.billing import estimate_call
from vgx.common.budget import BudgetedRequestCache, BudgetLimitError
from vgx.common.llm import Request
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.common.vertex import VertexBatchRunner
from vgx.common.vertex_partner import VertexPartnerRunner
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.prompt import _payload, _probability
from vgx.gpqa.score import binary_forecast_metrics, correctness_outcome, fit_verifier_likelihood
from vgx.gpqa.sequential import seal, unseal


PROMPT_VERSION = 'fixed_candidate_signal_v1'
SYSTEM = 'You evaluate whether a proposed answer to a graduate-level multiple-choice science question is correct.'
COMMON = ('Assess whether the fixed candidate answer is correct. Consider the question and all options. '
          'Do not suggest a replacement answer. Return only the requested JSON object, without explanation.\n')
INSTRUCTIONS = {
    'binary': 'Make a binary judgment. Return {"correct": true} if the fixed candidate is correct, otherwise {"correct": false}.',
    'probability': ('Report the probability that the fixed candidate answer is correct as a number from 0 to 1. '
                    'Intermediate values are allowed; do not restrict your answer to 0 or 1. '
                    'Return {"p_correct": <number>}.')}


def prompt(candidate, signal):
    if signal not in INSTRUCTIONS or candidate.answer not in ('A', 'B', 'C', 'D'):
        raise ValueError('supported signal and fixed answer required')
    options = '\n'.join(f'{letter}. {choice}' for letter, choice in zip('ABCD', candidate.choices))
    answer = candidate.choices['ABCD'.index(candidate.answer)]
    return (f'Subject: {candidate.subject}\n\nQuestion: {candidate.question}\n\n{options}\n\n'
            f'Fixed candidate answer: {candidate.answer}. {answer}\n\n{COMMON}{INSTRUCTIONS[signal]}')


def parse_signal(text, signal):
    if signal not in INSTRUCTIONS:
        raise ValueError('unknown signal format')
    value = _payload(text)
    if value is None:
        return None, 'invalid_json'
    if signal == 'binary':
        if set(value) != {'correct'} or type(value['correct']) is not bool:
            return None, 'expected_boolean_correct'
        return float(value['correct']), None
    if set(value) != {'p_correct'} or isinstance(value['p_correct'], bool):
        return None, 'expected_numeric_probability'
    score = _probability(value['p_correct'])
    return (score, None) if score is not None else (None, 'invalid_probability')


def runner_for(model, project):
    if model['transport'] == 'chat_completions':
        return VertexBatchRunner(model['model'], model['location'], project, **model['generation'])
    if model['transport'] == 'partner_raw_predict':
        return VertexPartnerRunner(model['model'], model['location'], project, **model['generation'])
    raise ValueError('unsupported Vertex transport')


def request_for(candidate, model, signal, operation_id=None):
    return Request(f"{model['id']}:{signal}|{candidate.item_id}", prompt(candidate, signal), SYSTEM,
                   {'role': 'verifier', 'partition': 'calibration', 'item_id': candidate.item_id,
                    'candidate': candidate.answer, 'prompt_version': PROMPT_VERSION,
                    'signal': signal, 'verifier_id': model['id'], 'operation_id': operation_id})


def candidates_for(config):
    manifest, candidates = load_candidates(Path(config['bundle']))
    eligible = [c for c in candidates if c.partition == 'calibration'
                and c.answer is not None and c.p_correct is not None]
    eligible.sort(key=lambda c: digest({'seed': config['seed'], 'item_id': c.item_id}))
    if not eligible:
        raise ValueError('no eligible calibration candidates')
    return manifest, eligible


def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ['gpqa/signal_study.py', 'gpqa/artifacts.py', 'gpqa/prompt.py', 'common/vertex.py',
             'common/vertex_partner.py', 'common/api.py', 'common/budget.py', 'common/billing.py']
    return {name: file_digest(root/name) for name in paths}


def prepare(config, output):
    manifest, candidates = candidates_for(config)
    if config['schema'] != 1 or config['project'] != 'llm-applications-490420':
        raise ValueError('unsupported schema or unauthorized project')
    if config['signals'] != ['binary', 'probability']:
        raise ValueError('paired binary/probability arms required')
    ids = [m['id'] for m in config['models']]
    names = [m['model'] for m in config['models']]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)) or not ids:
        raise ValueError('unique candidate models required')
    budget = BudgetedRequestCache(Path(output)/'cache', project=config['project'],
                                 pricing=config['pricing'], limit_usd=config['stage_budget_usd'])
    arms = []
    for model in config['models']:
        runner = runner_for(model, config['project'])
        rates = config['pricing']['usd_per_million_tokens'][model['model']]
        for signal in config['signals']:
            requests = [request_for(c, model, signal) for c in candidates]
            input_allowance = sum(len((r.prompt+r.system).encode())+1024 for r in requests)
            arms.append({'id': model['id']+':'+signal, 'runner': runner.identity,
                         'request_keys': [runner.cache_key(r) for r in requests],
                         'conservative_total_reservations_usd': sum(budget.reservation(runner,r) for r in requests),
                         'planning_usd_at_1024_output_tokens_per_call':
                         (input_allowance*rates['input']+len(requests)*1024*rates['output'])/1e6})
    value = seal({'schema': 1, 'config': config, 'bundle_id': manifest['bundle_id'],
                  'candidate_ids': [c.item_id for c in candidates],
                  'excluded_calibration_count': len(manifest['calibration_ids'])-len(candidates),
                  'candidate_hashes': [digest(asdict(c)) for c in candidates],
                  'implementation_hashes': implementation_hashes(),
                  'prompt': {'version': PROMPT_VERSION, 'system': SYSTEM, 'common': COMMON,
                             'instructions': INSTRUCTIONS},
                  'arms': arms, 'generator_calls_planned': 0, 'evaluation_calls_planned': 0,
                  'maximum_successful_requests': len(candidates)*len(arms),
                  'planning_estimated_usd': sum(a['planning_usd_at_1024_output_tokens_per_call'] for a in arms),
                  'cost_note': 'Planning estimate, not confirmed billing or a bound on Gemini reasoning tokens. '
                               'Per-request reservations and the stage ledger gate actual collection.'}, 'study_id')
    path = Path(output)/'plan.json'
    with file_lock(str(path)+'.lock'):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('study changed; use a new output directory')
        atomic_json(path, value)
    return value


def load_plan(output):
    plan = json.loads((Path(output)/'plan.json').read_text())
    unseal(plan, 'study_id')
    if plan['implementation_hashes'] != implementation_hashes():
        raise ValueError('collection implementation changed; prepare a new study')
    manifest, candidates = candidates_for(plan['config'])
    if manifest['bundle_id'] != plan['bundle_id'] or [digest(asdict(c)) for c in candidates] != plan['candidate_hashes']:
        raise ValueError('frozen inputs changed')
    return plan, candidates


def check_billing(project):
    """Read-only ADC check. Never emit credentials or billing account identifiers."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    with AuthorizedSession(credentials) as session:
        response = session.get(f'https://cloudbilling.googleapis.com/v1/projects/{project}/billingInfo', timeout=30)
        response.raise_for_status()
        body = response.json()
    if body.get('projectId') != project or body.get('billingEnabled') is not True:
        raise ValueError('authorized billing project is not enabled')
    return {'project': project, 'billing_enabled': True, 'checked_at': time.time(),
            'note': 'Resource and quota project bound on every call; not confirmation of billed dollar amounts.'}


def collect(output, *, limit=10, allow_api=False, workers=3):
    plan, candidates = load_plan(output)
    if type(limit) is not int or not 1 <= limit <= len(candidates) or type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError('positive candidate limit within cohort and workers in 1..5 required')
    config = plan['config']; candidates = candidates[:limit]
    cache = BudgetedRequestCache(Path(output)/'cache', project=config['project'],
                                 pricing=config['pricing'], limit_usd=config['stage_budget_usd'])
    if allow_api:
        atomic_json(Path(output)/'billing_check.json', check_billing(config['project']))
    def model_job(model):
        runner = runner_for(model, config['project'])
        completed = 0; reused = 0; failures = Counter()
        for c in candidates:
            # Balance signal order deterministically without labels.
            signals = sorted(config['signals'], key=lambda s:digest([config['seed'],c.item_id,s]))
            for signal in signals:
                request = request_for(c, model, signal, plan['study_id'])
                try:
                    call, hit = cache.get(runner, request, allow_api=allow_api)
                except (OfflineCacheMiss, ProviderRejectedError, UncertainRequestError, BudgetLimitError) as exc:
                    return {'model':model['id'], 'completed':completed, 'reused':reused,
                            'parse_failures':dict(failures), 'blocked':type(exc).__name__,
                            'http_status':getattr(exc,'status',None)}
                completed += 1; reused += int(hit)
                _, failure = parse_signal(call.response, signal)
                if failure:
                    failures[signal+':'+failure] += 1
        return {'model':model['id'], 'completed':completed, 'reused':reused,
                'parse_failures':dict(failures), 'blocked':None}
    with file_lock(Path(output)/'collection.lock'):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            statuses = []
            for result in pool.map(model_job, config['models']):
                statuses.append(result)
                atomic_json(Path(output)/f'collection_{limit}.json',
                            {'study_id':plan['study_id'],'candidate_limit':limit,'models':statuses,'budget':cache.summary()})
    return {'models':statuses,'budget':cache.summary()}


def report(output):
    """Calibration-only signal summary; a returned boolean is not a probability forecast."""
    plan, candidates = load_plan(output); config = plan['config']
    manifest, _ = load_candidates(Path(config['bundle']))
    label_path = Path(config['bundle'])/'calibration_records.jsonl'
    if file_digest(label_path) != manifest['files'][label_path.name]:
        raise ValueError('calibration labels changed')
    labels = {r['item_id']:r for r in read_jsonl(label_path)}
    if set(labels) != set(manifest['calibration_ids']) or any(r['partition'] != 'calibration' for r in labels.values()):
        raise ValueError('invalid calibration membership')
    y = {c.item_id:correctness_outcome(c.answer,labels[c.item_id]['correct_index']) for c in candidates}
    cache = RequestCache(Path(output)/'cache'); arms=[]; records=[]
    for model in config['models']:
        runner=runner_for(model,config['project'])
        for signal in config['signals']:
            scores=[]; outcomes=[]; failures=Counter(); usd=0.; unpriced=0; returned=0
            for candidate in candidates:
                try:
                    call,_=cache.get(runner,request_for(candidate,model,signal))
                except OfflineCacheMiss:
                    continue
                returned+=1
                score,failure=parse_signal(call.response,signal)
                estimate=estimate_call(asdict(call),config['pricing'])
                if estimate['estimated_usd'] is None: unpriced+=1
                else: usd+=estimate['estimated_usd']
                records.append({'item_id':candidate.item_id,'model':model['id'],'signal':signal,
                                'score':score,'failure':failure,'correct':y[candidate.item_id],
                                'request_key':call.key})
                if failure: failures[failure]+=1
                else: scores.append(score);outcomes.append(y[candidate.item_id])
            confusion = {str(label):dict(Counter('accept' if p>=.5 else 'reject'
                         for p,label_i in zip(scores,outcomes) if label_i==label)) for label in (0,1)}
            fit = fit_verifier_likelihood(outcomes,scores,bins=2 if signal=='binary' else 3) if len(set(outcomes))==2 else None
            arms.append({'model':model['id'],'signal':signal,'returned':returned,'valid':len(scores),
                         'missing':len(candidates)-returned,'parse_failures':dict(failures),
                         'score_counts':dict(Counter(str(p) for p in scores)),
                         'confusion_at_0_5_by_candidate_correctness':confusion,
                         'likelihood_diagnostic':asdict(fit) if fit else None,
                         'estimated_usd':usd,'unpriced_calls':unpriced})
    value={'study_id':plan['study_id'],'calibration_n':len(candidates),
           'calibration_wrong':sum(1-v for v in y.values()),
           'raw_generator':binary_forecast_metrics([y[c.item_id] for c in candidates],
                                                   [c.p_correct for c in candidates]).to_dict(),
           'arms':arms,'limitations':['Partial samples only check protocol behavior; do not rank models from them.',
                     'Likelihoods here are descriptive fits, not out-of-fold performance or a sealed policy.',
                     'No evaluation labels are opened. Joint calibration comparisons follow full collection.',
                     'Selection across ten arms needs independent confirmation; no winner selected.',
                     '0.5 is a diagnostic score threshold, not the stopping-policy release threshold.']}
    atomic_json(Path(output)/'signal_report.json',value)
    atomic_json(Path(output)/'signal_records.json',records)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','collect','report'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        if args.config is None: parser.error('--config required for prepare')
        plan=prepare(json.loads(args.config.read_text()),args.output)
        print(json.dumps({k:plan[k] for k in ('study_id','maximum_successful_requests','planning_estimated_usd')}))
    elif args.command=='collect':
        print(json.dumps(collect(args.output,limit=args.limit,allow_api=args.allow_api,workers=args.workers)))
    else:
        value=report(args.output)
        print(json.dumps({'calibration_n':value['calibration_n'],'calibration_wrong':value['calibration_wrong'],
                          'arms':[{k:a[k] for k in ('model','signal','returned','valid','missing','estimated_usd')} for a in value['arms']]}))


if __name__=='__main__':
    main()
