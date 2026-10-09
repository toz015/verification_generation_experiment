"""Frozen-candidate collection, calibration-only policy fitting, and live execution.

All commands are offline unless collect/execute explicitly receive --allow-api.
There is no generator collection command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

from vgx.common.api import RequestCache
from vgx.common.billing import estimate_call, summarize_vertex_usage
from vgx.common.storage import atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.score import correctness_outcome, fit_verifier_likelihood
from vgx.gpqa.sequential import execute_bundle, policy_parts, seal, unseal
from vgx.gpqa.verifiers import VerifierSpec, load_specs


def validate_original_specs(manifest, specs):
    """Prevent accidental changes to the original comparison conditions."""
    config = manifest['collection_config']
    originals = {f'verifier_{i}': model for i, model in enumerate(config['models']['verifiers'], 1)}
    for spec in specs:
        if spec.id not in originals:
            continue
        from vgx.gpqa.artifacts import vertex_from_config
        if spec.model != originals[spec.id] or spec.runner().identity != vertex_from_config(config, spec.model).identity:
            raise ValueError('original verifier model or generation settings changed; use a separate arm ID')


def collect_frozen(bundle: Path, specs, cache: RequestCache, output: Path,
                   *, partition='calibration', allow_api=False):
    manifest, candidates = load_candidates(bundle)
    validate_original_specs(manifest, specs)
    if partition not in ('calibration', 'evaluation'):
        raise ValueError('choose calibration or evaluation explicitly')
    selected = [c for c in candidates if c.partition == partition]
    identity = {'bundle_id': manifest['bundle_id'], 'partition': partition,
                'verifiers': [asdict(s) for s in specs], 'identities': [s.identity for s in specs]}
    output = Path(output)
    with file_lock(str(output)+'.lock'):
        run_path = Path(str(output)+'.manifest.json')
        if run_path.exists():
            run = json.loads(run_path.read_text())
            if run['identity'] != identity:
                raise ValueError('collection output belongs to a different frozen comparison')
        else:
            run = {'identity': identity, 'operation_id': str(uuid.uuid4())}
            atomic_json(run_path, run)
        observations = []
        for candidate in selected:
            for spec in specs:
                if candidate.answer is None:
                    observations.append({'item_id': candidate.item_id, 'verifier_id': spec.id,
                                         'score': None, 'failure': 'invalid_generator_answer'})
                    continue
                call, cached = cache.get(spec.runner(), spec.request(candidate, run['operation_id']), allow_api=allow_api)
                score, failure = spec.signal(call.response, candidate)
                observations.append({'item_id': candidate.item_id, 'verifier_id': spec.id,
                    'score': score, 'failure': failure, 'request_key': call.key,
                    'execution_id': call.meta.get('execution_id'), 'used_cached_response': cached})
        result = {**run, 'observations': observations}
        atomic_json(output, result)
        return result


def prepare_policy(bundle: Path, specs, cache: RequestCache, *, correct_reward: float, incorrect_loss: float,
                   normalized_costs=None, pricing=None, utility_per_usd=None, grid_size=1001):
    manifest, candidates = load_candidates(bundle)
    validate_original_specs(manifest, specs)
    calibration_path = Path(bundle)/'calibration_records.jsonl'
    if file_digest(calibration_path) != manifest['files']['calibration_records.jsonl']:
        raise ValueError('calibration labels changed')
    calibration = read_jsonl(calibration_path)
    labels = {r['item_id']: r for r in calibration}
    if len(labels) != len(calibration) or set(labels) != set(manifest['calibration_ids']):
        raise ValueError('calibration cohort differs from frozen split')
    if normalized_costs is not None:
        if utility_per_usd is not None or len(normalized_costs) != len(specs):
            raise ValueError('choose normalized costs or monetary conversion, not both')
    elif utility_per_usd is None or not isinstance(utility_per_usd, (int,float)) or isinstance(utility_per_usd, bool) or not 0 < utility_per_usd < float('inf'):
        raise ValueError('positive finite utility_per_usd required for monetary costs')
    likelihoods, fit_info, costs = [], [], []
    for index, spec in enumerate(specs):
        y, scores, estimates, keys = [], [], [], []
        for candidate in candidates:
            if candidate.partition != 'calibration' or candidate.answer is None or candidate.p_correct is None:
                continue
            row = labels[candidate.item_id]
            if (row['generator_answer'], row['generator_p_correct']) != (candidate.answer, candidate.p_correct):
                raise ValueError('calibration candidate differs from frozen public candidate')
            call, _ = cache.get(spec.runner(), spec.request(candidate), allow_api=False)
            keys.append(call.key)
            score, failure = spec.signal(call.response, candidate)
            if score is not None:
                scores.append(score)
                y.append(correctness_outcome(candidate.answer, row['correct_index']))
            if normalized_costs is None:
                estimate = estimate_call(asdict(call), pricing or {})
                if estimate['estimated_usd'] is None:
                    raise ValueError(f'unresolved calibration cost for {spec.id}: {estimate["issues"]}')
                estimates.append(estimate['estimated_usd'])
        likelihood = fit_verifier_likelihood(y, scores)
        likelihoods.append(likelihood)
        mean_usd = sum(estimates)/len(estimates) if estimates else None
        if normalized_costs is None and mean_usd is None:
            raise ValueError('no calibration usage available for expected query cost')
        costs.append(normalized_costs[index] if normalized_costs is not None else mean_usd*utility_per_usd)
        fit_info.append({'verifier_id': spec.id, 'valid_signal_n': len(y), 'correct_n': sum(y),
                         'incorrect_n': len(y)-sum(y), 'request_keys_sha256': digest(keys),
                         'cost_observation_n': len(estimates), 'mean_calibration_usd': mean_usd})
    planner = NestedPlanner(likelihoods, costs, correct_reward, incorrect_loss, grid_size=grid_size)
    mapping = {'kind': 'normalized_sensitivity', 'costs': costs} if normalized_costs is not None else {
        'kind': 'calibration_mean_usd', 'utility_per_usd': utility_per_usd, 'pricing_snapshot': pricing,
        'formula': 'query_cost_utility = utility_per_usd * mean_calibration_request_usd',
        'note': 'Expected costs fixed before evaluation; realized evaluation token counts never enter decisions.'}
    return seal({'schema': 1, 'bundle_id': manifest['bundle_id'],
                 'calibration_sha256': manifest['files']['calibration_records.jsonl'],
                 'verifiers': [asdict(s) for s in specs], 'verifier_identities': [s.identity for s in specs],
                 'fit_diagnostics': fit_info, 'cost_mapping': mapping, 'planner': planner.to_dict(),
                 'assumptions': ['raw generator report as prior', 'pooled conditional-independent verifier channels',
                                 'fixed order; singleton layers', 'no terminal audit or generator learning']}, 'policy_id')


def prepare_comparison(bundle: Path, config: dict, cache: RequestCache) -> dict:
    """Render a reviewable plan and cache inventory, without any inference."""
    manifest, candidates = load_candidates(bundle)
    specs = load_specs(config)
    validate_original_specs(manifest, specs)
    spec_ids = {s.id for s in specs}
    arms = config.get('arms', [])
    for arm in arms:
        if not arm['order'] or len(set(arm['order'])) != len(arm['order']) or set(arm['order']) - spec_ids:
            raise ValueError('invalid comparison arm')
        if sum(next(s.provider == 'typesafe' for s in specs if s.id == tag) for tag in arm['order']) > 1:
            raise ValueError('Jev Choice and Noul are separate arms, not independent layers')
    inventory = []
    for spec in specs:
        runner = spec.runner()
        counts = {p: {'candidate_n': 0, 'cached_n': 0, 'missing_n': 0} for p in ('calibration','evaluation')}
        for candidate in candidates:
            if candidate.answer is None:
                continue
            count = counts[candidate.partition]
            count['candidate_n'] += 1
            key = runner.cache_key(spec.request(candidate))
            found = any(r.get('key') == key and r.get('error') is None for r in cache.log(key).records())
            count['cached_n' if found else 'missing_n'] += 1
        inventory.append({'spec': asdict(spec), 'identity': spec.identity, 'cache': counts})
    return seal({'schema': 1, 'bundle_id': manifest['bundle_id'], 'generator_model': manifest['generator_model'],
                 'generator_calls_planned': 0, 'verifiers': inventory, 'arms': arms,
                 'primary_jev_arm': config.get('primary_jev_arm'),
                 'signal_semantics': 'Choice: probability of frozen generator option; Noul: yes probability. Never confidence.',
                 'evaluation_status': 'existing evaluation is exploratory; reserve unused items for confirmation',
                 'collection_status': 'prepared_only_no_calls', 'generator_and_original_prompts_frozen': True}, 'comparison_id')


def score_execution(bundle: Path, policy: dict, decisions: dict) -> dict:
    """Separate offline evaluation. This is the only live-workflow step opening evaluation labels."""
    manifest, candidates = load_candidates(bundle)
    unseal(decisions, 'decisions_id')
    specs, planner = policy_parts(policy)
    binding = decisions['binding']
    if binding['bundle_id'] != manifest['bundle_id'] or binding['policy_id'] != policy['policy_id'] or binding['partition'] != 'evaluation':
        raise ValueError('decision provenance mismatch')
    path = Path(bundle)/'evaluation_labels.jsonl'
    if file_digest(path) != manifest['files']['evaluation_labels.jsonl']:
        raise ValueError('evaluation labels changed')
    labels = {row['item_id']: row['correct_index'] for row in read_jsonl(path)}
    by_id = {c.item_id: c for c in candidates if c.partition == 'evaluation'}
    results = decisions['results']
    if len(results) != len(by_id) or {r['candidate_id'] for r in results} != set(by_id) or set(labels) != set(by_id):
        raise ValueError('incomplete evaluation decisions or labels')
    correct, released, wrong_released, wrong, utility, queries = 0, 0, 0, 0, [], []
    for row in results:
        candidate = by_id[row['candidate_id']]
        if row['candidate_sha256'] != digest(asdict(candidate)) or row['policy_id'] != policy['policy_id']:
            raise ValueError('decision candidate/policy mismatch')
        if row['action'] not in ('assert', 'abstain') or not 0 <= row['verifiers_used'] <= len(specs):
            raise ValueError('invalid final action or query count')
        y = correctness_outcome(candidate.answer, labels[candidate.item_id])
        release = row['action'] == 'assert'
        correct += y*release
        released += release
        wrong_released += (1-y)*release
        wrong += 1-y
        cost = sum(planner.costs[:row['verifiers_used']])
        utility.append(((planner.reward if y else -planner.loss) if release else 0.) - cost)
        queries.append(row['verifiers_used'])
    n = len(results)
    from vgx.gpqa.report import mean_interval, rate_interval
    return {'n': n, 'released_n': released, 'accuracy_among_released': correct/released if released else None,
            'accuracy_among_released_ci95': rate_interval([1]*correct+[0]*(released-correct)),
            'release_coverage': released/n if n else None,
            'release_coverage_ci95': rate_interval([1]*released+[0]*(n-released)),
            'wrong_release_fraction_all_items': wrong_released/n if n else None,
            'wrong_release_fraction_all_items_ci95': rate_interval([1]*wrong_released+[0]*(n-wrong_released)),
            'false_accept_rate_among_incorrect_candidates': wrong_released/wrong if wrong else None,
            'mean_queries': sum(queries)/n if n else None, 'utility': mean_interval(utility),
            'cost_mapping': policy['cost_mapping'], 'operation_id': decisions['operation_id'],
            'note': 'Utility uses expected configured costs; actual usage/billing is a separate ledger.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'prepare-policy', 'execute', 'prepare-comparison', 'score'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('results/request_cache'))
    parser.add_argument('--specs', type=Path)
    parser.add_argument('--verifiers', nargs='+')
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', choices=['calibration', 'evaluation'])
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--reward', type=float, default=1.)
    parser.add_argument('--loss', type=float, default=4.)
    parser.add_argument('--normalized-costs', type=float, nargs='+')
    parser.add_argument('--utility-per-usd', type=float)
    parser.add_argument('--pricing', type=Path)
    parser.add_argument('--grid-size', type=int, default=1001)
    args = parser.parse_args()
    if args.allow_api and args.command not in ('collect', 'execute'):
        parser.error('--allow-api is valid only for collect or execute')
    if args.verifiers and args.command == 'prepare-comparison':
        parser.error('prepare-comparison takes its complete arm definitions from --specs')
    cache = RequestCache(args.cache)
    pricing = json.loads(args.pricing.read_text()) if args.pricing else {}
    pricing = pricing.get('api_pricing', pricing)
    if args.command in ('execute', 'score'):
        if not args.policy:
            parser.error('--policy is required')
        policy = json.loads(args.policy.read_text())
        if args.command == 'execute':
            result = execute_bundle(args.bundle, policy, cache, args.output, partition=args.partition or 'evaluation', allow_api=args.allow_api)
            keys = [key for row in result['results'] for key in row['observed_request_keys']]
            ledger = summarize_vertex_usage(cache.logs(keys), pricing, operation_id=result['operation_id'])
            atomic_json(args.output/'incremental_usage_estimate.json', ledger)
        else:
            if not args.decisions:
                parser.error('--decisions is required')
            result = score_execution(args.bundle, policy, json.loads(args.decisions.read_text()))
            atomic_json(args.output, result)
    else:
        if not args.specs:
            parser.error('--specs is required')
        config = json.loads(args.specs.read_text())
        specs = load_specs(config)
        if args.verifiers:
            by_id = {s.id:s for s in specs}
            if len(set(args.verifiers)) != len(args.verifiers) or set(args.verifiers) - by_id.keys():
                parser.error('--verifiers must be distinct configured IDs')
            specs = tuple(by_id[key] for key in args.verifiers)
        if args.command == 'collect':
            result = collect_frozen(args.bundle, specs, cache, args.output, partition=args.partition or 'calibration', allow_api=args.allow_api)
            atomic_json(Path(str(args.output)+'.usage.json'),
                        summarize_vertex_usage(cache.logs([r['request_key'] for r in result['observations'] if r.get('request_key')]),
                                               pricing, operation_id=result['operation_id']))
        elif args.command == 'prepare-policy':
            result = prepare_policy(args.bundle, specs, cache, correct_reward=args.reward, incorrect_loss=args.loss,
                                    normalized_costs=args.normalized_costs, pricing=pricing,
                                    utility_per_usd=args.utility_per_usd, grid_size=args.grid_size)
            atomic_json(args.output, result)
        else:
            result = prepare_comparison(args.bundle, config, cache)
            atomic_json(args.output, result)
    print(json.dumps({'command': args.command, 'output': str(args.output), 'api_enabled': args.allow_api}))


if __name__ == '__main__':
    main()
