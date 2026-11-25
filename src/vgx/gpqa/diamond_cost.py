"""Diamond cost comparison: asking verifiers one at a time vs asking a fixed set every time.

Uses the sealed development plan and the collected Diamond scores; nothing is
refitted. Raw generator confidence and the independence model throughout.

Arms, for each loss L in the plan (output iff the final belief >= L/(1+L)):
  prior_only    no verifier
  adaptive      the plan's selected three-verifier order, asked one at a time
                (the policy already scored as triple_independence)
  fixed_triple  the same three verifiers, all asked for every question
  fixed_five    all five verifiers, all asked for every question
A missing verifier score makes that question abstain, as in the adaptive rule.
Invalid generator answers get no verifier call and abstain.

Utility charges the plan's expected cost per call (as diamond_score does); the
measured cost of the Diamond requests is reported beside it.

`decisions` is label-free. `score` adds the local outcomes and writes only
aggregates and descriptive paired-bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json
from vgx.gpqa.dependence_triples import ExactPlanner, bin_of, execute
from vgx.gpqa.diamond_score import boot, edges_of, joint_for, posterior, tag
from vgx.gpqa.raw_prior_analysis import baseline_decision, candidate_valid
from vgx.gpqa.sequential import seal, unseal

ARMS = ('prior_only', 'adaptive', 'fixed_triple', 'fixed_five')


def load(plan_path, signals_path):
    plan = json.loads(Path(plan_path).read_text())
    unseal(plan, 'plan_id')
    signals = json.loads(Path(signals_path).read_text())
    unseal(signals, 'signals_id')
    return plan, signals


def verifiers_of(fit):
    return sorted(v.split(':')[0] for v in fit['likelihoods'])


def decisions(plan, signals):
    """Label-free decisions per generator, loss and arm."""
    s = plan['settings']
    edges = edges_of(plan)
    out = {}
    for g, fit in plan['generators'].items():
        rows = signals['rows'][g]
        verifiers = verifiers_of(fit)
        costs = {v: fit['expected_cost_usd'][tag(v)] for v in verifiers}
        out[g] = {}
        for loss in s['losses']:
            loss = float(loss)
            triple = list(fit['adaptive_policies'][f'raw/{loss:g}']['independence']['selected_order'])

            def run(order, force_all):
                p = ExactPlanner(joint_for(fit, tuple(order), 'independence'),
                                 [s['utility_per_usd']*costs[v] for v in order], s['correct_reward'], loss)
                return [execute(r['prior'], candidate_valid(r), p, tuple(tag(v) for v in order),
                                [costs[v] for v in order], r['signals'], edges, force_all=force_all)
                        for r in rows]
            out[g][f'{loss:g}'] = {
                'triple': triple,
                'arms': {'prior_only': [baseline_decision(r['prior'], candidate_valid(r), loss, s['correct_reward'])
                                        for r in rows],
                         'adaptive': run(triple, False),
                         'fixed_triple': run(triple, True),
                         'fixed_five': run(verifiers, True)}}
    return out


def cost_summary(ds):
    n = len(ds)
    calls = sum(d['verifiers_used'] for d in ds)
    expected = sum(d['expected_cost_usd'] for d in ds)
    measured = sum(d.get('usage_estimated_cost_usd', 0.) for d in ds)
    return {'n': n, 'released': sum(d['action'] == 'assert' for d in ds), 'calls': calls,
            'calls_per_question': calls/n, 'expected_usd_per_question': expected/n,
            'measured_usd_per_question': measured/n, 'measured_usd_total': measured,
            'missing_score_abstentions': sum(d.get('failure') == 'missing_verifier_score' for d in ds)}


def preview(plan, signals):
    """Label-free: calls, cost and number released per arm."""
    return {g: {loss: {'triple': v['triple'], 'arms': {k: cost_summary(ds) for k, ds in v['arms'].items()}}
                for loss, v in by_loss.items()}
            for g, by_loss in decisions(plan, signals).items()}


def brier(plan, signals, flags, repeats, seed):
    """Brier of the always-asked posterior: confidence alone, the selected triple, all five."""
    edges = edges_of(plan)
    out = {}
    for g, fit in plan['generators'].items():
        verifiers = verifiers_of(fit)
        rows = [r for r in signals['rows'][g] if candidate_valid(r)
                and all(r['signals'][tag(v)]['score'] is not None for v in verifiers)]
        y = np.array([flags[g][r['item_id']] for r in rows], float)
        triple = sorted(fit['adaptive_policies'][f"raw/{float(plan['settings']['losses'][0]):g}"]
                        ['independence']['selected_order'])
        prob = {'prior_only': np.array([r['prior'] for r in rows])}
        for name, vs in (('triple', triple), ('five', verifiers)):
            j = joint_for(fit, tuple(vs), 'independence')
            prob[name] = np.array([posterior(r['prior'], j.likelihood_ratio(
                tuple(bin_of(r['signals'][tag(v)]['score'], edges) for v in vs))) for r in rows])
        sq = {k: (p - y)**2 for k, p in prob.items()}
        out[g] = {'n': len(rows), 'triple': triple, 'brier': {k: float(v.mean()) for k, v in sq.items()},
                  'triple_minus_prior': boot(sq['triple'] - sq['prior_only'], repeats, seed),
                  'five_minus_prior': boot(sq['five'] - sq['prior_only'], repeats, seed),
                  'five_minus_triple': boot(sq['five'] - sq['triple'], repeats, seed)}
    return out


def score(plan_path, signals_path, outcomes_path, output, repeats=10000):
    plan, signals = load(plan_path, signals_path)
    flags = json.loads(Path(outcomes_path).read_text())
    unseal(flags, 'outcomes_id')
    if flags['signals_id'] != signals['signals_id']:
        raise ValueError('outcomes belong to other signals')
    s = plan['settings']
    seed = int(plan['plan_id'][:8], 16)
    result = {'plan_id': plan['plan_id'], 'signals_id': signals['signals_id'], 'outcomes_id': flags['outcomes_id'],
              'brier': brier(plan, signals, flags['outcomes'], repeats, seed), 'policies': {}}
    for g, by_loss in decisions(plan, signals).items():
        rows = signals['rows'][g]
        y = np.array([flags['outcomes'][g][r['item_id']] or 0 for r in rows])
        result['policies'][g] = {}
        for loss, v in by_loss.items():
            util = {k: np.array([(s['correct_reward'] if yv == 1 else -float(loss)) if d['action'] == 'assert' else 0.
                                 for d, yv in zip(ds, y)]) - s['utility_per_usd']*np.array([d['expected_cost_usd'] for d in ds])
                    for k, ds in v['arms'].items()}
            arms = {}
            for k, ds in v['arms'].items():
                rel = np.array([d['action'] == 'assert' for d in ds])
                arms[k] = {**cost_summary(ds), 'wrong_released': int((rel & (y == 0)).sum()),
                           'mean_utility': float(util[k].mean())}
            result['policies'][g][loss] = {
                'triple': v['triple'], 'arms': arms,
                'adaptive_minus_prior': boot(util['adaptive'] - util['prior_only'], repeats, seed),
                'fixed_triple_minus_adaptive': boot(util['fixed_triple'] - util['adaptive'], repeats, seed),
                'fixed_five_minus_adaptive': boot(util['fixed_five'] - util['adaptive'], repeats, seed)}
    result = seal(result, 'result_id')
    atomic_json(Path(output), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('decisions', help='label-free calls, cost and outputs per arm')
    d.add_argument('--plan', required=True)
    d.add_argument('--signals', required=True)
    sc = sub.add_parser('score')
    sc.add_argument('--plan', required=True)
    sc.add_argument('--signals', required=True)
    sc.add_argument('--outcomes', required=True)
    sc.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'decisions':
        print(json.dumps(preview(*load(args.plan, args.signals)), indent=1))
    else:
        r = score(args.plan, args.signals, args.outcomes, args.output)
        print(json.dumps({'result_id': r['result_id']}))
