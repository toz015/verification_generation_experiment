"""Score the Diamond test exactly as the sealed development plan declares.

`outcomes` is the only step that opens Diamond answers; it runs locally after
the plan is sealed and the scores are collected, and writes nothing but one
correct/incorrect flag per (generator, question). `score` then evaluates the
pre-declared comparisons with the plan's frozen fits; nothing is refitted on
Diamond data.

Pre-declared (see the plan's Markdown):
  P1  Brier: Flash (asked always) vs generator confidence alone, raw confidence.
  P2  Brier: the declared triple (correlation-aware, asked always) vs Flash,
      raw and monotone-calibrated confidence.
  S1  Utility at L = 19 (and 99) of the adaptive policies with the frozen orders:
      abstain, confidence alone, Flash, selected triple (both models).
  Exploratory: Brier of every verifier subset under the independence model.
Brier uses questions with a valid answer and all five scores; policies use all.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vgx.common.storage import atomic_json, digest, file_digest, read_jsonl
from vgx.gpqa.dependence_triples import INF, ExactPlanner, JointLikelihood, bin_of, execute
from vgx.gpqa.raw_prior_analysis import baseline_decision, candidate_valid
from vgx.gpqa.sequential import seal, unseal

FLASH = 'gemini_flash'


def outcomes(dataset_root, signals_path, output):
    """Run only after the plan is sealed and approved: the one step that opens Diamond answers."""
    root = Path(dataset_root)
    manifest = json.loads((root/'dataset_manifest.json').read_text())
    unseal(manifest, 'dataset_id')
    if file_digest(root/'labels.jsonl') != manifest['files']['labels.jsonl']:
        raise ValueError('labels changed')
    signals = json.loads(Path(signals_path).read_text())
    unseal(signals, 'signals_id')
    if signals['provenance']['dataset_id'] != manifest['dataset_id']:
        raise ValueError('signals belong to another dataset')
    key = {r['item_id']: r['correct_index'] for r in read_jsonl(root/'labels.jsonl') if r['partition'] == 'evaluation'}
    flags = {g: {r['item_id']: (int('ABCD'.index(r['answer']) == key[r['item_id']]) if r['answer'] else None)
                 for r in rows} for g, rows in signals['rows'].items()}
    value = seal({'signals_id': signals['signals_id'], 'outcomes': flags}, 'outcomes_id')
    atomic_json(Path(output), value)
    return value


def tag(v):
    return v + ':probability'


def marginals(fit, verifiers):
    return [{1: fit['likelihoods'][tag(v)]['p_bin_if_correct'], 0: fit['likelihoods'][tag(v)]['p_bin_if_incorrect']}
            for v in verifiers]


def joint_for(fit, order, model):
    """Frozen joint model for an ordered subset; tables are stored in sorted-verifier axis order."""
    names = sorted(order)
    if model == 'independence' or len(order) == 1:
        return JointLikelihood(marginals(fit, order), None, INF)
    entry = fit['joint_models']['+'.join(names)]
    alpha = INF if entry['alpha'] == 'inf' else float(entry['alpha'])
    perm = [names.index(v) for v in order]
    counts = {int(y): np.transpose(np.array(c), perm) for y, c in entry['counts'].items()}
    return JointLikelihood(marginals(fit, order), counts, alpha)


def prior_map(fit, prior):
    if prior == 'raw':
        return lambda p: p
    c = fit['prior_calibration'][prior]
    if prior == 'monotone_binned':
        return lambda p: float(np.interp(p, c['interpolate_x'], c['interpolate_y']))
    return lambda p: 1/(1+math.exp(-(c['slope']*math.log(min(max(p, c['eps']), 1-c['eps'])
                                                      / (1-min(max(p, c['eps']), 1-c['eps']))) + c['intercept'])))


def edges_of(plan):
    return tuple(np.linspace(0., 1., plan['settings']['probability_bins']+1).tolist())


def posterior(prior, lr):
    return prior*lr/(prior*lr + 1-prior)


def boot(diff, repeats, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    draws = d[rng.integers(0, len(d), (repeats, len(d)))].mean(axis=1)
    return {'mean': float(d.mean()), 'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], 'n': len(d)}


def brier_block(rows, y, fit, plan, prior, verifiers_sets, model, repeats, seed):
    edges = edges_of(plan)
    m = prior_map(fit, prior)
    out = {'prior_only': np.array([m(r['prior']) for r in rows])}
    for name, vs in verifiers_sets.items():
        j = joint_for(fit, tuple(sorted(vs)), model)
        out[name] = np.array([posterior(m(r['prior']), j.likelihood_ratio(
            tuple(bin_of(r['signals'][tag(v)]['score'], edges) for v in sorted(vs)))) for r in rows])
    sq = {k: (v - y)**2 for k, v in out.items()}
    return {'brier': {k: float(v.mean()) for k, v in sq.items()},
            'auroc': {k: float(roc_auc_score(y, v)) if len(set(y)) == 2 else None for k, v in out.items()},
            'mean_posterior_minus_accuracy': {k: float(v.mean() - y.mean()) for k, v in out.items()}}, sq


def score(plan_path, signals_path, outcomes_path, output, repeats=10000):
    plan = json.loads(Path(plan_path).read_text())
    unseal(plan, 'plan_id')
    signals = json.loads(Path(signals_path).read_text())
    unseal(signals, 'signals_id')
    flags = json.loads(Path(outcomes_path).read_text())
    unseal(flags, 'outcomes_id')
    if flags['signals_id'] != signals['signals_id']:
        raise ValueError('outcomes belong to other signals')
    s = plan['settings']
    edges = edges_of(plan)
    seed = int(plan['plan_id'][:8], 16)
    result = {'plan_id': plan['plan_id'], 'signals_id': signals['signals_id'], 'outcomes_id': flags['outcomes_id'],
              'generators': {}}
    for g, fit in plan['generators'].items():
        rows = [dict(r, outcome=flags['outcomes'][g][r['item_id']]) for r in signals['rows'][g]]
        valid = [r for r in rows if candidate_valid(r)]
        verifiers = sorted(v.split(':')[0] for v in fit['likelihoods'])
        common = [r for r in valid if all(r['signals'][tag(v)]['score'] is not None for v in verifiers)]
        y = np.array([r['outcome'] for r in common])
        res = {'n': len(rows), 'valid_n': len(valid), 'brier_n': len(common), 'accuracy_valid': float(np.mean([r['outcome'] for r in valid])),
               'primary': {}, 'policies': {}, 'exploratory_brier_independence': {}}
        for prior in ('raw', 'monotone_binned', 'platt'):
            declared = fit['declared_brier_triple'][prior]['verifiers']
            block, sq = brier_block(common, y, fit, plan, prior, {'flash': [FLASH], 'declared_triple': declared,
                                                                  'all_five': verifiers}, 'dependence', repeats, seed)
            block['declared_triple'] = declared
            block['P1_flash_minus_prior'] = boot(sq['flash'] - sq['prior_only'], repeats, seed)
            block['P2_triple_minus_flash'] = boot(sq['declared_triple'] - sq['flash'], repeats, seed)
            block['five_minus_flash'] = boot(sq['all_five'] - sq['flash'], repeats, seed)
            res['primary'][prior] = block
            subsets = {'+'.join(c): list(c) for k in range(1, 6) for c in combinations(verifiers, k)}
            explore, _ = brier_block(common, y, fit, plan, prior, subsets, 'independence', repeats, seed)
            res['exploratory_brier_independence'][prior] = explore['brier']
            m = prior_map(fit, prior)
            for loss in s['losses']:
                key = f'{prior}/{float(loss):g}'
                pol = fit['adaptive_policies'][key]
                costs = {v: fit['expected_cost_usd'][tag(v)] for v in verifiers}

                def run(order, model):
                    j = joint_for(fit, tuple(order), model)
                    p = ExactPlanner(j, [s['utility_per_usd']*costs[v] for v in order], s['correct_reward'], float(loss))
                    return [execute(m(r['prior']), True, p, tuple(tag(v) for v in order), [costs[v] for v in order],
                                    r['signals'], edges) if candidate_valid(r)
                            else baseline_decision(r['prior'], False, float(loss), s['correct_reward']) for r in rows]
                decisions = {
                    'abstain_all': [baseline_decision(None, False, float(loss), s['correct_reward']) for _ in rows],
                    'prior_only': [baseline_decision(m(r['prior']) if candidate_valid(r) else r['prior'], candidate_valid(r),
                                                     float(loss), s['correct_reward']) for r in rows],
                    'flash': run([FLASH], 'independence'),
                    'triple_dependence': run(pol['dependence']['selected_order'], 'dependence'),
                    'triple_independence': run(pol['independence']['selected_order'], 'independence')}
                yy = np.array([r['outcome'] if r['outcome'] is not None else 0 for r in rows])
                util = {k: np.array([(s['correct_reward'] if yv == 1 else -float(loss)) if d['action'] == 'assert' else 0.
                                     for d, yv in zip(ds, yy)]) - s['utility_per_usd']*np.array([d['expected_cost_usd'] for d in ds])
                        for k, ds in decisions.items()}
                summary = {}
                for k, ds in decisions.items():
                    rel = np.array([d['action'] == 'assert' for d in ds])
                    summary[k] = {'released': int(rel.sum()), 'wrong_released': int((rel & (yy == 0)).sum()),
                                  'released_accuracy': float(yy[rel].mean()) if rel.any() else None,
                                  'mean_utility': float(util[k].mean()),
                                  'queries': int(sum(d['verifiers_used'] for d in ds)),
                                  'vs_flash': boot(util[k] - util['flash'], repeats, seed) if k != 'flash' else None}
                res['policies'][key] = {'orders': {'dependence': pol['dependence']['selected_order'],
                                                   'independence': pol['independence']['selected_order']},
                                        'summary': summary}
        result['generators'][g] = res
    result = seal(result, 'result_id')
    atomic_json(Path(output), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    o = sub.add_parser('outcomes', help='open Diamond answers (only after approval)')
    o.add_argument('--dataset-root', required=True)
    o.add_argument('--signals', required=True)
    o.add_argument('--output', required=True)
    sc = sub.add_parser('score')
    sc.add_argument('--plan', required=True)
    sc.add_argument('--signals', required=True)
    sc.add_argument('--outcomes', required=True)
    sc.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'outcomes':
        v = outcomes(args.dataset_root, args.signals, args.output)
        print(json.dumps({'outcomes_id': v['outcomes_id']}))
    else:
        r = score(args.plan, args.signals, args.outcomes, args.output)
        print(json.dumps({'result_id': r['result_id']}))
