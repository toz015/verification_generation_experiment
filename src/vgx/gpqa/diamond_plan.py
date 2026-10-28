"""Freeze the Diamond test on the 249 development questions only.

Every quantity the Diamond test uses is fitted here, before any Diamond verifier
score or label is seen: per-verifier likelihoods and expected costs, the joint
model's alpha, the generator-confidence calibration maps, the triple declared
for the Brier comparison and the orders of the adaptive policies. The output is
sealed and the scoring step must use it unchanged. Development data only; no
API calls, no evaluation labels.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.dependence_triples import (INF, ExactPlanner, alpha_label, all_orders, cell_counts, choose_alpha,
                                         complete, fit_prior_calibration, load_bundle, order_id)
from vgx.gpqa.raw_prior_analysis import candidate_valid
from vgx.gpqa.score import fit_verifier_likelihood
from vgx.gpqa.sequential import seal

PRIORS = ('raw', 'monotone_binned', 'platt')


def calibration_record(fit):
    if fit['method'] == 'monotone_binned':
        xs = [v for b in fit['summary']['blocks'] for v in b['raw_values']]
        ys = [b['calibrated'] for b in fit['summary']['blocks'] for _ in b['raw_values']]
        return {'method': 'monotone_binned', 'interpolate_x': xs, 'interpolate_y': ys,
                'blocks': fit['summary']['blocks'], 'outside_range': 'constant at the end values (numpy.interp)'}
    return {'method': 'platt', **fit['summary'],
            'map': '1/(1+exp(-(slope*logit(clip(raw, eps, 1-eps)) + intercept)))'}


def fit_generator(rows, cfg, tags, declared):
    edges = tuple(np.linspace(0., 1., cfg['probability_bins']+1).tolist())
    valid = [r for r in rows if candidate_valid(r)]
    marginals, likelihoods, costs = {}, {}, {}
    for t in tags:
        usable = [r for r in valid if r['signals'][t]['score'] is not None]
        m = fit_verifier_likelihood([r['outcome'] for r in usable], [r['signals'][t]['score'] for r in usable],
                                    bins=cfg['probability_bins'], laplace=cfg['laplace'])
        marginals[t] = {1: m.p_bin_if_correct, 0: m.p_bin_if_incorrect}
        likelihoods[t] = {'p_bin_if_correct': list(m.p_bin_if_correct), 'p_bin_if_incorrect': list(m.p_bin_if_incorrect),
                          'fit_correct': sum(r['outcome'] for r in usable), 'fit_wrong': sum(1-r['outcome'] for r in usable)}
        costs[t] = float(np.mean([r['signals'][t]['estimated_usd'] for r in valid]))
    calibrations = {p: fit_prior_calibration(rows, p) for p in PRIORS[1:]}
    triples = [tuple(c) for c in combinations(sorted(tags), 3)]
    five = (tuple(sorted(tags)),)
    alphas = {c: choose_alpha(rows, c, [float(a) for a in cfg['alpha_grid']] + [INF], edges, cfg, cfg['inner_seed'])
              for c in triples + list(five)}
    joints = {'+'.join(t.split(':')[0] for t in c): {
        'verifiers': list(c), 'alpha': alpha_label(alphas[c][0]), 'inner_cv_loglik': alphas[c][1],
        'counts': {str(y): v.tolist() for y, v in cell_counts(complete(rows, c), c, edges).items()}}
        for c in triples + list(five)}

    def planner(order, alpha, loss):
        from vgx.gpqa.dependence_triples import fit_joint
        j = fit_joint(rows, order, [marginals[t] for t in order], alpha, edges)
        return ExactPlanner(j, [cfg['utility_per_usd']*costs[t] for t in order], cfg['correct_reward'], loss)

    policies = {}
    for prior in PRIORS:
        priors = np.array([r['prior'] if prior == 'raw' else calibrations[prior]['map'](r['prior']) for r in valid])
        for loss in cfg['losses']:
            flash = planner((cfg['reference_verifier'],), INF, float(loss))
            entry = {'flash_training_value': float(flash.j(0, priors).mean())}
            for model in ('independence', 'dependence'):
                ranked = sorted(((float(planner(o, INF if model == 'independence' else alphas[tuple(sorted(o))][0],
                                                float(loss)).j(0, priors).mean()), list(o)) for o in all_orders(tags)),
                                key=lambda v: (-v[0], v[1]))
                entry[model] = {'selected_order': [t.split(':')[0] for t in ranked[0][1]],
                                'training_value': ranked[0][0],
                                'next_four': [{'order': order_id(o), 'training_value': v} for v, o in ranked[1:5]]}
            policies[f'{prior}/{float(loss):g}'] = entry
    return {'n': len(rows), 'valid_n': len(valid), 'wrong_n': sum(1-r['outcome'] for r in valid),
            'likelihoods': likelihoods, 'expected_cost_usd': costs,
            'prior_calibration': {p: calibration_record(f) for p, f in calibrations.items()},
            'joint_models': joints, 'adaptive_policies': policies, 'declared_brier_triple': declared}


def freeze(config_path, output, declared):
    cfg = json.loads(Path(config_path).read_text())
    bundle = Path(cfg['bundle'])
    if file_digest(bundle) != cfg['bundle_sha256']:
        raise ValueError('review bundle changed')
    protocol, inputs, _, _ = load_bundle(bundle)
    tags = [v+':probability' for v in protocol['config']['verifiers']]
    cfg['utility_per_usd'] = protocol['config']['primary_utility_per_usd']
    plan = {'schema': 1, 'purpose': 'one confirmatory Diamond test of methods fitted on the 249 development questions',
            'source_bundle_sha256': cfg['bundle_sha256'], 'config_sha256': file_digest(Path(config_path)),
            'implementation_sha256': {n: file_digest(Path(__file__).with_name(n))
                                      for n in ('diamond_plan.py', 'dependence_triples.py', 'score.py')},
            'settings': {k: cfg[k] for k in ('probability_bins', 'laplace', 'correct_reward', 'losses', 'alpha_grid',
                                             'inner_folds', 'inner_seed', 'reference_verifier', 'utility_per_usd')},
            'generators': {g: fit_generator(inputs['rows'][g], cfg, tags, declared[g]) for g in ('qwen', 'gemini')},
            'development_labels_only': True, 'evaluation_labels_read': False, 'api_requests': 0}
    plan = seal(plan, 'plan_id')
    atomic_json(Path(output), plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpqa_dependence_triples.json')
    parser.add_argument('--declared', required=True, help='JSON: generator -> prior -> declared Brier triple')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    p = freeze(args.config, args.output, json.loads(Path(args.declared).read_text()))
    print(json.dumps({'plan_id': p['plan_id']}))
