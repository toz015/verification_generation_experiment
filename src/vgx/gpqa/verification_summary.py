"""Offline summary of GPQA sequential verification for frozen generator runs (no API calls).

Reads the sealed development and Diamond signal files of a verification study
(sonnet_verification / gptoss_verification layout) and the dataset answer key, then
reproduces the study's raw-prior policy and adds the same secondary views used for
QuantumKatas: an acceptance-threshold grid, always asking the selected 3 or all 5
verifiers, per-verifier diagnostics, and (appendix) a development-fitted
monotone prior calibration. Fitting uses development rows only; Diamond is held out.
The likelihood, planner and order selection reuse the QuantumKatas cross-fitting
helpers, which call the unchanged GPQA ExactPlanner / JointLikelihood code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, file_digest, read_jsonl
from vgx.gpqa.dependence_triples import execute, fit_prior_calibration
from vgx.gpqa.diamond_score import boot
from vgx.gpqa.raw_prior_analysis import baseline_decision, candidate_valid
from vgx.gpqa.score import binary_forecast_metrics
from vgx.gpqa.sequential import unseal
from vgx.quantum_katas import crossfit

VERIFIERS = ['gemini_flash', 'gemini_lite', 'mistral_small', 'mistral_medium', 'claude_haiku']
GRID = (1.0, 1.5, 7/3, 4.0, 17/3, 9.0, 14.0, 19.0, 97/3, 49.0, 99.0)
EDGES = (0.0, 1/3, 2/3, 1.0)


def load_rows(study):
    study = Path(study)
    protocol = json.loads((study/'protocol.json').read_text()); unseal(protocol, 'protocol_id')
    labels = {r['item_id']: r['correct_index'] for r in read_jsonl(Path(protocol['sources']['dataset_root'])/'labels.jsonl')}
    rows = []
    for part, fold in (('calibration', 1), ('evaluation', 0)):
        value = json.loads((study/part/'signals.json').read_text()); unseal(value, 'signals_id')
        for r in value['rows']:
            valid = candidate_valid(r)
            sig = {v: {'score': (r['signals'].get(v+':probability') or {}).get('score'),
                       'estimated_usd': (r['signals'].get(v+':probability') or {}).get('estimated_usd', 0.0) or 0.0}
                   for v in VERIFIERS}
            rows.append({'item_id': r['item_id'], 'fold': fold, 'partition': part, 'valid': valid,
                         'prior': r['prior'] if valid else None, 'has_code': valid, 'labeled': True,
                         'outcome': int(r['answer'] is not None and r['answer'] == 'ABCD'[labels[r['item_id']]]),
                         'signals': sig})
    return rows, protocol


def arms(rows, cfg, loss, prior_map=None):
    """Fit on development (fold 1) only; decide on Diamond (fold 0)."""
    settings = cfg['settings']
    train = [r for r in rows if r['fold'] == 1]
    test = [r for r in rows if r['fold'] == 0]
    if prior_map:
        fit = fit_prior_calibration([{'prior': r['prior'], 'outcome': r['outcome'], 'answer': 'A'}
                                     for r in train if r['valid']], 'monotone_binned')
        def mapped(rs):
            return [dict(r, prior=fit['map'](r['prior'])) if r['valid'] else r for r in rs]
        train, test = mapped(train), mapped(test)
    fitted = crossfit.fit_fold(train, VERIFIERS, settings)
    triple = crossfit.select_order(train, fitted, VERIFIERS, loss, settings)[0]['order']
    out = {'prior_only': [baseline_decision(r['prior'] if r['valid'] else 0., r['valid'], loss,
                                            settings['correct_reward']) for r in test]}
    for name, order, force in (('adaptive_triple', triple, False), ('fixed_triple', triple, True),
                               ('fixed_five', VERIFIERS, True)):
        planner, costs = crossfit.planner_for(fitted, order, loss, settings)
        out[name] = [crossfit.decide(r, planner, order, costs, EDGES) if not force else
                     execute(r['prior'] if r['valid'] else 0., r['valid'], planner, order, costs,
                             {v: r['signals'][v] for v in order}, EDGES, force_all=True) for r in test]
    return out, test, triple, fitted


def summarize(test, ds, loss, settings):
    y = np.array([r['outcome'] for r in test])
    rel = np.array([d['action'] == 'assert' for d in ds])
    cost = np.array([d['expected_cost_usd'] for d in ds])
    u = rel*np.where(y == 1, settings['correct_reward'], -loss) - settings['utility_per_usd']*cost
    wrong = int((rel & (y == 0)).sum())
    return {'accepted': int(rel.sum()), 'wrong': wrong, 'coverage': float(rel.mean()),
            'error_rate': wrong/int(rel.sum()) if rel.any() else None,
            'calls_per_task': sum(d['verifiers_used'] for d in ds)/len(ds),
            'measured_usd_per_task': float(np.mean([d.get('usage_estimated_cost_usd', 0.) for d in ds])),
            'mean_utility': float(u.mean()), 'mean_utility_ci95': boot(u, settings['bootstrap_repeats'], 20261020)['ci95'],
            'utility_per_item': u}


def analyze(study, cfg):
    rows, protocol = load_rows(study)
    settings = cfg['settings']
    result = {'study': str(study), 'protocol_id': protocol['protocol_id'], 'n': {
        'development': sum(r['fold'] == 1 for r in rows), 'diamond': sum(r['fold'] == 0 for r in rows),
        'development_valid': sum(r['fold'] == 1 and r['valid'] for r in rows),
        'diamond_valid': sum(r['fold'] == 0 and r['valid'] for r in rows)},
        'pass_rate': {p: float(np.mean([r['outcome'] for r in rows if r['fold'] == f])) for p, f in (('development', 1), ('diamond', 0))},
        'curves': {}, 'paired': {}}
    for kind, pm in (('raw', False), ('calibrated', True)):
        curve = []
        for loss in GRID:
            ds_by_arm, test, triple, fitted = arms(rows, cfg, loss, pm)
            summ = {a: summarize(test, ds, loss, settings) for a, ds in ds_by_arm.items()}
            if round(loss, 6) == 19.0:
                reps = settings['bootstrap_repeats']
                result['paired'][kind] = {f'{a}_minus_{b}': (lambda d: {'mean': d['mean'], 'ci95': d['ci95']})(
                    boot(summ[a]['utility_per_item']-summ[b]['utility_per_item'], reps, 20261020))
                    for a, b in (('adaptive_triple', 'prior_only'), ('adaptive_triple', 'fixed_triple'),
                                 ('adaptive_triple', 'fixed_five'))}
                if kind == 'raw':
                    result['selected_triple_L19'] = triple
                    result['fitted_L19'] = {v: {'lr': [a/b for a, b in zip(fitted['likelihoods'][v]['p_bin_if_correct'],
                                                                         fitted['likelihoods'][v]['p_bin_if_incorrect'])],
                                                'usd': fitted['expected_cost_usd'][v]} for v in VERIFIERS}
            for s in summ.values():
                s.pop('utility_per_item')
            curve.append({'t': loss/(1+loss), 'loss': loss, 'arms': summ})
        result['curves'][kind] = curve
    valid_test = [r for r in rows if r['fold'] == 0 and r['valid']]
    result['verifier_auroc_diamond'] = {}
    for v in VERIFIERS:
        pairs = [(r['outcome'], r['signals'][v]['score']) for r in valid_test if r['signals'][v]['score'] is not None]
        result['verifier_auroc_diamond'][v] = binary_forecast_metrics([a for a, _ in pairs], [b for _, b in pairs]).auroc
    devv = [r for r in rows if r['fold'] == 1 and r['valid']]
    result['prior_distribution'] = {'share_ge_0.95': float(np.mean([r['prior'] >= .95 for r in devv + valid_test])),
                                    'pass_rate_valid': float(np.mean([r['outcome'] for r in devv + valid_test]))}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', action='append', required=True, help='name=path to a verification study directory')
    p.add_argument('--config', type=Path, default=Path('configs/gpqa_gptoss_verification_20261020.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    out = {'schema': 1, 'kind': 'offline_summary_no_api', 'settings': cfg['settings'], 'config_sha256': file_digest(a.config),
           'studies': {}}
    for spec in a.study:
        name, path = spec.split('=', 1)
        out['studies'][name] = analyze(path, cfg)
    atomic_json(a.output, out)
    print('wrote', a.output)


if __name__ == '__main__':
    main()
