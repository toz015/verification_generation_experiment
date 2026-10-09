"""Phase I: exploratory risk-coverage analysis on cached development data (offline).

Methods A (generator confidence), B (always-query Flash) and C (frozen sequential
Flash policy plus an acceptance gate) are scored out of fold: likelihoods and costs
come from training folds only, and every held-out score is built without held-out
labels. Thresholds read from these curves are chosen on the same data and are NOT
certified. Method C evaluates a fixed query policy with an added gate; it does not
establish optimal stopping for the risk objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.risk_control import (
    METHODS, accepts, binomial_upper_bound, export_observations, fit_pooled, generator_rows, likelihood_from,
    replay_check, rule_counts, score_method, sequential_planner, table_dict,
)
from vgx.gpqa.sequential import seal

IMPLEMENTATION = ('risk_control.py', 'risk_exploration.py', 'planner.py', 'raw_prior_analysis.py', 'score.py')


def _close(a, b, tol=1e-12):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=tol))


def score_run(export, generator, cohort, seed, cfg):
    """Out-of-fold scores for A/B/C on one frozen split, with checks against the frozen source fits."""
    settings, loss = cfg['frozen_settings'], cfg['sequential_base_loss']
    rows = generator_rows(export, generator)
    ids = export['generators'][generator]['cohorts'][cohort]['question_ids']
    split = export['generators'][generator]['cohorts'][cohort]['seeds'][str(seed)]
    assignment = dict(zip(ids, split['fold_assignment']))
    position = {q: i for i, q in enumerate(ids)}
    frozen = split['frozen_decisions'][str(loss)]
    scored = {m: {} for m in METHODS}
    checks = {'pooled_refit_matches': 0, 'always_query_matches': 0, 'sequential_matches': 0, 'replay_matches': 0}
    folds = []
    for k, fold in enumerate(split['folds']):
        train = [rows[q] for q in ids if assignment[q] != k]
        held = [q for q in ids if assignment[q] == k]
        refit, cost = fit_pooled(train, settings['probability_bins'], settings['laplace'])
        likelihood = likelihood_from(fold['pooled_likelihood'])
        if not (np.allclose(refit.p_bin_if_correct, likelihood.p_bin_if_correct, rtol=0, atol=1e-12)
                and np.allclose(refit.p_bin_if_incorrect, likelihood.p_bin_if_incorrect, rtol=0, atol=1e-12)
                and math.isclose(cost, fold['expected_cost_usd'], rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError('training-fold refit differs from the frozen pooled fit')
        checks['pooled_refit_matches'] += 1
        cost = fold['expected_cost_usd']
        planner = sequential_planner(likelihood, cost, settings, loss)
        folds.append({'fold': k, 'n_train': len(train), 'n_held_out': len(held), 'pooled_likelihood': table_dict(likelihood),
                      'expected_cost_usd': cost, 'fit_correct': fold['fit_correct'], 'fit_wrong': fold['fit_wrong']})
        for q in held:
            row = rows[q]
            a = score_method('A', row)
            b = score_method('B', row, likelihood=likelihood, expected_cost_usd=cost)
            c = score_method('C', row, expected_cost_usd=cost, planner=planner)
            fa, fs = frozen['always_query'][position[q]], frozen['sequential'][position[q]]
            if not (b['queries'] == fa['verifiers_used'] and _close(b['usage_cost_usd'], fa['usage_estimated_cost_usd'])
                    and _close(b['expected_cost_usd'], fa['expected_cost_usd'])
                    and (not b['eligible'] or _close(b['score'], fa['posterior']))):
                raise AssertionError('always-query scores do not reproduce the frozen decisions')
            checks['always_query_matches'] += 1
            if not (c['base_action'] == fs['action'] and c['queries'] == fs['verifiers_used']
                    and _close(c['terminal_belief'], fs['posterior'])
                    and _close(c['usage_cost_usd'], fs['usage_estimated_cost_usd'])
                    and _close(c['expected_cost_usd'], fs['expected_cost_usd'])):
                raise AssertionError('sequential scores do not reproduce the frozen decisions')
            checks['sequential_matches'] += 1
            if not replay_check(row, planner, c):
                raise AssertionError('sequential executor / replay mismatch')
            checks['replay_matches'] += 1
            for method, value in zip(METHODS, (a, b, c)):
                scored[method][q] = {**value, 'fold': k}
    return {m: [scored[m][q] for q in ids] for m in METHODS}, folds, checks


def curve(scored, correct, settings):
    """Every distinct eligible score is a threshold, strictest first; plus the zero-coverage end."""
    thresholds = sorted({s['score'] for s in scored if s['eligible']}, reverse=True)
    points = []
    for t in thresholds:
        c = rule_counts(scored, correct, t)
        utility = ((c['m'] - c['k'])*settings['correct_reward'] - c['k']*settings['primary_incorrect_loss']
                   - settings['utility_per_usd']*c['expected_cost_usd'])/c['N']
        accepted = [s['score'] for s in scored if s['eligible'] and s['score'] >= t]
        points.append({'threshold': t, **c, 'mean_score_accepted': float(np.mean(accepted)),
                       'secondary_mean_utility_L19': utility,
                       'upper_bound_single_rule_95': binomial_upper_bound(c['k'], c['m'], .05)})
    return points


def low_risk_region(points, alpha):
    """Exploratory, in-sample: the largest coverage whose empirical k/m <= alpha (optimistic by construction)."""
    passing = [p for p in points if p['m'] > 0 and p['k'] <= alpha*p['m']]
    if not passing:
        return None
    best = max(passing, key=lambda p: (p['m'], p['threshold']))
    return {k: best[k] for k in ('threshold', 'm', 'k', 'N', 'acceptance_rate', 'conditional_error',
                                 'upper_bound_single_rule_95')}


def matched(points, target):
    """Strictest threshold whose coverage reaches the target; actual coverage is reported (ties prevent exact matching)."""
    reach = [p for p in points if p['acceptance_rate'] >= target]
    if not reach:
        return None
    p = max(reach, key=lambda p: p['threshold'])
    return {k: p[k] for k in ('threshold', 'm', 'k', 'acceptance_rate', 'conditional_error')}


def ranking(scored, correct, repeats, seed):
    """AUROC of A vs B on matched valid candidates with a parsed Flash score; paired item bootstrap."""
    keep = [i for i, s in enumerate(scored['B']) if s['eligible']]
    y = np.array([correct[scored['A'][i]['question_id']] for i in keep])
    a = np.array([scored['A'][i]['score'] for i in keep])
    b = np.array([scored['B'][i]['score'] for i in keep])
    base = {'n': len(keep), 'n_incorrect': int((y == 0).sum()), 'auroc_A': float(roc_auc_score(y, a)),
            'auroc_B': float(roc_auc_score(y, b))}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repeats):
        ix = rng.integers(0, len(keep), len(keep))
        if len(set(y[ix])) < 2:
            continue
        diffs.append(roc_auc_score(y[ix], b[ix]) - roc_auc_score(y[ix], a[ix]))
    base['auroc_B_minus_A'] = base['auroc_B'] - base['auroc_A']
    base['ci95'] = [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))]
    return base


def run(cfg, export, output):
    output = Path(output)
    settings, exploration = cfg['frozen_settings'], cfg['exploration']
    alphas = [cfg['alpha']['primary'], cfg['alpha']['secondary']]
    summary, items, curves, params, regions, matches, ranks, checks_total = [], [], [], [], [], [], [], {}
    family_rules, overlap = [], []
    grid = cfg['certification']['family']['thresholds']
    for generator in cfg['generators']:
        correct = {i['question_id']: i['correct'] for i in export['generators'][generator]['items']}
        for cohort in cfg['cohorts']:
            for seed in cfg['seeds']:
                scored, folds, checks = score_run(export, generator, cohort, seed, cfg)
                for key, value in checks.items():
                    checks_total[key] = checks_total.get(key, 0) + value
                key = {'generator': generator, 'cohort': cohort, 'seed': seed}
                for fold in folds:
                    params.append({**key, **fold})
                r = ranking(scored, correct, exploration['bootstrap_repeats'], exploration['bootstrap_seed'])
                ranks.append({**key, **r})
                for method in METHODS:
                    points = curve(scored[method], correct, settings)
                    for p in points:
                        curves.append({**key, 'method': method, **p})
                    base = rule_counts(scored[method], correct, -math.inf)
                    summary.append({**key, 'method': method, 'N': base['N'],
                                    'valid_candidates': sum(1 for s in scored[method] if s['failure'] != 'invalid_generator'),
                                    'eligible': base['m'], 'eligible_wrong': base['k'],
                                    'invalid_generator': sum(1 for s in scored[method] if s['failure'] == 'invalid_generator'),
                                    'signal_failures': sum(1 for s in scored[method]
                                                           if s['failure'] not in (None, 'invalid_generator')),
                                    'queries': base['queries'], 'expected_cost_usd': base['expected_cost_usd'],
                                    'usage_cost_usd': base['usage_cost_usd']})
                    for alpha in alphas:
                        regions.append({**key, 'method': method, 'alpha': alpha, 'region': low_risk_region(points, alpha)})
                    for target in exploration['coverage_targets']:
                        matches.append({**key, 'method': method, 'coverage_target': target, 'point': matched(points, target)})
                    for t in grid[method]:
                        c = rule_counts(scored[method], correct, t)
                        family_rules.append({**key, 'method': method, 'threshold': t, **c,
                                             'upper_bound_single_rule_95': binomial_upper_bound(c['k'], c['m'], .05)})
                    for s in scored[method]:
                        items.append({**key, 'method': method, 'question_id': s['question_id'], 'fold': s['fold'],
                                      'correct': correct[s['question_id']], 'eligible': s['eligible'], 'score': s['score'],
                                      'queries': s['queries'], 'expected_cost_usd': s['expected_cost_usd'],
                                      'usage_cost_usd': s['usage_cost_usd'], 'failure': s['failure'],
                                      'base_action': s.get('base_action'), 'terminal_belief': s.get('terminal_belief')})
                for t in sorted(set(grid['B']) | set(grid['C'])):
                    accepted = {m: {s['question_id'] for s in scored[m] if accepts(s, t)} for m in ('B', 'C')}
                    overlap.append({**key, 'threshold': t, 'accepted_B': len(accepted['B']), 'accepted_C': len(accepted['C']),
                                    'accepted_both': len(accepted['B'] & accepted['C']),
                                    'identical_sets': accepted['B'] == accepted['C']})
    result = {'checks': checks_total, 'summary': summary, 'ranking': ranks, 'regions': regions, 'matched': matches,
              'fold_parameters': params, 'family_rules': family_rules, 'b_c_overlap': overlap}
    atomic_json(output/'exploration.json', result)
    _write_csv(output/'risk_coverage_curves.csv', curves)
    _write_csv(output/'item_scores.csv', items)
    _write_csv(output/'fold_parameters.csv', [{**{k: v for k, v in p.items() if k != 'pooled_likelihood'},
        **{f'p_bin{b}_given_correct': x for b, x in enumerate(p['pooled_likelihood']['p_bin_if_correct'])},
        **{f'p_bin{b}_given_incorrect': x for b, x in enumerate(p['pooled_likelihood']['p_bin_if_incorrect'])}}
        for p in params])
    _write_csv(output/'method_summary.csv', summary)
    _write_csv(output/'ranking_auroc.csv', [{**{k: v for k, v in r.items() if k != 'ci95'},
                                             'ci95_low': r['ci95'][0], 'ci95_high': r['ci95'][1]} for r in ranks])
    _write_csv(output/'low_risk_regions_exploratory.csv', [{**{k: v for k, v in r.items() if k != 'region'},
        **(r['region'] or {'threshold': None, 'm': 0})} for r in regions])
    _write_csv(output/'family_rules_exploratory.csv', family_rules)
    _write_csv(output/'b_c_acceptance_overlap.csv', overlap)
    _write_csv(output/'matched_coverage.csv', [{**{k: v for k, v in r.items() if k != 'point'},
        **(r['point'] or {'threshold': None})} for r in matches])
    return result


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protocol_for(cfg, config_path, export_path, extra=None):
    here = Path(__file__)
    return seal({'config': cfg, 'config_sha256': file_digest(config_path), 'observations_sha256': file_digest(export_path),
                 'implementation_sha256': {n: file_digest(here.with_name(n)) for n in IMPLEMENTATION},
                 'new_api_requests': 0, 'diamond_answer_keys_read': False, **(extra or {})}, 'protocol_id')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export', help='build the observation export from a frozen project or review package')
    e.add_argument('--config', default='configs/gpqa_risk_acceptance.json')
    e.add_argument('--root', required=True)
    e.add_argument('--output', required=True)
    r = sub.add_parser('run', help='exploratory risk-coverage analysis from an observation export')
    r.add_argument('--config', default='configs/gpqa_risk_acceptance.json')
    r.add_argument('--observations', required=True)
    r.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == 'export':
        value = export_observations(args.root, cfg)
        if digest(export_observations(args.root, cfg)) != digest(value):  # re-read: inputs unchanged
            raise ValueError('frozen inputs changed during export')
        atomic_json(args.output, value)
        print(json.dumps({'observations_sha256': file_digest(args.output), **value['provenance']['package_manifest_check']}))
        return
    export_path = Path(args.observations)
    export = json.loads(export_path.read_text())
    protocol = protocol_for(cfg, args.config, export_path)
    out = Path(args.output)
    existing = out/'protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError('protocol changed; use a new output directory')
    atomic_json(existing, protocol)
    result = run(cfg, export, out)
    if file_digest(export_path) != protocol['observations_sha256']:
        raise ValueError('observation export changed during the run')
    print(json.dumps({'protocol_id': protocol['protocol_id'], 'checks': result['checks']}))


if __name__ == '__main__':
    main()
