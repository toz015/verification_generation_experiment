"""Secondary QuantumKatas analyses mirroring the GPQA reports (added after the primary result).

Same rows, folds, likelihoods, planner and raw prior as crossfit.py; nothing is
recalibrated. For every held-out fold the fit uses the other four folds only.

  stopping       adaptive single / adaptive triple vs asking the same triple, or
                 all five verifiers, for every task (GPQA diamond_cost arms)
  verifiers      why these verifiers: per-verifier forecast quality, fitted
                 likelihood ratios, cost, single-verifier and 60-order J0
                 rankings per fold, selection frequency, score correlations
  loss_sweep     sensitivity of all arms to the incorrect-answer loss, the same
                 grid for both generators (not tuned per generator)
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.dependence_triples import execute
from vgx.gpqa.diamond_score import boot
from vgx.gpqa.raw_prior_analysis import baseline_decision, summarize_policy
from vgx.gpqa.report import rate_interval
from vgx.gpqa.score import binary_forecast_metrics
from vgx.quantum_katas import crossfit

SWEEP = (4.0, 9.0, 19.0, 49.0, 99.0)


def _summary(rows, ds, settings, loss, seed=20261019):
    s = summarize_policy(rows, ds, settings, loss)
    u = np.array(s.pop('utility_per_item'))
    s['output_error_rate'] = s['wrong_released']/s['released'] if s['released'] else None
    s['output_error_wilson95'] = rate_interval([1-r['outcome'] for r, d in zip(rows, ds) if d['action'] == 'assert'])
    s['calls_per_task'] = s['queries']/len(rows)
    s['expected_usd_per_task'] = s['expected_cost_usd']/len(rows)
    s['measured_usd_per_task'] = s['usage_estimated_cost_usd']/len(rows)
    s['query_histogram'] = dict(sorted(Counter(str(d['verifiers_used']) for d in ds).items()))
    s['mean_utility_ci95'] = boot(u, settings['bootstrap_repeats'], seed)['ci95']
    return s, u


def arms_for(rows, cfg, loss, transform=None):
    """Cross-fitted decisions for every arm at one loss, plus the per-fold fits.

    transform(k, rows) may return per-fold copies (e.g. a prior map fitted on the
    training folds of k); it must not use fold-k labels."""
    settings, verifiers = cfg['settings'], cfg['verifiers']
    edges = tuple(np.linspace(0., 1., settings['probability_bins']+1).tolist())
    names = ('prior_only', 'adaptive_single', 'adaptive_triple', 'fixed_triple', 'fixed_five')
    arms = {a: [None]*len(rows) for a in names}
    folds = {}
    original = rows
    for k in sorted({r['fold'] for r in original}):
        rows = transform(k, original) if transform else original
        train = [r for r in rows if r['fold'] != k]
        fitted = crossfit.fit_fold(train, verifiers, settings)
        ranked = crossfit.select_order(train, fitted, verifiers, loss, settings)
        triple = ranked[0]['order']
        folds[str(k)] = {'fitted': fitted, 'ranked': ranked, 'triple': triple}
        plans = {'adaptive_single': ([cfg['single_arm_verifier']], False), 'adaptive_triple': (triple, False),
                 'fixed_triple': (triple, True), 'fixed_five': (list(verifiers), True)}
        for name, (order, force) in plans.items():
            planner, costs = crossfit.planner_for(fitted, order, loss, settings)
            for i, r in enumerate(rows):
                if r['fold'] == k:
                    prior = r['prior'] if r['valid'] else 0.0
                    arms[name][i] = (crossfit.decide(r, planner, order, costs, edges) if not force else
                                     execute(prior, r['valid'], planner, order, costs,
                                             {v: r['signals'][v] for v in order}, edges, force_all=True))
        folds[str(k)]['priors'] = {i: r['prior'] for i, r in enumerate(rows) if r['fold'] == k and r['valid']}
        for i, r in enumerate(rows):
            if r['fold'] == k:
                arms['prior_only'][i] = baseline_decision(r['prior'] if r['valid'] else 0.0, r['valid'], loss,
                                                          settings['correct_reward'])
    return arms, folds


def stopping(rows, cfg, loss):
    settings = cfg['settings']
    arms, folds = arms_for(rows, cfg, loss)
    summaries, util = {}, {}
    for name, ds in arms.items():
        summaries[name], util[name] = _summary(rows, ds, settings, loss)
    reps = settings['bootstrap_repeats']
    paired = {f'{a}_minus_{b}': boot(util[a]-util[b], reps, 20261019) for a, b in (
        ('adaptive_triple', 'fixed_triple'), ('adaptive_triple', 'fixed_five'), ('fixed_five', 'fixed_triple'),
        ('adaptive_triple', 'prior_only'), ('adaptive_single', 'prior_only'), ('adaptive_triple', 'adaptive_single'))}
    queried = [d for d in arms['adaptive_triple'] if d['verifiers_used'] > 0]
    early = {'tasks_with_any_query': len(queried),
             'stopped_before_third': sum(d['verifiers_used'] < 3 for d in queried),
             'no_query': sum(d['verifiers_used'] == 0 for d in arms['adaptive_triple']),
             'calls_saved_vs_fixed_triple': summaries['fixed_triple']['queries'] - summaries['adaptive_triple']['queries'],
             'calls_saved_vs_fixed_five': summaries['fixed_five']['queries'] - summaries['adaptive_triple']['queries']}
    return {'arms': summaries, 'paired_utility': paired, 'early_stopping': early,
            'selected_triples': {k: f['triple'] for k, f in folds.items()}}, folds


def verifier_section(rows, cfg, folds_by_loss):
    verifiers = cfg['verifiers']
    valid = [r for r in rows if r['valid']]
    per = {}
    for v in verifiers:
        scored = [r for r in valid if r['signals'][v]['score'] is not None]
        m = binary_forecast_metrics([r['outcome'] for r in scored], [r['signals'][v]['score'] for r in scored]).to_dict()
        per[v] = {'valid_signal_rate': len(scored)/len(valid) if valid else None, 'n': len(scored),
                  'auroc': m['auroc'], 'brier': m['brier'], 'log_loss': m['log_loss'],
                  'mean_score_if_pass': float(np.mean([r['signals'][v]['score'] for r in scored if r['outcome'] == 1])),
                  'mean_score_if_fail': float(np.mean([r['signals'][v]['score'] for r in scored if r['outcome'] == 0])),
                  'mean_usd_per_call': float(np.mean([r['signals'][v]['estimated_usd'] for r in valid]))}
    by_loss = {}
    for loss, folds in folds_by_loss.items():
        lrs, singles, freq, ranks = {v: [] for v in verifiers}, {v: [] for v in verifiers}, Counter(), Counter()
        for k, f in folds.items():
            for v in verifiers:
                l = f['fitted']['likelihoods'][v]
                lrs[v].append([a/b for a, b in zip(l['p_bin_if_correct'], l['p_bin_if_incorrect'])])
            freq['>'.join(f['triple'])] += 1
            for v in f['triple']:
                ranks[v] += 1
            train = [r for r in rows if r['fold'] != int(k) and r['valid']]
            priors = np.array([r['prior'] for r in train])
            for v in verifiers:
                planner, _ = crossfit.planner_for(f['fitted'], [v], float(loss), cfg['settings'])
                singles[v].append(float(planner.j(0, priors).mean()))
            baseline = float(np.mean(np.maximum(0., (1+float(loss))*priors - float(loss))))
            for v in verifiers:
                singles[v][-1] -= baseline
        by_loss[loss] = {
            'fitted_likelihood_ratio_by_bin_mean_over_folds': {v: np.mean(lrs[v], axis=0).tolist() for v in verifiers},
            'single_verifier_training_J0_gain_over_prior_mean': {v: float(np.mean(singles[v])) for v in verifiers},
            'selected_triple_frequency': dict(freq), 'verifier_in_selected_triple_count': dict(ranks),
            'top3_orders_fold0': folds['0']['ranked'][:3]}
    corr = {}
    for y in (0, 1):
        sub = [r for r in valid if r['outcome'] == y and all(r['signals'][v]['score'] is not None for v in verifiers)]
        mat = [[None]*len(verifiers) for _ in verifiers]
        if len(sub) > 2:
            for i, a in enumerate(verifiers):
                for j, b in enumerate(verifiers):
                    x, z = [r['signals'][a]['score'] for r in sub], [r['signals'][b]['score'] for r in sub]
                    rho = spearmanr(x, z).statistic if len(set(x)) > 1 and len(set(z)) > 1 else float('nan')
                    mat[i][j] = None if np.isnan(rho) else float(rho)
        corr['pass' if y else 'fail'] = {'n': len(sub), 'order': verifiers, 'spearman': mat}
    return {'per_verifier': per, 'by_loss': by_loss, 'within_class_score_correlation': corr}


def analyze(config_path, output, sweep=SWEEP):
    cfg = json.loads(Path(config_path).read_text())
    output = Path(output)
    if output.exists():
        raise SystemExit('output exists; refusing to overwrite')
    split = {'value': json.loads(Path(cfg['split']).read_text()), 'sha256': file_digest(cfg['split'])}
    result = {'schema': 1, 'kind': 'secondary_analysis_added_after_primary_result', 'config_sha256':
              file_digest(config_path), 'loss_sweep': list(sweep), 'generators': {}}
    for name, root in cfg['runs'].items():
        rows = crossfit.load_rows(root, split, cfg['verifiers'])
        stop, folds_by_loss, sweep_out = {}, {}, {}
        for loss in sorted(set(sweep) | set(cfg['settings']['losses'])):
            key = f'{loss:g}'
            st, folds = stopping(rows, cfg, loss)
            sweep_out[key] = {a: {k: st['arms'][a][k] for k in (
                'released', 'wrong_released', 'output_error_rate', 'coverage', 'calls_per_task', 'mean_utility',
                'mean_utility_ci95')} for a in st['arms']}
            if loss in cfg['settings']['losses']:
                stop[key], folds_by_loss[key] = st, folds
        valid = [r for r in rows if r['valid']]
        result['generators'][name] = {
            'stopping': stop, 'verifiers': verifier_section(rows, cfg, folds_by_loss), 'loss_sweep': sweep_out,
            'prior_distribution': {'n': len(valid), 'share_ge_0.80': float(np.mean([r['prior'] >= .8 for r in valid])),
                                   'share_ge_0.95': float(np.mean([r['prior'] >= .95 for r in valid])),
                                   'share_ge_0.99': float(np.mean([r['prior'] >= .99 for r in valid])),
                                   'pass_rate': float(np.mean([r['outcome'] for r in valid]))}}
    output.mkdir(parents=True)
    atomic_json(output/'analysis.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.config, args.output)
    print(f'wrote {args.output}/analysis.json')


if __name__ == '__main__':
    main()
