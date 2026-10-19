"""Secondary analysis (decided after the primary result): calibrated prior and risk-coverage.

1. Calibrated prior: for each held-out fold, the GPQA `fit_prior_calibration`
   (monotone_binned) maps raw r to a prior using training-fold labels only; the
   same cross-fitted likelihoods, order selection and planner then run as before.
2. Risk-coverage: one grid of losses (acceptance thresholds 0.50-0.99) for both
   generators, raw and calibrated prior, realized error of accepted answers vs
   the threshold's target 1/(1+L).
3. Matched coverage (GPQA matched_coverage): for a verifier arm that accepts k
   tasks in a fold, a baseline accepts the k highest raw-r tasks of the same fold
   (ties shared fractionally) without verifier cost; compare errors and utility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.dependence_triples import fit_prior_calibration
from vgx.gpqa.diamond_score import boot
from vgx.quantum_katas import crossfit, crossfit_extended

GRID = (1.0, 1.5, 7/3, 4.0, 17/3, 9.0, 14.0, 19.0, 97/3, 49.0, 99.0)  # thresholds .5 .6 .7 .8 .85 .9 .933 .95 .97 .98 .99
ARMS = ('prior_only', 'adaptive_single', 'adaptive_triple', 'fixed_triple', 'fixed_five')


def calibrator(rows):
    cache = {}

    def transform(k, rows):
        if k not in cache:
            train = [{'prior': r['prior'], 'outcome': r['outcome'], 'answer': 'A'}
                     for r in rows if r['fold'] != k and r['valid']]
            cache[k] = fit_prior_calibration(train, 'monotone_binned')
        fit = cache[k]
        return [dict(r, raw_prior=r['prior'], prior=fit['map'](r['prior'])) if r['valid'] else r for r in rows]
    transform.fits = cache
    return transform


def per_item_utility(rows, ds, loss, settings):
    gross = np.array([(settings['correct_reward'] if r['outcome'] else -loss) if d['action'] == 'assert' else 0.
                      for r, d in zip(rows, ds)])
    return gross - settings['utility_per_usd']*np.array([d['expected_cost_usd'] for d in ds])


def matched(rows, ds, loss, settings):
    """Baseline weights: top-k raw r per fold among valid tasks, ties shared."""
    weights = np.zeros(len(rows))
    for k in sorted({r['fold'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['fold'] == k]
        quota = sum(ds[i]['action'] == 'assert' for i in idx)
        cand = sorted({rows[i]['prior'] for i in idx if rows[i]['valid']}, reverse=True)
        for value in cand:
            group = [i for i in idx if rows[i]['valid'] and rows[i]['prior'] == value]
            take = min(quota, len(group))
            for i in group:
                weights[i] = take/len(group)
            quota -= take
            if quota <= 0:
                break
    y = np.array([r['outcome'] for r in rows])
    base = weights*np.where(y == 1, settings['correct_reward'], -loss)
    arm = per_item_utility(rows, ds, loss, settings)
    return {'accepted': int(sum(d['action'] == 'assert' for d in ds)),
            'arm_wrong': int(sum(d['action'] == 'assert' and not r['outcome'] for r, d in zip(rows, ds))),
            'baseline_expected_wrong': float((weights*(1-y)).sum()),
            'utility_difference': boot(arm-base, settings['bootstrap_repeats'], 20261019)}


def summarize(rows, arms, loss, settings):
    out = {}
    target = 1/(1+loss)
    for name, ds in arms.items():
        released = [r['outcome'] for r, d in zip(rows, ds) if d['action'] == 'assert']
        u = per_item_utility(rows, ds, loss, settings)
        out[name] = {'accepted': len(released), 'wrong': len(released)-sum(released),
                     'coverage': len(released)/len(rows),
                     'error_rate': (len(released)-sum(released))/len(released) if released else None,
                     'target_error': target,
                     'meets_target': (len(released)-sum(released))/len(released) <= target if released else None,
                     'calls_per_task': sum(d['verifiers_used'] for d in ds)/len(rows),
                     'mean_utility': float(u.mean()),
                     'mean_utility_ci95': boot(u, settings['bootstrap_repeats'], 20261019)['ci95']}
    return out


def analyze(config_path, output):
    cfg = json.loads(Path(config_path).read_text())
    settings = cfg['settings']
    output = Path(output)
    if output.exists():
        raise SystemExit('output exists; refusing to overwrite')
    split = {'value': json.loads(Path(cfg['split']).read_text()), 'sha256': file_digest(cfg['split'])}
    result = {'schema': 1, 'kind': 'secondary_analysis_added_after_primary_result',
              'prior_calibration': 'monotone_binned (vgx.gpqa.dependence_triples.fit_prior_calibration), training folds only',
              'grid_losses': list(GRID), 'generators': {}}
    for name, root in cfg['runs'].items():
        rows = crossfit.load_rows(root, split, cfg['verifiers'])
        cal = calibrator(rows)
        gen = {'curves': {}, 'matched_coverage': {}, 'paired': {}}
        for prior_kind, transform in (('raw', None), ('calibrated', cal)):
            curve = {}
            for loss in GRID:
                arms, _ = crossfit_extended.arms_for(rows, cfg, loss, transform)
                key = f'{loss/(1+loss):.3f}'
                curve[key] = {'loss': loss, 'arms': summarize(rows, arms, loss, settings)}
                if round(loss, 6) in (19.0, 99.0):
                    lk = f'{loss:g}'
                    gen['matched_coverage'].setdefault(prior_kind, {})[lk] = {
                        a: matched(rows, arms[a], loss, settings) for a in ('adaptive_single', 'adaptive_triple')}
                    u = {a: per_item_utility(rows, arms[a], loss, settings) for a in arms}
                    reps = settings['bootstrap_repeats']
                    gen['paired'].setdefault(prior_kind, {})[lk] = {
                        'triple_minus_prior_only': boot(u['adaptive_triple']-u['prior_only'], reps, 20261019),
                        'single_minus_prior_only': boot(u['adaptive_single']-u['prior_only'], reps, 20261019),
                        'triple_minus_fixed_triple': boot(u['adaptive_triple']-u['fixed_triple'], reps, 20261019)}
                    if prior_kind == 'calibrated':
                        raw_arms, _ = crossfit_extended.arms_for(rows, cfg, loss)
                        ur = per_item_utility(rows, raw_arms['adaptive_triple'], loss, settings)
                        gen['paired'][prior_kind][lk]['calibrated_triple_minus_raw_triple'] = boot(
                            u['adaptive_triple']-ur, reps, 20261019)
            gen['curves'][prior_kind] = curve
        gen['calibration_maps'] = {str(k): f['summary'] for k, f in sorted(cal.fits.items())}
        result['generators'][name] = gen
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
