"""Reporting module r_phi on GPQA (Sec. 2.2-2.3 offline), input U = stated p_correct only; no API calls.

r_phi is fit per generator on the 249 calibration items by maximizing (1/n) sum [2 p_i Y_i - p_i^2],
i.e. minimizing the Brier loss, and then frozen. Two function classes are tried: isotonic (the
unrestricted monotone optimum, by pool-adjacent-violators, linear between knots) and linear in p
(clipped to [0, 1], the realizable linear model of Sec. 3.2 with X = (1, p)). Verifier likelihoods, the
ordered triple and the planner are refit on the same 249 items with the new priors; only the prior
changes. Results on the 196 Diamond items are compared with the raw-p_correct run.
Output: results/gpqa_reporting_trial_20261102/result.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vgx.baselines import LOSS, load, ours_and_fixed, splits, summarize
from vgx.common.storage import atomic_json
from vgx.gpqa.diamond_score import boot

OUT = Path('results/gpqa_reporting_trial_20261102')


def isotonic(p, y):
    """Pool adjacent violators; returns a map that interpolates between the pooled knots."""
    xs = np.unique(p)
    blocks = [[x, x, float(np.sum(y[p == x])), int(np.sum(p == x))] for x in xs]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][2]/out[-2][3] >= out[-1][2]/out[-1][3]:
            b2 = out.pop(); b1 = out.pop()
            out.append([b1[0], b2[1], b1[2]+b2[2], b1[3]+b2[3]])
    kx, ky = [], []
    for lo, hi, s, n in out:
        kx += [lo, hi] if hi > lo else [lo]
        ky += [s/n]*(2 if hi > lo else 1)
    return lambda q: np.interp(q, kx, ky), {'x': kx, 'y': ky}


def linear(p, y):
    X = np.stack([np.ones_like(p), p], 1)
    phi = np.linalg.lstsq(X, y, rcond=None)[0]
    return lambda q: np.clip(phi[0]+phi[1]*np.asarray(q), 0., 1.), {'phi': phi.tolist()}


def recalibrated(rows, fit_idx, cls):
    idx = [i for i in fit_idx if rows[i]['valid']]
    p = np.array([rows[i]['prior'] for i in idx], float); y = np.array([rows[i]['outcome'] for i in idx], float)
    f, info = cls(p, y)
    return [dict(r, prior=float(f(r['prior']))) if r['valid'] else r for r in rows], info


def brier(rows, idx):
    v = [i for i in idx if rows[i]['valid']]
    return float(np.mean([(rows[i]['prior']-rows[i]['outcome'])**2 for i in v]))


def main():
    data, cfg = load('gpqa')
    settings = cfg['settings']
    result = {'loss': LOSS, 'generators': {}}
    for gen, rows in data.items():
        (train, test), = splits('gpqa', rows)
        versions = {'raw': (rows, None)}
        for name, cls in (('isotonic', isotonic), ('linear', linear)):
            versions[name] = recalibrated(rows, train, cls)
        res, utils = {}, {}
        for name, (rs, info) in versions.items():
            arms, fit = ours_and_fixed(rs, train, test, settings)
            test_rows = [rs[i] for i in test]
            res[name] = {'fit': info, 'triple': fit['triple'], 'brier_calibration': brier(rs, train), 'brier_test': brier(rs, test),
                         'prior_ge_095_test': int(sum(r['valid'] and r['prior'] >= .95 for r in test_rows)), 'arms': {}}
            wrong = sum(r['valid'] and not r['outcome'] for r in test_rows) + sum(not r['valid'] for r in test_rows)
            for arm in ('generator_only', 'ours', 'fixed_3'):
                res[name]['arms'][arm], utils[(name, arm)] = summarize(test_rows, arms[arm], settings)
                res[name]['arms'][arm]['fpr'] = res[name]['arms'][arm]['wrong_released']/wrong if wrong else None
        for name in ('isotonic', 'linear'):
            res[name]['ours_minus_raw'] = {a: boot(utils[(name, a)]-utils[('raw', a)], settings['bootstrap_repeats'], 20261102)
                                           for a in ('generator_only', 'ours')}
            res[name]['ours_minus_generator_only'] = boot(utils[(name, 'ours')]-utils[(name, 'generator_only')],
                                                          settings['bootstrap_repeats'], 20261102)
        result['generators'][gen] = res
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/'result.json', result)
    for gen, res in result['generators'].items():
        for name, r in res.items():
            a = r['arms']
            print(gen, name, r['triple'], 'brier', round(r['brier_calibration'], 3), round(r['brier_test'], 3), 'p>=.95', r['prior_ge_095_test'],
                  {k: (v['released'], v['wrong_released'], round(v['mean_utility'], 3), round(v['expected_usd_per_task'], 4), round(v['fpr'], 3)) for k, v in a.items()},
                  {k: [round(v['mean'], 3), [round(x, 3) for x in v['ci95']]] for k, v in r.get('ours_minus_raw', {}).items()},
                  [round(r['ours_minus_generator_only']['mean'], 3), [round(x, 3) for x in r['ours_minus_generator_only']['ci95']]] if 'ours_minus_generator_only' in r else '')


if __name__ == '__main__':
    main()
