"""GPQA test results over L with the reporting module r_phi (isotonic on p_correct) and 5 verifier levels.

For each L, on the 196 Diamond items: calibrated prior alone (release iff p >= L/(1+L)), calibrated
prior plus our sequential verifier update, and the same two with the raw p_correct. r_phi, verifier
likelihoods and the ordered triple are fit on the 249 calibration items only, and the triple is
selected separately for each L. Output: results/gpqa_reporting_trial_20261102/sweep.json.
"""
from __future__ import annotations

import numpy as np

import vgx.baselines as B
from vgx.common.storage import atomic_json
from vgx.gpqa.diamond_score import boot
from vgx.gpqa.reporting_trial import OUT, isotonic, recalibrated

LEVELS = 5
LOSSES = (4., 9., 14., 19., 49., 99.)
SEED = 20261102


def arm(rows, decisions, settings, wrong_total):
    s, u = B.summarize(rows, decisions, settings)
    s['fpr'] = s['wrong_released']/wrong_total
    return s, u


def main():
    data, cfg = B.load('gpqa')
    settings = dict(cfg['settings'], probability_bins=LEVELS)
    B.EDGES = tuple(np.linspace(0, 1, LEVELS+1).tolist())
    result = {'levels': LEVELS, 'losses': LOSSES, 'generators': {}}
    for gen, rows in data.items():
        (train, test), = B.splits('gpqa', rows)
        versions = {'calibrated': recalibrated(rows, train, isotonic)[0], 'raw': rows}
        wrong_total = sum(1-r['outcome'] if r['valid'] else 1 for r in (rows[i] for i in test))
        out = {}
        for L in LOSSES:
            B.LOSS = L
            row = {}
            for name, rs in versions.items():
                arms, fit = B.ours_and_fixed(rs, train, test, settings)
                test_rows = [rs[i] for i in test]
                prior, u_prior = arm(test_rows, arms['generator_only'], settings, wrong_total)
                ours, u_ours = arm(test_rows, arms['ours'], settings, wrong_total)
                b = boot(u_ours-u_prior, settings['bootstrap_repeats'], SEED)
                row[name] = {'triple': fit['triple'], 'prior_only': prior, 'with_verifiers': ours,
                             'verifiers_minus_prior_only': {'mean': b['mean'], 'ci95': b['ci95']}, 'utility': u_ours}
            diff = boot(row['calibrated'].pop('utility')-row['raw'].pop('utility'), settings['bootstrap_repeats'], SEED)
            row['calibrated_minus_raw_with_verifiers'] = {'mean': diff['mean'], 'ci95': diff['ci95']}
            out[str(L)] = row
            c, r = row['calibrated'], row['raw']
            fmt = lambda a: (a['released'], a['wrong_released'], round(a['mean_utility'], 3), round(a['fpr'], 3))
            print(gen, L, 'cal prior', fmt(c['prior_only']), 'cal+v', fmt(c['with_verifiers']),
                  [round(x, 3) for x in c['verifiers_minus_prior_only']['ci95']],
                  'raw prior', fmt(r['prior_only']), 'raw+v', fmt(r['with_verifiers']),
                  [round(x, 3) for x in row['calibrated_minus_raw_with_verifiers']['ci95']], flush=True)
        result['generators'][gen] = out
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/'sweep.json', result)


if __name__ == '__main__':
    main()
