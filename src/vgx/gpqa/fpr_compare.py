"""MATH-500 and QuantumKatas version of fpr_baselines: same methods, tuned on true labels (no API calls).

Each of the five main-experiment folds is the test part once; everything, including the inner
out-of-fold design and the ordered triple, uses the other four folds. Results pool the five test parts.
The triple is picked on a thinned multiplier grid to keep the 60-order search short. Budgets are
1/12, 1/6, 1/3, 1/2 and 1 of the mean cost of three verifiers on each dataset (GPQA's grid has the same
shape). Output: results/gpqa_fpr_routing_20261102/baselines_<dataset>.json.
"""
from __future__ import annotations

import sys

import numpy as np

import vgx.baselines as B
from vgx.common.storage import atomic_json
from vgx.gpqa.dependence_triples import all_orders
from vgx.gpqa.fpr_baselines import lp, model_family, ours_mix, ssv_family, stats, vote_family, weaver_family
from vgx.gpqa.fpr_routing import CONSTRAINT, ALPHAS, Design, OMEGAS, OUT, RHOS, SEED, solve

BASE = [('abstain', 0., 0.), ('release', 0., 0.)]
GRID = BASE+[('bellman', o, r) for o in OMEGAS for r in RHOS]
THIN = BASE+[('bellman', o, r) for o in OMEGAS[::3] for r in RHOS[::3]]


def pick_order(des, budgets):
    best = None
    for order in all_orders(B.VERIFIERS):
        vals = [des.value(order, r) for r in THIN]
        score = float(np.mean([solve(vals, a, b, des.wrong)[1] for a in ALPHAS for b in budgets]))
        if best is None or (score, order) > best:
            best = (score, order)
    return best[1]


def budgets_for(data):
    cost = np.mean([sum(r['signals'][v]['estimated_usd'] for v in B.VERIFIERS) for rows in data.values() for r in rows if r['valid']])*3/5
    return tuple(float(cost*f) for f in (1/12, 1/6, 1/3, 1/2, 1))


def main(dataset):
    data, cfg = B.load(dataset)
    settings = cfg['settings']
    budgets = budgets_for(data)
    samples = B._raw_samples(dataset) if dataset in ('gpqa', 'math500') else None
    out = {'alphas': ALPHAS, 'budgets': budgets, 'generators': {}}
    for gen, rows in data.items():
        pools = {}
        for k, (train, test) in enumerate(B.splits(dataset, rows)):
            des, des_raw = Design(rows, train, settings, True), Design(rows, train, settings, False)
            order, order_raw = pick_order(des, budgets), pick_order(des_raw, budgets)
            print(gen, 'fold', k, order, order_raw, flush=True)
            vals = [des.value(order, r) for r in GRID]
            vals_raw = [des_raw.value(order_raw, r) for r in GRID]
            fam = {'confidence_only': model_family(rows, des, train, test, order, 'prior', settings, True),
                   'fixed_1': model_family(rows, des, train, test, order[:1], 'fixed', settings, True),
                   'fixed_2': model_family(rows, des, train, test, order[:2], 'fixed', settings, True),
                   'fixed_3': model_family(rows, des, train, test, order, 'fixed', settings, True),
                   'weaver_3': weaver_family(rows, des, train, test, list(order), settings),
                   'weaver_5': weaver_family(rows, des, train, test, B.VERIFIERS, settings),
                   'ssv_model': ssv_family(rows, des, train, test, settings)}
            if samples is not None:
                fam['sc_vote'] = vote_family(rows, des, train, test, gen, samples, dataset)
            for a in ALPHAS:
                for b in budgets:
                    cell = pools.setdefault(f'{a}|{b}', {})
                    cell.setdefault('ours', []).append(ours_mix(rows, des, train, test, order, GRID, vals, a, b, settings, True))
                    cell.setdefault('ours_raw', []).append(ours_mix(rows, des_raw, train, test, order_raw, GRID, vals_raw, a, b, settings, False))
                    for name, f in fam.items():
                        fa, fc = lp(f, float((1-f['y_des']).mean()), a, b)
                        cell.setdefault(name, []).append((fa, fc, f['y_test']))
        n = len(next(iter(pools.values()))['ours'][0][0])*0+sum(len(x[0]) for x in next(iter(pools.values()))['ours'])
        idx = [np.random.default_rng(SEED+k).integers(0, n, n) for k in range(2000)]
        cells = {}
        for key, cell in pools.items():
            joined = {m: tuple(np.concatenate([p[j] for p in parts]) for j in range(3)) for m, parts in cell.items()}
            ao = joined['ours'][0]
            res = {}
            for m, (a_, c_, y_) in joined.items():
                res[m] = stats(a_, c_, y_, idx)
                if m not in ('ours', 'ours_raw'):
                    d = np.array([ao[i].mean()-a_[i].mean() for i in idx])
                    res[m]['ours_minus_pub'] = {'mean': float(ao.mean()-a_.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist()}
            cells[key] = res
        out['generators'][gen] = cells
        atomic_json(OUT/f'baselines_{dataset}{"_fdr" if CONSTRAINT == "fdr" else ""}.json', out)
        print(gen, 'done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
