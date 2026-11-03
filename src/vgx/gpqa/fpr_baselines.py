"""GPQA: publication under FPR <= alpha and mean cost <= B for our routing and the baselines (no API calls).

Every method is tuned the same way: its candidate rules (thresholds or parameters) are counted with
true labels on the 249 calibration items, out of fold where a method fits anything, and the deployment
LP mixes them. Mixtures are then scored on the 196 Diamond items in expectation. Self-consistency
releases the majority answer of the 10 raw samples, so its correctness is that answer's, and its cost
is the 10 samples. Weaver uses r_phi as prior; EM uses no labels. SSV follows baselines.SSV with its
target error as the tuned parameter. Output: results/gpqa_fpr_routing_20261102/baselines.json.
"""
from __future__ import annotations

import json

import numpy as np

import vgx.baselines as B
from vgx.common.storage import atomic_json
from vgx.gpqa.dependence_triples import bin_of
from vgx.gpqa.fpr_routing import ALPHAS, BUDGETS, Design, EDGES, Model, OMEGAS, OUT, RHOS, SEED, design, solve
from vgx.quantum_katas import crossfit


def lp(fam, wrong, alpha, budget):
    vals = [(a.mean(), (a*(1-fam['y_des'])).mean(), c.mean()) for a, c, _, _ in fam['rules']]
    w = solve(vals, alpha, budget, wrong)[0]
    a = sum(wi*r[2] for wi, r in zip(w, fam['rules']))
    c = sum(wi*r[3] for wi, r in zip(w, fam['rules']))
    return a, c


def stats(a, c, y, rng_idx):
    wrong = 1-y
    def f(i):
        return a[i].mean(), (a[i]*wrong[i]).sum()/max(wrong[i].sum(), 1), c[i].mean()
    pub, fpr, usd = f(np.arange(len(a)))
    b = np.array([f(i) for i in rng_idx])
    lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
    return {'released': float(a.sum()), 'wrong_released': float((a*wrong).sum()), 'pub': float(pub), 'fpr': float(fpr), 'usd': float(usd),
            'fpr_ci': [float(lo[1]), float(hi[1])], 'pub_ci': [float(lo[0]), float(hi[0])], 'wrong_total': float(wrong.sum())}


def base_rules(n_des, n_test):
    z, o = np.zeros(n_des), np.zeros(n_test)
    return [(z, z, o, o)]


def threshold_family(s_des, c_des, y_des, s_test, c_test, y_test):
    s_des, s_test = np.array(s_des, float), np.array(s_test, float)
    finite = np.unique(s_des[np.isfinite(s_des)])
    taus = np.quantile(finite, np.linspace(0, 1, min(60, len(finite)))) if len(finite) else []
    rules = base_rules(len(s_des), len(s_test))
    for t in taus:
        a, b = (s_des >= t).astype(float), (s_test >= t).astype(float)
        rules.append((a, a*0+c_des, b, b*0+c_test))
    return {'rules': rules, 'y_des': np.array(y_des, float), 'y_test': np.array(y_test, float)}


def model_family(rows, des, cal, test, order, kind, settings, calibrate):
    """Fixed-k and confidence-only rules: same priors and likelihoods as ours, one threshold."""
    m = Model(rows, cal, test, settings, calibrate)
    rules = base_rules(des.n, len(test))
    for o in OMEGAS:
        rule = (kind, o, 0.)
        parts = [x.run(order, rule) for x in des.models]
        ran = m.run(order, rule)
        rules.append((np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), *ran))
    return {'rules': rules, 'y_des': np.concatenate([x.y for x in des.models]), 'y_test': m.y}


def ours_mix(rows, des, cal, test, order, grid, vals, alpha, budget, settings, calibrate):
    rules, w, *_ = design(des, order, grid, vals, alpha, budget)
    m = Model(rows, cal, test, settings, calibrate)
    a, c = np.zeros(len(test)), np.zeros(len(test))
    for r, wi in zip(rules, w):
        if wi > 1e-9:
            ai, ci = m.run(order, r); a += wi*ai; c += wi*ci
    return a, c, m.y


def weaver_family(rows, des, cal, test, pool, settings):
    """Posterior from EM on all items, with out-of-fold reports on calibration and r_phi on test."""
    ids = [i for f in des.folds for i in f]
    prior = {i: p for f, m in zip(des.folds, des.models) for i, p in zip(f, m.prior)}
    mt = Model(rows, cal, test, settings, True)
    prior.update(dict(zip(test, mt.prior)))
    use = [i for i in ids+list(test) if rows[i]['valid']]
    bins = [[-1 if rows[i]['signals'][v]['score'] is None else bin_of(rows[i]['signals'][v]['score'], EDGES) for v in pool] for i in use]
    q, *_ = B.weaver_binned_posterior(bins, [prior[i] for i in use], len(EDGES)-1)
    post = dict(zip(use, q))
    usd = lambda i: sum(rows[i]['signals'][v]['estimated_usd'] for v in pool) if rows[i]['valid'] else 0.
    y = lambda i: rows[i]['outcome'] if rows[i]['valid'] else 0
    s = lambda idx: [post.get(i, -np.inf) for i in idx]
    return threshold_family(s(ids), np.array([usd(i) for i in ids]), [y(i) for i in ids], s(test), np.array([usd(i) for i in test]), [y(i) for i in test])


def vote_family(rows, des, cal, test, gen, samples, dataset='gpqa'):
    ids = [i for f in des.folds for i in f]
    vrows, usage = B.vote_rows(dataset, gen, rows, samples)
    s = lambda idx: [vrows[i]['prior'] if vrows[i]['valid'] else -np.inf for i in idx]
    y = lambda idx: [vrows[i]['outcome'] if vrows[i]['valid'] else 0 for i in idx]
    c = lambda idx: np.array([usage[i] for i in idx])
    return threshold_family(s(ids), c(ids), y(ids), s(test), c(test), y(test))


def pick_ssv(rows, train, settings3):
    B.EDGES = (0., 1/3, 2/3, 1.)
    B.LOSS = 19.
    tr = [rows[i] for i in train]
    fitted = crossfit.fit_fold(tr, B.VERIFIERS, settings3)
    priors = np.array([r['prior'] for r in tr if r['valid']])
    weak = max(B.VERIFIERS, key=lambda v: (float(crossfit.planner_for(fitted, [v], 19., settings3)[0].j(0, priors).mean()), v))
    strong = max((v for v in B.VERIFIERS if v != weak), key=lambda v: (B.auroc(rows, train, v), v))
    return weak, strong, fitted


def ssv_family(rows, des, cal, test, settings3):
    ids = [i for f in des.folds for i in f]
    rules = base_rules(len(ids), len(test))
    wk, st, ft = pick_ssv(rows, cal, settings3)
    picks = [pick_ssv(rows, [i for g, f in enumerate(des.folds) if g != k for i in f], settings3) for k in range(len(des.folds))]
    for target in (.005, .01, .02, .05, .1, .15, .2, .3, .4, .5):
        B.LOSS = 1/target-1
        a_d, c_d = [], []
        for k, f in enumerate(des.folds):
            w, s, fit = picks[k]
            ds, _ = B.ssv(rows, [i for g, x in enumerate(des.folds) if g != k for i in x], f, w, s, fit)
            a_d += [float(d['action'] == 'assert') for d in ds]; c_d += [d['usage_estimated_cost_usd'] for d in ds]
        ds, _ = B.ssv(rows, cal, test, wk, st, ft)
        rules.append((np.array(a_d), np.array(c_d), np.array([float(d['action'] == 'assert') for d in ds]),
                      np.array([d['usage_estimated_cost_usd'] for d in ds])))
    y = lambda idx: np.array([rows[i]['outcome'] if rows[i]['valid'] else 0 for i in idx], float)
    return {'rules': rules, 'y_des': y(ids), 'y_test': y(test)}


def main():
    data, cfg = B.load('gpqa')
    settings = cfg['settings']
    samples = B._raw_samples('gpqa')
    orders = json.loads((OUT/'result.json').read_text())['generators']
    base = [('abstain', 0., 0.), ('release', 0., 0.)]
    grid = base+[('bellman', o, r) for o in OMEGAS for r in RHOS]
    out = {'alphas': ALPHAS, 'budgets': BUDGETS, 'generators': {}}
    for gen, rows in data.items():
        (cal, test), = B.splits('gpqa', rows)
        des = Design(rows, cal, settings, True)
        des_raw = Design(rows, cal, settings, False)
        order = tuple(orders[gen]['calibrated']['order'])
        order_raw = tuple(orders[gen]['raw']['order'])
        vals = [des.value(order, r) for r in grid]
        vals_raw = [des_raw.value(order_raw, r) for r in grid]
        families = {'confidence_only': model_family(rows, des, cal, test, order, 'prior', settings, True),
                    'fixed_1': model_family(rows, des, cal, test, order[:1], 'fixed', settings, True),
                    'fixed_2': model_family(rows, des, cal, test, order[:2], 'fixed', settings, True),
                    'fixed_3': model_family(rows, des, cal, test, order, 'fixed', settings, True),
                    'weaver_3': weaver_family(rows, des, cal, test, list(order), settings),
                    'weaver_5': weaver_family(rows, des, cal, test, B.VERIFIERS, settings),
                    'sc_vote': vote_family(rows, des, cal, test, gen, samples),
                    'ssv_model': ssv_family(rows, des, cal, test, settings)}
        idx = [np.random.default_rng(SEED+k).integers(0, len(test), len(test)) for k in range(2000)]
        cells = {}
        for a in ALPHAS:
            for b in BUDGETS:
                ao, co, y = ours_mix(rows, des, cal, test, order, grid, vals, a, b, settings, True)
                ar, cr, yr = ours_mix(rows, des_raw, cal, test, order_raw, grid, vals_raw, a, b, settings, False)
                cell = {'ours': stats(ao, co, y, idx), 'ours_raw': stats(ar, cr, yr, idx)}
                for name, fam in families.items():
                    wrong = float((1-fam['y_des']).mean())
                    fa, fc = lp(fam, wrong, a, b)
                    cell[name] = stats(fa, fc, fam['y_test'], idx)
                    diffs = np.array([ao[i].mean()-fa[i].mean() for i in idx])
                    cell[name]['ours_minus_pub'] = {'mean': float(ao.mean()-fa.mean()), 'ci95': np.percentile(diffs, [2.5, 97.5]).tolist()}
                cells[f'{a}|{b}'] = cell
                print(gen, a, b, *[(k, round(v['released'], 1), round(v['wrong_released'], 1), round(v['usd'], 4)) for k, v in cell.items()], flush=True)
        out['generators'][gen] = cells
    atomic_json(OUT/'baselines.json', out)


if __name__ == '__main__':
    main()
