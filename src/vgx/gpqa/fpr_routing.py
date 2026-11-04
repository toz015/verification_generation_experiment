"""GPQA prototype of DesignRouting: maximize publication subject to FPR <= alpha and mean cost <= B (no API calls).

Prior r_phi (isotonic on p_correct) and verifier likelihoods (5 levels) as in reporting_sweep. A rule is
the Bellman policy for one pair of multipliers (omega, rho), so its release threshold is 1 - 1/omega.
Rule values (p_i, e_i, c_i) are counted with true labels on the 249 calibration items, out of fold: in
each of 5 inner folds r_phi and the likelihoods are fit on the other folds and the rule runs on the
held-out items. The rule set is a grid of multipliers plus column-generation rounds, and the deployment
linear program mixes rules. One ordered triple per generator: the highest mean design publication over
the (alpha, B) grid. For deployment, r_phi and the likelihoods are refit on all 249 items, the mixture
is kept, and the expected decisions are scored on the 196 Diamond items. Comparators on the same
constraints: confidence alone, and always asking the triple.
Output: results/gpqa_fpr_routing_20261102/result.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import vgx.baselines as B
from vgx.common.storage import atomic_json
from vgx.gpqa.dependence_triples import all_orders, bin_of
from vgx.gpqa.reporting_trial import isotonic
from vgx.quantum_katas import crossfit

OUT = Path('results/gpqa_fpr_routing_20261102')
LEVELS = 5
EDGES = tuple(np.linspace(0, 1, LEVELS+1).tolist())
ALPHAS = (.01, .02, .05, .1, .2)
BUDGETS = (.0005, .001, .002, .003, .006)
OMEGAS = tuple((1+np.logspace(-.5, 3, 18)).tolist())
RHOS = (0., *np.logspace(-1, 3.5, 10).tolist())
GRID = np.r_[0., 1/(1+np.exp(-np.linspace(-14, 14, 1999))), 1.]
FOLDS, ROUNDS, TOL, SEED = 5, 15, 1e-6, 20261102


class Model:
    """r_phi and likelihoods fit on `fit`, prepared for running rules on `items`."""

    def __init__(self, rows, fit, items, settings, calibrate=True):
        fit_valid = [i for i in fit if rows[i]['valid']]
        f = (isotonic(np.array([rows[i]['prior'] for i in fit_valid], float),
                      np.array([rows[i]['outcome'] for i in fit_valid], float))[0] if calibrate else (lambda q: q))
        self.fitted = crossfit.fit_fold([rows[i] for i in fit], B.VERIFIERS, {'probability_bins': LEVELS, 'laplace': 1.0})
        self.valid = np.array([rows[i]['valid'] for i in items])
        self.prior = np.array([float(f(rows[i]['prior'])) if rows[i]['valid'] else 0. for i in items])
        self.y = np.array([rows[i]['outcome'] if rows[i]['valid'] else 0 for i in items], float)
        self.bins = {v: np.array([-1 if rows[i]['signals'][v]['score'] is None else bin_of(rows[i]['signals'][v]['score'], EDGES)
                                  for i in items]) for v in B.VERIFIERS}
        self.usd = {v: np.array([rows[i]['signals'][v]['estimated_usd'] for i in items]) for v in B.VERIFIERS}

    def margins(self, order, omega, rho):
        """Q_k - G_k on GRID for each stage; positive means ask the next verifier."""
        stop = np.maximum(0., 1-omega*(1-GRID))
        value, out = stop, []
        for v in reversed(order):
            lk = self.fitted['likelihoods'][v]
            p1, p0 = np.array(lk['p_bin_if_correct']), np.array(lk['p_bin_if_incorrect'])
            mass = np.outer(GRID, p1)+np.outer(1-GRID, p0)
            post = np.outer(GRID, p1)/np.maximum(mass, 1e-300)
            q = -rho*self.fitted['expected_cost_usd'][v]+(mass*np.interp(post.ravel(), GRID, value).reshape(post.shape)).sum(1)
            out.append(q-stop)
            value = np.maximum(stop, q)
        return out[::-1]

    def run(self, order, rule):
        """Release indicator and realized cost of each item under a rule."""
        kind, omega, rho = rule
        n = len(self.prior)
        if kind == 'release':
            return self.valid.astype(float), np.zeros(n)
        if kind == 'abstain':
            return np.zeros(n), np.zeros(n)
        tau = 1-1/omega
        margins = self.margins(order, omega, rho) if kind == 'bellman' else None
        b, cost = self.prior.copy(), np.zeros(n)
        alive, failed = self.valid.copy(), np.zeros(n, bool)
        for k, v in enumerate(order if kind != 'prior' else ()):
            ask = alive.copy() if kind == 'fixed' else alive & (np.interp(b, GRID, margins[k]) > 0)
            cost[ask] += self.usd[v][ask]
            x = self.bins[v]
            failed |= ask & (x < 0)
            step = ask & (x >= 0)
            lk = self.fitted['likelihoods'][v]
            p1, p0 = np.array(lk['p_bin_if_correct'])[x[step]], np.array(lk['p_bin_if_incorrect'])[x[step]]
            b[step] = b[step]*p1/(b[step]*p1+(1-b[step])*p0)
            alive = ask & ~failed
        return (self.valid & ~failed & (b >= tau)).astype(float), cost


class Design:
    """Out-of-fold counts of rules on the calibration items."""

    def __init__(self, rows, cal, settings, calibrate):
        perm = np.random.default_rng(SEED).permutation(cal)
        self.folds = [[int(i) for i in f] for f in np.array_split(perm, FOLDS)]
        self.models = [Model(rows, [i for g, f in enumerate(self.folds) if g != k for i in f], self.folds[k], settings, calibrate)
                       for k in range(FOLDS)]
        self.n = len(cal)
        self.wrong = sum(float((1-m.y).sum()) for m in self.models)/self.n

    def value(self, order, rule):
        p = e = c = 0.
        for m in self.models:
            a, cost = m.run(order, rule)
            p += a.sum(); e += (a*(1-m.y)).sum(); c += cost.sum()
        return p/self.n, e/self.n, c/self.n


class ModelDesign(Design):
    """Rule values under the fitted model as in the draft: the same out-of-fold reports, no labels."""

    def __init__(self, rows, cal, settings, calibrate):
        super().__init__(rows, cal, settings, calibrate)
        self.wrong = sum(float((1-m.prior).sum()) for m in self.models)/self.n

    def value(self, order, rule):
        kind, omega, rho = rule
        p = e = c = 0.
        for m in self.models:
            pr = m.prior[m.valid]
            if kind == 'release':
                p += len(pr); e += (1-pr).sum()
            elif kind != 'abstain':
                margins = m.margins(order, omega, rho) if kind == 'bellman' else None
                rel = (GRID >= 1-1/omega).astype(float)
                ids = order if kind != 'prior' else ()
                wr, cost = rel*(1-GRID), np.zeros_like(GRID)
                for k in range(len(ids)-1, -1, -1):
                    lk = m.fitted['likelihoods'][ids[k]]
                    p1, p0 = np.array(lk['p_bin_if_correct']), np.array(lk['p_bin_if_incorrect'])
                    mass = np.outer(GRID, p1)+np.outer(1-GRID, p0)
                    post = (np.outer(GRID, p1)/np.maximum(mass, 1e-300)).ravel()
                    next_ = [(mass*np.interp(post, GRID, t).reshape(mass.shape)).sum(1) for t in (rel, wr, cost)]
                    ask = np.ones_like(GRID, bool) if kind == 'fixed' else margins[k] > 0
                    stop = (GRID >= 1-1/omega).astype(float)
                    rel = np.where(ask, next_[0], stop)
                    wr = np.where(ask, next_[1], stop*(1-GRID))
                    cost = np.where(ask, next_[2]+m.fitted['expected_cost_usd'][ids[k]], 0.)
                p += np.interp(pr, GRID, rel).sum(); e += np.interp(pr, GRID, wr).sum(); c += np.interp(pr, GRID, cost).sum()
        return p/self.n, e/self.n, c/self.n


# 'fpr': erroneous releases <= alpha * share wrong; 'fdr': erroneous releases <= alpha * releases.
CONSTRAINT = os.environ.get('VGX_CONSTRAINT', 'fpr')


def solve(values, alpha, budget, wrong):
    """Deployment LP; returns weights, publication and the multipliers of the two constraints."""
    v = np.array(values)
    err = v[:, 1]-alpha*v[:, 0] if CONSTRAINT == 'fdr' else v[:, 1]
    res = linprog(-v[:, 0], A_ub=np.stack([err, v[:, 2]]), b_ub=[0. if CONSTRAINT == 'fdr' else alpha*wrong, budget], A_eq=np.ones((1, len(v))), b_eq=[1.], bounds=(0, None), method='highs')
    return res.x, float(v[:, 0]@res.x), max(0., -res.ineqlin.marginals[0]), max(0., -res.ineqlin.marginals[1])


# VGX_SINGLE=1: deploy the one rule with the most training publication among those meeting both
# constraints (each item gets a definite decision), instead of the LP mixture of rules.
SINGLE = os.environ.get('VGX_SINGLE') == '1'


def deploy(values, alpha, budget, wrong):
    """Deployment weights and training publication: the LP mixture, or one feasible rule."""
    if not SINGLE:
        w, pub, *_ = solve(values, alpha, budget, wrong)
        return w, pub
    v = np.array(values)
    err_ok = v[:, 1] <= alpha*v[:, 0]+1e-12 if CONSTRAINT == 'fdr' else v[:, 1] <= alpha*wrong+1e-12
    ok = np.flatnonzero(err_ok & (v[:, 2] <= budget+1e-12))
    best = ok[np.argmax(v[ok, 0])]
    w = np.zeros(len(v)); w[best] = 1.
    return w, float(v[best, 0])


def design(des, order, rules, values, alpha, budget):
    rules, values = list(rules), list(values)
    for _ in range(ROUNDS):
        w, _, omega, rho = solve(values, alpha, budget, des.wrong)
        # Under the release-error constraint the Lagrangian is p(1 + omega alpha) - omega e - rho c,
        # the same Bellman rule as (omega, rho) / (1 + omega alpha).
        scale = 1+omega*alpha if CONSTRAINT == 'fdr' else 1.
        z = max(p*scale-omega*e-rho*c for p, e, c in values)
        if omega/scale <= 1:
            break
        rule = ('bellman', omega/scale, rho/scale)
        val = des.value(order, rule)
        if val[0]*scale-omega*val[1]-rho*val[2] <= z+TOL:
            break
        rules.append(rule); values.append(val)
    w, pub = deploy(values, alpha, budget, des.wrong)
    return rules, w, pub, float(np.array(values)[:, 1]@w)/des.wrong, float(np.array(values)[:, 2]@w)


def score(rows, test, cal, order, rules, weights, settings, calibrate):
    m = Model(rows, cal, test, settings, calibrate)
    a, c = np.zeros(len(test)), np.zeros(len(test))
    for rule, w in zip(rules, weights):
        if w > 1e-9:
            ai, ci = m.run(order, rule)
            a += w*ai; c += w*ci
    wrong = 1-m.y
    stat = lambda i: (a[i].mean(), (a[i]*wrong[i]).sum()/max(wrong[i].sum(), 1), c[i].mean(),
                      (a[i]*wrong[i]).sum()/max(a[i].sum(), 1e-12))
    rng = np.random.default_rng(SEED)
    boots = np.array([stat(rng.integers(0, len(test), len(test))) for _ in range(2000)])
    pub, fpr, cost, fdr = stat(np.arange(len(test)))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {'released': float(a.sum()), 'wrong_released': float((a*wrong).sum()), 'pub': pub, 'fpr': fpr, 'usd': cost, 'release_error': fdr,
            'pub_ci': [lo[0], hi[0]], 'fpr_ci': [lo[1], hi[1]], 'wrong_total': float(wrong.sum())}


def main(design_cls=Design, name='result.json'):
    data, cfg = B.load('gpqa')
    settings = cfg['settings']
    base = [('abstain', 0., 0.), ('release', 0., 0.)]
    result = {'alphas': ALPHAS, 'budgets': BUDGETS, 'generators': {}}
    for gen, rows in data.items():
        (cal, test), = B.splits('gpqa', rows)
        out = {}
        for version, calibrate in (('calibrated', True), ('raw', False)):
            des = design_cls(rows, cal, settings, calibrate)
            grid = base+[('bellman', o, r) for o in OMEGAS for r in RHOS]
            fixed = base+[('fixed', o, 0.) for o in OMEGAS]
            prior = base+[('prior', o, 0.) for o in OMEGAS]
            scored = []
            for order in all_orders(B.VERIFIERS):
                vals = [des.value(order, r) for r in grid]
                scored.append((np.mean([solve(vals, a, b, des.wrong)[1] for a in ALPHAS for b in BUDGETS]), order, vals))
            mean_pub, order, vals = max(scored, key=lambda s: (s[0], s[1]))
            vals_f, vals_p = [des.value(order, r) for r in fixed], [des.value(order, r) for r in prior]
            table = {}
            for a in ALPHAS:
                for b in BUDGETS:
                    rules, w, pub, fpr, usd = design(des, order, grid, vals, a, b)
                    wf, *_ = solve(vals_f, a, b, des.wrong)
                    wp, *_ = solve(vals_p, a, b, des.wrong)
                    table[f'{a}|{b}'] = {
                        'design': {'pub': pub, 'fpr': fpr, 'usd': usd},
                        'ours': score(rows, test, cal, order, rules, w, settings, calibrate),
                        'fixed_3': score(rows, test, cal, order, fixed, wf, settings, calibrate),
                        'confidence_only': score(rows, test, cal, order, prior, wp, settings, calibrate)}
                    t = table[f'{a}|{b}']
                    print(gen, version, a, b, 'design', round(pub, 3), round(fpr, 3), round(usd, 4),
                          *[(k, round(t[k]['released'], 1), round(t[k]['wrong_released'], 1), round(t[k]['fpr'], 3), round(t[k]['usd'], 4))
                            for k in ('ours', 'fixed_3', 'confidence_only')], flush=True)
            out[version] = {'order': order, 'design_wrong_share': des.wrong, 'cells': table}
        result['generators'][gen] = out
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/name, result)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fitted-model':
        main(ModelDesign, 'result_fitted_model.json')
    else:
        main()
