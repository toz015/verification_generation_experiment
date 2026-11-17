"""Offline multi-risk comparison on cached candidates (docs/offline-multirisk-agent-handbook-20261116.md).

Three datasets x three generators x three risk constraints (release error, FPR, FNR) x their alpha grids x six
budgets. Every method is fit on Fit, picks one working point on Validation by the shared rule of its risk, is
frozen, and is scored once on Test. Policies read features only; labels reach the evaluator after decisions
are fixed. No generator, verifier or grading API is called: the network is blocked for every command.

  python -m vgx.offline_multirisk preflight --config configs/offline_multirisk_20261116.json
  python -m vgx.offline_multirisk run       --config ... [--only gpqa:qwen3] [--out DIR]
  python -m vgx.offline_multirisk verify    --config ... [--out DIR]
  python -m vgx.offline_multirisk report    --config ... [--out DIR]

Notation (handbook section 4): Y_i = 1 when the fixed candidate is correct, a_i the release probability,
p = mean a, e = mean a (1 - Y), m = mean (1 - a) Y, q = mean Y.
  selective_error  max p  s.t.  e <= alpha p
  fpr              max p  s.t.  e <= alpha (1 - q)
  fnr              min e  s.t.  m <= alpha q
each with mean screening cost <= B.

Rules. Every planned rule is a Bellman policy for a release value g(b) = b - kappa (1 - b) and a cost weight mu
(ask the next verifier iff -mu c_v + E V_next > max(0, g(b))); it releases iff b >= kappa / (1 + kappa).
The handbook's per-risk values are positive multiples of this form (derivation in
docs/offline-multirisk-implementation-20261116.md, checked by tests):
  selective_error  g = 1 - lam((1 - b) - alpha)  ->  kappa = (lam (1 - alpha) - 1) / (1 + lam alpha), mu' = mu / (1 + lam alpha)
  fpr              g = 1 - lam (1 - b)           ->  kappa = lam - 1,                             mu' = mu
  fnr              g = lam b - (1 - b)           ->  kappa = 1 / lam,                             mu' = mu / lam
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linprog

import vgx.baselines as B
import vgx.offline_frozen as O
from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa import fpr_routing as R
from vgx.gpqa.dependence_triples import all_orders

CONFIG = Path('configs/offline_multirisk_20261116.json')
RISKS = ('selective_error', 'fpr', 'fnr')
GRID = np.r_[0., 1/(1+np.exp(-np.linspace(-14, 14, 399))), 1.]
KAPPAS = tuple((10**np.linspace(-2.5, 3, 22)).tolist())
MUS = (0., *np.logspace(-1, 3.5, 10).tolist())
ROUNDS, CG_TOL = 15, 1e-6
TOL_COUNT, TOL_USD = 1e-9, 1e-12
METHODS = ('Ours-Offline', 'AT-Frozen', 'AT+Cal-Frozen', 'Raw-threshold', 'Fixed-1-Mix', 'Fixed-3-Mix')
REFERENCES = ('Release-eligible', 'Abstain-all')
BASELINES = METHODS[1:]
BASE_RULES = (('abstain', 0., 0.), ('release', 0., 0.))
FALLBACK = {'selective_error': 'abstain_all', 'fpr': 'abstain_all', 'fnr': 'release_eligible'}
CODE_FILES = ('src/vgx/offline_multirisk.py', 'src/vgx/offline_frozen.py', 'src/vgx/gpqa/fpr_routing.py',
              'src/vgx/gpqa/reporting_trial.py', 'src/vgx/selective_auditing.py', 'src/vgx/baselines.py',
              'src/vgx/quantum_katas/crossfit.py')


# ----------------------------------------------------------------------------- execution guard

NETWORK_ATTEMPTS = []


def clean(x):
    """JSON-safe copy: arrays to lists, non-finite floats to the strings 'inf', '-inf', 'nan'."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, 'tolist'):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else ('inf' if x > 0 else '-inf' if x < 0 else 'nan')
    return x


def block_network():
    """Every socket connection or DNS lookup raises; attempts are recorded as evidence."""
    def refuse(*args, **kwargs):
        NETWORK_ATTEMPTS.append(repr(args[:2]))
        raise RuntimeError('network disabled: cache-only experiment')
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    return {'blocked': ['socket.connect', 'socket.connect_ex', 'socket.create_connection', 'socket.getaddrinfo']}


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------- risk

class Risk:
    """One risk branch: model LP, dual-to-rule map, empirical counts, feasibility and selection order."""

    def __init__(self, name):
        if name not in RISKS:
            raise ValueError(name)
        self.name = name

    # --- model side (planning): V rows are (p, e, c) per rule; q is the model share correct
    def lp(self, V, alpha, budget, q):
        V = np.asarray(V, float)
        p, e, c = V[:, 0], V[:, 1], V[:, 2]
        if self.name == 'selective_error':
            obj, A, b = -p, [e-alpha*p, c], [0., budget]
        elif self.name == 'fpr':
            obj, A, b = -p, [e, c], [alpha*(1-q), budget]
        else:   # m = q - p + e <= alpha q  <=>  e - p <= (alpha - 1) q
            obj, A, b = e, [e-p, c], [(alpha-1)*q, budget]
        res = linprog(obj, A_ub=np.stack(A), b_ub=b, A_eq=np.ones((1, len(V))), b_eq=[1.], bounds=(0, None), method='highs')
        if res.status != 0:
            return {'status': int(res.status), 'message': res.message}
        w = np.maximum(res.x, 0.)
        w = w/w.sum()
        return {'status': 0, 'w': w, 'p': float(p@w), 'e': float(e@w), 'c': float(c@w), 'm': float(q-p@w+e@w),
                'lam': max(0., -float(res.ineqlin.marginals[0])), 'mu': max(0., -float(res.ineqlin.marginals[1])),
                'residual': (np.stack(A)@w-np.asarray(b)).tolist(), 'objective': float(-obj@w) if self.name != 'fnr' else float(-(e@w))}

    def lagrangian(self, v, lam, mu, alpha):
        p, e, c = v
        if self.name == 'selective_error':
            return p-lam*(e-alpha*p)-mu*c
        if self.name == 'fpr':
            return p-lam*e-mu*c
        return -e-lam*(e-p)-mu*c

    def release_value(self, b, lam, alpha):
        """The handbook's g(b) (relative to abstaining at 0)."""
        b = np.asarray(b, float)
        if self.name == 'selective_error':
            return 1-lam*((1-b)-alpha)
        if self.name == 'fpr':
            return 1-lam*(1-b)
        return lam*b-(1-b)

    def canonical(self, lam, mu, alpha):
        """(kappa, mu') of the Bellman rule maximizing the Lagrangian, or None when that rule is a boundary rule
        (release every eligible candidate, or abstain on all), which the LP always holds."""
        if self.name == 'selective_error':
            u = 1+lam*alpha
            kappa = (lam*(1-alpha)-1)/u
            return None if kappa <= 0 else (kappa, mu/u)
        if self.name == 'fpr':
            return None if lam <= 1 else (lam-1, mu)
        return None if lam <= 1e-12 else (1/lam, mu/lam)

    # --- empirical side (validation selection and test evaluation)
    @staticmethod
    def counts(a, y):
        a = np.atleast_2d(np.asarray(a, float)); y = np.asarray(y, float)
        P = a.sum(axis=1); E = (a*(1-y)).sum(axis=1); M = ((1-a)*y).sum(axis=1)
        return P, E, M, float(y.sum()), float(len(y)-y.sum())

    def ratio(self, P, E, M, Q, W):
        if self.name == 'selective_error':
            return None if P <= TOL_COUNT else E/P
        if self.name == 'fpr':
            return None if W <= 0 else E/W
        return None if Q <= 0 else M/Q

    def estimable_on(self, Q, W):
        return W > 0 if self.name == 'fpr' else Q > 0 if self.name == 'fnr' else True

    def feasible_counts(self, P, E, M, Q, W, alpha):
        if self.name == 'selective_error':
            return (P > TOL_COUNT) & (E-alpha*P <= TOL_COUNT)
        if self.name == 'fpr':
            return (P > TOL_COUNT) & (E-alpha*W <= TOL_COUNT)
        return M-alpha*Q <= TOL_COUNT

    def select(self, A, C, y, alpha, budget, simple=None):
        """(index or None, status) by the shared Validation rule (config 'selection')."""
        A, C = np.atleast_2d(A), np.atleast_2d(C)
        P, E, M, Q, W = self.counts(A, y)
        if not self.estimable_on(Q, W):
            return None, 'risk_not_estimable'
        cost = C.mean(axis=1)
        ok = self.feasible_counts(P, E, M, Q, W, alpha) & (cost <= budget+TOL_USD)
        if not ok.any():
            return None, 'no_feasible_candidate'
        simple = np.zeros(len(A)) if simple is None else np.asarray(simple, float)
        idx = np.flatnonzero(ok)
        if self.name == 'fnr':
            order = np.lexsort((idx, simple[idx], M[idx], cost[idx], E[idx]))
        else:
            order = np.lexsort((idx, simple[idx], cost[idx], -P[idx]))
        return int(idx[order[0]]), 'selected'


# ----------------------------------------------------------------------------- rules under the fitted model

def threshold(kappa):
    return np.inf if not np.isfinite(kappa) else kappa/(1+kappa)


def margins(m, order, kappa, mu):
    """Q_k - G_k on GRID for each stage; positive means ask the next verifier."""
    stop = np.maximum(0., GRID-kappa*(1-GRID))
    value, out = stop, []
    for v in reversed(order):
        lk = m.fitted['likelihoods'][v]
        p1, p0 = np.array(lk['p_bin_if_correct']), np.array(lk['p_bin_if_incorrect'])
        mass = np.outer(GRID, p1)+np.outer(1-GRID, p0)
        post = np.outer(GRID, p1)/np.maximum(mass, 1e-300)
        q = -mu*m.fitted['expected_cost_usd'][v]+(mass*np.interp(post.ravel(), GRID, value).reshape(post.shape)).sum(1)
        out.append(q-stop)
        value = np.maximum(stop, q)
    return out[::-1]


def model_value(m, order, rule):
    """(p, e, c) of a rule per item under the fitted model on m's items (reports only, no labels)."""
    kind, kappa, mu = rule
    n = len(m.prior)
    pr = m.prior[m.valid]
    if kind == 'abstain':
        return 0., 0., 0.
    if kind == 'release':
        return len(pr)/n, float((1-pr).sum())/n, 0.
    t = threshold(kappa)
    rel = (GRID >= t).astype(float)
    wr, cost = rel*(1-GRID), np.zeros_like(GRID)
    mg = margins(m, order, kappa, mu) if kind == 'bellman' else None
    for k in range(len(order)-1, -1, -1):
        lk = m.fitted['likelihoods'][order[k]]
        p1, p0 = np.array(lk['p_bin_if_correct']), np.array(lk['p_bin_if_incorrect'])
        mass = np.outer(GRID, p1)+np.outer(1-GRID, p0)
        post = (np.outer(GRID, p1)/np.maximum(mass, 1e-300)).ravel()
        nxt = [(mass*np.interp(post, GRID, x).reshape(mass.shape)).sum(1) for x in (rel, wr, cost)]
        ask = np.ones_like(GRID, bool) if kind == 'fixed' else mg[k] > 0
        stop = (GRID >= t).astype(float)
        rel = np.where(ask, nxt[0], stop)
        wr = np.where(ask, nxt[1], stop*(1-GRID))
        cost = np.where(ask, nxt[2]+m.fitted['expected_cost_usd'][order[k]], 0.)
    return (float(np.interp(pr, GRID, rel).sum())/n, float(np.interp(pr, GRID, wr).sum())/n,
            float(np.interp(pr, GRID, cost).sum())/n)


def run_rule(m, order, rule):
    """Execute one rule on m's items, reading only the prefix of verifier outputs it asks for.
    Returns release indicator, realized cost, asked (items x stages), final belief."""
    kind, kappa, mu = rule
    n = len(m.prior)
    asked = np.zeros((n, len(order)), bool)
    if kind == 'release':
        return m.valid.astype(float), np.zeros(n), asked, m.prior.copy()
    if kind == 'abstain':
        return np.zeros(n), np.zeros(n), asked, m.prior.copy()
    t = threshold(kappa)
    mg = margins(m, order, kappa, mu) if kind == 'bellman' else None
    b, cost = m.prior.copy(), np.zeros(n)
    alive, failed = m.valid.copy(), np.zeros(n, bool)
    for k, v in enumerate(order):
        # Fixed panels ask every member (no confidence-based early stop); a failed call stops and defers.
        ask = alive.copy() if kind == 'fixed' else alive & (np.interp(b, GRID, mg[k]) > 0)
        asked[:, k] = ask
        cost[ask] += m.usd[v][ask]
        x = m.bins[v]
        failed |= ask & (x < 0)
        step = ask & (x >= 0)
        lk = m.fitted['likelihoods'][v]
        p1, p0 = np.array(lk['p_bin_if_correct'])[x[step]], np.array(lk['p_bin_if_incorrect'])[x[step]]
        b[step] = b[step]*p1/(b[step]*p1+(1-b[step])*p0)
        alive = ask & ~failed
    return (m.valid & ~failed & (b >= t)).astype(float), cost, asked, b


class Runs:
    """Per item set: cached rule executions keyed by (order, rule)."""

    def __init__(self, model):
        self.m = model
        self.cache = {}

    def get(self, order, rule):
        key = (tuple(order), tuple(rule))
        if key not in self.cache:
            self.cache[key] = run_rule(self.m, tuple(order), tuple(rule))
        return self.cache[key]

    def mixture(self, cand):
        """Expected release, cost, calls and P(at least one call) of a frozen mixture on these items."""
        w = np.asarray(cand['weights'], float)
        per = [self.get(cand['order'], r) for r in cand['rules']]
        a = sum(wi*p[0] for wi, p in zip(w, per))
        c = sum(wi*p[1] for wi, p in zip(w, per))
        calls = sum(wi*p[2].sum(axis=1) for wi, p in zip(w, per))
        qprob = sum(wi*(p[2].sum(axis=1) > 0) for wi, p in zip(w, per))
        return a, c, calls, qprob, per


class ValueCache:
    def __init__(self, model):
        self.m = model
        self.cache = {}

    def get(self, order, rule):
        key = (tuple(order), tuple(rule))
        if key not in self.cache:
            self.cache[key] = model_value(self.m, tuple(order), tuple(rule))
        return self.cache[key]


def design(vc, order, risk, alpha, budget, q, rules, values):
    """LP over the rule set plus column generation from the risk's duals. Returns rules, values and the LP."""
    rules, values = list(rules), list(values)
    for _ in range(ROUNDS):
        sol = risk.lp(values, alpha, budget, q)
        if sol['status'] != 0:
            break
        z = max(risk.lagrangian(v, sol['lam'], sol['mu'], alpha) for v in values)
        can = risk.canonical(sol['lam'], sol['mu'], alpha)
        if can is None:
            break
        rule = ('bellman', float(can[0]), float(can[1]))
        val = vc.get(order, rule)
        if risk.lagrangian(val, sol['lam'], sol['mu'], alpha) <= z+CG_TOL:
            break
        rules.append(rule); values.append(val)
    return rules, values, risk.lp(values, alpha, budget, q)


def candidate(method, order, rules, values, sol, q, **extra):
    keep = [j for j, x in enumerate(sol['w']) if x > 1e-12]
    w = np.array([sol['w'][j] for j in keep]); w = w/w.sum()
    V = np.array([values[j] for j in keep])
    return {'method': method, 'order': list(order), 'rules': [list(rules[j]) for j in keep], 'weights': w.tolist(),
            'planned': {'p': float(V[:, 0]@w), 'e': float(V[:, 1]@w), 'c': float(V[:, 2]@w), 'q_model': q,
                        'm': float(q-V[:, 0]@w+V[:, 1]@w)},
            'rule_values': [list(values[j]) for j in keep], 'lp_status': sol['status'], 'lp_residual': sol['residual'],
            'lp_multipliers': [sol['lam'], sol['mu']], **extra}


def plan_objective(risk, sol):
    return -np.inf if sol['status'] != 0 else sol['objective']


def ours_candidates(vc, risk, alpha_plans, bplans, cfg):
    """Shortlist orders by the label-free model LP objective on the thin grid, then plan mixtures."""
    thin = list(BASE_RULES)+[('bellman', k, u) for k in KAPPAS[::3] for u in MUS[::3]]
    grid = list(BASE_RULES)+[('bellman', k, u) for k in KAPPAS for u in MUS]
    q = float(vc.m.prior.mean())
    scored = []
    for order in all_orders(B.VERIFIERS):
        vals = [vc.get(order, r) for r in thin]
        score = float(np.mean([plan_objective(risk, risk.lp(vals, a, b, q)) for a in alpha_plans for _, b in bplans]))
        scored.append((score, order))
    scored.sort(key=lambda s: (-s[0], s[1]))
    out = []
    for score, order in scored[:cfg['ours']['orders_shortlist']]:
        vals = [vc.get(order, r) for r in grid]
        for ap in alpha_plans:
            for bf, bp in bplans:
                rules, values, sol = design(vc, order, risk, ap, bp, q, grid, vals)
                if sol['status'] != 0:
                    out.append({'method': 'Ours-Offline', 'order': list(order), 'lp_status': sol['status'], 'alpha_plan': ap,
                                'budget_plan_fraction': bf, 'invalid': True})
                    continue
                out.append(candidate('Ours-Offline', order, rules, values, sol, q, alpha_plan=ap, budget_plan_fraction=bf,
                                     order_score=score, pool='adaptive'))
    if cfg['ours'].get('fixed_panels_in_pool'):
        # Fixed-panel mixtures over Ours' own shortlist (each single verifier in it, and each shortlisted triple),
        # added to the pool Validation selects from; the adaptive LP above is unchanged.
        short = [tuple(sorted(o)) for _, o in scored[:cfg['ours']['orders_shortlist']]]
        panels = sorted({(v,) for p in short for v in p}) + sorted(set(short))
        fx, bad = fixed_candidates(vc, risk, None, alpha_plans, bplans, 'Ours-Offline', panels=panels)
        out += [dict(c, pool='fixed_panel') for c in fx]
        out += [dict(b, method='Ours-Offline', invalid=True, pool='fixed_panel') for b in bad]
    ranking = [{'order': list(o), 'score': s} for s, o in scored]
    return [c for c in out if not c.get('invalid')], ranking, [c for c in out if c.get('invalid')]


def fixed_candidates(vc, risk, k, alpha_plans, bplans, method, panels=None):
    rules = list(BASE_RULES)+[('fixed', kap, 0.) for kap in KAPPAS]
    q = float(vc.m.prior.mean())
    out, bad = [], []
    for panel in (panels if panels is not None else combinations(sorted(B.VERIFIERS), k)):
        vals = [vc.get(panel, r) for r in rules]
        for ap in alpha_plans:
            for bf, bp in bplans:
                sol = risk.lp(vals, ap, bp, q)
                if sol['status'] != 0:
                    bad.append({'panel': list(panel), 'alpha_plan': ap, 'budget_plan_fraction': bf, 'lp_status': sol['status']})
                    continue
                out.append(candidate(method, panel, rules, vals, sol, q, alpha_plan=ap, budget_plan_fraction=bf))
    return out, bad


# ----------------------------------------------------------------------------- data

def eligible(rows, idx):
    return np.array([bool(rows[i]['valid']) for i in idx])


def truth(rows, idx):
    """Evaluator side only: Y = 1 when the fixed candidate is correct; no answer counts wrong (official rule)."""
    return np.array([int(rows[i]['outcome']) if rows[i]['valid'] else 0 for i in idx], int)


def label_source(r):
    if r['valid']:
        return 'official_checker'
    return 'no_answer_official_rule'


def conditions(cfg):
    return [(r, float(a), float(f)) for r in cfg['risk_metrics'] for a in cfg['alpha_target'][r] for f in cfg['budget_fractions']]


def splits(ds, rows):
    return O.splits(ds, rows)


# ----------------------------------------------------------------------------- one generator

CONFIDENCE_INPUTS = ('mixed', 'calibrated', 'raw')


def fold_rows(rows, fit, mode):
    """Rows every method of one outer fold reads, and whether the planning model still calibrates.
    mixed: the original setup (Ours and Fixed calibrate on Fit; AT, AT+Cal and the threshold read raw p_correct).
    calibrated: every method reads the Fit-calibrated p_correct (isotonic PAV fit on Fit, applied to all items).
    raw: every method reads raw p_correct (Ours and Fixed plan with the raw value as their prior)."""
    if mode not in CONFIDENCE_INPUTS:
        raise ValueError(f'unknown confidence_input {mode}')
    if mode == 'mixed':
        return rows, True
    if mode == 'raw':
        return rows, False
    fv = [i for i in fit if rows[i]['valid']]
    f = R.isotonic(np.array([rows[i]['prior'] for i in fv], float), np.array([rows[i]['outcome'] for i in fv], float))[0]
    return [dict(r, prior=float(f(r['prior']))) if r['valid'] else r for r in rows], False


def run_generator(ds, gen, rows, cfg, groups=None, quick=False):
    t0 = time.perf_counter()
    seeds = cfg['at']['seeds'][:2] if quick else cfg['at']['seeds']
    risks = {r: Risk(r) for r in cfg['risk_metrics']}
    fracs = [float(f) for f in cfg['budget_fractions']]
    conds = conditions(cfg)
    sp = splits(ds, rows)
    folds, cells, configs, traj = [], {}, {}, {}
    brier = {'raw': [], 'ours_prior': [], 'at_cal': {}}
    mode = cfg.get('confidence_input', 'mixed')
    raw_rows = rows
    for k, fit, val, test in sp:
        tf = time.perf_counter()
        rows, cal = fold_rows(raw_rows, fit, mode)
        cref = O.c_ref(rows, fit)
        bplans = [(f, f*cref) for f in fracs]
        yv, yt = truth(rows, val), truth(rows, test)
        ev, et = eligible(rows, val), eligible(rows, test)
        mv, mt = R.Model(rows, fit, val, {}, cal), R.Model(rows, fit, test, {}, cal)
        vc = ValueCache(mv)
        rv, rt = Runs(mv), Runs(mt)
        fv, ft = O.Features(rows, val), O.Features(rows, test)
        folds.append({'fold': k, 'n_fit': len(fit), 'n_validation': len(val), 'n_test': len(test), 'c_ref_usd': cref,
                      'c_ref_n': int(sum(rows[i]['valid'] for i in fit)),
                      'c_ref_per_verifier_usd': {v: float(np.mean([rows[i]['signals'][v]['estimated_usd'] for i in fit if rows[i]['valid']]))
                                                 for v in B.VERIFIERS},
                      'budget_usd': {str(f): f*cref for f in fracs},
                      'validation_correct': int(yv.sum()), 'validation_wrong': int(len(yv)-yv.sum()),
                      'test_correct': int(yt.sum()), 'test_wrong': int(len(yt)-yt.sum()),
                      'test_eligible': int(et.sum()), 'test_ids': [rows[i]['item_id'] for i in test],
                      'confidence_input': mode, 'test_prior_eligible': [float(x) for x in mt.prior[et]]})
        raw_conf = O.Features(raw_rows, test).confidence
        brier['raw'].append([float(np.sum((raw_conf[et]-yt[et])**2)), int(et.sum())])
        brier['ours_prior'].append([float(np.sum((mt.prior[et]-yt[et])**2)), int(et.sum())])
        brier.setdefault('ours_map', {}).update({rows[i]['item_id']: float(mt.prior[j]) for j, i in enumerate(test) if et[j]})
        # --- threshold methods: frozen states per seed (risk independent)
        thr = {}
        for s in seeds:
            st = O.threshold_methods(rows, fit, s)
            for name, kind in (('AT-Frozen', 'AT'), ('AT+Cal-Frozen', 'AT+CC')):
                av, _ = O.threshold_decisions(kind, st[kind], fv, 0.0, s)
                at_, score = O.threshold_decisions(kind, st[kind], ft, 0.0, s)
                thr.setdefault(name, []).append({'seed': s, 'av': av, 'at': at_, 'score': score, 'configs': st[kind]['configs'],
                                                 'tau': st[kind]['tau'], 'fit_reveals': st[kind]['fit_reveals'],
                                                 'state_hash': O.state_hash({k_: np.asarray(v_).tolist() for k_, v_ in st[kind].items() if k_ != 'configs'})})
        rawtau = np.r_[-np.inf, np.unique(fv.confidence[fv.valid]), np.inf]
        raw_av = ((np.nan_to_num(fv.confidence, nan=-np.inf)[None, :] >= rawtau[:, None]) & fv.valid[None, :]).astype(float)
        raw_at = ((np.nan_to_num(ft.confidence, nan=-np.inf)[None, :] >= rawtau[:, None]) & ft.valid[None, :]).astype(float)
        thr['Raw-threshold'] = [{'seed': 0, 'av': raw_av, 'at': raw_at, 'configs': [[float(x)] for x in rawtau], 'tau': rawtau}]
        # --- model-planned methods per risk
        for rname, risk in risks.items():
            aps = cfg['alpha_plan'][rname]
            cands = {}
            oc, ranking, bad_o = ours_candidates(vc, risk, aps, bplans, cfg)
            cands['Ours-Offline'] = oc
            cands['Fixed-1-Mix'], bad_1 = fixed_candidates(vc, risk, 1, aps, bplans, 'Fixed-1-Mix')
            cands['Fixed-3-Mix'], bad_3 = fixed_candidates(vc, risk, 3, aps, bplans, 'Fixed-3-Mix')
            folds[-1].setdefault('planning', {})[rname] = {
                'order_ranking_top10': ranking[:10], 'n_candidates': {m_: len(c) for m_, c in cands.items()},
                'lp_failures': {'Ours-Offline': bad_o, 'Fixed-1-Mix': bad_1, 'Fixed-3-Mix': bad_3}}
            vmix = {}
            for mname, cs in cands.items():
                for j, c in enumerate(cs):
                    c['id'] = f'{ds}|{gen}|f{k}|{rname}|{mname}|c{j}'
                mixes = [rv.mixture(c) for c in cs]
                vmix[mname] = (np.stack([x[0] for x in mixes]), np.stack([x[1] for x in mixes]), [len(c['rules']) for c in cs])
            for alpha in cfg['alpha_target'][rname]:
                for f, budget in bplans:
                    key = (rname, float(alpha), f)
                    for mname in ('Ours-Offline', 'Fixed-1-Mix', 'Fixed-3-Mix'):
                        A, C, simple = vmix[mname]
                        j, status = risk.select(A, C, yv, alpha, budget, simple)
                        rec = fold_record(rname, status, j is None)
                        if j is None:
                            fill_fallback(rec, rname, et)
                        else:
                            c = cands[mname][j]
                            a, cost, calls, qprob, per = rt.mixture(c)
                            rec.update(a=a[None], cost=cost[None], calls=calls[None], qprob=qprob[None], config_id=[c['id']],
                                       draws=draw(per, c['weights'], len(test), cfg['ours']['mixture_draws']))
                            configs[c['id']] = c
                            for r_ in c['rules']:
                                traj[(k, tuple(c['order']), tuple(r_))] = (rt.get(c['order'], r_), test)
                        cells.setdefault(key, {}).setdefault(mname, []).append(rec)
                    for mname, states in thr.items():
                        rec = fold_record(rname, None, None)
                        recs = []
                        for st in states:
                            j, status = risk.select(st['av'], np.zeros_like(st['av']), yv, alpha, budget,
                                                    np.ones(len(st['av'])))
                            if j is None:
                                fb = np.zeros(len(test)) if FALLBACK[rname] == 'abstain_all' else et.astype(float)
                                recs.append((fb, status, True, None))
                            else:
                                cid = f'{ds}|{gen}|f{k}|{mname}|seed{st["seed"]}|c{j}'
                                configs.setdefault(cid, {'method': mname, 'seed': st['seed'], 'config': list(map(float, st['configs'][j])),
                                                         'threshold': float(st['tau'][j]), 'state_hash': st.get('state_hash'),
                                                         'fit_reveals': st.get('fit_reveals', 0)})
                                recs.append((st['at'][j], status, False, cid))
                            if mname == 'AT+Cal-Frozen' and alpha == cfg['alpha_main'][rname] and f == max(fracs):
                                sc = None if j is None else np.clip(st['score'][j][et], 0, 1)
                                brier['at_cal'].setdefault(rname, []).append(
                                    {'fold': k, 'seed': st['seed'], 'status': 'selected' if j is not None else 'no_selected_calibrator',
                                     'sum': None if sc is None else float(np.sum((sc-yt[et])**2)), 'n': int(et.sum()),
                                     'items': [rows[i]['item_id'] for i in np.array(test)[et]] if sc is not None else None,
                                     'scores': None if sc is None else sc.tolist()})
                        rec.update(a=np.stack([x[0] for x in recs]), cost=np.zeros((len(recs), len(test))),
                                   calls=np.zeros((len(recs), len(test))), qprob=np.zeros((len(recs), len(test))),
                                   status=[x[1] for x in recs], fallback=[x[2] for x in recs], config_id=[x[3] for x in recs],
                                   seeds=[st['seed'] for st in states])
                        cells.setdefault(key, {}).setdefault(mname, []).append(rec)
                    for mname in REFERENCES:
                        rec = fold_record(rname, 'reference', False)
                        a = et.astype(float) if mname == 'Release-eligible' else np.zeros(len(test))
                        rec.update(a=a[None], cost=np.zeros((1, len(test))), calls=np.zeros((1, len(test))), qprob=np.zeros((1, len(test))))
                        cells.setdefault(key, {}).setdefault(mname, []).append(rec)
        folds[-1]['seconds'] = time.perf_counter()-tf
    rows = raw_rows
    out = summarise(ds, gen, rows, sp, folds, cells, cfg, groups)
    out.update({'folds': folds, 'brier': brier_summary(brier, rows, sp), 'seconds': time.perf_counter()-t0})
    return out, cells, configs, traj


def fold_record(rname, status, fallback):
    return {'status': [status], 'fallback': [fallback], 'config_id': [None], 'seeds': [0], 'draws': None}


def fill_fallback(rec, rname, et):
    n = len(et)
    a = np.zeros(n) if FALLBACK[rname] == 'abstain_all' else et.astype(float)
    rec.update(a=a[None], cost=np.zeros((1, n)), calls=np.zeros((1, n)), qprob=np.zeros((1, n)), fallback=[True],
               config_id=[f'fallback:{FALLBACK[rname]}'])


def draw(per, w, n, draws):
    """Sampled deployments: each item gets one rule drawn with the mixture weights (preset seeds 0..draws-1)."""
    w = np.asarray(w, float)
    A, C = np.zeros((draws, n)), np.zeros((draws, n))
    for s in range(draws):
        pick = np.random.default_rng(s).choice(len(w), size=n, p=w/w.sum())
        A[s] = np.array([per[r][0][i] for i, r in enumerate(pick)])
        C[s] = np.array([per[r][1][i] for i, r in enumerate(pick)])
    return {'a': A, 'cost': C}


# ----------------------------------------------------------------------------- evaluation

def metrics(risk, alpha, a, cost, calls, qprob, y, elig, budget):
    """Evaluator: every number from frozen decisions and the private labels (handbook 10.1)."""
    a = np.asarray(a, float); y = np.asarray(y, float); n = len(a)
    P, E, M, Q, W = (float(x) if np.ndim(x) == 0 else float(x[0]) for x in Risk.counts(a, y))
    p, e, m, q = P/n, E/n, M/n, Q/n
    if abs(m-(q-p+e)) > 1e-9:
        raise AssertionError('m != q - p + e')
    sel = None if P <= TOL_COUNT else E/P
    fpr = None if W <= 0 else E/W
    fnr = None if Q <= 0 else M/Q
    ew = float(((1-y)*elig).sum())
    r = risk.ratio(P, E, M, Q, W)
    mc = float(np.mean(cost))
    out = {'n_tasks': n, 'n_eligible': int(elig.sum()), 'n_labeled': n, 'n_correct': Q, 'n_wrong': W,
           'n_released': P, 'n_wrong_released': E, 'n_correct_deferred': M,
           'coverage': p, 'release_error_rate': sel, 'fpr': fpr, 'fnr': fnr,
           'fpr_eligible_subset': None if ew <= 0 else E/ew,
           'joint_wrong_release_rate': e, 'correct_retention_rate': None if fnr is None else 1-fnr,
           'mean_screen_cost': mc, 'budget_usd': budget, 'cost_excess_usd': max(0., mc-budget),
           'cost_excess_relative': None if budget <= 0 else max(0., mc-budget)/budget,
           'weak_verifier_calls_per_task': float(np.mean(calls)), 'verifier_query_rate': float(np.mean(qprob)),
           'risk': r, 'risk_estimable': r is not None,
           'risk_excess': None if r is None else r-alpha, 'zero_release': P <= TOL_COUNT}
    risk_ok = r is not None and bool(risk.feasible_counts(P, E, M, Q, W, alpha))
    out['risk_met'] = risk_ok
    out['budget_met'] = mc <= budget+TOL_USD
    out['empirical_feasible'] = bool(risk_ok and out['budget_met'] and not (risk.name != 'fnr' and out['zero_release']))
    return out


def summarise(ds, gen, rows, sp, folds, cells, cfg, groups):
    """Pool folds per condition and method; seed-average release probabilities before pooling (handbook 10.1)."""
    idx = [i for _, _, _, t in sp for i in t]
    y = truth(rows, idx); elig = eligible(rows, idx)
    nt = np.array([len(t) for _, _, _, t in sp], float)
    out = {'dataset': ds, 'generator': gen, 'n_test': len(idx), 'cells': {}}
    for (rname, alpha, f), by_method in cells.items():
        risk = Risk(rname)
        bud_f = np.array([fo['budget_usd'][str(f)] for fo in folds])
        budget = float((bud_f*nt).sum()/nt.sum())
        row = {}
        for mname, parts in by_method.items():
            a = np.concatenate([p['a'].mean(axis=0) for p in parts])
            cost = np.concatenate([p['cost'].mean(axis=0) for p in parts])
            calls = np.concatenate([p['calls'].mean(axis=0) for p in parts])
            qprob = np.concatenate([p['qprob'].mean(axis=0) for p in parts])
            mt = metrics(risk, alpha, a, cost, calls, qprob, y, elig, budget)
            st = [s for p in parts for s in p['status']]
            fb = [x for p in parts for x in p['fallback']]
            mt['selection_status'] = {s: st.count(s) for s in sorted(set(map(str, st)))}
            mt['fallback_used'] = bool(any(fb))
            mt['fallback_share'] = float(np.mean([bool(x) for x in fb]))
            mt['fold_budget_exceeded'] = [int(fo['fold']) for fo, p in zip(folds, parts)
                                          if p['cost'].mean(axis=0).mean() > fo['budget_usd'][str(f)]+TOL_USD]
            mt['budget_usd_by_fold'] = bud_f.tolist()
            mt['config_ids'] = [p['config_id'] for p in parts]
            nseed = parts[0]['a'].shape[0]
            if nseed > 1:
                per = []
                for s in range(nseed):
                    a_s = np.concatenate([p['a'][s] for p in parts])
                    per.append(metrics(risk, alpha, a_s, np.zeros_like(a_s), np.zeros_like(a_s), np.zeros_like(a_s), y, elig, budget))
                mt['per_seed'] = {k: [x[k] for x in per] for k in ('coverage', 'risk', 'joint_wrong_release_rate', 'empirical_feasible')}
                mt['seed_risk_exceeded_share'] = float(np.mean([x['risk'] is None or not x['risk_met'] for x in per]))
            if all(p.get('draws') is not None or p['fallback'][0] for p in parts) and any(p.get('draws') is not None for p in parts):
                D = len(next(p['draws']['a'] for p in parts if p.get('draws') is not None))
                sampled = []
                for s in range(D):
                    a_s = np.concatenate([p['draws']['a'][s] if p.get('draws') is not None else p['a'][0] for p in parts])
                    c_s = np.concatenate([p['draws']['cost'][s] if p.get('draws') is not None else p['cost'][0] for p in parts])
                    sampled.append(metrics(risk, alpha, a_s, c_s, np.zeros_like(a_s), np.zeros_like(a_s), y, elig, budget))
                mt['sampled_draws'] = {k: [x[k] for x in sampled] for k in ('coverage', 'risk', 'joint_wrong_release_rate', 'mean_screen_cost', 'empirical_feasible')}
            mt['_items'] = (a, cost)
            row[mname] = mt
        out['cells'][f'{rname}|{alpha}|{f}'] = row
    bootstrap(out, rows, sp, idx, y, cfg, groups)
    for row in out['cells'].values():
        for mt in row.values():
            mt.pop('_items', None)
    return out


def bootstrap(out, rows, sp, idx, y, cfg, groups):
    """Paired bootstrap of Ours minus each baseline: the same resampled units for every comparison of a generator,
    stratified by outer test fold; QuantumKatas resamples merged kata groups."""
    unit = np.array([str(groups.get(rows[i]['item_id'], rows[i]['item_id'])) if groups else str(i) for i in idx])
    fold = np.array([rows[i]['fold'] for i in idx]) if len(sp) > 1 else np.zeros(len(idx), int)
    keys, inv = np.unique(unit, return_inverse=True)
    kfold = np.array([fold[inv == g][0] for g in range(len(keys))])
    if any(len(set(fold[inv == g])) > 1 for g in range(len(keys))):
        raise AssertionError('a resampling unit spans test folds')
    rng = np.random.default_rng(cfg['bootstrap']['seed'])
    R_ = cfg['bootstrap']['replicates']
    Wt = np.zeros((R_, len(keys)))
    for fo in np.unique(kfold):
        g = np.flatnonzero(kfold == fo)
        draws = rng.integers(0, len(g), (R_, len(g)))
        for r in range(R_):
            Wt[r] += np.bincount(g[draws[r]], minlength=len(keys))
    gs = lambda v: np.bincount(inv, weights=v, minlength=len(keys))
    Y = gs(y.astype(float)); N = gs(np.ones(len(y)))
    for key, row in out['cells'].items():
        rname = key.split('|')[0]
        ao = row['Ours-Offline']['_items'][0]

        def stats(a):
            P, E = Wt@gs(a), Wt@gs(a*(1-y))
            M = Wt@gs((1-a)*y)
            n, q = Wt@N, Wt@Y
            if rname == 'selective_error':
                r = np.where(P > TOL_COUNT, E/np.where(P > TOL_COUNT, P, 1), np.nan)
            elif rname == 'fpr':
                r = np.where(n-q > 0, E/np.where(n-q > 0, n-q, 1), np.nan)
            else:
                r = np.where(q > 0, M/np.where(q > 0, q, 1), np.nan)
            return P/n, E/n, r
        po, eo, ro = stats(ao)
        row['Ours-Offline']['paired'] = {}
        for b in BASELINES:
            ab = row[b]['_items'][0]
            pb, eb, rb = stats(ab)
            main = (po-pb) if rname != 'fnr' else (eo-eb)
            pt_main = (row['Ours-Offline']['coverage']-row[b]['coverage']) if rname != 'fnr' else \
                (row['Ours-Offline']['joint_wrong_release_rate']-row[b]['joint_wrong_release_rate'])
            dr = ro-rb
            ok = ~np.isnan(dr)
            r1, r2 = row['Ours-Offline']['risk'], row[b]['risk']
            row['Ours-Offline']['paired'][b] = {
                'field': 'ours_coverage_minus_baseline_coverage' if rname != 'fnr' else 'ours_e_minus_baseline_e',
                'point': pt_main, 'ci95': np.percentile(main, [2.5, 97.5]).tolist(),
                'risk_point': None if r1 is None or r2 is None else r1-r2,
                'risk_ci95': np.percentile(dr[ok], [2.5, 97.5]).tolist() if ok.any() else None,
                'risk_valid_replicates': int(ok.sum()), 'risk_zero_denominator_share': float(1-ok.mean())}


def brier_summary(brier, rows, sp):
    out = {}
    for k in ('raw', 'ours_prior'):
        s = sum(x[0] for x in brier[k]); n = sum(x[1] for x in brier[k])
        out[k] = {'brier': s/n, 'n': n, 'folds': len(brier[k])}
    out['at_cal'] = {}
    for rname, recs in brier['at_cal'].items():
        ok = [r for r in recs if r['status'] == 'selected']
        tot = len(recs)
        common = None
        if ok:
            out_r = {'status_counts': {'selected': len(ok), 'no_selected_calibrator': tot-len(ok)},
                     'brier_selected_task_seeds': sum(r['sum'] for r in ok)/sum(r['n'] for r in ok),
                     'task_seed_pairs': sum(r['n'] for r in ok), 'fold_seed_coverage': len(ok)/tot}
            # Paired with Raw and Ours prior on the same task x seed pairs
            raw_map = {rows[i]['item_id']: (rows[i]['prior'], int(rows[i]['outcome'])) for _, _, _, t in sp for i in t if rows[i]['valid']}
            npair = sum(r['n'] for r in ok)
            out_r['raw_on_same_pairs'] = sum((raw_map[t][0]-raw_map[t][1])**2 for r in ok for t in r['items'])/npair
            out_r['ours_prior_on_same_pairs'] = sum((brier['ours_map'][t]-raw_map[t][1])**2 for r in ok for t in r['items'])/npair
            out_r['distinct_tasks'] = len({t for r in ok for t in r['items']})
            common = out_r
        else:
            common = {'status_counts': {'selected': 0, 'no_selected_calibrator': tot}}
        out['at_cal'][rname] = common
    return out


# ----------------------------------------------------------------------------- logs

LOG_FIELDS = ['dataset', 'generator', 'task_id', 'group_id', 'fold', 'risk_metric', 'alpha_target', 'budget_id', 'budget_usd',
              'method', 'seed', 'selected_config_id', 'selection_status', 'fallback_used', 'input_eligible', 'raw_confidence',
              'calibrated_confidence', 'release_probability', 'expected_cost', 'expected_verifier_calls',
              'verifier_query_probability', 'rule_ids', 'mixture_weights']


def write_logs(out_dir, ds, gen, rows, sp, folds, cells, configs, traj, groups, cfg):
    """Per-item decisions (seed means for seeded methods; per-seed releases in a separate file), per-rule traces."""
    out_dir = Path(out_dir)
    prior_t = {}
    for k, fit, val, test in sp:
        rk, cal = fold_rows(rows, fit, cfg.get('confidence_input', 'mixed'))
        prior_t[k] = R.Model(rk, fit, test, {}, cal).prior
    dpath = out_dir/f'decisions_{ds}_{gen}.csv.gz'
    with gzip.open(dpath, 'wt', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(LOG_FIELDS)
        for (rname, alpha, f), by_method in cells.items():
            for mname, parts in by_method.items():
                for (k, _, _, test), p, fo in zip(sp, parts, folds):
                    a = p['a'].mean(axis=0); c = p['cost'].mean(axis=0); n_ = p['calls'].mean(axis=0); u = p['qprob'].mean(axis=0)
                    cid = p['config_id']
                    single = len(cid) == 1
                    cand = configs.get(cid[0]) if single and cid[0] in configs else None
                    rules = json.dumps(cand['rules']) if cand and 'rules' in cand else ''
                    wts = json.dumps([round(x, 12) for x in cand['weights']]) if cand and 'weights' in cand else ''
                    status = p['status'][0] if single else json.dumps(p['status'])
                    fb = p['fallback'][0] if single else round(float(np.mean(p['fallback'])), 6)
                    seed = p['seeds'][0] if single else 'mean'
                    cids = cid[0] if single else f'per_seed:{len(cid)}'
                    for j, i in enumerate(test):
                        r = rows[i]
                        w.writerow([ds, gen, r['item_id'], groups.get(r['item_id'], r['item_id']) if groups else r['item_id'], k,
                                    rname, alpha, f'frac:{f}', fo['budget_usd'][str(f)], mname, seed, cids, status, fb,
                                    int(bool(r['valid'])), '' if r['prior'] is None else r['prior'], round(float(prior_t[k][j]), 12),
                                    round(float(a[j]), 12), round(float(c[j]), 15), round(float(n_[j]), 12), round(float(u[j]), 12), rules, wts])
    # per-seed releases of seeded methods (compact)
    seeded = {}
    for (rname, alpha, f), by_method in cells.items():
        for mname in ('AT-Frozen', 'AT+Cal-Frozen'):
            seeded[f'{rname}|{alpha}|{f}|{mname}'] = np.concatenate([p['a'] for p in by_method[mname]], axis=1).astype(np.uint8)
    np.savez_compressed(out_dir/f'seed_releases_{ds}_{gen}.npz', **seeded)
    tpath = out_dir/f'rule_traces_{ds}_{gen}.jsonl.gz'
    with gzip.open(tpath, 'wt') as fh:
        for (k, order, rule), ((a, cost, asked, b), test) in sorted(traj.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
            for j, i in enumerate(test):
                r = rows[i]
                seq = [order[s] for s in range(len(order)) if asked[j, s]]
                fh.write(json.dumps({'dataset': ds, 'generator': gen, 'fold': k, 'order': list(order), 'rule': list(rule),
                                     'task_id': r['item_id'], 'queried': seq,
                                     'scores_read': [r['signals'][v]['score'] for v in seq],
                                     'call_usd': [r['signals'][v]['estimated_usd'] for v in seq],
                                     'stop': 'failure' if any(r['signals'][v]['score'] is None for v in seq) else
                                             ('ineligible' if not r['valid'] else ('end_of_order' if len(seq) == len(order) else 'policy_stop')),
                                     'final_belief': float(b[j]), 'release': int(a[j]), 'cost': float(cost[j])})+'\n')
    sel = {cid: c for cid, c in configs.items()}
    atomic_json(out_dir/f'selected_configs_{ds}_{gen}.json', clean(sel))
    return [dpath, tpath]


# ----------------------------------------------------------------------------- commands

def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get('schema') != 2 or cfg.get('kind') != 'offline_multirisk_cached':
        raise ValueError('not an offline multi-risk config')
    if cfg['execution'].get('network_allowed'):
        raise ValueError('this experiment is cache-only')
    return cfg


def source_files(ds):
    cfg = json.loads(Path(B.CONFIGS[ds]).read_text())
    files = [B.CONFIGS[ds]]
    if ds == 'gpqa':
        for study in B.GPQA_STUDIES.values():
            for part in ('calibration', 'evaluation'):
                files.append(str(Path(study)/part/'signals.json'))
    else:
        if 'split' in cfg:
            files.append(cfg['split'])
    if ds == 'qkatas':
        files.append('configs/quantum_katas_split_20261019.json')
    return [f for f in files if Path(f).exists()]


def preflight(cfg, out_dir):
    """Cache completeness, eligibility and label-source counts per dataset, generator and fold; input hashes."""
    rep = {'network_guard': block_network(), 'config_sha256': file_digest(CONFIG) if CONFIG.exists() else None,
           'git_head': git_head(), 'code_hash': code_hash(), 'datasets': {}}
    problems = []
    for ds in cfg['datasets']:
        data, _ = B.load(ds)
        groups = O.qk_groups() if ds == 'qkatas' else None
        d = {'sources': {f: file_digest(f) for f in source_files(ds)}, 'generators': {}}
        for gen in cfg['generators']:
            if gen not in data:
                problems.append(f'{ds}:{gen} missing'); continue
            rows = data[gen]
            g = {'n': len(rows), 'folds': {}}
            for k, fit, val, test in splits(ds, rows):
                if set(fit) & set(val) or set(fit) & set(test) or set(val) & set(test):
                    problems.append(f'{ds}:{gen}:fold{k} overlapping splits')
                if groups:
                    gp = {part: {groups.get(rows[i]['item_id'], rows[i]['item_id']) for i in ix} for part, ix in (('fit', fit), ('val', val), ('test', test))}
                    if gp['fit'] & gp['val'] or gp['fit'] & gp['test'] or gp['val'] & gp['test']:
                        problems.append(f'{ds}:{gen}:fold{k} kata group crosses splits')
                fd = {}
                for part, ix in (('fit', fit), ('validation', val), ('test', test)):
                    src = {}
                    for i in ix:
                        src[label_source(rows[i])] = src.get(label_source(rows[i]), 0)+1
                    fd[part] = {'n': len(ix), 'eligible': int(sum(rows[i]['valid'] for i in ix)),
                                'correct': int(sum(truth(rows, ix))), 'label_source': src,
                                'missing_verifier_scores_eligible': int(sum(rows[i]['signals'][v]['score'] is None for i in ix if rows[i]['valid'] for v in B.VERIFIERS)),
                                'generator_usd': float(sum(rows[i].get('generator_usd') or 0. for i in ix))}
                g['folds'][str(k)] = fd
            for r in rows:
                if r['valid'] and r['prior'] is None:
                    problems.append(f'{ds}:{gen}:{r["item_id"]} eligible without confidence')
                for v in B.VERIFIERS:
                    if r['signals'][v].get('estimated_usd') is None:
                        problems.append(f'{ds}:{gen}:{r["item_id"]}:{v} no cached price')
            d['generators'][gen] = g
        rep['datasets'][ds] = d
    rep['problems'] = problems
    rep['network_attempts'] = list(NETWORK_ATTEMPTS)
    rep['conditions'] = len(conditions(cfg))*len(cfg['datasets'])*len(cfg['generators'])
    atomic_json(Path(out_dir)/'preflight.json', rep)
    print('preflight:', 'ok' if not problems else problems, '| main conditions:', rep['conditions'])
    return rep


def run(cfg, out_dir, only=None, quick=False):
    guard = block_network()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    chash = code_hash()
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    for ds in cfg['datasets']:
        data = None
        for gen in cfg['generators']:
            if only and f'{ds}:{gen}' not in only and ds not in only:
                continue
            path = out_dir/f'{ds}_{gen}.json'
            if path.exists():
                prev = json.loads(path.read_text())
                if prev.get('code_hash') == chash and prev.get('config_hash') == cfg_hash and prev.get('complete'):
                    print('skip (hash match)', ds, gen, flush=True)
                    continue
            if data is None:
                data, _ = B.load(ds)
                groups = O.qk_groups() if ds == 'qkatas' else None
            t0 = time.time()
            res, cells, configs, traj = run_generator(ds, gen, data[gen], cfg, groups, quick)
            files = write_logs(out_dir, ds, gen, data[gen], splits(ds, data[gen]), res['folds'], cells, configs, traj, groups, cfg)
            res.update({'code_hash': chash, 'config_hash': cfg_hash, 'git_head': git_head(), 'network_guard': guard,
                        'network_attempts': list(NETWORK_ATTEMPTS), 'started': t0, 'finished': time.time(), 'complete': True,
                        'quick': quick, 'log_files': {str(f): file_digest(f) for f in files}})
            atomic_json(path, clean(res))
            print(ds, gen, 'done', round(res['seconds'], 1), 's', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['preflight', 'run', 'verify', 'report'])
    ap.add_argument('--config', default=str(CONFIG))
    ap.add_argument('--out')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out or cfg['output_dir'])
    if args.cmd == 'preflight':
        preflight(cfg, out)
    elif args.cmd == 'run':
        run(cfg, out, args.only, args.quick)
    elif args.cmd == 'verify':
        block_network()
        from vgx.offline_multirisk_verify import verify
        verify(cfg, out)
    else:
        block_network()
        from vgx.offline_multirisk_report import report
        report(cfg, out, Path(cfg['report_dir']) if not args.out else out/'report')


if __name__ == '__main__':
    main()
