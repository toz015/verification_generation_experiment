"""Offline frozen comparison (docs/offline-at-frozen-experiment-protocol-20261110.md): AT-Frozen,
AT+Calibration-Frozen and Ours-Offline on cached candidates (no API calls).

Each method is fit on Fit, picks one working point on Validation under the shared rule
  max coverage  s.t.  sum_i a_i (e_i - alpha_target) <= 0  and  mean screening cost <= B,
is frozen, and is scored once on Test. The risk is the selective (release) error throughout
(`risk_metric: selective_error`). Policies never see test labels: they get features through `Features`, and
labels reach only the Evaluator after all decisions are fixed. "Deferred" (not released) is not an audit:
no oracle label is acquired at test time; verifier queries are counted separately.

  python -m vgx.offline_frozen config     # write the frozen config
  python -m vgx.offline_frozen verify     # leakage, freezing, LP and cost checks on cached tasks
  python -m vgx.offline_frozen run [ds]   # full runs
  python -m vgx.offline_frozen report
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import vgx.baselines as B
from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa import fpr_routing as R
from vgx.gpqa.dependence_triples import all_orders
from vgx.gpqa.reporting_trial import isotonic
from vgx.selective_auditing import Oracle

# The release-error constraint for the shared LP / Bellman machinery, set from the config, not the environment.
R.CONSTRAINT = 'fdr'
R.SINGLE = False
# Planning grid on beliefs: 401 points instead of the prototype's 2001 (same logistic spacing), for run time.
R.GRID = np.r_[0., 1/(1+np.exp(-np.linspace(-14, 14, 399))), 1.]

CONFIG = Path('configs/offline_at_frozen_20261111.json')
OUT = Path('results/offline_at_frozen_20261111')
DATASETS = ('gpqa', 'math500', 'qkatas')
SETTINGS = {
    'risk_metric': 'selective_error',
    'alpha_target': [0.01, 0.05, 0.10], 'alpha_main': 0.05,
    'alpha_at': [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90],
    'gamma_fit': 0.10, 'gamma_test_main': 0.0,
    'eta': [0.001, 0.003, 0.01, 0.03, 0.10],
    'beta': [0.001, 0.01, 0.10, 1.00],
    'calibration_grid': [round(0.05*i, 10) for i in range(21)],
    'tau1': 0.5, 'fit_epochs': 1,
    'alpha_plan': [0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
    'budget_fractions': [0, 0.1, 0.25, 0.5, 0.75, 1.0],
    'gpqa_usd_budgets': [0.0005, 0.001, 0.002, 0.003, 0.006],
    'ours_orders_shortlist': 3,
    'seeds': list(range(20)), 'gpqa_split_seed': 20261111, 'gpqa_validation_share': 0.2,
    'bootstrap': 2000, 'bootstrap_seed': 20261111, 'mixture_draws': 20,
    'audit_price_ratios': [1, 5, 20],
}


# ----------------------------------------------------------------------------- data

class Features:
    """What a policy may read for one task set: validity, raw confidence and an accessor for verifier outputs.
    Labels are not here."""

    def __init__(self, rows, idx):
        self.idx = list(idx)
        self.valid = np.array([rows[i]['valid'] for i in idx])
        self.confidence = np.array([rows[i]['prior'] if rows[i]['valid'] else np.nan for i in idx], float)


def splits(ds, rows):
    """(outer fold, fit, validation, test) index lists; the same for every method."""
    if ds == 'gpqa':
        cal = [i for i, r in enumerate(rows) if r['fold'] == 1]
        test = [i for i, r in enumerate(rows) if r['fold'] == 0]
        perm = list(np.random.default_rng(SETTINGS['gpqa_split_seed']).permutation(cal))
        n_val = round(SETTINGS['gpqa_validation_share']*len(cal))
        return [(0, sorted(int(i) for i in perm[n_val:]), sorted(int(i) for i in perm[:n_val]), test)]
    out = []
    for k in range(5):
        v = (k+1) % 5
        out.append((k, [i for i, r in enumerate(rows) if r['fold'] not in (k, v)],
                    [i for i, r in enumerate(rows) if r['fold'] == v], [i for i, r in enumerate(rows) if r['fold'] == k]))
    return out


def labels(rows, idx):
    """Evaluator side only."""
    return np.array([1-rows[i]['outcome'] if rows[i]['valid'] else 1 for i in idx], int)


def c_ref(rows, fit):
    return float(np.mean([sum(rows[i]['signals'][v]['estimated_usd'] for v in B.VERIFIERS) for i in fit if rows[i]['valid']]))


def budgets(ds, cref):
    out = [('frac', f, f*cref) for f in SETTINGS['budget_fractions']]
    if ds == 'gpqa':
        out += [('usd', b, b) for b in SETTINGS['gpqa_usd_budgets']]
    return out


def qk_groups():
    """task id -> kata group as frozen in the split (its `groups` are already the merged final groups)."""
    s = json.loads(Path('configs/quantum_katas_split_20261019.json').read_text())
    if len(s['groups']) != s['final_groups']:
        raise ValueError('split groups are not the final merged groups')
    return {t: g for g, ts in s['groups'].items() for t in ts}


def state_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- AT and AT+CC on Fit (vectorised over configurations)

def at_fit(conf, errors, alpha, eta, gamma, seed):
    """One pass of eq. (1)-(2) over Fit in a seeded order, all configurations at once. Labels only via the oracle."""
    perm = np.random.default_rng(seed).permutation(len(conf))
    u = np.random.default_rng(10_000+seed).random(len(conf))
    K = len(alpha)
    oracle = Oracle(np.asarray(errors)[perm], K)
    tau = np.full(K, SETTINGS['tau1'])
    for t, j in enumerate(perm):
        pi = np.where(conf[j] < tau, 1.0, gamma)
        H = u[t] < pi
        e = oracle.audit(t, H)
        tau = tau+np.where(H, eta*(1-pi-alpha)*np.where(H, e, 0)/pi, 0.0)
    return tau, int(oracle.revealed.sum())


def cc_fit(conf, errors, alpha, eta, beta, gamma, grid, seed):
    """Algorithm 1 over Fit, all configurations at once; returns frozen log-weights and thresholds."""
    perm = np.random.default_rng(seed).permutation(len(conf))
    u = np.random.default_rng(10_000+seed).random(len(conf))
    theta = np.append(np.asarray(grid, float), np.inf)
    K = len(alpha)
    oracle = Oracle(np.asarray(errors)[perm], K)
    logw = np.full((K, len(theta)), -np.log(len(theta)))
    tau = np.full(K, SETTINGS['tau1'])
    for t, j in enumerate(perm):
        x = (conf[j] >= theta).astype(float)
        w = np.exp(logw-logw.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
        s = w@x
        pi = np.where(s < tau, 1.0, gamma)
        H = u[t] < pi
        e = oracle.audit(t, H)
        y = 1-np.where(H, e, 0)
        g = np.where(H, (s-y)/pi, 0.0)[:, None]*x[None, :]
        logw = logw-beta[:, None]*g
        m = logw.max(axis=1, keepdims=True)
        logw = logw-(m+np.log(np.exp(logw-m).sum(axis=1, keepdims=True)))
        tau = tau+np.where(H, eta*(1-pi-alpha)*np.where(H, e, 0)/pi, 0.0)
    return logw, tau, theta, int(oracle.revealed.sum())


def cc_score(logw, theta, conf):
    w = np.exp(logw-logw.max(axis=-1, keepdims=True)); w /= w.sum(axis=-1, keepdims=True)
    x = (np.nan_to_num(conf, nan=-1.0)[:, None] >= theta[None, :]).astype(float)
    return x@w.T                                             # (items, configs)


def at_candidates():
    out = []
    for a in SETTINGS['alpha_at']:
        for eta in SETTINGS['eta']:
            if SETTINGS['gamma_fit'] <= 1-a:
                out.append((a, eta))
    return out


def calibrated_rows(rows, fit):
    """A copy of rows whose confidence is Ours' isotonic map (fit on Fit labels only); for the AT-on-calibrated control."""
    fv = [i for i in fit if rows[i]['valid']]
    f, _ = isotonic(np.array([rows[i]['prior'] for i in fv], float), np.array([rows[i]['outcome'] for i in fv], float))
    return [dict(r, prior=float(f(r['prior']))) if r['valid'] else r for r in rows]


def threshold_methods(rows, fit, seed):
    """Frozen states of AT and AT+CC for every configuration, trained on Fit with one seed."""
    fv = [i for i in fit if rows[i]['valid']]
    conf = np.array([rows[i]['prior'] for i in fv], float)
    err = labels(rows, fv)
    cands = at_candidates()
    alpha = np.array([a for a, _ in cands]); eta = np.array([e for _, e in cands])
    tau, reveals_at = at_fit(conf, err, alpha, eta, SETTINGS['gamma_fit'], seed)
    cc_c = [(a, e, b) for a, e in cands for b in SETTINGS['beta']]
    logw, ctau, theta, reveals_cc = cc_fit(conf, err, np.array([c[0] for c in cc_c]), np.array([c[1] for c in cc_c]),
                                           np.array([c[2] for c in cc_c]), SETTINGS['gamma_fit'], SETTINGS['calibration_grid'], seed)
    return {'AT': {'configs': cands, 'tau': tau, 'fit_reveals': reveals_at},
            'AT+CC': {'configs': cc_c, 'tau': ctau, 'logw': logw, 'theta': theta, 'fit_reveals': reveals_cc}}


def threshold_decisions(method, state, feats, gamma_test, seed):
    """Release probability (configs x items) under the frozen state; gamma_test > 0 keeps the paper's random deferral
    in expectation (the sensitivity version)."""
    c = np.nan_to_num(feats.confidence, nan=-np.inf)
    if method == 'AT':
        score = np.repeat(c[None, :], len(state['tau']), axis=0)
    elif method == 'AT+CC':
        score = cc_score(state['logw'], state['theta'], feats.confidence).T
    else:  # raw threshold: state['tau'] is a list of thresholds on raw confidence
        score = np.repeat(c[None, :], len(state['tau']), axis=0)
    a = (score >= state['tau'][:, None]) & feats.valid[None, :]
    return a.astype(float)*(1-gamma_test), score


# ----------------------------------------------------------------------------- Ours-Offline

class Planner:
    """Model fit on Fit, planning over the Validation items' reports (no labels): the draft's model-based values."""

    def __init__(self, rows, fit, items, calibrate):
        self.models = [R.Model(rows, fit, items, {}, calibrate)]
        self.n = len(items)
        self.wrong = float((1-self.models[0].prior).sum())/self.n

    value = R.ModelDesign.value


def run_rule(model, order, rule):
    """Execute one rule on prepared items: release, cost, which verifiers were read, the final belief."""
    kind, omega, rho = rule
    n = len(model.prior)
    asked = np.zeros((n, len(order)), bool)
    if kind == 'release':
        return model.valid.astype(float), np.zeros(n), asked, model.prior.copy()
    if kind == 'abstain':
        return np.zeros(n), np.zeros(n), asked, model.prior.copy()
    tau = 1-1/omega
    margins = model.margins(order, omega, rho) if kind == 'bellman' else None
    b, cost = model.prior.copy(), np.zeros(n)
    alive, failed = model.valid.copy(), np.zeros(n, bool)
    for k, v in enumerate(order if kind != 'prior' else ()):
        ask = alive.copy() if kind == 'fixed' else alive & (np.interp(b, R.GRID, margins[k]) > 0)
        asked[:, k] = ask
        cost[ask] += model.usd[v][ask]
        x = model.bins[v]
        failed |= ask & (x < 0)
        step = ask & (x >= 0)
        lk = model.fitted['likelihoods'][v]
        p1, p0 = np.array(lk['p_bin_if_correct'])[x[step]], np.array(lk['p_bin_if_incorrect'])[x[step]]
        b[step] = b[step]*p1/(b[step]*p1+(1-b[step])*p0)
        alive = ask & ~failed
    return (model.valid & ~failed & (b >= tau)).astype(float), cost, asked, b


def ours_candidates(rows, fit, val, cref, calibrate):
    """Model-planned mixtures for a shortlist of verifier orders x alpha_plan x B_plan (Validation reports, no labels)."""
    plan = Planner(rows, fit, val, calibrate)
    base = [('abstain', 0., 0.), ('release', 0., 0.)]
    grid = base+[('bellman', o, r) for o in R.OMEGAS for r in R.RHOS]
    thin_rules = base+[('bellman', o, r) for o in R.OMEGAS[::3] for r in R.RHOS[::3]]
    bplans = [f*cref for f in SETTINGS['budget_fractions']]
    scored = []
    for order in all_orders(B.VERIFIERS):
        vals = [plan.value(order, r) for r in thin_rules]
        score = np.mean([R.solve(vals, a, b, plan.wrong)[1] for a in SETTINGS['alpha_plan'] for b in bplans])
        scored.append((score, order))
    scored.sort(key=lambda s: (-s[0], s[1]))
    out = []
    for _, order in scored[:SETTINGS['ours_orders_shortlist']]:
        vals = [plan.value(order, r) for r in grid]
        for ap in SETTINGS['alpha_plan']:
            for bf, bp in zip(SETTINGS['budget_fractions'], bplans):
                rules, w, pub, _, planned_cost = R.design(plan, order, grid, vals, ap, bp)
                keep = [(r, float(x)) for r, x in zip(rules, w) if x > 1e-9]
                out.append({'order': list(order), 'alpha_plan': ap, 'budget_plan_fraction': bf, 'rules': [list(r) for r, _ in keep],
                            'weights': [x for _, x in keep], 'planned_coverage': pub, 'planned_cost': planned_cost})
    return out


def ours_apply(rows, fit, items, cand, calibrate, draws=None):
    """Expected release / cost per item of a frozen mixture; with `draws`, also sampled executions (one rule per item)."""
    model = R.Model(rows, fit, items, {}, calibrate)
    order = tuple(cand['order'])
    per = [run_rule(model, order, tuple(r)) for r in cand['rules']]
    w = np.array(cand['weights'])/np.sum(cand['weights'])
    a = sum(wi*p[0] for wi, p in zip(w, per)); c = sum(wi*p[1] for wi, p in zip(w, per))
    calls = sum(wi*p[2].sum(axis=1) for wi, p in zip(w, per))
    out = {'a': a, 'cost': c, 'calls': calls, 'per_rule': per, 'model': model}
    if draws:
        out['draws'] = []
        for s in draws:
            pick = np.random.default_rng(s).choice(len(w), size=len(items), p=w)
            out['draws'].append({'a': np.array([per[k][0][i] for i, k in enumerate(pick)]),
                                 'cost': np.array([per[k][1][i] for i, k in enumerate(pick)]),
                                 'calls': np.array([per[k][2][i].sum() for i, k in enumerate(pick)]), 'rule': pick})
    return out


# ----------------------------------------------------------------------------- shared selection and evaluation

def select(A, C, err_val, alpha, budget, simple=None):
    """Index of the chosen candidate by the shared rule on Validation, or None (no_nontrivial_feasible_policy).
    A, C: candidates x validation items (release probability, cost). Ties: lower cost, then simpler, then lower index."""
    A, C = np.atleast_2d(A), np.atleast_2d(C)
    released = A.sum(axis=1)
    risk = (A*(err_val[None, :]-alpha)).sum(axis=1)
    cost = C.mean(axis=1)
    ok = (released > 1e-9) & (risk <= 1e-9) & (cost <= budget+1e-12)
    if not ok.any():
        return None
    simple = np.zeros(len(A)) if simple is None else np.asarray(simple, float)
    idx = np.flatnonzero(ok)
    order = np.lexsort((idx, simple[idx], cost[idx], -released[idx]))
    return int(idx[order[0]])


def evaluate(a, cost, err, valid, calls=None, external=0):
    """Evaluator: every count from the frozen decisions and the private labels."""
    a = np.asarray(a, float); d = 1-a; e = np.asarray(err, float)
    ev = valid.astype(float)
    n = len(a)
    rel = a.sum()
    out = {'n_tasks': n, 'n_valid': int(valid.sum()), 'n_released': float(rel), 'n_wrong_released': float((a*e).sum()),
           'release_error_rate': None if rel < 1e-9 else float((a*e).sum()/rel), 'coverage': float(rel/n),
           'defer_rate': float(d.sum()/n), 'mean_screen_cost': float(np.mean(cost)),
           'weak_verifier_calls_per_task': None if calls is None else float(np.mean(calls)),
           'verifier_query_rate': None if calls is None else float(np.mean(np.asarray(calls) > 0)),
           'external_labels_acquired': external}
    E = (e*ev).sum(); C = ((1-e)*ev).sum(); D = (d*ev).sum()
    out['error_capture_rate'] = None if E == 0 else float((d*e*ev).sum()/E)
    out['error_share_in_deferred'] = None if D == 0 else float((d*e*ev).sum()/D)
    out['correct_deferral_rate'] = None if C == 0 else float((d*(1-e)*ev).sum()/C)
    return out


# ----------------------------------------------------------------------------- one dataset

def run_dataset(ds, quick=False):
    data, _ = B.load(ds)
    groups = qk_groups() if ds == 'qkatas' else None
    seeds = SETTINGS['seeds'][:2] if quick else SETTINGS['seeds']
    res = {'dataset': ds, 'generators': {}}
    logs = []
    for gen, rows in data.items():
        t0 = time.perf_counter()
        sp = splits(ds, rows)
        G = {'splits': [{'fold': k, 'fit': len(f), 'validation': len(v), 'test': len(t)} for k, f, v, t in sp], 'cells': {}}
        pooled = {}   # (method, alpha, budget_key) -> lists over folds of per-item arrays
        fold_info = []
        for k, fit, val, test in sp:
            cref = c_ref(rows, fit)
            bud = budgets(ds, cref)
            fv, ft = Features(rows, val), Features(rows, test)
            ev, et = labels(rows, val), labels(rows, test)
            # --- threshold methods (AT, AT+CC, raw threshold), one frozen state per seed
            rows_cal = calibrated_rows(rows, fit)
            fvc, ftc = Features(rows_cal, val), Features(rows_cal, test)
            for s in seeds:
                st = threshold_methods(rows, fit, s)
                stc = threshold_methods(rows_cal, fit, s)
                raw = {'tau': np.append(np.unique(fv.confidence[fv.valid]), np.inf)}
                variants = [('AT-Frozen', 'AT', st['AT'], 0.0, fv, ft), ('AT+Cal-Frozen', 'AT+CC', st['AT+CC'], 0.0, fv, ft),
                            ('AT-Frozen (gamma_test=gamma_fit)', 'AT', st['AT'], SETTINGS['gamma_fit'], fv, ft),
                            ('AT+Cal-Frozen (gamma_test=gamma_fit)', 'AT+CC', st['AT+CC'], SETTINGS['gamma_fit'], fv, ft),
                            ('AT-Frozen (calibrated confidence)', 'AT', stc['AT'], 0.0, fvc, ftc)]
                if s == seeds[0]:
                    variants.append(('Raw-threshold', 'raw', raw, 0.0, fv, ft))
                for name, kind, state, gt, fv_, ft_ in variants:
                    h0 = state_hash({k_: np.asarray(v_).tolist() for k_, v_ in state.items() if k_ != 'configs'})
                    av, _ = threshold_decisions(kind, state, fv_, gt, s)
                    at_, score = threshold_decisions(kind, state, ft_, gt, s)
                    if state_hash({k_: np.asarray(v_).tolist() for k_, v_ in state.items() if k_ != 'configs'}) != h0:
                        raise AssertionError('state changed during evaluation')
                    for a_t in SETTINGS['alpha_target']:
                        for bk, bv, budget in bud:
                            j = select(av, np.zeros_like(av), ev, a_t, budget)
                            arr = np.zeros(len(test)) if j is None else at_[j]
                            key = (name, a_t, f'{bk}:{bv}')
                            pooled.setdefault(key, {}).setdefault(s, []).append(
                                {'a': arr, 'cost': np.zeros(len(test)), 'calls': np.zeros(len(test)), 'err': et, 'valid': ft.valid,
                                 'idx': test, 'feasible': j is not None, 'config': None if j is None else (float(state['tau'][j]) if kind == 'raw' else list(state['configs'][j])),
                                 'brier': None if kind == 'raw' else float(np.mean((np.clip(score[0 if j is None else j][ft.valid], 0, 1)-(1-et[ft.valid]))**2)),
                                 'hash': h0, 'fit_reveals': state.get('fit_reveals', 0)})
            # --- Ours-Offline (calibrated prior) and the raw-prior variant
            for name, calibrate in (('Ours-Offline', True), ('Ours-Offline (raw prior)', False)):
                cands = ours_candidates(rows, fit, val, cref, calibrate)
                h0 = state_hash(cands)
                vals = [ours_apply(rows, fit, val, c, calibrate) for c in cands]
                applied = {}
                for a_t in SETTINGS['alpha_target']:
                    for bk, bv, budget in bud:
                        j = select(np.stack([v['a'] for v in vals]), np.stack([v['cost'] for v in vals]), ev, a_t, budget,
                                   [len(c['rules']) for c in cands])
                        key = (name, a_t, f'{bk}:{bv}')
                        if j is None:
                            item = {'a': np.zeros(len(test)), 'cost': np.zeros(len(test)), 'calls': np.zeros(len(test)), 'draws': None}
                        else:
                            if j not in applied:
                                applied[j] = ours_apply(rows, fit, test, cands[j], calibrate, draws=range(SETTINGS['mixture_draws']))
                            item = applied[j]
                        pooled.setdefault(key, {}).setdefault(0, []).append(
                            {'a': item['a'], 'cost': item['cost'], 'calls': item['calls'], 'err': et, 'valid': ft.valid, 'idx': test,
                             'feasible': j is not None, 'config': None if j is None else {k_: cands[j][k_] for k_ in ('order', 'alpha_plan', 'budget_plan_fraction', 'weights')},
                             'draws': item.get('draws'), 'hash': h0,
                             'brier': None if j is None else float(np.mean((applied[j]['model'].prior[ft.valid]-(1-et[ft.valid]))**2))})
                if state_hash(cands) != h0:
                    raise AssertionError('Ours state changed during evaluation')
            fold_info.append({'fold': k, 'c_ref': cref})
        G['folds'] = fold_info
        # --- pool folds and summarise
        for (name, a_t, bkey), by_seed in pooled.items():
            runs = []
            for s, parts in by_seed.items():
                a = np.concatenate([p['a'] for p in parts]); c = np.concatenate([p['cost'] for p in parts])
                calls = np.concatenate([p['calls'] for p in parts]); e = np.concatenate([p['err'] for p in parts])
                valid = np.concatenate([p['valid'] for p in parts]); idx = [i for p in parts for i in p['idx']]
                m = evaluate(a, c, e, valid, calls)
                m.update({'seed': s, 'infeasible_folds': sum(not p['feasible'] for p in parts),
                          'configs': [p['config'] for p in parts], 'brier': [p['brier'] for p in parts], 'hashes': [p['hash'] for p in parts]})
                if parts[0].get('draws') is not None or any(p.get('draws') for p in parts):
                    sampled = []
                    for dnum in range(SETTINGS['mixture_draws']):
                        da = np.concatenate([p['draws'][dnum]['a'] if p.get('draws') else np.zeros(len(p['idx'])) for p in parts])
                        dc = np.concatenate([p['draws'][dnum]['cost'] if p.get('draws') else np.zeros(len(p['idx'])) for p in parts])
                        dk = np.concatenate([p['draws'][dnum]['calls'] if p.get('draws') else np.zeros(len(p['idx'])) for p in parts])
                        sampled.append(evaluate(da, dc, e, valid, dk))
                    m['sampled_runs'] = {k_: [x[k_] for x in sampled] for k_ in ('n_released', 'n_wrong_released', 'release_error_rate', 'mean_screen_cost')}
                runs.append((m, a, c, e, valid, idx))
            G['cells'].setdefault(str(a_t), {}).setdefault(bkey, {})[name] = {'runs': [r[0] for r in runs]}
            G['cells'][str(a_t)][bkey][name]['_items'] = (np.mean([r[1] for r in runs], axis=0), runs[0][3], runs[0][4], runs[0][5],
                                                          np.mean([r[2] for r in runs], axis=0))
        # --- paired bootstrap of Ours minus each threshold method (seed-averaged release per item)
        rng_seed = SETTINGS['bootstrap_seed']
        for a_t, row in G['cells'].items():
            for bkey, cell in row.items():
                if 'Ours-Offline' not in cell:
                    continue
                ao, e, valid, idx, _ = cell['Ours-Offline']['_items']
                unit = np.array([groups.get(rows[i]['item_id'], rows[i]['item_id']) for i in idx]) if groups else np.array(idx)
                keys, inv = np.unique(unit, return_inverse=True)
                rng = np.random.default_rng(rng_seed)
                Wt = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)) for _ in range(SETTINGS['bootstrap'])])
                gn = np.bincount(inv, minlength=len(keys)).astype(float)
                gs = lambda v: np.bincount(inv, weights=v, minlength=len(keys))
                cov = lambda a: (Wt@gs(a))/(Wt@gn)
                rer = lambda a: (Wt@gs(a*e))/np.maximum(Wt@gs(a), 1e-9)
                cell['paired_vs_ours'] = {}
                for name, other in cell.items():
                    if name in ('Ours-Offline', 'paired_vs_ours') or '_items' not in other:
                        continue
                    ab = other['_items'][0]
                    dc, dr = cov(ao)-cov(ab), rer(ao)-rer(ab)
                    cell['paired_vs_ours'][name] = {'coverage_diff': [float(np.mean(dc)), *np.percentile(dc, [2.5, 97.5]).tolist()],
                                                    'release_error_diff': [float(np.mean(dr)), *np.percentile(dr, [2.5, 97.5]).tolist()],
                                                    'unit': 'merged kata group' if groups else 'question'}
        # --- decision log for the main seed
        for a_t, row in G['cells'].items():
            for bkey, cell in row.items():
                for name, x in cell.items():
                    if '_items' not in x:
                        continue
                    a, e, valid, idx, c = x['_items']
                    for j, i in enumerate(idx):
                        logs.append({'dataset': ds, 'generator': gen, 'task_id': rows[i]['item_id'], 'fold': int(rows[i]['fold']),
                                     'method': name, 'alpha_target': float(a_t), 'budget': bkey,
                                     'raw_confidence': rows[i]['prior'], 'release_probability': float(a[j]), 'screen_cost': float(c[j]),
                                     'external_audit_count': 0})
                    x.pop('_items')
        G['seconds'] = time.perf_counter()-t0
        res['generators'][gen] = G
        print(ds, gen, 'done', round(G['seconds'], 1), 's', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/f'{ds}{"_quick" if quick else ""}.json', res)
    with gzip.open(OUT/f'{ds}{"_quick" if quick else ""}_decisions.jsonl.gz', 'wt') as f:
        for r in logs:
            f.write(json.dumps(r)+'\n')
    return res


def write_config():
    cfg = {'schema': 1, 'kind': 'offline_frozen_comparison', 'protocol': 'docs/offline-at-frozen-experiment-protocol-20261110.md',
           'paper': 'arXiv:2610.31662v1 (offline adaptation; the paper\'s online guarantees do not apply)',
           'settings': SETTINGS, 'datasets': list(DATASETS),
           'notes': ['Settings are this project\'s choices, not the paper\'s defaults.',
                     'Ours: Model (isotonic PAV on Fit labels = Brier-optimal monotone fit; verifier likelihoods with Laplace 1, 5 bins) fit on Fit; model-planned mixtures built on Validation reports without labels; working point chosen on Validation with labels by the shared rule.',
                     'Verifier order shortlist: top 3 of 60 by label-free model-planned coverage averaged over alpha_plan x B_plan.',
                     'Invalid candidates (no answer or no confidence; never label-based) are deferred by every method and counted in n.']}
    atomic_json(CONFIG, cfg)
    print('wrote', CONFIG, file_digest(CONFIG))


def main():
    cmd = sys.argv[1]
    if cmd == 'config':
        write_config()
    elif cmd == 'run':
        for ds in (sys.argv[2:] or DATASETS):
            run_dataset(ds)
    elif cmd == 'quick':
        run_dataset(sys.argv[2], quick=True)


if __name__ == '__main__':
    main()
