"""One sequential replay run: labeled initialization on D_init, then online learning on the stream.

Arms (all share the initialization, hierarchy, costs, design set, rule draws and audit draws):
  A    static: phi, theta and routing design frozen; no learning audits.
  B    phi updated from noise-corrected feedback Z-hat; theta fixed at theta_0.
  C1   phi updated as in B; theta by full-history EM (history recomputed with the round-start phi).
  C2   phi updated as in B; theta by recursive online EM (stored soft counts never recomputed).
  D    labeled reference: same audits, the true label Y replaces Z-hat; theta from labeled counts.
  C1F, C2F  diagnostics: phi frozen at phi_0, theta updated as in C1 / C2.

Settings.likelihood = 'dependence' replaces the routing likelihood by a joint table over the hierarchy
(vgx.online.joint); arms A, B, C2 and D only. theta is then the pair (g, T).

Round t, with (phi_t, theta_t, R_t) fixed at the start:
  1 read the fixed candidate's public fields; 2 p_hat = r_phi_t(r); 3 draw a rule from R_t with u_rule[t];
  4 NestedVerify (scores read only when queried); 5 the decision, belief and cost are fixed and logged;
  6 audit iff u_audit[t] < q (independent of everything above); 7 an audit completes the panel, reusing
  outputs already read; 8 updates use phi_t and theta_t only, then R_{t+1} is redesigned on D_design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import json
import time

import numpy as np

from vgx.online import calibrator as cal
from vgx.online import joint as jm
from vgx.online import likelihood as lk
from vgx.online.routing import design_routing, nested_verify, pick_rule
from vgx.online.views import CODES, InitLabels, LabelGate, ScoreSource

ARMS = ('A', 'B', 'C1', 'C2', 'D', 'C1F', 'C2F')
UPDATES_PHI = {'A': None, 'B': 'zhat', 'C1': 'zhat', 'C2': 'zhat', 'D': 'label', 'C1F': None, 'C2F': None}
UPDATES_THETA = {'A': None, 'B': None, 'C1': 'em_full', 'C2': 'em_online', 'D': 'labeled',
                 'C1F': 'em_full', 'C2F': 'em_online'}


@dataclass
class Settings:
    alpha: float = 0.05
    budget_fraction: float = 0.25
    eta0: float = 0.05
    gamma: float = lk.GAMMA
    em_max_iter: int = 500
    em_tol: float = 1e-9
    hierarchy_size: int = 3
    likelihood: str = 'independence'          # or 'dependence'
    alpha_grid: tuple = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256)
    inner_folds: int = 5
    inner_seed: int = 20261109


def roles(items, rotation, folds=5):
    init_fold, design_fold = rotation % folds, (rotation+1) % folds
    init = [it for it in items if it.fold == init_fold]
    design = [it for it in items if it.fold == design_fold]
    stream = [it for it in items if it.fold not in (init_fold, design_fold)]
    return init, design, stream


def draws(master_seed, rotation, order, n_stream):
    """Stream permutation and the common random numbers for rule and audit draws."""
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, rotation, order]))
    perm = rng.permutation(n_stream)
    return {'perm': perm.tolist(), 'u_rule': rng.random(n_stream).tolist(), 'u_audit': rng.random(n_stream).tolist()}


def _panel(source, task_id, verifiers, purpose):
    acc = source.open(task_id, purpose)
    try:
        return [acc.read(v) for v in verifiers]
    finally:
        acc.close()


@dataclass
class Init:
    verifiers: tuple
    phi0: np.ndarray
    theta0: np.ndarray
    audit_j: int
    hierarchy: tuple
    costs: np.ndarray
    budget: float
    init_z: np.ndarray       # complete-panel init records (unlabeled use in C1/C2)
    init_bins: np.ndarray
    init_y: np.ndarray       # kept only for arm D (labeled reference)
    design_z: np.ndarray     # nan for non-processable design tasks
    info: dict = field(default_factory=dict)
    T0: np.ndarray | None = None     # joint table over the hierarchy (dependence model only)
    joint_alpha: float = jm.INF


def initialize(init_items, design_items, source: ScoreSource, init_labels: InitLabels, st: Settings) -> Init:
    """Everything here uses D_init (labels allowed) and the public fields of D_design only."""
    V = source.verifiers
    proc = [it for it in init_items if it.processable]
    z = cal.feature([it.raw_r for it in proc])
    y = np.array([init_labels.outcome(it.task_id) for it in proc], int)
    phi0, phi_info = cal.fit_brier(z, y)
    panels = [_panel(source, it.task_id, V, 'init') for it in proc]
    costs = np.array([np.mean([p[j]['usd'] for p in panels]) if panels else 0. for j in range(len(V))])
    complete = [i for i, p in enumerate(panels) if all(o['bin'] is not None for o in p)]
    bins = np.array([[panels[i][j]['bin'] for j in range(len(V))] for i in complete], int).reshape(-1, len(V))
    yc, zc = y[complete], z[complete]
    # Audit verifier: smallest estimated Var(Z-hat) = E[Var(A|Y)] / gap^2 under the unconstrained labeled fit.
    g_free, _ = lk.labeled_fit(bins, yc, 0, gamma=-np.inf)
    pbar = float(np.mean(cal.predict(phi0, z))) if len(z) else .5
    cand = []
    for j in range(len(V)):
        m0, m1 = CODES @ g_free[j, 0], CODES @ g_free[j, 1]
        v0, v1 = CODES**2 @ g_free[j, 0] - m0**2, CODES**2 @ g_free[j, 1] - m1**2
        gap = m1 - m0
        cand.append({'verifier': V[j], 'gap': float(gap), 'zhat_var_proxy':
                     float((pbar*v1 + (1-pbar)*v0)/gap**2) if gap > 0 else None})
    ok = [j for j in range(len(V)) if cand[j]['gap'] >= st.gamma]
    # gap >= gamma > 0 for every candidate in ok, so the proxy is finite there
    audit_j = min(ok, key=lambda j: (cand[j]['zhat_var_proxy'], j)) if ok else max(range(len(V)), key=lambda j: (cand[j]['gap'], -j))
    theta0, active = lk.labeled_fit(bins, yc, audit_j, gamma=st.gamma)
    budget = st.budget_fraction*float(costs.sum())
    pi_init = np.full(len(init_items), np.nan)
    for i, it in enumerate(init_items):
        if it.processable:
            pi_init[i] = float(cal.predict(phi0, cal.feature(it.raw_r)))
    dep = st.likelihood == 'dependence'
    if st.likelihood not in ('independence', 'dependence'):
        raise ValueError(st.likelihood)
    alphas, alpha_info = {}, {}
    if dep:  # alpha per verifier set, D_init labels only
        grid = [float(a) for a in st.alpha_grid] + [jm.INF]
        for combo in sorted({tuple(sorted(o)) for o in permutations(range(len(V)), st.hierarchy_size)}):
            a, sc = jm.choose_alpha(bins, yc, combo, audit_j, grid, st.inner_folds,
                                    np.random.SeedSequence([st.inner_seed, *combo]), st.gamma)
            alphas[combo] = a
            alpha_info['+'.join(V[j] for j in combo)] = {'chosen': 'inf' if np.isinf(a) else a, 'inner_cv_loglik': sc}

    def lik_for(order):
        if not dep:
            return theta0[list(order)]
        return jm.JointLik(jm.joint_table(theta0[list(order)], jm.hard_cells(bins[:, list(order)], yc),
                                          alphas[tuple(sorted(order))]))
    ranked = []
    for order in permutations(range(len(V)), st.hierarchy_size):
        d = design_routing(pi_init, lik_for(order), costs[list(order)], st.alpha, budget)
        ranked.append((-d['fitted_pub'], order, d['fitted_pub']))
    ranked.sort(key=lambda r: (r[0], r[1]))  # ties: lexicographic in configured verifier order
    hierarchy = ranked[0][1]
    T0, jalpha = None, jm.INF
    if dep:
        jalpha = alphas[tuple(sorted(hierarchy))]
        T0 = jm.joint_table(theta0[list(hierarchy)], jm.hard_cells(bins[:, list(hierarchy)], yc), jalpha)
    design_z = np.array([cal.feature(it.raw_r) if it.processable else np.nan for it in design_items])
    info = {'phi0': phi0.tolist(), 'phi_fit': phi_info, 'audit_candidates': cand, 'audit_verifier': V[audit_j],
            'theta0_constraint_active': bool(active), 'costs_usd': dict(zip(V, costs.tolist())),
            'full_panel_cost_usd': float(costs.sum()), 'budget_usd': budget,
            'hierarchy': [V[j] for j in hierarchy], 'hierarchy_top5': [[[V[j] for j in r[1]], r[2]] for r in ranked[:5]],
            'init_processable': len(proc), 'init_complete_panels': len(complete),
            'init_correct': int(yc.sum()), 'init_classes_present': sorted(set(yc.tolist())),
            'likelihood': st.likelihood}
    if dep:
        info['joint_alpha_by_set'] = alpha_info
        info['joint_alpha'] = 'inf' if np.isinf(jalpha) else jalpha
    return Init(V, phi0, theta0, audit_j, tuple(hierarchy), costs, budget, zc, bins, yc, design_z, info, T0, jalpha)


class Run:
    def __init__(self, arm, q, init: Init, stream, source: ScoreSource, rand, st: Settings, gate: LabelGate | None = None):
        if arm not in ARMS:
            raise ValueError(arm)
        if arm == 'D' and gate is None:
            raise ValueError('arm D needs a label gate')
        if arm != 'D' and gate is not None:
            raise ValueError('only arm D may hold a label gate')
        self.arm, self.q, self.init, self.source, self.st, self.gate = arm, q, init, source, st, gate
        self.dep = init.T0 is not None
        if self.dep and arm not in ('A', 'B', 'C2', 'D'):
            raise ValueError(f'arm {arm} is not implemented for the dependence model')
        self.stream = [stream[i] for i in rand['perm']]
        self.u_rule, self.u_audit = rand['u_rule'], rand['u_audit']
        self.hier = list(init.hierarchy)
        self.d = init.costs[self.hier]
        self.t = 0
        self.phi = init.phi0.copy()
        self.theta = init.theta0.copy()
        self.T = None if init.T0 is None else init.T0.copy()
        self.n_phi = 0
        self.hist_z = [float(v) for v in init.init_z]
        self.hist_bins = [list(map(int, r)) for r in init.init_bins]
        self.c2 = None
        if UPDATES_THETA[arm] == 'em_online':
            if self.dep:
                self.c2 = jm.JointOnlineEM(cal.predict(init.phi0, init.init_z), init.init_bins, init.theta0, init.T0,
                                           self.hier, init.joint_alpha, init.audit_j, st.gamma)
            else:
                self.c2 = lk.OnlineEM(cal.predict(init.phi0, init.init_z), init.init_bins, init.theta0, init.audit_j,
                                      st.gamma)
        self.d_counts = lk.hard_counts(init.init_bins, init.init_y, len(init.verifiers)) if arm == 'D' else None
        self.d_cells = (jm.hard_cells(init.init_bins[:, self.hier], init.init_y) if arm == 'D' and self.dep else None)
        self.design = self._design()
        self.design_seconds_initial = self.design['seconds']
        self.log = []

    # ---- helpers -------------------------------------------------------------------------------------
    def _lik(self, theta, T):
        return jm.JointLik(T) if T is not None else theta[self.hier]

    def _design(self):
        pi = np.where(np.isfinite(self.init.design_z), cal.predict(self.phi, np.nan_to_num(self.init.design_z)), np.nan)
        return design_routing(pi, self._lik(self.theta, self.T), self.d, self.st.alpha, self.init.budget)

    def eta(self):
        return self.st.eta0/np.sqrt(1. + self.n_phi)

    # ---- one round -----------------------------------------------------------------------------------
    def step(self):
        t, item = self.t, self.stream[self.t]
        V, aj = self.init.verifiers, self.init.audit_j
        phi_t, theta_t = self.phi.copy(), self.theta.copy()
        T_t = None if self.T is None else self.T.copy()
        rec = {'t': t, 'task_id': item.task_id, 'group': item.group, 'fold': item.fold, 'arm': self.arm, 'q': self.q,
               'processable': item.processable, 'phi_start': phi_t.tolist(), 'theta_start': theta_t.round(10).tolist(),
               'design_fitted': {k: self.design[k] for k in ('fitted_pub', 'fitted_risk', 'fitted_cost', 'omega', 'rho')},
               'audited': False, 'audit_cost_usd': 0., 'phi_update': None, 'theta_update': None}
        if T_t is not None:
            rec['joint_start'] = T_t.round(10).tolist()
        acc = self.source.open(item.task_id, 'round')
        try:
            if not item.processable:
                rec.update({'raw_r': item.raw_r, 'p_hat': None, 'action': 'abstain', 'reason': 'not_processable',
                            'decision_reads': [], 'decision_cost_usd': 0., 'belief': None, 'rule': None})
            else:
                z = float(cal.feature(item.raw_r))
                p_hat = float(cal.predict(phi_t, z))
                rule = pick_rule(self.design, self.u_rule[t])

                def read(k):
                    return acc.read(V[self.hier[k]])['bin']
                res = nested_verify(p_hat, rule, self._lik(theta_t, T_t), self.d, self.st.alpha, read)
                reads = list(acc.reads)
                dec_cost = float(sum(acc.read(v)['usd'] for v in reads))
                rec.update({'raw_r': item.raw_r, 'z': z, 'p_hat': p_hat, 'action': res['action'],
                            'reason': res['failure'], 'decision_reads': reads, 'decision_cost_usd': dec_cost,
                            'belief': res['belief'], 'layers_used': res['layers_used'],
                            'rule': {k: rule[k] for k in ('kind', 'omega', 'rho')}})
                rec['decision_fixed'] = True  # nothing below changes the fields above
                if UPDATES_PHI[self.arm] is not None or UPDATES_THETA[self.arm] is not None:
                    if self.u_audit[t] < self.q:
                        self._audit(rec, acc, z, phi_t, theta_t, T_t)
            rec['reads_all'] = list(acc.reads)
        finally:
            acc.close()
            if self.gate is not None:
                self.gate.revoke()
        self.log.append(rec)
        self.t += 1
        return rec

    def _audit(self, rec, acc, z, phi_t, theta_t, T_t=None):
        V, aj = self.init.verifiers, self.init.audit_j
        before = set(acc.reads)
        panel = [acc.read(v) for v in V]  # reuse outputs already read; read the rest once
        rec['audited'] = True
        rec['audit_new_reads'] = [v for v in V if v not in before]
        rec['audit_cost_usd'] = float(sum(acc.read(v)['usd'] for v in rec['audit_new_reads']))
        bins = [o['bin'] for o in panel]
        complete = all(b is not None for b in bins)
        rec['audit_panel'] = bins
        changed = False
        # --- confidence update (uses phi_t and theta_t only) ---
        kind = UPDATES_PHI[self.arm]
        target = None
        if kind == 'zhat':
            if bins[aj] is not None:
                mu0, mu1 = lk.audit_means(theta_t, aj)
                target = (CODES[bins[aj]] - mu0)/(mu1 - mu0)
                rec['zhat'] = float(target)
                rec['audit_mu'] = [mu0, mu1]
        elif kind == 'label':
            self.gate.allow(rec['task_id'])
            target = self.gate.outcome(rec['task_id'])
        if kind is not None:
            if target is None:
                rec['phi_update'] = {'status': 'skipped_missing_audit_score'}
            else:
                eta = self.eta()
                new, flag = cal.step(phi_t, z, target, eta, self.q)
                rec['phi_update'] = {'status': flag or 'ok', 'eta': float(eta), 'target': float(target)}
                if flag != 'nonfinite_step_skipped':
                    self.phi, self.n_phi, changed = new, self.n_phi + 1, True
        # --- likelihood update ---
        tk = UPDATES_THETA[self.arm]
        if tk is not None:
            if not complete:
                rec['theta_update'] = {'status': 'skipped_incomplete_panel'}
            elif tk == 'em_full':
                self.hist_z.append(z)
                self.hist_bins.append(bins)
                pis = cal.predict(phi_t, np.array(self.hist_z))
                g, info = lk.full_history_em(pis, np.array(self.hist_bins), theta_t, aj, gamma=self.st.gamma,
                                             max_iter=self.st.em_max_iter, tol=self.st.em_tol)
                rec['theta_update'] = {'status': info['stop_reason'], **info}
                self.theta, changed = g, True
            elif tk == 'em_online':
                if self.dep:
                    g, self.T, info = self.c2.update(float(cal.predict(phi_t, z)), bins, theta_t, T_t)
                else:
                    g, info = self.c2.update(float(cal.predict(phi_t, z)), bins, theta_t)
                rec['theta_update'] = {'status': 'ok', **info}
                self.theta, changed = g, True
            elif tk == 'labeled':
                t0 = time.perf_counter()
                self.d_counts += lk.hard_counts(np.array([bins]), [target], len(V))
                g, active = lk.mstep(self.d_counts, aj, self.st.gamma)
                if self.dep:
                    self.d_cells += jm.hard_cells(np.array([[bins[h] for h in self.hier]]), [target])
                    self.T = jm.joint_table(g[self.hier], self.d_cells, self.init.joint_alpha)
                rec['theta_update'] = {'status': 'ok', 'constraint_active': bool(active), 'records_processed': 1,
                                       'em_iterations': 0, 'seconds': time.perf_counter()-t0}
                self.theta, changed = g, True
        if changed:
            self.design = self._design()
            rec['redesign_seconds'] = self.design['seconds']
            rec['redesign_rounds'] = self.design['rounds']
            rec['redesign_status'] = self.design['status']
        rec['state_bytes'] = {'c1_history': len(json.dumps([self.hist_z, self.hist_bins])) if tk == 'em_full' else 0,
                              'c2_stats': len(json.dumps(self.c2.state())) if self.c2 is not None else 0}

    def run(self, until=None):
        end = len(self.stream) if until is None else min(until, len(self.stream))
        while self.t < end:
            self.step()
        return self.log

    # ---- checkpointing -------------------------------------------------------------------------------
    def state(self):
        return json.loads(json.dumps({
            't': self.t, 'phi': self.phi.tolist(), 'theta': self.theta.tolist(), 'n_phi': self.n_phi,
            'T': None if self.T is None else self.T.tolist(),
            'd_cells': None if self.d_cells is None else self.d_cells.tolist(),
            'hist_z': self.hist_z, 'hist_bins': self.hist_bins, 'c2': self.c2.state() if self.c2 else None,
            'd_counts': self.d_counts.tolist() if self.d_counts is not None else None, 'log': self.log}))

    def restore(self, s):
        self.t, self.phi, self.theta, self.n_phi = s['t'], np.array(s['phi']), np.array(s['theta']), s['n_phi']
        self.hist_z, self.hist_bins = list(s['hist_z']), [list(r) for r in s['hist_bins']]
        self.T = None if s.get('T') is None else np.array(s['T'])
        self.d_cells = None if s.get('d_cells') is None else np.array(s['d_cells'])
        if s['c2'] is not None:
            if self.dep:
                self.c2 = jm.JointOnlineEM.from_state(s['c2'], self.hier, self.init.joint_alpha, self.init.audit_j,
                                                      self.st.gamma)
            else:
                self.c2 = lk.OnlineEM.from_state(s['c2'], self.init.audit_j, self.st.gamma)
        if s['d_counts'] is not None:
            self.d_counts = np.array(s['d_counts'])
        self.log = list(s['log'])
        self.design = self._design()
