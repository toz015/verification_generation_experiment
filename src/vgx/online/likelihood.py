"""Three-bin verifier observation models and their updates.

theta is an array g[j, y, b] = P(bin b of verifier j | Y = y), y = 0 incorrect, 1 correct.
Feasible set Theta: every g[j, y] is a strictly positive distribution over the 3 bins, and for the
designated audit verifier a the coded mean gap mu_1 - mu_0 = CODES.(g[a,1] - g[a,0]) >= GAMMA.
This fixes the orientation of the latent label and keeps the Z-hat denominator away from zero.

All fits are MAP with a symmetric Dirichlet(1 + LAPLACE) prior per (j, y), i.e. LAPLACE pseudo-counts
per cell (the existing fit_verifier_likelihood convention). Objectives below include that log prior;
they are not unpenalized maximum likelihood.

  labeled (theta_0, arm D):  maximize sum_s log g[., Y_s, A_s] + LAPLACE * sum log g   over Theta
  C1 (full-history EM):      maximize L_t(theta) + LAPLACE * sum log g, with
                             L_t = sum_s log[pi_s prod_j g[j,1,A_sj] + (1-pi_s) prod_j g[j,0,A_sj]],
                             pi_s = r_phi_t(U_s) fixed during EM, started at theta_t.
  C2 (recursive online EM):  S_n = (1-gamma_n) S_{n-1} + gamma_n s_n, gamma_n = 1/(N0+n);
                             M-step on counts (N0+n) S_n + LAPLACE over Theta.
With gamma_n = 1/(N0+n), (N0+n) S_n is the sum of the stored soft counts, so the C2 M-step has the
same penalized form as C1's; the difference is that C2's soft counts keep the weights computed when
each record arrived.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import brentq

from vgx.online.views import BINS, CODES

LAPLACE = 1.0
GAMMA = 0.1


def _solve_lambda(n, c, tau, sign):
    """Find lam with sum_b n_b / (lam - sign*tau*c_b) = 1 on the admissible branch."""
    lo = (sign*tau*c).max() + 1e-300
    f = lambda lam: float(np.sum(n/(lam - sign*tau*c))) - 1.
    hi = lo + n.sum() + 1.
    while f(hi) > 0:
        hi = lo + 2*(hi-lo)
    a = lo + 1e-12*(hi-lo)
    while f(a) < 0:  # move toward the pole until the sum exceeds one
        a = lo + (a-lo)*1e-3
    return brentq(f, a, hi, xtol=1e-15, rtol=1e-14, maxiter=500)


def constrained_pair(n1, n0, gamma=GAMMA, c=CODES):
    """argmax sum n1 log g1 + sum n0 log g0 over simplices with c.(g1-g0) >= gamma (exact KKT solve).

    n1, n0 are strictly positive (pseudo-counts included). Returns (g1, g0, active)."""
    n1, n0 = np.asarray(n1, float), np.asarray(n0, float)
    g1, g0 = n1/n1.sum(), n0/n0.sum()
    if c @ (g1-g0) >= gamma:
        return g1, g0, False

    def at(tau):
        l1 = _solve_lambda(n1, c, tau, +1)
        l0 = _solve_lambda(n0, c, tau, -1)
        return n1/(l1 - tau*c), n0/(l0 + tau*c)

    def gap(tau):
        a, b = at(tau)
        return float(c @ (a-b)) - gamma

    hi = 1.
    while gap(hi) < 0:
        hi *= 2
        if hi > 1e12:
            raise RuntimeError('mean-gap constraint infeasible')
    tau = brentq(gap, 0., hi, xtol=1e-14, rtol=1e-13, maxiter=500)
    a, b = at(tau)
    return a/a.sum(), b/b.sum(), True


def mstep(counts, audit_j, gamma=GAMMA, laplace=LAPLACE):
    """MAP M-step over Theta from (soft or hard) counts[j, y, b] (pseudo-counts added here)."""
    n = np.asarray(counts, float) + laplace
    g = n/n.sum(axis=2, keepdims=True)
    g1, g0, active = constrained_pair(n[audit_j, 1], n[audit_j, 0], gamma)
    g[audit_j, 1], g[audit_j, 0] = g1, g0
    return g, active


def log_prior(g, laplace=LAPLACE):
    return float(laplace*np.log(g).sum())


def audit_means(g, audit_j):
    return float(CODES @ g[audit_j, 0]), float(CODES @ g[audit_j, 1])


def hard_counts(bins, y, m):
    """bins: (n, m) int array of complete panels; y: (n,) 0/1."""
    counts = np.zeros((m, 2, BINS))
    for j in range(m):
        np.add.at(counts[j], (np.asarray(y, int), bins[:, j]), 1.)
    return counts


def soft_counts(bins, w, m):
    counts = np.zeros((m, 2, BINS))
    w = np.asarray(w, float)
    for j in range(m):
        np.add.at(counts[j, 1], bins[:, j], w)
        np.add.at(counts[j, 0], bins[:, j], 1-w)
    return counts


def panel_logs(g, bins):
    """log prod_j g[j,1,A_j] and log prod_j g[j,0,A_j] for each complete panel (rows of bins)."""
    bins = np.asarray(bins, int).reshape(-1, g.shape[0])
    lg = np.log(g)
    j = np.arange(g.shape[0])[None, :]
    return lg[j, 1, bins].sum(axis=1), lg[j, 0, bins].sum(axis=1)


def posterior(pi, bins, g):
    l1, l0 = panel_logs(g, np.asarray(bins, int).reshape(-1, g.shape[0]))
    pi = np.clip(np.asarray(pi, float), 1e-300, 1-1e-16)
    a, b = np.log(pi) + l1, np.log1p(-pi) + l0
    return 1/(1+np.exp(b-a))


def marginal_loglik(pi, bins, g):
    l1, l0 = panel_logs(g, bins)
    pi = np.clip(np.asarray(pi, float), 1e-300, 1-1e-16)
    return float(np.logaddexp(np.log(pi)+l1, np.log1p(-pi)+l0).sum())


def labeled_fit(bins, y, audit_j, gamma=GAMMA):
    g, active = mstep(hard_counts(np.asarray(bins, int), y, bins.shape[1] if len(bins) else 0), audit_j, gamma)
    return g, active


def full_history_em(pi, bins, g_start, audit_j, *, gamma=GAMMA, max_iter=500, tol=1e-9):
    """C1: EM on the fixed penalized marginal objective, mixing weights pi fixed, started at g_start."""
    t0 = time.perf_counter()
    bins = np.asarray(bins, int)
    m = g_start.shape[0]
    g = g_start.copy()
    obj = marginal_loglik(pi, bins, g) + log_prior(g)
    start_obj, violations, it, reason = obj, 0, 0, 'max_iter'
    for it in range(1, max_iter+1):
        w = posterior(pi, bins, g)
        g_new, _ = mstep(soft_counts(bins, w, m), audit_j, gamma)
        new = marginal_loglik(pi, bins, g_new) + log_prior(g_new)
        if not math.isfinite(new):
            reason = 'nonfinite_objective'
            break
        if new < obj - 1e-8*(1+abs(obj)):
            violations += 1
        g, done = g_new, abs(new-obj) <= tol*(1+abs(obj))
        obj = new
        if done:
            reason = 'converged'
            break
    return g, {'em_iterations': it, 'objective_start': start_obj, 'objective_end': obj, 'stop_reason': reason,
               'monotone_violations': violations, 'records': int(len(bins)), 'records_processed': int(len(bins)*it),
               'seconds': time.perf_counter()-t0}


class OnlineEM:
    """C2: recursive averaged sufficient statistics S[j, y, b] (soft counts per record)."""

    def __init__(self, pi0, bins0, g0, audit_j, gamma=GAMMA):
        bins0 = np.asarray(bins0, int)
        self.audit_j, self.gamma = audit_j, gamma
        self.N0 = int(len(bins0))
        if self.N0 == 0:
            raise ValueError('C2 needs at least one complete initialization panel')
        w = posterior(pi0, bins0, g0)
        self.S = soft_counts(bins0, w, g0.shape[0]) / self.N0
        self.n = 0

    def update(self, pi, bins_row, g_t):
        t0 = time.perf_counter()
        bins_row = np.asarray(bins_row, int).reshape(1, -1)
        w = posterior([pi], bins_row, g_t)
        s = soft_counts(bins_row, w, g_t.shape[0])
        self.n += 1
        step = 1./(self.N0 + self.n)
        self.S = (1-step)*self.S + step*s
        g, active = mstep((self.N0+self.n)*self.S, self.audit_j, self.gamma)
        return g, {'em_iterations': 1, 'records_processed': 1, 'records': 1, 'gamma_n': step,
                   'posterior_weight': float(w[0]), 'constraint_active': bool(active),
                   'seconds': time.perf_counter()-t0}

    def state(self):
        return {'S': self.S.tolist(), 'n': self.n, 'N0': self.N0}

    @classmethod
    def from_state(cls, st, audit_j, gamma=GAMMA):
        obj = cls.__new__(cls)
        obj.S, obj.n, obj.N0, obj.audit_j, obj.gamma = np.array(st['S']), st['n'], st['N0'], audit_j, gamma
        return obj
