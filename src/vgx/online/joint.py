"""Dependence-aware observation model for the online replay (joint table over the routing hierarchy).

The independence model (vgx.online.likelihood) multiplies per-verifier bin likelihoods g[j, y, b].
Here the K hierarchy verifiers are modelled jointly per class, as in vgx.gpqa.dependence_triples:

  T[y] = (C[y] + alpha * prod_k g[h_k, y]) / (C[y].sum() + alpha),

C[y] = (hard or soft) counts of the K-bin cells, g = the usual MAP marginals (audit mean-gap
constraint included), alpha = Dirichlet shrinkage weight toward the independence product. alpha = inf
is exactly the independence model. Verifiers outside the hierarchy keep their marginal g and are
treated as independent of the hierarchy given Y; they enter only the full-panel posterior (EM
E-step, evaluator), never routing.

Z-hat still uses the audit verifier's constrained marginal g[a] (so B's confidence path is the same
under both models; only routing and the evaluator's panel posterior change).

alpha is chosen once per rotation on D_init only (labels allowed there): inner K-fold cross-validated
class-conditional log-likelihood of held-out cells; inf is a candidate; ties go to the larger alpha.
"""
from __future__ import annotations

from functools import reduce
import math
import time

import numpy as np

from vgx.online import likelihood as lk
from vgx.online.views import BINS

INF = math.inf


class IndepLik:
    """Routing likelihood under conditional independence: layer k bins ~ G[k, y]."""

    def __init__(self, G):
        self.G = np.asarray(G, float)
        self.K = self.G.shape[0]

    def cond(self, k, hist):
        return self.G[k, 1], self.G[k, 0]


class JointLik:
    """Routing likelihood from a joint table T[y] over the K hierarchy bins (axes in hierarchy order)."""

    def __init__(self, T):
        self.T = np.asarray(T, float)
        self.K = self.T.ndim - 1
        self._cache = {}

    def cond(self, k, hist):
        key = (k, tuple(hist))
        if key not in self._cache:
            out = []
            for y in (1, 0):
                t = self.T[y]
                m = t.sum(axis=tuple(range(k+1, self.K))) if k+1 < self.K else t
                v = m[tuple(hist)]
                out.append(v/v.sum())
            self._cache[key] = tuple(out)
        return self._cache[key]


def as_lik(G):
    return G if hasattr(G, 'cond') else IndepLik(G)


def product_table(g_hier):
    """prod_k g_hier[k, y] as an array T[y, b_1, ..., b_K]."""
    g_hier = np.asarray(g_hier, float)
    return np.stack([reduce(np.multiply.outer, [g_hier[k, y] for k in range(len(g_hier))]) for y in (0, 1)])


def joint_table(g_hier, cells, alpha):
    base = product_table(g_hier)
    if math.isinf(alpha):
        return base
    c = np.asarray(cells, float)
    n = c.reshape(2, -1).sum(axis=1).reshape((2,) + (1,)*(c.ndim-1))
    T = (c + alpha*base)/(n + alpha)
    if not np.allclose(T.reshape(2, -1).sum(axis=1), 1.) or (T <= 0).any():
        raise ValueError('joint table must be a strictly positive distribution per class')
    return T


def empty_cells(K):
    return np.zeros((2,) + (BINS,)*K)


def hard_cells(bins_h, y):
    bins_h = np.asarray(bins_h, int)
    c = empty_cells(bins_h.shape[1])
    for row, yy in zip(bins_h, np.asarray(y, int)):
        c[(int(yy),) + tuple(row)] += 1.
    return c


def soft_cells(bins_h, w):
    bins_h = np.asarray(bins_h, int)
    c = empty_cells(bins_h.shape[1])
    for row, ww in zip(bins_h, np.asarray(w, float)):
        c[(1,) + tuple(row)] += ww
        c[(0,) + tuple(row)] += 1. - ww
    return c


def panel_logs(g, T, hier, bins):
    """log P(full panel | Y=1), log P(full panel | Y=0) for rows of bins (n, m); T None = independence."""
    bins = np.asarray(bins, int).reshape(-1, g.shape[0])
    if T is None:
        return lk.panel_logs(g, bins)
    hier = list(hier)
    rest = [j for j in range(g.shape[0]) if j not in hier]
    lg = np.log(g)
    l1 = np.zeros(len(bins))
    l0 = np.zeros(len(bins))
    if rest:
        jr = np.array(rest)[None, :]
        l1 += lg[jr, 1, bins[:, rest]].sum(axis=1)
        l0 += lg[jr, 0, bins[:, rest]].sum(axis=1)
    idx = tuple(bins[:, h] for h in hier)
    lT = np.log(T)
    return l1 + lT[(1,) + idx], l0 + lT[(0,) + idx]


def posterior(pi, bins, g, T, hier):
    l1, l0 = panel_logs(g, T, hier, bins)
    pi = np.clip(np.asarray(pi, float), 1e-300, 1-1e-16)
    a, b = np.log(pi) + l1, np.log1p(-pi) + l0
    return 1/(1+np.exp(b-a))


def label_loglik(g, T, hier, bins, y):
    l1, l0 = panel_logs(g, T, hier, bins)
    return np.where(np.asarray(y, int) == 1, l1, l0)


def fit_labeled(bins, y, hier, audit_j, alpha, gamma=lk.GAMMA):
    """Labeled fit: g by MAP (audit constraint), T from hard cell counts shrunk to prod g[hier]."""
    bins = np.asarray(bins, int)
    g, active = lk.labeled_fit(bins, y, audit_j, gamma=gamma)
    T = joint_table(g[list(hier)], hard_cells(bins[:, list(hier)], y), alpha)
    return g, T, active


def choose_alpha(bins, y, hier, audit_j, alphas, folds, seed, gamma=lk.GAMMA):
    """Inner CV on labeled complete panels: sum of held-out log T[y][cell]. Ties go to the larger alpha.
    The order of hier does not matter (cells are permuted consistently)."""
    bins, y = np.asarray(bins, int), np.asarray(y, int)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(bins)) % folds
    scores = {}
    for alpha in alphas:
        total = 0.
        for f in range(folds):
            tr, te = fold != f, fold == f
            if not te.any():
                continue
            _, T, _ = fit_labeled(bins[tr], y[tr], hier, audit_j, alpha, gamma)
            idx = tuple(bins[te][:, h] for h in hier)
            total += float(np.log(T[(y[te],) + idx]).sum())
        scores[alpha] = total
    best = max(alphas, key=lambda a: (scores[a], a))
    return best, {('inf' if math.isinf(a) else f'{a:g}'): s for a, s in scores.items()}


class JointOnlineEM:
    """C2 under the joint model: recursive averages of soft marginal counts S and soft cell counts SC."""

    def __init__(self, pi0, bins0, g0, T0, hier, alpha, audit_j, gamma=lk.GAMMA):
        bins0 = np.asarray(bins0, int)
        self.hier, self.alpha, self.audit_j, self.gamma = list(hier), alpha, audit_j, gamma
        self.N0 = int(len(bins0))
        if self.N0 == 0:
            raise ValueError('C2 needs at least one complete initialization panel')
        w = posterior(pi0, bins0, g0, T0, self.hier)
        self.S = lk.soft_counts(bins0, w, g0.shape[0])/self.N0
        self.SC = soft_cells(bins0[:, self.hier], w)/self.N0
        self.n = 0

    def update(self, pi, bins_row, g_t, T_t):
        t0 = time.perf_counter()
        row = np.asarray(bins_row, int).reshape(1, -1)
        w = posterior([pi], row, g_t, T_t, self.hier)
        self.n += 1
        step = 1./(self.N0 + self.n)
        self.S = (1-step)*self.S + step*lk.soft_counts(row, w, g_t.shape[0])
        self.SC = (1-step)*self.SC + step*soft_cells(row[:, self.hier], w)
        g, active = lk.mstep((self.N0+self.n)*self.S, self.audit_j, self.gamma)
        T = joint_table(g[self.hier], (self.N0+self.n)*self.SC, self.alpha)
        return g, T, {'em_iterations': 1, 'records_processed': 1, 'records': 1, 'gamma_n': step,
                      'posterior_weight': float(w[0]), 'constraint_active': bool(active),
                      'seconds': time.perf_counter()-t0}

    def state(self):
        return {'S': self.S.tolist(), 'SC': self.SC.tolist(), 'n': self.n, 'N0': self.N0}

    @classmethod
    def from_state(cls, st, hier, alpha, audit_j, gamma=lk.GAMMA):
        obj = cls.__new__(cls)
        obj.S, obj.SC, obj.n, obj.N0 = np.array(st['S']), np.array(st['SC']), st['n'], st['N0']
        obj.hier, obj.alpha, obj.audit_j, obj.gamma = list(hier), alpha, audit_j, gamma
        return obj
