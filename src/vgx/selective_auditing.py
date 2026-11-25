"""Selective-auditing baselines from Gibbs, Tibshirani, Han and Zrnic, "Reliable AI-Assisted Decisions via
Selective Auditing" (arXiv:2610.31662): additive thresholding (AT, Sec. 3.1, eqs. 1-2) and AT with online
confidence calibration (AT+CC, Sec. 3.2, Algorithm 1). No API calls.

Semantics (protocol in docs/selective-auditing-baselines.md):
  H_t = 0  accept the raw candidate automatically; its label is never read.
  H_t = 1  hand the candidate to a reliable auditor; the auditor's verdict (the official checker label) is
           revealed to the learner. The candidate and its original label are unchanged.

Learners see confidences one round at a time and obtain labels only through `Oracle.audit`, which records
every reveal. Many independent audit-randomness runs ("seeds") of one method on one stream are simulated
together as arrays; each run has its own state and its own reveals. Common random numbers: run r of every
method uses the same uniforms u[r, t], and audits iff u[r, t] < pi[r, t] under its own pi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


class Oracle:
    """Holds a stream's error labels; a learner reads e_t only for the runs that audit round t."""

    def __init__(self, errors, runs):
        self._errors = np.asarray(errors, dtype=np.int8)
        self.revealed = np.zeros((runs, len(self._errors)), dtype=bool)

    def audit(self, t, mask):
        """e_t for runs where mask is True, -1 elsewhere. The audit decision must be fixed before this call."""
        mask = np.asarray(mask, dtype=bool)
        if self.revealed[mask, t].any():
            raise RuntimeError(f'round {t} audited twice')
        self.revealed[mask, t] = True
        out = np.full(mask.shape, -1, dtype=np.int8)
        out[mask] = self._errors[t]
        return out


@dataclass
class Trace:
    """Per-round records for every run: arrays of shape (runs, T)."""
    pi: np.ndarray
    audited: np.ndarray
    score: np.ndarray          # the score compared with the threshold, before this round's update
    tau_before: np.ndarray
    tau_after: np.ndarray
    seconds: float = 0.0
    state_bytes: int = 0
    extra: dict = field(default_factory=dict)


class AdditiveThresholding:
    """Eqs. (1)-(2): audit with probability 1 if C_t < tau_t, else gamma; tau moves only on audited rounds."""

    name = 'AT'

    def __init__(self, alpha, gamma, eta, runs, tau0=0.5):
        if not 0 < alpha < 1 or not 0 < gamma <= 1-alpha or eta <= 0:
            raise ValueError('need 0 < alpha < 1, 0 < gamma <= 1 - alpha, eta > 0')
        self.alpha, self.gamma, self.eta = alpha, gamma, eta
        self.tau = np.full(runs, float(tau0))

    def score(self, c):
        return np.full(self.tau.shape, float(c))

    def audit_prob(self, s):
        return np.where(s < self.tau, 1.0, self.gamma)

    def update(self, c, s, pi, audited, e):
        step = self.eta*(1-pi-self.alpha)*np.where(audited, e, 0)/pi
        self.tau = self.tau+np.where(audited, step, 0.0)

    def state(self):
        return {'tau': self.tau.tolist()}

    def load(self, state):
        self.tau = np.asarray(state['tau'], dtype=float)


class CalibratedThresholding(AdditiveThresholding):
    """Algorithm 1: S_t = sum_j w_j 1{C_t >= theta_j} with theta_{k+1} = +inf; exponentiated-gradient weights
    on inverse-probability-weighted squared-loss gradients; the same threshold update applied to S_t."""

    name = 'AT+CC'

    def __init__(self, alpha, gamma, eta, beta, grid, runs, tau0=0.5):
        super().__init__(alpha, gamma, eta, runs, tau0)
        grid = np.asarray(grid, dtype=float)
        if grid[0] != 0 or np.any(np.diff(grid) <= 0) or grid[-1] > 1 or beta <= 0:
            raise ValueError('need 0 = theta_1 < ... < theta_k <= 1 and beta > 0')
        self.beta = beta
        self.theta = np.append(grid, np.inf)
        self.logw = np.full((runs, len(self.theta)), -np.log(len(self.theta)))

    def weights(self):
        z = self.logw-self.logw.max(axis=1, keepdims=True)
        w = np.exp(z)
        return w/w.sum(axis=1, keepdims=True)

    def score(self, c):
        x = (c >= self.theta).astype(float)
        return self.weights()@x

    def update(self, c, s, pi, audited, e):
        x = (c >= self.theta).astype(float)
        y = 1-e
        g = np.where(audited, (s-y)/pi, 0.0)[:, None]*x[None, :]
        logw = self.logw-self.beta*g
        m = logw.max(axis=1, keepdims=True)
        self.logw = logw-(m+np.log(np.exp(logw-m).sum(axis=1, keepdims=True)))
        super().update(c, s, pi, audited, e)

    def state(self):
        return {**super().state(), 'logw': self.logw.tolist()}

    def load(self, state):
        super().load(state)
        self.logw = np.asarray(state['logw'], dtype=float)


class RandomAudit:
    """Audits every round with probability 1 - alpha (valid by construction); learns nothing."""

    name = 'random'

    def __init__(self, alpha, runs):
        self.alpha, self.tau = alpha, np.full(runs, np.nan)

    def score(self, c):
        return np.full(self.tau.shape, float(c))

    def audit_prob(self, s):
        return np.full(s.shape, 1-self.alpha)

    def update(self, c, s, pi, audited, e):
        pass

    def state(self):
        return {}

    def load(self, state):
        pass


def uniforms(seeds, T):
    """Common random numbers: row r is the audit randomness of run r, shared by every method."""
    return np.stack([np.random.default_rng(int(s)).random(T) for s in seeds])


def run(method, confidence, oracle, u, start=0, stop=None, trace=None):
    """Replay rounds [start, stop). The learner sees C_t at round t only; labels come only from `oracle`."""
    confidence = np.asarray(confidence, dtype=float)
    T = len(confidence)
    stop = T if stop is None else stop
    runs = u.shape[0]
    if trace is None:
        z = lambda dtype=float: np.zeros((runs, T), dtype=dtype)
        trace = Trace(z(), z(bool), z(), z(), z())
    t0 = time.perf_counter()
    for t in range(start, stop):
        c = confidence[t]
        s = method.score(c)
        pi = method.audit_prob(s)
        audited = u[:, t] < pi
        trace.pi[:, t], trace.audited[:, t], trace.score[:, t] = pi, audited, s
        trace.tau_before[:, t] = method.tau
        e = oracle.audit(t, audited)
        method.update(c, s, pi, audited, e)
        trace.tau_after[:, t] = method.tau
    trace.seconds += time.perf_counter()-t0
    trace.state_bytes = sum(np.asarray(v).nbytes for v in method.state().values())
    return trace


def oracle_threshold(confidence, errors, alpha):
    """Hindsight reference (Sec. 6): the least conservative fixed threshold c, auditing iff C_t < c, whose
    realized leak rate is <= alpha. Uses every label; not a deployable method."""
    confidence, errors = np.asarray(confidence, float), np.asarray(errors, int)
    E = errors.sum()
    best = (np.inf, len(confidence))
    for c in np.append(np.unique(confidence), np.inf):
        audit = confidence < c
        leak = (errors*(~audit)).sum()
        if E == 0 or leak <= alpha*E:
            if audit.sum() < best[1]:
                best = (c, int(audit.sum()))
    c = best[0]
    return c, confidence < c


def metrics(audited, pi, errors, score=None):
    """Counts and rates per run, from the evaluator's side (all labels). Ratios with a zero denominator are NaN."""
    errors = np.asarray(errors, dtype=int)
    T = audited.shape[1]
    H = audited.astype(int)
    M = (errors[None, :]*(1-H)).sum(axis=1)
    E = int(errors.sum())
    A = (1-H).sum(axis=1)
    out = {'T': T, 'E': E, 'M': M, 'A': A, 'audits': H.sum(axis=1),
           'mandatory_audits': (H*(pi >= 1)).sum(axis=1), 'spot_audits': (H*(pi < 1)).sum(axis=1)}
    with np.errstate(invalid='ignore', divide='ignore'):
        out['leak_error_rate'] = M/E if E else np.full(M.shape, np.nan)
        out['selective_risk'] = np.where(A > 0, M/np.maximum(A, 1), np.nan)
    out['total_unaudited_error_rate'] = M/T
    out['auto_accept_rate'] = A/T
    out['audit_rate'] = H.sum(axis=1)/T
    if score is not None:
        out['brier'] = ((score-(1-errors)[None, :])**2).mean(axis=1)
    return out
