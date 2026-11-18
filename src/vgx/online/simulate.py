"""Synthetic checks for the online replay (no model calls, no real labels).

Each scenario draws tasks with a true correctness probability p* = sigmoid(a* + b* logit r), a label
Y ~ Bernoulli(p*), and five verifier bins from known 3-bin likelihoods (scores placed at bin centres).
Scenarios:
  ci_known      conditional independence; phi_0 and theta_0 set to the true values
  biased_conf   conditional independence; theta_0 true, phi_0 overconfident (a shifted by +1.5)
  biased_lik    conditional independence; phi_0 true, theta_0 with the audit verifier's classes
                pulled halfway toward each other and another verifier's classes swapped toward each other
  correlated    a shared per-task nuisance shifts all verifiers' bins together given Y; phi_0, theta_0
                fitted from the labeled initialization fold as in the real replay
The same Run class and DesignRouting are used as on the cached data.
"""
from __future__ import annotations

import numpy as np

from vgx.online import calibrator as cal
from vgx.online.runner import Init, Run, Settings, draws, initialize, roles
from vgx.online.views import InitLabels, LabelGate, LabelStore, PublicItem, ScoreSource

VERIFIERS = ('v_strong', 'v_cheap1', 'v_cheap2', 'v_weak', 'v_mid')
COSTS = np.array([0.0022, 0.0004, 0.0001, 0.0005, 0.0015])
TRUE_G = np.array([
    [[.70, .15, .15], [.05, .10, .85]],
    [[.45, .25, .30], [.15, .20, .65]],
    [[.40, .30, .30], [.15, .25, .60]],
    [[.30, .35, .35], [.25, .30, .45]],
    [[.55, .25, .20], [.10, .20, .70]],
])
TRUE_PHI = np.array([0.0, 1.0])
SCENARIOS = ('ci_known', 'biased_conf', 'biased_lik', 'correlated')


def dataset(scenario, seed, n_init=64, n_design=64, n_stream=194):
    rng = np.random.default_rng(seed)
    n = n_init + n_design + n_stream
    raw = rng.beta(5, 1.6, n)  # mass near high confidence, like the generators
    p = cal.predict(TRUE_PHI, cal.feature(raw))
    y = (rng.random(n) < p).astype(int)
    shift = rng.normal(0, 1.2, n) if scenario == 'correlated' else np.zeros(n)
    items, signals, labels = [], {}, {}
    folds = np.r_[np.zeros(n_init, int), np.ones(n_design, int), 2 + np.arange(n_stream) % 3]
    for i in range(n):
        tid = f'sim/{i}'
        sig = {}
        for j, v in enumerate(VERIFIERS):
            probs = TRUE_G[j, y[i]].copy()
            if scenario == 'correlated':  # tilt toward higher (shift>0) or lower bins, same for every verifier
                tilt = np.exp(shift[i]*np.array([-1., 0., 1.]))
                probs = probs*tilt/np.sum(probs*tilt)
            b = rng.choice(3, p=probs)
            sig[v] = {'score': (b + .5)/3, 'failure': None, 'estimated_usd': float(COSTS[j])}
        items.append(PublicItem(tid, int(folds[i]), f'g{i % 40}', float(raw[i]), True))
        signals[tid], labels[tid] = sig, int(y[i])
    return items, signals, labels


def biased_theta(g, audit_j):
    g = g.copy()
    a1, a0 = g[audit_j, 1].copy(), g[audit_j, 0].copy()
    g[audit_j, 1], g[audit_j, 0] = .75*a1 + .25*a0, .75*a0 + .25*a1
    k = 4 if audit_j != 4 else 1
    b1, b0 = g[k, 1].copy(), g[k, 0].copy()
    g[k, 1], g[k, 0] = .5*b1 + .5*b0 + 1e-3, .5*b0 + .5*b1 + 1e-3
    g[k] /= g[k].sum(axis=1, keepdims=True)
    return g


def setup(scenario, seed, st: Settings, **kw):
    items, signals, labels = dataset(scenario, seed, **kw)
    src, store = ScoreSource(signals, VERIFIERS), LabelStore(labels)
    init, design, stream = roles(items, 0)
    I = initialize(init, design, src, InitLabels(store, [i.task_id for i in init]), st)
    if scenario in ('ci_known', 'biased_lik'):
        I.phi0 = TRUE_PHI.copy()
    if scenario == 'biased_conf':
        I.phi0 = TRUE_PHI + np.array([1.5, 0.])
    if scenario in ('ci_known', 'biased_conf'):
        I.theta0 = TRUE_G.copy()
    if scenario == 'biased_lik':
        I.theta0 = biased_theta(TRUE_G, I.audit_j)
    if scenario != 'correlated':  # keep the shared hierarchy rule but under the overridden parameters
        I.info['overridden'] = True
    return src, store, I, stream


def run_scenario(scenario, seed, arms, q, st: Settings, order=0, **kw):
    src, store, I, stream = setup(scenario, seed, st, **kw)
    rand = draws(seed, 0, order, len(stream))
    out = {}
    for arm in arms:
        r = Run(arm, q, I, stream, src, rand, st, LabelGate(store) if arm == 'D' else None)
        out[arm] = (r.run(), r)
    return out, store, I


def summarize(log, store, run, scenario=None):
    y = np.array([store.outcome(x['task_id']) for x in log])
    p = np.array([x['p_hat'] for x in log])
    rel = np.array([x['action'] == 'release' for x in log])
    n = len(log)
    thirds = [slice(0, n//3), slice(n//3, 2*n//3), slice(2*n//3, n)]
    out = {'brier': float(np.mean((p-y)**2)), 'brier_last_third': float(np.mean((p[thirds[2]]-y[thirds[2]])**2)),
           'pub': float(rel.mean()), 'released': int(rel.sum()), 'wrong': int((rel & (y == 0)).sum()),
           'risk': float((rel & (y == 0)).sum()/max(rel.sum(), 1)),
           'decision_cost': float(np.mean([x['decision_cost_usd'] for x in log])),
           'phi_end': run.phi.tolist(), 'phi_err_end': float(np.abs(run.phi - TRUE_PHI).sum()),
           'theta_err_end': float(np.abs(run.theta - TRUE_G).mean()),
           'theta_err_start': float(np.abs(run.init.theta0 - TRUE_G).mean())}
    return out
