"""Reporting function r_phi: monotone logistic recalibration of the raw generator confidence.

  p_hat = sigmoid(a + b * logit(clip(r, EPS, 1-EPS))),  phi = (a, b),  b >= 0 (monotone)

Boundary handling: r is clipped to [EPS, 1-EPS] before the logit. Feasible set: a in [-A_MAX, A_MAX],
b in [0, B_MAX]; every stochastic step is followed by projection (clipping) onto this box.
Initialization minimizes the empirical Brier loss directly (L-BFGS-B on the box), not cross-entropy.
A step whose target, prediction or gradient is not finite is skipped and logged.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

EPS = 1e-3
A_MAX, B_MAX = 10., 10.
BOUNDS = ((-A_MAX, A_MAX), (0., B_MAX))


def feature(r):
    r = np.clip(np.asarray(r, float), EPS, 1-EPS)
    return np.log(r) - np.log1p(-r)


def predict(phi, z):
    a, b = phi
    return 1/(1+np.exp(-(a + b*np.asarray(z, float))))


def grad(phi, z):
    """d p_hat / d phi at one input z."""
    p = float(predict(phi, z))
    return p*(1-p)*np.array([1., float(z)])


def project(phi):
    return np.array([min(max(phi[0], -A_MAX), A_MAX), min(max(phi[1], 0.), B_MAX)])


def fit_brier(z, y):
    z, y = np.asarray(z, float), np.asarray(y, float)
    if len(z) == 0:
        return np.array([0., 1.]), {'status': 'empty_init_identity'}

    def f(phi):
        p = predict(phi, z)
        r = p - y
        s = p*(1-p)
        return float(np.mean(r**2)), np.array([np.mean(2*r*s), np.mean(2*r*s*z)])

    best = None
    for start in ((0., 1.), (0., .5), (1., 1.), (-1., 1.)):  # fixed multi-start, best objective wins
        res = minimize(f, np.array(start), jac=True, method='L-BFGS-B', bounds=BOUNDS,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if best is None or res.fun < best.fun - 1e-15:
            best = res
    return project(best.x), {'status': 'ok' if best.success else f'lbfgsb:{best.message}',
                             'brier': float(best.fun), 'n': int(len(z))}


def step(phi, z, target, eta, q):
    """phi + (2 eta / q) (target - p_hat) grad p_hat, evaluated at the old phi, then projected."""
    p = float(predict(phi, z))
    g = grad(phi, z)
    delta = (2*eta/q)*(float(target) - p)*g
    if not (math.isfinite(p) and np.all(np.isfinite(g)) and np.all(np.isfinite(delta))):
        return np.array(phi, float), 'nonfinite_step_skipped'
    new = project(np.asarray(phi, float) + delta)
    return new, ('projected' if not np.allclose(new, np.asarray(phi)+delta) else None)
