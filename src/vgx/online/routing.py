"""DesignRouting (main-text version) and NestedVerify for a fixed hierarchy of singleton layers.

Fitted law on the design set: Y ~ Bernoulli(pi_i) with pi_i = r_phi(U_i), layer-k bins ~ g[k, Y]
(conditional independence). A rule is the Bellman policy for multipliers (omega, rho):

  A(b) = 1 - omega (1 - b - alpha),  G(b) = max(0, A(b)),  J_K = G,
  Q_k(b) = -rho d_{k+1} + sum_x M_k(x; b) J_{k+1}(Phi_k(b, x)),  J_k = max(G, Q_k);
  continue iff Q_k > G + TIE (ties stop); on stopping release iff A(b) >= 0.

Values are computed exactly on the 3^K tree of bin paths (no belief grid). For each rule i,
p_i, e_i, c_i are means over all design tasks (a non-processable task contributes zero). The
deployment LP is

  max sum w_i p_i  s.t.  sum w_i (e_i - alpha p_i) <= 0,  sum w_i c_i <= B,  sum w_i = 1, w >= 0,

its duals give (omega, rho), and a new Bellman rule is added while its penalized value
p - omega (e - alpha p) - rho c exceeds max_i [p_i - omega (e_i - alpha p_i) - rho c_i] + EPS.
(This corrects the appendix Algorithm 2, whose LP constraint and pricing omit the alpha p terms.)
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog

TIE = 1e-12
EPS = 1e-9
MAX_ROUNDS = 200


def _node(k, b, mass, G, d, omega, rho, alpha):
    """Returns J_k(b), and per-item contributions (p, e, c) of the Bellman rule from this node."""
    A = 1 - omega*(1 - b - alpha)
    stop = np.maximum(0., A)
    rel = (A >= 0).astype(float)
    sp, se = mass*rel, mass*rel*(1-b)
    if k == len(d):
        return stop, sp, se, np.zeros_like(b)
    q = np.full(b.shape, -rho*d[k])
    cp, ce, cc = np.zeros_like(b), np.zeros_like(b), np.full(b.shape, 0.)
    g1, g0 = G[k, 1], G[k, 0]
    for x in range(G.shape[2]):
        px = b*g1[x] + (1-b)*g0[x]
        bx = np.divide(b*g1[x], px, out=np.zeros_like(b), where=px > 0)
        J, p_, e_, c_ = _node(k+1, bx, mass*px, G, d, omega, rho, alpha)
        q += px*J
        cp, ce, cc = cp+p_, ce+e_, cc+c_
    go = q > stop + TIE
    J = np.where(go, q, stop)
    return J, np.where(go, cp, sp), np.where(go, ce, se), np.where(go, cc + mass*d[k], 0.)


def rule_values(pi, G, d, omega, rho, alpha):
    """Exact fitted (p, e, c, penalized value) of the Bellman rule over design priors pi (nan = not processable)."""
    pi = np.asarray(pi, float)
    ok = np.isfinite(pi)
    n = len(pi)
    b = np.where(ok, pi, 0.)
    J, p, e, c = _node(0, b[ok], np.ones(ok.sum()), np.asarray(G), np.asarray(d, float), omega, rho, alpha)
    P, E, C, V = p.sum()/n, e.sum()/n, c.sum()/n, J.sum()/n
    return float(P), float(E), float(C), float(V)


def decide(k, b, G, d, omega, rho, alpha):
    """Rule action at layer k with current belief b: 'query', 'release' or 'abstain'."""
    J, *_ = _node(k, np.array([b]), np.array([1.]), np.asarray(G), np.asarray(d, float), omega, rho, alpha)
    A = 1 - omega*(1 - b - alpha)
    if k < len(d) and J[0] > max(0., A) + TIE:
        return 'query'
    return 'release' if A >= 0 else 'abstain'


def _lp(cols, alpha, budget):
    v = np.array(cols)
    res = linprog(-v[:, 0], A_ub=np.stack([v[:, 1]-alpha*v[:, 0], v[:, 2]]), b_ub=[0., budget],
                  A_eq=np.ones((1, len(v))), b_eq=[1.], bounds=(0, None), method='highs')
    if res.status != 0:
        raise RuntimeError(f'deployment LP failed: {res.message}')
    omega, rho = max(0., -res.ineqlin.marginals[0]), max(0., -res.ineqlin.marginals[1])
    return res.x, -res.fun, omega, rho


def design_routing(pi, G, d, alpha, budget):
    """Column generation from {release-all, abstain-all}. Returns the rule mixture and diagnostics."""
    t0 = time.perf_counter()
    pi = np.asarray(pi, float)
    n = len(pi)
    ok = np.isfinite(pi)
    rel_all = (ok.sum()/n, float(np.sum(1-pi[ok]))/n, 0.)
    rules = [{'kind': 'release_all', 'omega': 0., 'rho': 0.}, {'kind': 'abstain_all', 'omega': None, 'rho': None}]
    cols = [rel_all, (0., 0., 0.)]
    status, rounds = 'max_rounds', 0
    for rounds in range(1, MAX_ROUNDS+1):
        w, pub, omega, rho = _lp(cols, alpha, budget)
        z = max(p - omega*(e - alpha*p) - rho*c for p, e, c in cols)
        P, E, C, V = rule_values(pi, G, d, omega, rho, alpha)
        if V <= z + EPS:
            status = 'converged'
            break
        rules.append({'kind': 'bellman', 'omega': omega, 'rho': rho})
        cols.append((P, E, C))
    w, pub, omega, rho = _lp(cols, alpha, budget)
    w = np.where(w > 1e-12, w, 0.)
    w = w/w.sum()
    keep = [i for i in range(len(rules)) if w[i] > 0]
    mix = [{**rules[i], 'weight': float(w[i]), 'p': cols[i][0], 'e': cols[i][1], 'c': cols[i][2]} for i in keep]
    P = float(sum(m['weight']*m['p'] for m in mix))
    E = float(sum(m['weight']*m['e'] for m in mix))
    C = float(sum(m['weight']*m['c'] for m in mix))
    return {'rules': mix, 'fitted_pub': P, 'fitted_err_pub': E, 'fitted_risk': E/P if P > 0 else 0.,
            'fitted_cost': C, 'omega': omega, 'rho': rho, 'status': status, 'rounds': rounds,
            'candidates': len(rules), 'seconds': time.perf_counter()-t0}


def pick_rule(design, u):
    """Inverse-CDF draw over the mixture in candidate order (common random number u in [0,1))."""
    acc = 0.
    for r in design['rules']:
        acc += r['weight']
        if u < acc:
            return r
    return design['rules'][-1]


def nested_verify(prior, rule, G, d, alpha, read):
    """Algorithm 3. read(k) returns the bin of layer k (or None on failure) and is called only on query."""
    if rule['kind'] == 'release_all':
        return {'action': 'release', 'layers_used': 0, 'belief': prior, 'failure': None, 'bins': []}
    if rule['kind'] == 'abstain_all':
        return {'action': 'abstain', 'layers_used': 0, 'belief': prior, 'failure': None, 'bins': []}
    b, k, bins = prior, 0, []
    G = np.asarray(G)
    while True:
        act = decide(k, b, G, d, rule['omega'], rule['rho'], alpha)
        if act != 'query':
            return {'action': act, 'layers_used': k, 'belief': b, 'failure': None, 'bins': bins}
        x = read(k)
        k += 1
        if x is None:
            return {'action': 'abstain', 'layers_used': k, 'belief': b, 'failure': 'missing_verifier_score',
                    'bins': bins}
        mass = b*G[k-1, 1, x] + (1-b)*G[k-1, 0, x]
        if mass <= 0:
            return {'action': 'abstain', 'layers_used': k, 'belief': None, 'failure': 'zero_predictive_likelihood',
                    'bins': bins}
        b = b*G[k-1, 1, x]/mass
        bins.append(int(x))
