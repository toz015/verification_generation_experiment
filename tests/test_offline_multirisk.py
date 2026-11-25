"""Tests for the offline multi-risk comparison (handbook section 14, stage D)."""
import copy
from pathlib import Path
import socket

import numpy as np
import pytest

import vgx.offline_multirisk as M

HAS_CACHE = Path('results/gpqa_sonnet55_verification_20261016').exists()


# ----------------------------------------------------------------------------- synthetic model

class Fake:
    """A fitted-model stand-in with two verifiers and hand-set likelihoods (5 bins)."""

    def __init__(self, prior, valid=None, bins=None, usd=None):
        self.prior = np.asarray(prior, float)
        n = len(self.prior)
        self.valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
        self.prior = np.where(self.valid, self.prior, 0.)
        self.fitted = {'likelihoods': {'v1': {'p_bin_if_correct': [.05, .1, .15, .3, .4], 'p_bin_if_incorrect': [.4, .3, .15, .1, .05]},
                                       'v2': {'p_bin_if_correct': [.1, .1, .2, .3, .3], 'p_bin_if_incorrect': [.3, .3, .2, .1, .1]}},
                       'expected_cost_usd': {'v1': .001, 'v2': .002}}
        rng = np.random.default_rng(0)
        self.bins = bins or {v: rng.integers(0, 5, n) for v in ('v1', 'v2')}
        self.usd = usd or {'v1': np.full(n, .001), 'v2': np.full(n, .002)}


def exact_value(m, order, b, kappa, mu):
    """Exact recursion without the belief grid: V = max(0, b - kappa (1 - b), -mu c + sum_z P(z|b) V(b_z))."""
    stop = max(0., b-kappa*(1-b))
    if not order:
        return stop
    v = order[0]
    lk = m.fitted['likelihoods'][v]
    p1, p0 = np.array(lk['p_bin_if_correct']), np.array(lk['p_bin_if_incorrect'])
    cont = -mu*m.fitted['expected_cost_usd'][v]
    for z in range(5):
        pz = b*p1[z]+(1-b)*p0[z]
        cont += pz*exact_value(m, order[1:], b*p1[z]/pz, kappa, mu)
    return max(stop, cont)


# ----------------------------------------------------------------------------- 1. risks: denominators, direction, LP

def test_risk_ratios_denominators_and_identity():
    y = np.array([1, 1, 1, 0, 0])
    a = np.array([1, 1, 0, 1, 0.])
    for name, expect in (('selective_error', 1/3), ('fpr', 1/2), ('fnr', 1/3)):
        r = M.Risk(name)
        P, E, M_, Q, W = (np.ravel(x)[0] if np.ndim(x) else x for x in r.counts(a, y))
        assert (P, E, M_, Q, W) == (3, 1, 1, 3, 2)
        assert np.isclose(r.ratio(P, E, M_, Q, W), expect)
        assert np.isclose(M_/5, Q/5-P/5+E/5)


def test_lp_constraints_and_direction():
    V = [(0., 0., 0.), (1., .2, 0.), (.6, .02, 0.), (.9, .05, .01)]
    q = .8
    sel = M.Risk('selective_error').lp(V, .05, 1., q)
    v = np.array(V)
    assert sel['w']@(v[:, 1]-.05*v[:, 0]) <= 1e-9 and sel['p'] > .6
    fpr = M.Risk('fpr').lp(V, .1, 1., q)
    assert fpr['e'] <= .1*(1-q)+1e-9
    fnr = M.Risk('fnr').lp(V, .05, 1., q)
    assert fnr['m'] <= .05*q+1e-9
    # FNR minimizes joint error: it must not simply release everything when a cheaper-error feasible mix exists
    assert fnr['e'] <= fnr['w']@v[:, 1]+1e-12 and fnr['e'] < .2
    # budget binds
    tight = M.Risk('selective_error').lp(V, .05, .001, q)
    assert tight['c'] <= .001+1e-12


def test_handbook_release_value_matches_canonical_rule():
    rng = np.random.default_rng(1)
    for name in M.RISKS:
        r = M.Risk(name)
        for _ in range(200):
            lam, alpha, b = rng.uniform(.01, 50), rng.uniform(.005, .4), rng.uniform(0, 1)
            can = r.canonical(lam, 0., alpha)
            g = float(r.release_value(b, lam, alpha))
            if can is None:
                # boundary rule: release always (sel, fpr) or never (fnr with lam = 0) -- here lam > 0
                assert name != 'fnr' and g >= -1e-12
                continue
            kappa = can[0]
            canon = b-kappa*(1-b)
            assert np.sign(round(g, 12)) == np.sign(round(canon, 12)) or abs(g) < 1e-9
        if name == 'fnr':
            lam = 3.
            assert np.isclose(M.threshold(r.canonical(lam, 0., .05)[0]), 1/(1+lam))


# ----------------------------------------------------------------------------- 2. FNR boundaries

def test_fnr_boundaries():
    r = M.Risk('fnr')
    y = np.array([1, 1, 0, 0])
    elig = np.array([1, 1, 1, 0.])
    release = elig
    abstain = np.zeros(4)
    j, status = r.select(np.stack([abstain, release]), np.zeros((2, 4)), y, .05, 0.)
    assert (j, status) == (1, 'selected')            # Abstain-all violates FNR; Release-eligible meets it
    assert r.select(abstain[None], np.zeros((1, 4)), y, .05, 0.) == (None, 'no_feasible_candidate')
    assert r.select(release[None], np.zeros((1, 4)), np.zeros(4), .05, 0.) == (None, 'risk_not_estimable')
    mt = M.metrics(r, .05, abstain, np.zeros(4), np.zeros(4), np.zeros(4), y, elig.astype(bool), 0.)
    assert mt['fnr'] == 1. and not mt['empirical_feasible']
    mt = M.metrics(r, .05, release, np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), elig.astype(bool), 0.)
    assert mt['fnr'] is None and not mt['risk_estimable'] and not mt['empirical_feasible']


def test_selection_orders():
    y = np.array([1, 1, 1, 0])
    A = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0.]])
    C = np.array([[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1.]])
    assert M.Risk('selective_error').select(A, C, y, .05, 5.) == (3, 'selected')   # max release, then cheaper
    assert M.Risk('fnr').select(A, C, y, .4, 5.) == (3, 'selected')                # 2 misses FNR; min e, then cheaper
    assert M.Risk('fpr').select(A, C, np.ones(4), .1, 5.) == (None, 'risk_not_estimable')


# ----------------------------------------------------------------------------- 3. Bellman, LP and column generation

def test_bellman_value_matches_exact_recursion():
    m = Fake([.3, .55, .7, .9, .97])
    order = ('v1', 'v2')
    for kappa, mu in ((1., 0.), (4., 50.), (.2, 300.), (20., 10.)):
        rule = ('bellman', kappa, mu)
        p, e, c = M.model_value(m, order, rule)
        # Lagrangian of the canonical form per item: E[a (b - kappa (1 - b))] - mu c = mean_i V(b_i)
        lag = (p-e)-kappa*e-mu*c
        exact = np.mean([exact_value(m, order, b, kappa, mu) for b in m.prior])
        assert abs(lag-exact) < 2e-3
        # optimal against alternatives: never ask, ask both
        for alt in (('fixed', kappa, 0.), ('bellman', kappa, 1e9)):
            pa, ea, ca = M.model_value(m, order, alt)
            assert lag >= (pa-ea)-kappa*ea-mu*ca-2e-3


def test_column_generation_improves_and_satisfies_constraints():
    rng = np.random.default_rng(3)
    m = Fake(rng.uniform(.2, .99, 60))
    vc = M.ValueCache(m)
    order = ('v1', 'v2')
    base = list(M.BASE_RULES)+[('bellman', k, u) for k in M.KAPPAS[::6] for u in M.MUS[::5]]
    vals = [vc.get(order, r) for r in base]
    q = float(m.prior.mean())
    for name, alpha in (('selective_error', .05), ('fpr', .1), ('fnr', .05)):
        risk = M.Risk(name)
        sol0 = risk.lp(vals, alpha, .0015, q)
        rules, values, sol = M.design(vc, order, risk, alpha, .0015, q, base, vals)
        assert sol['status'] == 0 and sol['objective'] >= sol0['objective']-1e-9
        assert max(sol['residual']) <= 1e-8
        # no remaining column with positive reduced cost on a dense rule set
        z = max(risk.lagrangian(v, sol['lam'], sol['mu'], alpha) for v in values)
        dense = [('bellman', k, u) for k in M.KAPPAS for u in M.MUS]
        best = max(risk.lagrangian(vc.get(order, r), sol['lam'], sol['mu'], alpha) for r in dense)
        assert best <= z+1e-3


def test_lp_failure_is_not_a_zero_policy():
    r = M.Risk('fnr')
    sol = r.lp([(0., 0., 0.)], .05, 1., .8)   # only Abstain-all: m = q > alpha q, infeasible
    assert sol['status'] != 0 and 'w' not in sol


# ----------------------------------------------------------------------------- 4. mixture semantics

def test_mixture_cost_query_rate_and_counts():
    m = Fake([.9, .9])
    runs = M.Runs(m)
    cand = {'order': ['v1', 'v2'], 'rules': [['release', 0., 0.], ['fixed', 1., 0.]], 'weights': [.5, .5]}
    a, c, calls, qprob, per = runs.mixture(cand)
    assert np.allclose(qprob, .5)                       # not 1{0.5 * 2 calls > 0} = 1
    assert np.allclose(calls, 1.)                       # 0.5 * 0 + 0.5 * 2
    assert np.allclose(c, .5*(.001+.002))
    assert np.allclose(a, .5*1+.5*per[1][0])
    y = np.array([1, 0])
    mt = M.metrics(M.Risk('selective_error'), .05, a, c, calls, qprob, y, m.valid, 1.)
    assert np.isclose(mt['release_error_rate'], (a*(1-y)).sum()/a.sum())   # pooled numerators, not a mean of ratios
    assert np.isclose(mt['verifier_query_rate'], .5)


def test_draws_execute_one_rule_per_item():
    m = Fake([.9]*50)
    runs = M.Runs(m)
    cand = {'order': ['v1', 'v2'], 'rules': [['abstain', 0., 0.], ['fixed', 1., 0.]], 'weights': [.3, .7]}
    *_, per = runs.mixture(cand)
    d = M.draw(per, cand['weights'], 50, 5)
    for s in range(5):
        for i in range(50):
            assert d['cost'][s, i] in (0., per[1][1][i])


# ----------------------------------------------------------------------------- 5. AT grid

def test_at_boundary_keeps_alpha_09():
    import vgx.offline_frozen as O
    cands = O.at_candidates()
    assert len(cands) == 40 and any(np.isclose(a, .9) for a, _ in cands)
    assert len(cands)*len(O.SETTINGS['beta']) == 160


# ----------------------------------------------------------------------------- 6. Brier

def test_brier_pools_unequal_folds_by_samples_and_flags_missing_calibrators():
    brier = {'raw': [[1., 10], [3., 30]], 'ours_prior': [[0., 10], [6., 30]], 'ours_map': {'t1': .5},
             'at_cal': {'fpr': [{'fold': 0, 'seed': 0, 'status': 'no_selected_calibrator', 'sum': None, 'n': 1, 'items': None}]}}
    out = M.brier_summary(brier, [], [])
    assert np.isclose(out['raw']['brier'], 4/40) and np.isclose(out['ours_prior']['brier'], 6/40)
    assert out['at_cal']['fpr']['status_counts'] == {'selected': 0, 'no_selected_calibrator': 1}
    assert 'brier_selected_task_seeds' not in out['at_cal']['fpr']


# ----------------------------------------------------------------------------- 12. network guard

def test_network_is_blocked():
    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
    try:
        M.block_network()
        with pytest.raises(RuntimeError):
            socket.create_connection(('example.com', 80))
        with pytest.raises(RuntimeError):
            socket.getaddrinfo('example.com', 80)
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = saved


def test_fixed_panel_asks_every_member():
    m = Fake([.999, .001, .5])
    a, cost, asked, b = M.run_rule(m, ('v1', 'v2'), ('fixed', 1., 0.))
    assert asked.all() and np.allclose(cost, .003)
    a, cost, asked, b = M.run_rule(m, ('v1', 'v2'), ('bellman', 1., 50.))
    assert not asked[0].any() and not asked[1].any()      # adaptive rule stops on confident beliefs


# ----------------------------------------------------------------------------- cached-data tests

@pytest.fixture(scope='module')
def gpqa():
    if not HAS_CACHE:
        pytest.skip('cached GPQA data not present')
    import vgx.baselines as B
    cfg = M.load_config(M.CONFIG)
    data, _ = B.load('gpqa')
    return cfg, data['sonnet55']


def plan(cfg, rows, risk='fnr'):
    from vgx.gpqa import fpr_routing as R
    k, fit, val, test = M.splits('gpqa', rows)[0]
    cref = M.O.c_ref(rows, fit)
    mv, mt = R.Model(rows, fit, val, {}, True), R.Model(rows, fit, test, {}, True)
    cands, *_ = M.ours_candidates(M.ValueCache(mv), M.Risk(risk), cfg['alpha_plan'][risk], [(1., cref)], cfg)
    return fit, val, test, cref, cands, mt


def test_ids_disjoint_and_qk_groups_isolated(gpqa):
    import vgx.baselines as B
    cfg, rows = gpqa
    for ds in ('gpqa', 'math500', 'qkatas'):
        data, _ = B.load(ds)
        groups = M.O.qk_groups() if ds == 'qkatas' else None
        for gen, rs in data.items():
            for k, fit, val, test in M.splits(ds, rs):
                ids = [{rs[i]['item_id'] for i in x} for x in (fit, val, test)]
                assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
                if groups:
                    g = [{groups[rs[i]['item_id']] for i in x} for x in (fit, val, test)]
                    assert not g[0] & g[1] and not g[0] & g[2] and not g[1] & g[2]


def test_test_labels_and_prices_do_not_change_fit_budget_candidates_or_actions(gpqa):
    cfg, rows = gpqa
    fit, val, test, cref, cands, mt = plan(cfg, rows)
    other = copy.deepcopy(rows)
    for i in test:
        other[i]['outcome'] = 1-other[i]['outcome']
        for v in M.B.VERIFIERS:
            other[i]['signals'][v]['estimated_usd'] *= 3
    fit2, val2, test2, cref2, cands2, mt2 = plan(cfg, other)
    assert cref == cref2
    assert [c['rules'] for c in cands] == [c['rules'] for c in cands2] and [c['weights'] for c in cands] == [c['weights'] for c in cands2]
    r1, r2 = M.Runs(mt), M.Runs(mt2)
    for c in cands[:20]:
        a1, c1, *_ = r1.mixture(c)
        a2, c2, *_ = r2.mixture(c)
        assert np.array_equal(a1, a2)                    # actions unchanged
        assert np.allclose(c2, 3*c1)                     # only the realized cost changes


def test_unqueried_verifier_outputs_do_not_change_actions(gpqa):
    from vgx.gpqa import fpr_routing as R
    cfg, rows = gpqa
    fit, val, test, cref, cands, mt = plan(cfg, rows, 'selective_error')
    cand = max(cands, key=lambda c: c['planned']['c'])
    runs = M.Runs(mt)
    a, cost, calls, qprob, per = runs.mixture(cand)
    asked = np.zeros((len(test), 3), bool)
    for p in per:
        asked |= p[2]
    other = copy.deepcopy(rows)
    for j, i in enumerate(test):
        for k, v in enumerate(cand['order']):
            if not asked[j, k]:
                other[i]['signals'][v]['score'] = 1.0-(other[i]['signals'][v]['score'] or 0.0)
    a2, cost2, *_ = M.Runs(R.Model(other, fit, test, {}, True)).mixture(cand)
    assert np.array_equal(a, a2) and np.array_equal(cost, cost2)


def test_risk_order_in_one_process_does_not_matter(gpqa):
    cfg, rows = gpqa
    _, _, _, _, c_fnr_first, _ = plan(cfg, rows, 'fnr')
    _, _, _, _, c_sel, _ = plan(cfg, rows, 'selective_error')
    _, _, _, _, c_fnr_again, _ = plan(cfg, rows, 'fnr')
    assert [c['rules'] for c in c_fnr_first] == [c['rules'] for c in c_fnr_again]
    assert [c['weights'] for c in c_fnr_first] == [c['weights'] for c in c_fnr_again]


def test_confidence_inputs_give_the_same_prior_as_the_model(gpqa):
    """calibrated: every method reads the Fit-calibrated p, equal to the model's own calibration; raw: the raw p;
    calibration uses Fit labels only."""
    from vgx.gpqa import fpr_routing as R
    cfg, rows = gpqa
    k, fit, val, test = M.splits('gpqa', rows)[0]
    rc, cal = M.fold_rows(rows, fit, 'calibrated')
    assert cal is False
    np.testing.assert_allclose(R.Model(rc, fit, test, {}, False).prior, R.Model(rows, fit, test, {}, True).prior, atol=1e-12)
    rr, cal = M.fold_rows(rows, fit, 'raw')
    assert rr is rows and cal is False
    flipped = [dict(r, outcome=1-r['outcome']) if i in set(test) | set(val) else r for i, r in enumerate(rows)]
    rf, _ = M.fold_rows(flipped, fit, 'calibrated')
    assert [r['prior'] for r in rf] == [r['prior'] for r in rc]


def test_threshold_example():
    from vgx.offline_multirisk_report import threshold_example
    b = [0.99, 0.95, 0.9, 0.5, 0.2]
    t, share = threshold_example(b, 0, 'selective_error', 0.05)   # wrong mass .01,.06,.16 over 1,2,3 released
    assert t == 0.95 and share == 0.4
    t, share = threshold_example(b, 0, 'fnr', 0.1)                # q = 3.54; blocking .2 (<= .354) but not .2+.5
    assert t == 0.5 and share == 0.8
    t, share = threshold_example(b, 1, 'fpr', 0.3)                # wrong total 1.46+1; cap .738: .01+.05+.1+.5 = .66
    assert t == 0.5 and share == 4/6


def test_shrunk_isotonic_and_gamma():
    rng = np.random.default_rng(3)
    p = rng.choice([.5, .6, .7, .8, .9, .95, .99], 300)
    y = (rng.random(300) < p*.8).astype(float)
    from vgx.gpqa.reporting_trial import isotonic
    q = np.linspace(.4, 1, 61)
    assert np.allclose(isotonic(p, y)[0](q), M.shrunk_isotonic(p, y, 0)(q))
    f = M.shrunk_isotonic(p, y, 20)(q)
    assert np.all(np.diff(f) >= -1e-12) and f.max() < isotonic(p, y)[0](q).max() + 1e-12
    grid = np.linspace(0, 1, 11); p1 = np.array([.1, .9]); p0 = np.array([.6, .4])
    mass = np.outer(grid, p1)+np.outer(1-grid, p0)
    full, half = M.posterior_grid(grid, p1, p0, 1.0, mass), M.posterior_grid(grid, p1, p0, .5, mass)
    assert np.all(np.abs(half-grid[:, None]) <= np.abs(full-grid[:, None]) + 1e-12)
