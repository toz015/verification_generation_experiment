"""Checks for the dependence-aware joint likelihood in vgx.online (synthetic data only)."""
import copy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from vgx.online import calibrator as cal
from vgx.online import joint as jm
from vgx.online import likelihood as lk
from vgx.online import simulate as sim
from vgx.online.routing import design_routing, nested_verify
from vgx.online.runner import Run, Settings, draws, initialize, roles
from vgx.online.views import InitLabels, LabelGate, LabelStore, ScoreSource

ST = Settings(eta0=0.1, likelihood='dependence', alpha_grid=(1, 8, 64))


def build(labels_override=None, signals_override=None, seed=7, n_stream=60, st=ST):
    items, signals, labels = sim.dataset('correlated', seed, n_stream=n_stream)
    if labels_override:
        labels = labels_override(items, labels)
    if signals_override:
        signals = signals_override(items, signals)
    src, store = ScoreSource(signals, sim.VERIFIERS), LabelStore(labels)
    init, design, stream = roles(items, 0)
    I = initialize(init, design, src, InitLabels(store, [i.task_id for i in init]), st)
    return items, src, store, I, stream


def run(arm, q=0.5, st=ST, **kw):
    items, src, store, I, stream = build(st=st, **kw)
    r = Run(arm, q, I, stream, src, draws(5, 0, 0, len(stream)), st, LabelGate(store) if arm == 'D' else None)
    return r.run(), r, store


def strip(log):
    drop = {'seconds', 'redesign_seconds'}

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if k not in drop}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    return clean(log)


def random_g(rng, m=3):
    return rng.dirichlet([1, 1, 1], size=(m, 2))


def test_product_table_reproduces_independence_exactly():
    rng = np.random.default_rng(1)
    G = random_g(rng)
    T = jm.product_table(G)
    pi, d = rng.random(80), np.array([.001, .002, .003])
    a = design_routing(pi, G, d, .05, .003)
    b = design_routing(pi, jm.JointLik(T), d, .05, .003)
    for k in ('fitted_pub', 'fitted_risk', 'fitted_cost'):
        assert a[k] == pytest.approx(b[k], abs=1e-12)
    rule = next(r for r in a['rules'] if r['kind'] == 'bellman')
    for cell in product(range(3), repeat=3):
        x = nested_verify(.7, rule, G, d, .05, lambda k: cell[k])
        y = nested_verify(.7, rule, jm.JointLik(T), d, .05, lambda k: cell[k])
        assert x['action'] == y['action'] and x['belief'] == pytest.approx(y['belief'], abs=1e-12)
    g5 = random_g(rng, 5)
    bins = rng.integers(0, 3, (40, 5))
    hier = [3, 0, 2]
    assert np.allclose(jm.posterior(rng.random(40)*0 + .6, bins, g5, jm.product_table(g5[hier]), hier),
                       lk.posterior(np.full(40, .6), bins, g5))


def test_joint_conditionals_chain_to_table_and_sequential_bayes_matches_joint():
    rng = np.random.default_rng(2)
    T = np.stack([rng.dirichlet(np.ones(27)).reshape(3, 3, 3) for _ in (0, 1)])
    L = jm.JointLik(T)
    for cell in product(range(3), repeat=3):
        for y, idx in ((1, 0), (0, 1)):
            p = np.prod([L.cond(k, cell[:k])[idx][cell[k]] for k in range(3)])
            assert p == pytest.approx(T[(y,) + cell])
    # a rule that always queries: zero cost and a release threshold no belief can reach early
    rule = {'kind': 'bellman', 'omega': 1e6, 'rho': 0.}
    prior = .4
    for cell in product(range(3), repeat=3):
        res = nested_verify(prior, rule, L, np.zeros(3), .05, lambda k: cell[k])
        if res['layers_used'] == 3:
            lr = T[(1,) + cell]/T[(0,) + cell]
            assert res['belief'] == pytest.approx(prior*lr/(prior*lr + 1-prior))


def test_joint_table_shrinkage_limits():
    rng = np.random.default_rng(3)
    g = random_g(rng)
    cells = rng.integers(0, 5, (2, 3, 3, 3)).astype(float)
    assert np.allclose(jm.joint_table(g, cells, np.inf), jm.product_table(g))
    small = jm.joint_table(g, cells, 1e-9)
    assert np.allclose(small, cells/cells.reshape(2, -1).sum(axis=1)[:, None, None, None], atol=1e-6)
    mid = jm.joint_table(g, cells, 7.)
    assert np.allclose(mid.reshape(2, -1).sum(axis=1), 1.) and (mid > 0).all()


def test_alpha_selection_detects_dependence():
    rng = np.random.default_rng(4)
    n = 400
    y = (rng.random(n) < .5).astype(int)
    indep = np.stack([np.where(rng.random(n) < .5 + .3*(2*y-1)*.5, 2, 0) for _ in range(3)], 1)
    shared = np.where(rng.random(n) < .5 + .3*(2*y-1)*.5, 2, 0)
    dep = np.stack([np.where(rng.random(n) < .9, shared, rng.integers(0, 3, n)) for _ in range(3)], 1)
    grid = [.5, 2, 8, 32, 128, np.inf]
    a_ind, _ = jm.choose_alpha(indep, y, (0, 1, 2), 0, grid, 5, 1)
    a_dep, _ = jm.choose_alpha(dep, y, (0, 1, 2), 0, grid, 5, 1)
    assert np.isfinite(a_dep) and a_dep < a_ind


def test_initialization_uses_d_init_labels_only_and_alpha_inf_is_independence():
    def flip(items, labels):
        return {t: (1-y if it.fold != 0 else y) for it in items for t, y in [(it.task_id, labels[it.task_id])]}
    _, _, _, a, _ = build()
    _, _, _, b, _ = build(labels_override=flip)
    assert a.hierarchy == b.hierarchy and a.joint_alpha == b.joint_alpha and np.array_equal(a.T0, b.T0)
    # with only alpha = inf available the dependence model is the independence model
    st_inf = replace(ST, alpha_grid=())
    _, _, _, c, _ = build(st=st_inf)
    _, _, _, d, _ = build(st=Settings(eta0=0.1))
    assert c.joint_alpha == np.inf and c.hierarchy == d.hierarchy
    assert np.allclose(c.T0, jm.product_table(d.theta0[list(d.hierarchy)]))


@pytest.mark.parametrize('arm', ('A', 'B', 'C2'))
def test_hidden_stream_labels_do_not_change_unlabeled_trajectories(arm):
    def flip(items, labels):
        stream_ids = {it.task_id for it in items if it.fold >= 2}
        return {t: (1-y if t in stream_ids else y) for t, y in labels.items()}
    a, _, _ = run(arm)
    b, _, _ = run(arm, labels_override=flip)
    assert strip(a) == strip(b)


@pytest.mark.parametrize('arm', ('B', 'C2', 'D'))
def test_future_and_unread_scores_do_not_change_current_decision(arm):
    base, r, _ = run(arm)
    t0 = 25
    order = [x['task_id'] for x in base]
    unread_now = set(sim.VERIFIERS) - set(base[t0]['reads_all'])

    def perturb(items, signals):
        s = copy.deepcopy(signals)
        for tid in order[t0+1:]:
            for v in s[tid]:
                s[tid][v]['score'] = 1 - s[tid][v]['score']
        for v in unread_now:
            s[order[t0]][v]['score'] = 1 - s[order[t0]][v]['score']
        return s
    other, _, _ = run(arm, signals_override=perturb)
    assert strip(base[:t0+1]) == strip(other[:t0+1])


def test_joint_c2_recursion_uses_stored_soft_counts_only():
    log, r, _ = run('C2')
    I, hier = r.init, list(r.init.hierarchy)
    w0 = jm.posterior(cal.predict(I.phi0, I.init_z), I.init_bins, I.theta0, I.T0, hier)
    N0 = len(I.init_bins)
    S, SC = lk.soft_counts(I.init_bins, w0, 5)/N0, jm.soft_cells(I.init_bins[:, hier], w0)/N0
    theta, T, n = I.theta0.copy(), I.T0.copy(), 0
    for rec in log:
        assert np.allclose(np.array(rec['theta_start']), theta, atol=1e-9)
        assert np.allclose(np.array(rec['joint_start']), T, atol=1e-9)
        if rec['theta_update'] and rec['theta_update'].get('status') == 'ok':
            n += 1
            row = np.array([rec['audit_panel']])
            w = jm.posterior([cal.predict(np.array(rec['phi_start']), rec['z'])], row, theta, T, hier)
            step = 1/(N0+n)
            S = (1-step)*S + step*lk.soft_counts(row, w, 5)
            SC = (1-step)*SC + step*jm.soft_cells(row[:, hier], w)
            theta, _ = lk.mstep((N0+n)*S, I.audit_j)
            T = jm.joint_table(theta[hier], (N0+n)*SC, I.joint_alpha)
    assert n > 3 and np.allclose(r.theta, theta) and np.allclose(r.T, T)


def test_labeled_reference_counts_cells():
    log, r, store = run('D')
    I, hier = r.init, list(r.init.hierarchy)
    audited = [x for x in log if x['theta_update'] and x['theta_update'].get('status') == 'ok']
    bins = np.vstack([I.init_bins] + [np.array([x['audit_panel']]) for x in audited])
    y = np.r_[I.init_y, [store.outcome(x['task_id']) for x in audited]]
    g, _ = lk.labeled_fit(bins, y, I.audit_j)
    assert np.allclose(r.T, jm.joint_table(g[hier], jm.hard_cells(bins[:, hier], y), I.joint_alpha))


def test_resume_equals_continuous_and_unsupported_arms_refuse():
    items, src, store, I, stream = build()
    rand = draws(5, 0, 0, len(stream))
    full = Run('C2', .5, I, stream, src, rand, ST).run()
    a = Run('C2', .5, I, stream, src, rand, ST)
    a.run(until=30)
    b = Run('C2', .5, I, stream, src, rand, ST)
    b.restore(a.state())
    assert strip(b.run()) == strip(full)
    for arm in ('C1', 'C1F', 'C2F'):
        with pytest.raises(ValueError):
            Run(arm, .5, I, stream, src, rand, ST)
