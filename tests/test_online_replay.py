"""Engineering checks for vgx.online (synthetic data only)."""
import copy

import numpy as np
import pytest
from scipy.optimize import minimize

from vgx.online import calibrator as cal
from vgx.online import likelihood as lk
from vgx.online import simulate as sim
from vgx.online.routing import _lp, design_routing, rule_values
from vgx.online.runner import Run, Settings, draws, initialize, roles
from vgx.online.views import CODES, AccessError, InitLabels, LabelGate, LabelStore, ScoreSource

ST = Settings(eta0=0.1)
UNLABELED = ('A', 'B', 'C1', 'C2', 'C1F', 'C2F')


def build(labels_override=None, signals_override=None, seed=7, n_stream=60):
    items, signals, labels = sim.dataset('correlated', seed, n_stream=n_stream)
    if labels_override:
        labels = labels_override(items, labels)
    if signals_override:
        signals = signals_override(items, signals)
    src, store = ScoreSource(signals, sim.VERIFIERS), LabelStore(labels)
    init, design, stream = roles(items, 0)
    I = initialize(init, design, src, InitLabels(store, [i.task_id for i in init]), ST)
    return items, src, store, I, stream


def run(arm, q=0.5, **kw):
    items, src, store, I, stream = build(**kw)
    r = Run(arm, q, I, stream, src, draws(5, 0, 0, len(stream)), ST, LabelGate(store) if arm == 'D' else None)
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


@pytest.mark.parametrize('arm', UNLABELED)
def test_hidden_stream_labels_do_not_change_unlabeled_trajectories(arm):
    def flip(items, labels):
        stream_ids = {it.task_id for it in items if it.fold >= 2}
        return {t: (1-y if t in stream_ids else y) for t, y in labels.items()}
    a, _, _ = run(arm)
    b, _, _ = run(arm, labels_override=flip)
    assert strip(a) == strip(b)


def test_labeled_reference_does_depend_on_labels():
    def flip(items, labels):
        stream_ids = {it.task_id for it in items if it.fold >= 2}
        return {t: (1-y if t in stream_ids else y) for t, y in labels.items()}
    a, _, _ = run('D')
    b, _, _ = run('D', labels_override=flip)
    assert strip(a) != strip(b)


@pytest.mark.parametrize('arm', ('B', 'C1', 'C2', 'D'))
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


def test_c1_recomputes_history_confidence_with_round_start_phi(monkeypatch):
    seen = []
    orig = lk.full_history_em

    def spy(pi, bins, g_start, audit_j, **kw):
        seen.append((np.array(pi), np.array(bins), g_start.copy()))
        return orig(pi, bins, g_start, audit_j, **kw)
    monkeypatch.setattr(lk, 'full_history_em', spy)
    log, r, _ = run('C1')
    upd = [x for x in log if x['theta_update'] and x['theta_update'].get('records')]
    assert len(upd) == len(seen) > 3
    hz = list(r.init.init_z)
    for rec, (pi, bins, g0) in zip(upd, seen):
        hz.append(rec['z'])
        assert np.allclose(pi, cal.predict(np.array(rec['phi_start']), np.array(hz)))
        assert np.allclose(g0, np.array(rec['theta_start']), atol=1e-9)
        assert len(bins) == len(hz)


def test_c2_recursion_uses_stored_soft_counts_only():
    log, r, _ = run('C2')
    I = r.init
    S = lk.soft_counts(I.init_bins, lk.posterior(cal.predict(I.phi0, I.init_z), I.init_bins, I.theta0), 5)/len(I.init_bins)
    n = 0
    theta = I.theta0.copy()
    for rec in log:
        assert np.allclose(np.array(rec['theta_start']), theta, atol=1e-9)
        if rec['theta_update'] and rec['theta_update'].get('status') == 'ok':
            n += 1
            w = lk.posterior([cal.predict(np.array(rec['phi_start']), rec['z'])], [rec['audit_panel']], theta)
            g = 1/(len(I.init_bins)+n)
            S = (1-g)*S + g*lk.soft_counts(np.array([rec['audit_panel']]), w, 5)
            theta, _ = lk.mstep((len(I.init_bins)+n)*S, I.audit_j)
    assert n > 3 and np.allclose(r.c2.S, S) and np.allclose(r.theta, theta)


@pytest.mark.parametrize('arm', ('B', 'C1', 'C2', 'D'))
def test_updates_use_round_start_parameters(arm):
    log, r, store = run(arm)
    for prev, rec in zip(log, log[1:]):
        pu = prev['phi_update']
        if pu and pu['status'] in ('ok', 'projected'):
            expect, _ = cal.step(np.array(prev['phi_start']), prev['z'], pu['target'], pu['eta'], 0.5)
            assert np.allclose(rec['phi_start'], expect)
        elif not (prev['theta_update'] and prev['theta_update'].get('status') not in (None, 'skipped_incomplete_panel')):
            assert rec['phi_start'] == prev['phi_start'] and rec['theta_start'] == prev['theta_start']
        if prev.get('zhat') is not None:
            mu0, mu1 = lk.audit_means(np.array(prev['theta_start']), r.init.audit_j)
            a = CODES[prev['audit_panel'][r.init.audit_j]]
            assert np.isclose(prev['zhat'], (a-mu0)/(mu1-mu0), atol=1e-6)


def test_em_ascends_its_fixed_objective():
    log, _, _ = run('C1')
    for rec in log:
        u = rec['theta_update']
        if u and 'objective_end' in u:
            assert u['monotone_violations'] == 0
            assert u['objective_end'] >= u['objective_start'] - 1e-9


def test_resume_equals_continuous():
    items, src, store, I, stream = build()
    rand = draws(5, 0, 0, len(stream))
    for arm in ('C1', 'C2', 'D'):
        full = Run(arm, .5, I, stream, src, rand, ST, LabelGate(store) if arm == 'D' else None).run()
        a = Run(arm, .5, I, stream, src, rand, ST, LabelGate(store) if arm == 'D' else None)
        a.run(until=23)
        state = a.state()
        b = Run(arm, .5, I, stream, src, rand, ST, LabelGate(store) if arm == 'D' else None)
        b.restore(state)
        b.run()
        assert strip(b.log) == strip(full)


def test_audit_reuses_outputs_and_costs_add_up():
    log, r, _ = run('C1')
    usd = dict(zip(sim.VERIFIERS, sim.COSTS))
    audited = [x for x in log if x['audited']]
    assert audited
    for x in log:
        assert np.isclose(x['decision_cost_usd'], sum(usd[v] for v in x['decision_reads']))
        if x['audited']:
            assert not set(x['audit_new_reads']) & set(x['decision_reads'])
            assert set(x['audit_new_reads']) | set(x['decision_reads']) == set(sim.VERIFIERS)
            assert np.isclose(x['audit_cost_usd'], sum(usd[v] for v in x['audit_new_reads']))
        else:
            assert x['reads_all'] == x['decision_reads'] and x['audit_cost_usd'] == 0


def test_static_arm_never_audits_and_audits_are_shared():
    logs = {a: run(a)[0] for a in ('A', 'B', 'C1', 'C2', 'D')}
    assert not any(x['audited'] for x in logs['A'])
    pattern = [x['audited'] for x in logs['B']]
    for a in ('C1', 'C2', 'D'):
        assert [x['audited'] for x in logs[a]] == pattern


def test_access_guards():
    items, signals, labels = sim.dataset('ci_known', 1, n_stream=10)
    src, store = ScoreSource(signals, sim.VERIFIERS), LabelStore(labels)
    acc = src.open(items[0].task_id, 'x')
    with pytest.raises(AccessError):
        src.open(items[1].task_id, 'y')
    acc.close()
    with pytest.raises(AccessError):
        acc.read(sim.VERIFIERS[0])
    gate = LabelGate(store)
    with pytest.raises(AccessError):
        gate.outcome(items[0].task_id)
    init = InitLabels(store, [items[0].task_id])
    with pytest.raises(AccessError):
        init.outcome(items[1].task_id)


def test_constrained_mstep_matches_generic_solver():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n1, n0 = rng.uniform(.5, 20, 3), rng.uniform(.5, 20, 3)
        gamma = rng.uniform(.05, .6)
        g1, g0, _ = lk.constrained_pair(n1, n0, gamma)
        assert CODES @ (g1-g0) >= gamma - 1e-9
        f = lambda x: -(n1 @ np.log(x[:3]) + n0 @ np.log(x[3:]))
        cons = [{'type': 'eq', 'fun': lambda x: x[:3].sum()-1}, {'type': 'eq', 'fun': lambda x: x[3:].sum()-1},
                {'type': 'ineq', 'fun': lambda x: CODES @ (x[:3]-x[3:]) - gamma}]
        r = minimize(f, np.r_[np.ones(3)/3, np.ones(3)/3], constraints=cons, bounds=[(1e-9, 1)]*6, method='SLSQP',
                     options={'ftol': 1e-14, 'maxiter': 1000})
        assert -f(np.r_[g1, g0]) >= -r.fun - 1e-6


def test_design_routing_matches_dense_multiplier_grid_and_meets_fitted_constraints():
    rng = np.random.default_rng(0)
    pi = rng.beta(4, 2, 50)
    G = sim.TRUE_G[[2, 1, 0]]
    d = sim.COSTS[[2, 1, 0]]
    for alpha, B in ((.05, 5e-4), (.1, 2e-4)):
        D = design_routing(pi, G, d, alpha, B)
        assert D['status'] == 'converged'
        assert D['fitted_err_pub'] <= alpha*D['fitted_pub'] + 1e-9 and D['fitted_cost'] <= B + 1e-12
        cols = [(1., float(np.mean(1-pi)), 0.), (0., 0., 0.)]
        for om in np.r_[0, np.logspace(-2, 4, 40)]:
            for rho in np.r_[0, np.logspace(-1, 5, 40)]:
                cols.append(rule_values(pi, G, d, om, rho, alpha)[:3])
        assert D['fitted_pub'] >= _lp(cols, alpha, B)[1] - 1e-9


def test_initial_calibrator_minimizes_brier():
    rng = np.random.default_rng(0)
    r = rng.beta(5, 1.5, 200)
    y = (rng.random(200) < cal.predict([-.5, .8], cal.feature(r))).astype(int)
    phi, info = cal.fit_brier(cal.feature(r), y)
    z = cal.feature(r)
    best = np.mean((cal.predict(phi, z)-y)**2)
    for da in (-.05, .05):
        for db in (-.05, .05):
            assert np.mean((cal.predict(phi+np.array([da, db]), z)-y)**2) >= best - 1e-12
