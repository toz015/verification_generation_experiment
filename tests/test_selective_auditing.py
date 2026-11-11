import json
import socket

import numpy as np
import pytest

from vgx import selective_auditing as sa

GRID = np.round(np.arange(0, 1.0001, 0.1), 10)


def stream(T=400, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.random(T)
    e = (rng.random(T) > c).astype(int)
    return c, e


def make(kind, runs, alpha=0.1, gamma=0.1, eta=0.05, beta=1.0):
    if kind == 'AT':
        return sa.AdditiveThresholding(alpha, gamma, eta, runs)
    return sa.CalibratedThresholding(alpha, gamma, eta, beta, GRID, runs)


def replay(kind, c, e, u, **kw):
    m = make(kind, u.shape[0], **kw)
    o = sa.Oracle(e, u.shape[0])
    tr = sa.run(m, c, o, u)
    return m, o, tr


@pytest.mark.parametrize('kind', ['AT', 'AT+CC'])
def test_unrevealed_labels_do_not_change_the_trajectory(kind):
    c, e = stream()
    u = sa.uniforms(range(20), len(c))
    _, o, tr = replay(kind, c, e, u)
    never = ~o.revealed.any(axis=0)
    assert never.any()
    e2 = e.copy(); e2[never] = 1-e2[never]
    _, o2, tr2 = replay(kind, c, e2, u)
    for f in ('pi', 'audited', 'score', 'tau_before', 'tau_after'):
        assert np.array_equal(getattr(tr, f), getattr(tr2, f))


@pytest.mark.parametrize('kind', ['AT', 'AT+CC'])
def test_a_label_matters_only_after_its_first_reveal(kind):
    c, e = stream()
    u = sa.uniforms([3], len(c))
    _, o, tr = replay(kind, c, e, u)
    t = int(np.flatnonzero(o.revealed[0])[5])
    e2 = e.copy(); e2[t] = 1-e2[t]
    _, _, tr2 = replay(kind, c, e2, u)
    assert np.array_equal(tr.audited[:, :t+1], tr2.audited[:, :t+1])
    assert np.array_equal(tr.tau_before[:, :t+1], tr2.tau_before[:, :t+1])


@pytest.mark.parametrize('kind', ['AT', 'AT+CC'])
def test_future_confidences_do_not_change_past_decisions(kind):
    c, e = stream()
    u = sa.uniforms(range(10), len(c))
    _, _, tr = replay(kind, c, e, u)
    c2 = c.copy(); c2[200:] = 1-c2[200:]
    _, _, tr2 = replay(kind, c2, e, u)
    assert np.array_equal(tr.audited[:, :200], tr2.audited[:, :200])
    assert np.array_equal(tr.score[:, :200], tr2.score[:, :200])


def test_at_update_matches_equation_1():
    c, e = stream()
    u = sa.uniforms(range(10), len(c))
    alpha, eta = 0.1, 0.05
    _, _, tr = replay('AT', c, e, u, alpha=alpha, eta=eta)
    expect = tr.tau_before+np.where(tr.audited, eta*(1-tr.pi-alpha)*e[None, :]/tr.pi, 0.0)
    assert np.allclose(tr.tau_after, expect)
    assert np.allclose(tr.tau_before[:, 1:], tr.tau_after[:, :-1])
    assert np.all(tr.tau_before[:, 0] == 0.5)
    assert set(np.unique(tr.pi)) <= {1.0, 0.1}
    assert np.all((tr.pi == 1.0) == (tr.score < tr.tau_before))


def test_at_cc_weights_and_gradient_match_algorithm_1():
    c, e = stream(T=60)
    u = sa.uniforms([7], len(c))
    alpha, gamma, eta, beta = 0.1, 0.1, 0.05, 0.7
    m = make('AT+CC', 1, alpha=alpha, gamma=gamma, eta=eta, beta=beta)
    o = sa.Oracle(e, 1)
    w = np.full(len(GRID)+1, 1/(len(GRID)+1)); tau = 0.5
    theta = np.append(GRID, np.inf)
    for t in range(len(c)):
        x = (c[t] >= theta).astype(float)
        s = w@x
        pi = 1.0 if s < tau else gamma
        assert np.isclose(m.score(c[t])[0], s)
        assert np.isclose(m.audit_prob(m.score(c[t]))[0], pi)
        sa.run(m, c, o, u, start=t, stop=t+1)
        if u[0, t] < pi:
            g = (s-(1-e[t]))*x/pi
            w = w*np.exp(-beta*g); w /= w.sum()
            tau += eta*(1-pi-alpha)*e[t]/pi
        assert np.allclose(m.weights()[0], w)
        assert np.isclose(m.tau[0], tau)


def test_calibrated_map_is_monotone_normalised_and_in_unit_interval():
    c, e = stream(T=800)
    u = sa.uniforms(range(5), len(c))
    m, _, tr = replay('AT+CC', c, e, u)
    w = m.weights()
    assert np.allclose(w.sum(axis=1), 1) and np.all(w > 0)
    grid = np.linspace(0, 1, 101)
    s = np.stack([m.score(g) for g in grid], axis=1)
    assert np.all(np.diff(s, axis=1) >= -1e-12) and s.min() >= 0 and s.max() <= 1
    assert tr.score.min() >= 0 and tr.score.max() <= 1


def test_threshold_is_not_clipped():
    T = 300
    c, e = np.full(T, 0.9), np.ones(T, dtype=int)
    u = sa.uniforms([1], T)
    _, _, tr = replay('AT', c, e, u)
    assert tr.tau_after.max() > 1
    c, e = np.full(T, 0.05), np.zeros(T, dtype=int)
    _, _, tr = replay('AT', c, e, u)
    assert tr.tau_after.max() == 0.5  # no audited error, no move


@pytest.mark.parametrize('kind', ['AT', 'AT+CC'])
def test_same_seed_reproduces_and_resume_matches_continuous(kind):
    c, e = stream()
    u = sa.uniforms(range(8), len(c))
    _, _, a = replay(kind, c, e, u)
    _, _, b = replay(kind, c, e, u)
    assert np.array_equal(a.audited, b.audited)
    m = make(kind, 8); o = sa.Oracle(e, 8)
    tr = sa.run(m, c, o, u, stop=150)
    saved = json.loads(json.dumps(m.state()))
    m2 = make(kind, 8); m2.load(saved)
    sa.run(m2, c, o, u, start=150, trace=tr)
    assert np.array_equal(tr.audited, a.audited) and np.allclose(tr.tau_after, a.tau_after)


@pytest.mark.parametrize('kind', ['AT', 'AT+CC'])
def test_audit_counts_match_reveals(kind):
    c, e = stream()
    u = sa.uniforms(range(12), len(c))
    _, o, tr = replay(kind, c, e, u)
    assert np.array_equal(o.revealed, tr.audited)
    mt = sa.metrics(tr.audited, tr.pi, e, tr.score)
    assert np.array_equal(mt['audits'], o.revealed.sum(axis=1))
    assert np.array_equal(mt['audits'], mt['mandatory_audits']+mt['spot_audits'])
    assert np.all(mt['A']+mt['audits'] == len(c))


def test_oracle_refuses_a_second_reveal():
    o = sa.Oracle([1, 0], 2)
    o.audit(0, [True, False])
    with pytest.raises(RuntimeError):
        o.audit(0, [True, True])


def test_ratio_metrics_are_nan_without_denominator():
    mt = sa.metrics(np.ones((2, 3), bool), np.ones((2, 3)), [0, 0, 0])
    assert np.all(np.isnan(mt['leak_error_rate'])) and np.all(np.isnan(mt['selective_risk']))
    assert np.all(mt['M'] == 0)


def test_random_audit_leak_rate_is_alpha_in_expectation():
    c, e = stream(T=500)
    alpha = 0.2
    u = sa.uniforms(range(4000), len(c))
    m = sa.RandomAudit(alpha, u.shape[0]); o = sa.Oracle(e, u.shape[0])
    tr = sa.run(m, c, o, u)
    mt = sa.metrics(tr.audited, tr.pi, e)
    assert abs(mt['M'].mean()/mt['E']-alpha) < 0.01


def test_no_network_during_replay(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError('network access during replay')
    monkeypatch.setattr(socket, 'socket', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    c, e = stream()
    u = sa.uniforms(range(4), len(c))
    replay('AT', c, e, u); replay('AT+CC', c, e, u)
