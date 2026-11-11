import copy
from pathlib import Path

import numpy as np
import pytest

from vgx import selective_auditing as sa

pytestmark = pytest.mark.skipif(not Path('results/gpqa_sonnet55_verification_20261016').exists(), reason='cached GPQA data not present')


@pytest.fixture(scope='module')
def gpqa():
    import vgx.baselines as B
    import vgx.offline_frozen as O
    data, _ = B.load('gpqa')
    rows = data['sonnet55']
    k, fit, val, test = O.splits('gpqa', rows)[0]
    return O, rows, fit, val, test


def flipped(rows, idx):
    out = copy.deepcopy(rows)
    for i in idx:
        out[i]['outcome'] = 1-out[i]['outcome']
    return out


def test_splits_are_disjoint(gpqa):
    O, rows, fit, val, test = gpqa
    assert not set(fit) & set(val) and not set(fit) & set(test) and not set(val) & set(test)
    assert all(rows[i]['fold'] == 0 for i in test)


def test_hidden_test_labels_do_not_change_decisions(gpqa):
    O, rows, fit, val, test = gpqa
    other = flipped(rows, test)
    st, st2 = O.threshold_methods(rows, fit, 0), O.threshold_methods(other, fit, 0)
    for m in ('AT', 'AT+CC'):
        a1, _ = O.threshold_decisions(m, st[m], O.Features(rows, test), 0.0, 0)
        a2, _ = O.threshold_decisions(m, st2[m], O.Features(other, test), 0.0, 0)
        assert np.array_equal(a1, a2)
    cand = O.ours_candidates(rows, fit, val, O.c_ref(rows, fit), True)[10]
    r1, r2 = O.ours_apply(rows, fit, test, cand, True), O.ours_apply(other, fit, test, cand, True)
    assert np.array_equal(r1['a'], r2['a']) and np.array_equal(r1['cost'], r2['cost'])


def test_unqueried_verifier_outputs_do_not_change_decisions(gpqa):
    O, rows, fit, val, test = gpqa
    cands = O.ours_candidates(rows, fit, val, O.c_ref(rows, fit), True)
    cand = max(cands, key=lambda c: c['planned_cost'])
    r = O.ours_apply(rows, fit, test, cand, True)
    other = copy.deepcopy(rows)
    order = cand['order']
    asked_any = np.zeros((len(test), len(order)), bool)
    for p in r['per_rule']:
        asked_any |= p[2]
    for j, i in enumerate(test):
        for k, v in enumerate(order):
            if not asked_any[j, k]:
                other[i]['signals'][v]['score'] = 1.0-(other[i]['signals'][v]['score'] or 0.0)
        for v in set(O.B.VERIFIERS)-set(order):
            other[i]['signals'][v]['score'] = 0.123
    r2 = O.ours_apply(other, fit, test, cand, True)
    assert np.array_equal(r['a'], r2['a']) and np.array_equal(r['cost'], r2['cost'])


def test_frozen_state_is_unchanged_by_evaluation(gpqa):
    O, rows, fit, val, test = gpqa
    st = O.threshold_methods(rows, fit, 3)
    h = O.state_hash({k: np.asarray(v).tolist() for k, v in st['AT+CC'].items() if k != 'configs'})
    O.threshold_decisions('AT+CC', st['AT+CC'], O.Features(rows, test), 0.0, 3)
    assert O.state_hash({k: np.asarray(v).tolist() for k, v in st['AT+CC'].items() if k != 'configs'}) == h
    cands = O.ours_candidates(rows, fit, val, O.c_ref(rows, fit), True)
    h = O.state_hash(cands)
    O.ours_apply(rows, fit, test, cands[5], True)
    assert O.state_hash(cands) == h


def test_vectorised_fit_matches_the_reference_updates(gpqa):
    O, rows, fit, val, test = gpqa
    fv = [i for i in fit if rows[i]['valid']]
    conf = np.array([rows[i]['prior'] for i in fv], float)
    err = O.labels(rows, fv)
    seed = 4
    perm = np.random.default_rng(seed).permutation(len(conf))
    u = np.random.default_rng(10_000+seed).random(len(conf))[None, :]
    tau, _ = O.at_fit(conf, err, np.array([0.2]), np.array([0.03]), 0.1, seed)
    m = sa.AdditiveThresholding(0.2, 0.1, 0.03, 1)
    sa.run(m, conf[perm], sa.Oracle(err[perm], 1), u)
    assert np.isclose(tau[0], m.tau[0])
    logw, ctau, theta, _ = O.cc_fit(conf, err, np.array([0.2]), np.array([0.03]), np.array([0.1]), 0.1, O.SETTINGS['calibration_grid'], seed)
    m = sa.CalibratedThresholding(0.2, 0.1, 0.03, 0.1, O.SETTINGS['calibration_grid'], 1)
    sa.run(m, conf[perm], sa.Oracle(err[perm], 1), u)
    assert np.isclose(ctau[0], m.tau[0]) and np.allclose(np.exp(logw[0]-logw[0].max())/np.exp(logw[0]-logw[0].max()).sum(), m.weights()[0])


def test_lp_constrains_release_error_not_the_fpr_denominator():
    from vgx.gpqa import fpr_routing as R
    import vgx.offline_frozen  # noqa: F401  (sets the release-error constraint)
    assert R.CONSTRAINT == 'fdr'
    values = [(0.0, 0.0, 0.0), (1.0, 0.2, 0.0), (0.5, 0.01, 0.0)]
    w, pub, *_ = R.solve(values, 0.05, 1.0, 0.2)
    v = np.array(values)
    assert w@(v[:, 1]-0.05*v[:, 0]) <= 1e-9
    assert pub > 0.5   # the FPR form (e <= alpha * wrong = 0.01) would cap coverage at 0.5


def test_costs_are_the_calls_actually_read_and_mixtures_charge_one_rule(gpqa):
    O, rows, fit, val, test = gpqa
    cands = O.ours_candidates(rows, fit, val, O.c_ref(rows, fit), True)
    cand = next(c for c in cands if len(c['rules']) > 1 and c['planned_cost'] > 0)
    r = O.ours_apply(rows, fit, test, cand, True, draws=[0])
    order = cand['order']
    for a, cost, asked, _ in r['per_rule']:
        expect = np.array([sum(rows[i]['signals'][v]['estimated_usd'] for k, v in enumerate(order) if asked[j, k]) for j, i in enumerate(test)])
        assert np.allclose(cost, expect)
    d = r['draws'][0]
    for j in range(len(test)):
        assert np.isclose(d['cost'][j], r['per_rule'][d['rule'][j]][1][j])
    m = O.evaluate(r['a'], r['cost'], O.labels(rows, test), O.Features(rows, test).valid, r['calls'])
    assert m['external_labels_acquired'] == 0
    assert m['defer_rate'] != m['verifier_query_rate']


def test_selection_returns_none_without_a_feasible_nonzero_release():
    import vgx.offline_frozen as O
    A = np.array([[1, 1, 0], [0, 0, 0]], float)
    err = np.array([1, 1, 0])
    assert O.select(A, np.zeros_like(A), err, 0.05, 1.0) is None
    A = np.array([[1, 0, 1], [0, 0, 1]], float)
    assert O.select(A, np.zeros_like(A), np.array([0, 1, 0]), 0.05, 1.0) == 0
