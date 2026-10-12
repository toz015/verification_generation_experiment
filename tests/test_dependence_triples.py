import math
from itertools import permutations, product

import numpy as np
import pytest

from vgx.gpqa.dependence_triples import (INF, ExactPlanner, JointLikelihood, choose_alpha, execute,
                                         fit_joint)
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.score import VerifierLikelihood

EDGES = (0., 1/3, 2/3, 1.)


def marginals(rng):
    return [{y: tuple(rng.dirichlet([2, 1, 3])) for y in (0, 1)} for _ in range(3)]


def test_infinite_alpha_is_the_independence_model():
    rng = np.random.default_rng(1)
    m = marginals(rng)
    joint = JointLikelihood(m, {0: np.ones((3, 3, 3)), 1: np.ones((3, 3, 3))}, INF)
    for history in [(), (0,), (2,), (1, 2)]:
        for y in (0, 1):
            assert np.allclose(joint.conditional(history, y), m[len(history)][y])


def test_exact_recursion_matches_grid_planner_under_independence():
    rng = np.random.default_rng(2)
    m = marginals(rng)
    joint = JointLikelihood(m, {0: np.zeros((3, 3, 3)), 1: np.zeros((3, 3, 3))}, INF)
    costs = [0.02, 0.01, 0.03]
    exact = ExactPlanner(joint, costs, 1., 19.)
    grid = NestedPlanner([VerifierLikelihood(EDGES, m[k][1], m[k][0]) for k in range(3)], costs, 1., 19.,
                         grid_size=20001)
    b = np.linspace(0.5, 1., 101)
    assert np.allclose(exact.j(0, b), np.interp(b, grid.grid, grid.j[0]), atol=2e-4)


def test_sequential_update_equals_joint_table_bayes():
    rng = np.random.default_rng(3)
    counts = {y: rng.integers(0, 9, (3, 3, 3)) for y in (0, 1)}
    joint = JointLikelihood(marginals(rng), counts, 2.)
    planner = ExactPlanner(joint, [0., 0., 0.], 1., 19.)
    for cell in product(range(3), repeat=3):
        b, history = 0.8, ()
        for x in cell:
            b = planner.update(b, history, x)
            history = (*history, x)
        lr = joint.likelihood_ratio(cell)
        assert math.isclose(b, 0.8*lr/(0.8*lr + 0.2), rel_tol=1e-12)


class Tracking(dict):
    def __init__(self, *a):
        super().__init__(*a)
        self.read = []

    def __getitem__(self, key):
        self.read.append(key)
        return super().__getitem__(key)


def test_executor_reads_only_queried_verifiers_and_failures_abstain():
    rng = np.random.default_rng(4)
    joint = JointLikelihood(marginals(rng), {y: rng.integers(0, 5, (3, 3, 3)) for y in (0, 1)}, 4.)
    planner = ExactPlanner(joint, [0.001, 0.001, 0.001], 1., 19.)
    order = ('a', 'b', 'c')
    signals = Tracking({t: {'score': 0.9, 'estimated_usd': 0.001} for t in order})
    d = execute(0.9, True, planner, order, [1e-4]*3, signals, EDGES)
    assert set(signals.read) <= set(order[:d['verifiers_used']])
    signals = Tracking({'a': {'score': None, 'estimated_usd': 0.001}, 'b': {'score': 0.9, 'estimated_usd': 0.001},
                        'c': {'score': 0.9, 'estimated_usd': 0.001}})
    d = execute(0.9, True, planner, order, [1e-4]*3, signals, EDGES)
    if d['verifiers_used']:
        assert d['action'] == 'abstain' and d['failure'] == 'missing_verifier_score' and d['verifiers_used'] == 1


def test_query_all_posterior_does_not_depend_on_order():
    rng = np.random.default_rng(5)
    tags = ('a', 'b', 'c')
    rows = [{'answer': 'A', 'prior': 0.9, 'outcome': int(y),
             'signals': {t: {'score': float(s), 'estimated_usd': 0.} for t, s in zip(tags, rng.random(3))}}
            for y in rng.integers(0, 2, 60)]
    m = marginals(rng)
    posts = set()
    for order in permutations(range(3)):
        o = tuple(tags[i] for i in order)
        joint = fit_joint(rows, o, [m[i] for i in order], 3., EDGES)
        d = execute(0.9, True, ExactPlanner(joint, [0.]*3, 1., 19.), o, [0.]*3, rows[0]['signals'], EDGES,
                    force_all=True)
        posts.add(round(d['posterior'], 12))
    assert len(posts) == 1


@pytest.mark.parametrize('dependent', [False, True])
def test_inner_cv_shrinks_less_when_verifiers_are_dependent(dependent):
    rng = np.random.default_rng(6)
    tags = ('a', 'b', 'c')
    rows = []
    for _ in range(600):
        y = int(rng.random() < 0.6)
        shared = rng.random()
        scores = [(shared if dependent else rng.random())*0.5 + 0.5*y*rng.random() for _ in tags]
        rows.append({'answer': 'A', 'prior': 0.9, 'outcome': y,
                     'signals': {t: {'score': float(min(s, 1.)), 'estimated_usd': 0.} for t, s in zip(tags, scores)}})
    cfg = {'inner_folds': 5, 'probability_bins': 3, 'laplace': 1.}
    alpha, _ = choose_alpha(rows, tags, [0.5, 2., 8., 32., 128., INF], EDGES, cfg, 7)
    assert (alpha <= 32.) if dependent else (alpha >= 32.)


@pytest.mark.parametrize('method', ['monotone_binned', 'platt'])
def test_prior_calibration_is_monotone_and_keeps_raw_values(method):
    from vgx.gpqa.dependence_triples import calibrated_rows, fit_prior_calibration
    rng = np.random.default_rng(8)
    raw = rng.choice([0.85, 0.9, 0.95, 0.98, 0.99, 1.0], 300)
    rows = [{'answer': 'A', 'prior': float(p), 'outcome': int(rng.random() < p - 0.15)} for p in raw]
    fit = fit_prior_calibration(rows, method)
    grid = np.linspace(0., 1., 201)
    mapped = [fit['map'](p) for p in grid]
    assert all(b >= a - 1e-12 for a, b in zip(mapped, mapped[1:]))
    assert all(0. < m < 1. for m in mapped)
    out = calibrated_rows(rows + [{'answer': None, 'prior': None, 'outcome': 0}], fit['map'])
    assert [r['raw_prior'] for r in out[:-1]] == [r['prior'] for r in rows] and out[-1]['prior'] is None
    assert fit['map'](0.95) < 0.95  # overconfident input is pulled down
