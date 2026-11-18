from itertools import permutations, product

import numpy as np

from vgx.gpqa.diamond_score import joint_for, prior_map


def fit(rng):
    vs = ['a', 'b', 'c']
    lik = {v+':probability': {'p_bin_if_correct': list(rng.dirichlet([1, 1, 4])),
                              'p_bin_if_incorrect': list(rng.dirichlet([3, 1, 2]))} for v in vs}
    counts = {str(y): rng.integers(0, 9, (3, 3, 3)).tolist() for y in (0, 1)}
    return {'likelihoods': lik, 'joint_models': {'a+b+c': {'alpha': '4', 'counts': counts}},
            'prior_calibration': {'monotone_binned': {'interpolate_x': [0.5, 0.9], 'interpolate_y': [0.4, 0.7]}}}


def test_frozen_joint_does_not_depend_on_query_order():
    f = fit(np.random.default_rng(0))
    for model in ('dependence', 'independence'):
        base = joint_for(f, ('a', 'b', 'c'), model)
        for order in permutations('abc'):
            j = joint_for(f, order, model)
            for cell in product(range(3), repeat=3):
                permuted = tuple(cell['abc'.index(v)] for v in order)
                assert np.isclose(j.likelihood_ratio(permuted), base.likelihood_ratio(cell))


def test_monotone_map_is_the_frozen_interpolation():
    m = prior_map(fit(np.random.default_rng(1)), 'monotone_binned')
    assert m(0.3) == 0.4 and m(0.99) == 0.7 and np.isclose(m(0.7), 0.55)
