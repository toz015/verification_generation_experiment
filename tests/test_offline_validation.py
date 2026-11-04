"""Offline comparisons: fitting boundary, no-peeking, accounting and exact oracle."""
from copy import deepcopy
import json
from pathlib import Path
import socket

import numpy as np
import pytest

from vgx.common.storage import file_digest
from vgx.gpqa.offline_validation import (
    CorrectnessForecast, choose_rules, decide_policy, fit_channels,
    matched_coverage, policy_metrics, run, run_policy,
)
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.score import VerifierLikelihood
from vgx.gpqa.simulation_validation import VerificationWorld, joint_distribution


def rows(n=30, partition='calibration'):
    return [dict(item_id=f'{partition}_{i}', partition=partition,
                 generator_answer='A', correct_index=int(i % 3 == 0),
                 generator_p_correct=.5 + .04 * (i % 10),
                 verifiers={tag: {'p_correct': .15 if i % 3 == 0 else .9}
                            for tag in ('a', 'b', 'c')}) for i in range(n)]


def config():
    value = json.loads(Path('configs/gpqa_offline_validation.json').read_text())
    value.update(primary_order=['a', 'b', 'c'], diagnostic_verifiers=['a', 'b', 'c'],
                 folds=2, bootstrap_evaluation=5, bootstrap_refit=3,
                 simulation_learning_repeats=2, simulation_learning_sizes=[30],
                 correct_reward=1., incorrect_loss=1., cost_multipliers=[1.],
                 loss_sensitivity=[1.], gate_widths=[0., .1, 1.])
    return value


@pytest.mark.parametrize('factory', [lambda: CorrectnessForecast(),
                                     lambda: CorrectnessForecast(('a',))])
def test_forecast_rejects_evaluation_fitting_and_never_predicts_from_labels(factory):
    with pytest.raises(ValueError, match='calibration'):
        factory().fit(rows(partition='evaluation'))
    model = factory().fit(rows())
    public = rows(partition='evaluation')
    expected = model.predict(public)
    for row in public:
        del row['correct_index']
        del row['partition']
    assert model.predict(public) == pytest.approx(expected)


def test_missing_feature_imputation_is_fitted_on_calibration_only():
    train = rows()
    model = CorrectnessForecast(('a',)).fit(train)
    median = model.medians[:]
    test = rows(2, 'evaluation')
    test[0]['verifiers']['a']['p_correct'] = None
    test[1]['verifiers']['a']['p_correct'] = 0.
    assert np.isfinite(model.predict(test)).all()
    assert model.medians == median


def test_likelihood_and_rule_selection_reject_evaluation_labels():
    with pytest.raises(ValueError, match='calibration'):
        fit_channels(rows(partition='evaluation'), ['a'])
    with pytest.raises(ValueError, match='calibration'):
        choose_rules(rows(partition='evaluation'), ['a', 'b', 'c'],
                     {'a': .001, 'b': .01, 'c': .02}, config())


class NoFutureSignals(dict):
    def get(self, tag, default=None):
        assert tag == 'a', 'future signal inspected'
        return {'p_correct': .9}


@pytest.mark.parametrize('kind', ['none', 'myopic', 'nested'])
def test_no_answer_key_or_future_signal_access(kind):
    perfect = VerifierLikelihood((0., .5, 1.), (0., 1.), (1., 0.))
    planner = NestedPlanner([perfect, perfect], [.01, .01], 1., 1.)
    row = {'item_id': 'public', 'generator_answer': 'A', 'generator_p_correct': .5,
           'verifiers': NoFutureSignals()}
    result = run_policy([row], [.5], planner, ['a', 'b'], kind)[0]
    assert result['used'] == (0 if kind == 'none' else 1)
    assert result['cost'] == (0 if kind == 'none' else .01)


@pytest.mark.parametrize('kind', ['always', 'gate', 'myopic', 'nested'])
def test_missing_queried_signal_abstains_and_charges_only_attempted_stage(kind):
    perfect = VerifierLikelihood((0., .5, 1.), (0., 1.), (1., 0.))
    planner = NestedPlanner([perfect, perfect], [.01, .01], 1., 1.)
    row = rows(1, 'evaluation')[0]
    row['verifiers'] = {'a': {'p_correct': None}, 'b': {'p_correct': .9}}
    result = run_policy([row], [.5], planner, ['a', 'b'], kind)[0]
    assert result['used'] == 1 and result['cost'] == .01
    assert result['action'] == 'abstain'
    assert result['failure'] == 'missing_verifier_score'


def test_tie_randomization_matches_hypergeometric_expectation():
    # Two certain releases; draw two from a boundary group of four with one error.
    result = matched_coverage([1, 1, 1, 1, 1, 0], [.99, .99, .95, .95, .95, .95], 4)
    assert result['expected_wrong'] == .5
    assert result['expected_accuracy'] == .875
    assert result['boundary_tie_n'] == 4
    assert result['tie_randomization_error_interval95'] == {'low': 0., 'high': 1.}
    with pytest.raises(ValueError):
        matched_coverage([1], [.9], 2)


def test_full_cohort_denominator_and_expected_query_cost():
    sample = rows(3, 'evaluation')
    sample[1]['generator_answer'] = None
    dec = [{'action': 'assert', 'used': 1, 'cost': .01, 'failure': None},
           {'action': 'abstain', 'used': 0, 'cost': 0., 'failure': 'invalid_candidate'},
           {'action': 'assert', 'used': 2, 'cost': .03, 'failure': None}]
    metric = policy_metrics(sample, dec, 1., 19., 10, 1)
    assert metric['coverage'] == 2 / 3
    assert metric['queries'] == 3
    assert metric['wrong_released'] == 1
    assert metric['mean_utility']['estimate'] == pytest.approx((-19 + 1 - .04) / 3)


@pytest.mark.parametrize('kind', ['none', 'always', 'gate', 'myopic', 'nested'])
def test_invalid_prior_validation(kind):
    planner = NestedPlanner([], [], 1., 1.)
    with pytest.raises(ValueError, match='belief'):
        decide_policy(float('nan'), [], planner, kind)


def test_unknown_policy_and_length_mismatches_are_rejected():
    planner = NestedPlanner([], [], 1., 1.)
    with pytest.raises(ValueError, match='unknown'):
        decide_policy(.5, [], planner, 'typo')
    with pytest.raises(ValueError, match='length'):
        run_policy(rows(2), [.5], planner, [], 'none')


def test_one_layer_nested_equals_myopic_on_grid():
    world = VerificationWorld([.8], [.75], [.015], loss=4.)
    planner = NestedPlanner(world.channels, world.costs, 1., 4.)
    for b in np.linspace(0., 1., 101):
        for score in (.25, .75):
            nested = decide_policy(float(b), [score], planner, 'nested')
            myopic = decide_policy(float(b), [score], planner, 'myopic')
            assert nested == myopic


@pytest.mark.parametrize('rho', [0., .5, 1.])
def test_correlated_joint_preserves_marginals(rho):
    probabilities = [.2, .6, .8]
    joint = joint_distribution(probabilities, rho)
    assert sum(joint.values()) == pytest.approx(1.)
    for i, p in enumerate(probabilities):
        assert sum(mass for bits, mass in joint.items() if bits[i]) == pytest.approx(p)


def test_uninformative_positive_cost_stops():
    world = VerificationWorld([.5, .5], [.5, .5], [.01, .01], loss=1.)
    for kind in ('nested', 'myopic', 'oracle'):
        result = world.expected(.5, kind)
        assert result['queries'] == 0
        assert result['utility'] == pytest.approx(0.)


def test_lookahead_pays_to_reach_later_useful_verifier():
    world = VerificationWorld([.5, .9], [.5, .9], [.001, .01], loss=1.)
    nested = world.expected(.5, 'nested')
    assert nested['utility'] == pytest.approx(.389)
    assert nested['queries'] == pytest.approx(2.)
    assert nested['regret'] == pytest.approx(0., abs=1e-12)
    assert world.expected(.5, 'myopic')['utility'] == pytest.approx(0.)


@pytest.mark.parametrize('rho', [0., .8])
def test_exact_oracle_value_equals_integrated_policy_utility(rho):
    world = VerificationWorld([.6, .75, .9], [.6, .75, .9], [.002, .01, .05], rho=rho)
    for b in (.25, .5, .9, .95, .99):
        result = world.expected(b, 'oracle')
        assert result['utility'] == pytest.approx(world.oracle(b)['value'], abs=1e-12)
        assert world.expected(b, 'nested')['utility'] <= result['utility'] + 1e-10


def test_grid_values_converge_to_exact_independent_oracle():
    world = VerificationWorld([.6, .75, .9], [.6, .75, .9], [.002, .01, .05])
    errors = []
    for size in (101, 10001):
        planner = NestedPlanner(world.channels, world.costs, 1., 19., grid_size=size)
        errors.append(max(abs(planner.decide(0, float(b))['value'] - world.oracle(float(b))['value'])
                          for b in np.linspace(.01, .99, 37)))
    assert errors[1] < errors[0]
    assert errors[1] < 1e-4


def test_end_to_end_offline_is_deterministic_and_preserves_sources(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('offline analysis attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    source = tmp_path / 'records.json'
    source.write_text(json.dumps(rows() + rows(12, 'evaluation')))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'cost_mapping': {'utility_per_usd': 10.},
                                 'verifiers': [{'id': t} for t in ('a', 'b', 'c')],
                                 'planner': {'costs': [.001, .01, .02]}}))
    cfg = config()
    cfg.update(records=str(source), vertex_policy=str(policy), jev_policy=str(policy))
    before = [file_digest(p) for p in (source, policy)]
    first = run(cfg, tmp_path / 'first')
    second = run(cfg, tmp_path / 'second')
    assert first == second
    assert first['api_calls'] == 0 and first['source_artifacts_unchanged']
    assert before == [file_digest(p) for p in (source, policy)]
    changed = deepcopy(cfg)
    changed['seed'] += 1
    with pytest.raises(ValueError, match='different analysis protocol'):
        run(changed, tmp_path / 'first')
