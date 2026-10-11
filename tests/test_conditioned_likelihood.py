"""Confidence-conditioned likelihoods: fitting boundary, shrinkage, fixed table choice, no peeking, integrity."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa import conditioned_likelihood as cl
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.raw_prior_analysis import evaluate as raw_prior_evaluate
from vgx.gpqa.score import VerifierLikelihood

TAG = 'gemini_flash:probability'
BINARY = 'gemini_flash:binary'
SETTINGS = {'correct_reward': 1., 'primary_incorrect_loss': 19., 'sensitivity_incorrect_losses': [99.],
            'utility_per_usd': 10., 'grid_size': 1001, 'probability_bins': 3, 'laplace': 1., 'folds': 3,
            'bootstrap_repeats': 50, 'prior': 'unmodified_generator_p_correct_including_endpoints'}
SHRINKAGE = {'tau_grid': [1, 5, 20, 100], 'inner_folds': 3, 'sensitivity_tau': [5, 1e9]}


def cfg():
    return {'verifier': TAG, 'confidence_threshold': .95, 'shrinkage': SHRINKAGE}


def source_rows(n=120):
    """Synthetic raw-prior rows; one invalid generator answer and one unparsable verifier response."""
    priors = [.8, .9, .95, .98, .99, 1.]
    rows = []
    for i in range(n):
        prior = priors[i % len(priors)]
        outcome = int((i * 7) % 10 < (8 if prior >= .95 else 5))
        score = (.9 if outcome else (.95 if i % 3 else .1)) if prior >= .95 else (.85 if outcome else (.2 if i % 2 else .8))
        signal = {'score': score, 'failure': None, 'estimated_usd': .001 + 1e-5 * (i % 4), 'request_key': f'sha256:{i}'}
        if i == 7:
            signal = {'score': None, 'failure': 'invalid_json', 'estimated_usd': .002, 'request_key': f'sha256:{i}'}
        binary = {'score': float(score is not None and score > .5), 'failure': None, 'estimated_usd': .001,
                  'request_key': f'sha256:b{i}'}
        rows.append({'item_id': f'q{i:03d}', 'partition': 'calibration', 'answer': None if i == 11 else 'A',
                     'prior': prior, 'outcome': outcome, 'signals': {TAG: signal, BINARY: binary}})
    return rows


def frozen_fit(rows, seed=3):
    config = {**SETTINGS, 'binary_bins': 2}
    fit = raw_prior_evaluate(deepcopy(rows), [BINARY, TAG], {BINARY: 'binary', TAG: 'probability'}, config, seed)
    assert fit['status'] == 'complete_exploratory_oof'
    return json.loads(json.dumps(fit))


def signals_of(rows):
    return {r['item_id']: r['signals'][TAG] for r in rows if r['answer'] == 'A'}


def run_evaluate(rows=None, seed=3, config=None):
    rows = rows or source_rows()
    fit = frozen_fit(rows, seed)
    mine = cl.load_rows(fit, signals_of(rows), TAG)
    frozen = {loss: fit['decisions'][loss]['sequential:' + TAG] for loss in fit['decisions']}
    return mine, fit, cl.evaluate(mine, fit, config or cfg(), SETTINGS, frozen)


def test_shrinkage_gives_valid_distributions_and_empty_cells_fall_back_to_pooled():
    pooled = (.2, .1, .7)
    for counts in ([0, 0, 5], [3, 1, 0], [10, 20, 30]):
        for tau in (.5, 5, 500):
            values, fallback = cl.shrink(counts, pooled, tau)
            assert not fallback and min(values) > 0 and sum(values) == pytest.approx(1, abs=1e-12)
    assert cl.shrink([0, 0, 0], pooled, 5) == (pooled, True)
    assert cl.shrink([0, 0, 4], pooled, 1e9)[0] == pytest.approx(pooled, abs=1e-7)
    for bad in (0, -1, float('inf'), True):
        with pytest.raises(ValueError):
            cl.shrink([1, 1, 1], pooled, bad)
    with pytest.raises(ValueError):
        cl.shrink([1, 1], pooled, 5)


def test_empty_confidence_group_uses_training_pooled_table():
    fitting = [{'item_id': str(i), 'prior': .98, 'outcome': i % 2, 'valid': True,
                'signal': {'score': .9 if i % 2 else .2}} for i in range(12)]
    pooled = cl.fit_pooled(fitting, 3, 1.)
    tables, fallbacks = cl.conditioned_tables(pooled, cl.group_counts(fitting, pooled.edges, .95), 5)
    assert sorted(fallbacks) == ['low:correct', 'low:incorrect']
    assert tables['low'] == pooled
    for table in tables.values():
        NestedPlanner([table], [.01], 1., 19.)  # validates both distributions


def test_confidence_group_uses_original_confidence_with_inclusive_threshold():
    assert cl.confidence_group(.95, .95) == 'high'
    assert cl.confidence_group(.9499999, .95) == 'low'
    assert cl.confidence_group(1., .95) == 'high'
    with pytest.raises(ValueError):
        cl.confidence_group(None, .95)


def test_fitting_and_tau_selection_receive_training_rows_only(monkeypatch):
    received = []
    original = cl.fit_fold

    def spy(train, *args, **kwargs):
        received.append([r['item_id'] for r in train])
        return original(train, *args, **kwargs)
    monkeypatch.setattr(cl, 'fit_fold', spy)
    _, fit, _ = run_evaluate()
    assert received == [fold['train_ids'] for fold in fit['folds']]
    for ids, fold in zip(received, fit['folds']):
        assert not set(ids) & set(fold['validation_ids'])


def test_fitted_tables_and_tau_do_not_depend_on_held_out_rows():
    rows = cl.load_rows(frozen_fit(source_rows()), signals_of(source_rows()), TAG)
    fit = frozen_fit(source_rows())
    fold = fit['folds'][0]
    lookup = {r['item_id']: r for r in rows}
    train = [lookup[i] for i in fold['train_ids']]
    first = cl.fit_fold(deepcopy(train), fold['arms'][TAG], SETTINGS, SHRINKAGE, .95, 3)
    for item in fold['validation_ids']:  # held-out labels and signals are not inputs
        lookup[item]['outcome'] = 1 - lookup[item]['outcome']
        if lookup[item]['signal']:
            lookup[item]['signal']['score'] = 0.
    second = cl.fit_fold(deepcopy(train), fold['arms'][TAG], SETTINGS, SHRINKAGE, .95, 3)
    assert first['tables'] == second['tables'] and first['tau_selection'] == second['tau_selection']


def test_tau_selection_ties_go_to_larger_tau_and_degenerate_strata_fall_back():
    fitting = [{'item_id': str(i), 'prior': .98, 'outcome': int(i > 0), 'valid': True,
                'signal': {'score': .9}} for i in range(6)]
    selected = cl.select_tau(fitting, [1, 5, 20], 5, 0, .95, 3, 1.)
    assert selected['tau'] == 20 and selected['reason'] == 'insufficient_inner_strata_largest_tau'


class SpyPlanner(NestedPlanner):
    def __init__(self, name, calls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name, self.calls = name, calls

    def decide(self, stage, belief):
        self.calls.append((self.name, stage))
        return super().decide(stage, belief)

    def update(self, stage, belief, score):
        self.calls.append((self.name, 'update'))
        return super().update(stage, belief, score)


def test_table_is_fixed_by_original_confidence_even_after_posterior_crosses_threshold():
    calls = []
    strong = VerifierLikelihood((0, 1/3, 2/3, 1), (.05, .05, .9), (.6, .2, .2))
    weak = VerifierLikelihood((0, 1/3, 2/3, 1), (.3, .3, .4), (.3, .3, .4))
    planners = {'low': SpyPlanner('low', calls, [strong], [.001], 1., 19.),
                'high': SpyPlanner('high', calls, [weak], [.001], 1., 19.)}
    signal = {'score': .9, 'failure': None, 'estimated_usd': 1e-4}
    result = cl.decide_conditioned(.94, True, planners, .95, lambda: signal, expected_cost_usd=1e-4)
    assert result['likelihood_group'] == 'low'
    assert result['posterior'] > .95 and result['action'] == 'assert'
    assert result['posterior'] == pytest.approx(strong.posterior(.94, .9))
    assert {name for name, _ in calls} == {'low'}  # the high table is never consulted


def test_no_signal_read_before_a_selected_query_and_no_label_input():
    assert 'outcome' not in inspect.signature(cl.decide_conditioned).parameters
    table = VerifierLikelihood((0, 1/3, 2/3, 1), (.05, .05, .9), (.6, .2, .2))
    planners = {g: NestedPlanner([table], [.001], 1., 19.) for g in cl.GROUPS}

    def forbidden():
        raise AssertionError('signal read without a query decision')
    for prior in (0., .05, 1.):  # stop immediately: abstain, abstain, release
        result = cl.decide_conditioned(prior, True, planners, .95, forbidden, expected_cost_usd=.001)
        assert result['verifiers_used'] == 0 and result['expected_cost_usd'] == 0
    reads = []
    result = cl.decide_conditioned(.96, True, planners, .95,
                                   lambda: reads.append(1) or {'score': .9, 'failure': None, 'estimated_usd': 3e-4},
                                   expected_cost_usd=.001)
    assert reads == [1] and result['verifiers_used'] == 1
    assert result['decision']['action'] == 'query'  # chosen before the read
    invalid = cl.decide_conditioned(None, False, planners, .95, forbidden, expected_cost_usd=.001)
    assert invalid['action'] == 'abstain' and invalid['likelihood_group'] is None


def test_end_to_end_reproduces_frozen_pooled_policy_and_accounts_costs():
    rows, fit, result = run_evaluate()
    n_losses = len(fit['decisions'])
    assert result['frozen_pooled_agreements'] == len(rows) * n_losses
    assert result['replay_agreements'] == sum(r['valid'] for r in rows) * n_losses * (len(result['policy_names']) - 1)
    cost = {i: fold['arms'][TAG]['expected_cost_usd'] for fold in fit['folds'] for i in fold['validation_ids']}
    for loss, policies in result['decisions'].items():
        for name, decisions in policies.items():
            for row, d in zip(rows, decisions):
                if d['verifiers_used']:
                    assert d['expected_cost_usd'] == cost[row['item_id']]
                    assert d['usage_estimated_cost_usd'] == row['signal']['estimated_usd']
                else:
                    assert d['expected_cost_usd'] == 0 and d['usage_estimated_cost_usd'] == 0
                if not row['valid']:
                    assert d['action'] == 'abstain' and d['failure'] == 'invalid_generator'
                elif name != 'raw_confidence':
                    assert d['likelihood_group'] == cl.confidence_group(row['prior'], .95)
                if row['signal'] and row['signal']['score'] is None and d['verifiers_used']:
                    assert d['action'] == 'abstain'
    # tau -> infinity reproduces the pooled decisions exactly.
    for policies in result['decisions'].values():
        assert [(d['action'], d['verifiers_used']) for d in policies['conditioned_tau_1e+09']] == [
            (d['action'], d['verifiers_used']) for d in policies['pooled']]
    for fold in result['folds']:
        for tables in fold['tables'].values():
            for table in tables.values():
                assert sum(table['p_bin_if_correct']) == pytest.approx(1) and sum(table['p_bin_if_incorrect']) == pytest.approx(1)


def test_tampered_signal_breaks_frozen_reproduction():
    rows = source_rows()
    fit = frozen_fit(rows)
    signals = signals_of(rows)
    victim = fit['folds'][0]['train_ids'][0]
    signals[victim] = {**signals[victim], 'score': 1 - (signals[victim]['score'] or 0)}
    mine = cl.load_rows(fit, signals, TAG)
    frozen = {loss: fit['decisions'][loss]['sequential:' + TAG] for loss in fit['decisions']}
    with pytest.raises(ValueError, match='frozen'):
        cl.evaluate(mine, fit, cfg(), SETTINGS, frozen)


def write_package(root, rows, seed=3):
    """A miniature frozen package: source config, raw-prior fit and nested-trace observations."""
    fit = frozen_fit(rows, seed)
    atomic_json(root/'configs/source.json', {**SETTINGS, 'seeds': [seed], 'cohorts': ['all_calibration'],
                                            'binary_bins': 2})
    atomic_json(root/f'results/raw/gemini/all_calibration/seed_{seed}.json', fit)
    trace = root/f'results/nested/gemini/all_calibration/seed_{seed}'
    atomic_json(trace/'results.json', {'item_ids': [r['item_id'] for r in rows]})
    atomic_json(trace/'decisions_loss_19.0.json', {'always_selected_three': [
        {'queried_tags': [TAG], 'observations': [r['signals'][TAG]]} if r['answer'] == 'A' else {}
        for r in rows]})
    config = json.loads(Path('configs/gpqa_conditioned_likelihood.json').read_text())
    config.update(source_config='configs/source.json', source_results='results/raw',
                  signal_trace_results='results/nested', generators=['gemini'], cohorts=['all_calibration'],
                  seeds=[seed], frozen_settings=SETTINGS, shrinkage={**config['shrinkage'], **SHRINKAGE})
    atomic_json(root/'configs/conditioned.json', config)
    files = [p for p in root.rglob('*') if p.is_file()]
    atomic_json(root/'FILE_MANIFEST.json', {'files': {str(p.relative_to(root)): file_digest(p) for p in files}})
    return root/'configs/conditioned.json'


def test_analysis_leaves_frozen_inputs_unchanged_and_checks_manifest(tmp_path, monkeypatch):
    rows = source_rows(90)
    config = write_package(tmp_path, rows)
    before = {str(p): file_digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    report = cl.analyze(config, tmp_path, tmp_path/'out')
    assert report['source_integrity_unchanged'] and report['new_api_spend_usd'] == 0
    assert {p: file_digest(p) for p in before} == before
    protocol = json.loads((tmp_path/'out/protocol.json').read_text())
    assert protocol['package_manifest_check']['files_checked'] == len(protocol['input_sha256'])
    assert cl.analyze(config, tmp_path, tmp_path/'out')['analysis_id'] == report['analysis_id']

    source = tmp_path/'results/raw/gemini/all_calibration/seed_3.json'
    original_evaluate = cl.evaluate

    def mutating(*args, **kwargs):
        value = original_evaluate(*args, **kwargs)
        source.write_text(source.read_text() + ' ')
        return value
    monkeypatch.setattr(cl, 'evaluate', mutating)
    with pytest.raises(ValueError, match='changed'):
        cl.analyze(config, tmp_path, tmp_path/'out2')
    monkeypatch.undo()
    with pytest.raises(ValueError, match='manifest'):
        cl.analyze(config, tmp_path, tmp_path/'out3')
