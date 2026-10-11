"""Risk-controlled acceptance: exact bounds, multiplicity, partitions, leakage, failures, costs, replay, integrity."""
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import socket

import numpy as np
import pytest
from scipy.stats import beta

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa import risk_certification as cert
from vgx.gpqa import risk_control as rc
from vgx.gpqa import risk_exploration as rx
from vgx.gpqa.raw_prior_analysis import evaluate as raw_prior_evaluate

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'configs'/'gpqa_risk_acceptance.json').read_text())
TAG = CONFIG['verifier']
BINARY = 'gemini_flash:binary'
SETTINGS = CONFIG['frozen_settings']
PRIORS = [.6, .8, .9, .95, .98, .99, 1.]


# ---------------------------------------------------------------- synthetic data

def synthetic_items(generator, n=300):
    """One invalid candidate and one malformed (paid) verifier response per generator."""
    shift = 0 if generator == 'qwen' else 3
    items = []
    for i in range(n):
        prior = PRIORS[(i + shift) % len(PRIORS)]
        correct = int((i*7 + shift) % 10 < (9 if prior >= .95 else 5))
        score = (.9 if correct else (.92 if i % 4 == 0 else .1)) if prior >= .9 else (.8 if correct else .3)
        signal = {'score': score, 'failure': None, 'usage_usd': .001 + 1e-5*(i % 5), 'request_key': f'sha256:{generator}{i}'}
        if i == 7:
            signal = {'score': None, 'failure': 'invalid_json', 'usage_usd': .002, 'request_key': f'sha256:{generator}{i}'}
        valid = i != 11
        items.append({'question_id': f'q{i:04d}', 'valid': valid, 'raw_confidence': prior, 'correct': correct,
                      'signal': signal if valid else None})
    return items


def synthetic_export(n=300):
    out = {'schema': 1, 'verifier': TAG, 'generators': {}}
    for g in ('qwen', 'gemini'):
        items = synthetic_items(g, n)
        out['generators'][g] = {'items': items, 'cohorts': {'all_calibration': {'question_ids': [i['question_id'] for i in items]}}}
    return out


def roles_for(export, seed=5):
    ids = export['generators']['qwen']['cohorts']['all_calibration']['question_ids']
    return rc.partition_questions(ids, CONFIG['certification']['dry_run_partition']['fractions'], seed)


def relabel(export, roles, role, flip=True):
    """Flip correctness labels of one role only."""
    out = deepcopy(export)
    for g in out['generators'].values():
        for item in g['items']:
            if roles[item['question_id']] == role and flip:
                item['correct'] = 1 - item['correct']
    return out


def planner_parts(rows):
    likelihood, cost = rc.fit_pooled(rows, SETTINGS['probability_bins'], SETTINGS['laplace'])
    return likelihood, cost, rc.sequential_planner(likelihood, cost, SETTINGS, CONFIG['sequential_base_loss'])


# ---------------------------------------------------------------- exact bounds

def test_binomial_upper_bound_zero_errors_all_errors_and_empty():
    assert rc.binomial_upper_bound(0, 0, .05) is None
    assert rc.binomial_upper_bound(5, 5, .01) == 1.
    for m in (1, 10, 59, 400):
        assert rc.binomial_upper_bound(0, m, .05) == pytest.approx(1 - .05**(1/m), rel=1e-10)
    assert rc.binomial_upper_bound(3, 100, .05) == pytest.approx(beta.ppf(.95, 4, 97), rel=1e-12)
    bounds = [rc.binomial_upper_bound(k, 100, .05) for k in range(101)]
    assert all(a < b for a, b in zip(bounds, bounds[1:]))
    assert rc.binomial_upper_bound(2, 50, .05/26) > rc.binomial_upper_bound(2, 50, .05)
    for bad in ((3, 2, .05), (-1, 2, .05), (1., 2, .05), (1, 2, 0.), (1, 2, 1.)):
        with pytest.raises(ValueError):
            rc.binomial_upper_bound(*bad)


def test_reference_sample_sizes_and_vectorized_error_allowance():
    assert rc.min_accepted(0, .05, .05) == 59 and rc.binomial_upper_bound(0, 58, .05) > .05
    assert rc.min_accepted(0, .05, .05/20) == 117 and rc.binomial_upper_bound(0, 116, .05/20) > .05
    plan = cert.sample_size_plan(CONFIG)
    assert all(r['computed'] == r['expected'] for r in plan['reference_checks'])
    for alpha, e in ((.05, .05), (.1, .05/26)):
        ms = np.arange(1, 800)
        assert (rc.max_errors_allowed_many(ms, alpha, e) == [rc.max_errors_allowed(int(m), alpha, e) for m in ms]).all()


def test_power_requirement_is_none_at_or_above_alpha_and_met_when_returned():
    assert rc.accepted_for_power(.05, .05, .05) is None
    m = rc.accepted_for_power(.02, .05, .05/26, .8)
    assert all(rc.pass_probability(m + j, .02, .05, .05/26) >= .8 for j in range(26))
    assert rc.accepted_for_power(.02, .05, .05/26, .8) > rc.accepted_for_power(.02, .05, .05, .8)


# ---------------------------------------------------------------- multiplicity and selection

def rule(rule_id, m, k, n=500, cost=0.):
    return {'rule_id': rule_id, 'm': m, 'k': k, 'N': n, 'expected_cost_usd': cost}


def test_bonferroni_counts_every_declared_rule_including_empty_ones():
    single, _ = rc.certify_bonferroni([rule('a', 59, 0)], .05, .05)
    assert single[0]['certified']
    family = [rule('a', 59, 0)] + [rule(f'empty{i}', 0, 0) for i in range(25)]
    table, chosen = rc.certify_bonferroni(family, .05, .05)
    assert {r['family_size'] for r in table} == {26} and {r['per_rule_error'] for r in table} == {.05/26}
    assert chosen is None and not any(r['certified'] for r in table)
    assert all(r['upper_bound'] is None for r in table if r['m'] == 0)


def test_declared_family_matches_config_and_order():
    family = cert.declared_family(CONFIG)
    assert len(family) == 26 == len({r['rule_id'] for r in family})
    assert family[0]['rule_id'] == 'qwen:A:0.95' and family[-1]['rule_id'] == 'gemini:C:0.995'
    thresholds = [r['threshold'] for r in family if (r['generator'], r['method']) == ('qwen', 'B')]
    assert thresholds == sorted(thresholds)


def test_no_rule_certified_reports_no_positive_coverage_rule():
    table, chosen = rc.certify_bonferroni([rule('a', 30, 0), rule('b', 0, 0)], .05, .05)
    assert chosen is None and CONFIG['certification']['no_rule_message'] == 'No positive-coverage rule certified.'


def test_selection_prefers_coverage_then_cost_then_declared_order_never_labels():
    rules = [rule('first', 300, 2, cost=2.), rule('cheap', 300, 0, cost=1.), rule('narrow', 200, 0, cost=0.)]
    _, chosen = rc.certify_bonferroni(rules, .05, .05)
    assert chosen['rule_id'] == 'cheap'
    rules[1]['k'] = 3  # different labels, still certified: selection unchanged
    table, chosen = rc.certify_bonferroni(rules, .05, .05)
    assert all(r['certified'] for r in table) and chosen['rule_id'] == 'cheap'
    rules[1]['expected_cost_usd'] = 2.
    assert rc.certify_bonferroni(rules, .05, .05)[1]['rule_id'] == 'first'


def test_fixed_sequence_stops_at_first_failure_and_splits_delta_over_chains():
    chains = [[rule('strict', 10, 0), rule('lenient', 400, 2)], [rule('x', 400, 0)]]
    table, chosen = rc.certify_fixed_sequence(chains, .05, .05)
    by = {r['rule_id']: r for r in table}
    assert {r['per_test_error'] for r in table} == {.025}
    assert not by['strict']['certified'] and not by['lenient']['tested'] and not by['lenient']['certified']
    assert by['x']['certified'] and chosen['rule_id'] == 'x'


@pytest.mark.parametrize('scenario', ['all_rules_at_alpha', 'non_monotone_top_error'])
def test_monte_carlo_familywise_error_is_within_delta(scenario):
    chains = cert.validation_scenarios(.05)[scenario]
    result = cert.simulate_procedures(chains, 600, .05, .05, 1500, 11)
    for name, rate in result['familywise_error'].items():
        assert rate <= .05 + 3*math.sqrt(.05*.95/1500), name


# ---------------------------------------------------------------- partitions and leakage

def test_partition_is_question_level_label_free_disjoint_and_deterministic():
    ids = [f'q{i:04d}' for i in range(249)]
    fractions = CONFIG['certification']['dry_run_partition']['fractions']
    roles = rc.partition_questions(ids, fractions, 7)
    assert set(roles) == set(ids) and set(roles.values()) == {'fit', 'calibration', 'evaluation'}
    counts = {r: sum(v == r for v in roles.values()) for r in fractions}
    assert counts == {'fit': 85, 'calibration': 82, 'evaluation': 82}
    assert rc.partition_questions(list(reversed(ids)) + ids[:5], fractions, 7) == roles
    assert rc.partition_questions(ids, fractions, 8) != roles
    assert list(inspect.signature(rc.partition_questions).parameters) == ['question_ids', 'fractions', 'seed']


def test_fitting_reads_only_fit_role_labels():
    export = synthetic_export()
    roles = roles_for(export)
    base = cert.fit_models(export, roles, CONFIG)
    for role in ('calibration', 'evaluation'):
        other = cert.fit_models(relabel(export, roles, role), roles, CONFIG)
        for g in base:
            assert rc.table_dict(other[g]['likelihood']) == rc.table_dict(base[g]['likelihood'])
            assert other[g]['expected_cost_usd'] == base[g]['expected_cost_usd']
    changed = cert.fit_models(relabel(export, roles, 'fit'), roles, CONFIG)
    assert rc.table_dict(changed['qwen']['likelihood']) != rc.table_dict(base['qwen']['likelihood'])


def test_selection_reads_calibration_labels_but_never_evaluation_labels():
    export = synthetic_export()
    roles = roles_for(export)
    base = cert.calibrate(export, roles, CONFIG, .3)
    same = cert.calibrate(relabel(export, roles, 'evaluation'), roles, CONFIG, .3)
    assert same['selection'] == base['selection'] and same['certification_table'] == base['certification_table']
    moved = cert.calibrate(relabel(export, roles, 'calibration'), roles, CONFIG, .3)
    assert [r['k'] for r in moved['certification_table']] != [r['k'] for r in base['certification_table']]
    assert {s['question_id'] for s in base['calibration_items']['qwen']['A']} == \
        {q for q, r in roles.items() if r == 'calibration'} == {s['question_id'] for s in base['calibration_items']['gemini']['B']}


def test_scores_are_computed_without_labels():
    export = synthetic_export()
    roles = roles_for(export)
    models = cert.fit_models(export, roles, CONFIG)
    scored = cert.score_role(export, models, roles, 'evaluation', CONFIG)
    flipped = cert.score_role(relabel(export, roles, 'evaluation'), models, roles, 'evaluation', CONFIG)
    assert scored == flipped
    row = {k: v for k, v in rc.generator_rows(export, 'qwen')['q0003'].items() if k != 'correct'}
    for method in rc.METHODS:  # no 'correct' key anywhere: would raise KeyError if read
        rc.score_method(method, row, likelihood=models['qwen']['likelihood'],
                        expected_cost_usd=models['qwen']['expected_cost_usd'], planner=models['qwen']['planner'])


def test_evaluation_needs_the_sealed_selection_and_same_partition():
    export = synthetic_export()
    roles = roles_for(export)
    selection = cert.calibrate(export, roles, CONFIG, .3)['selection']
    tampered = {**selection, 'selected_rule': 'qwen:A:0.95'}
    with pytest.raises(ValueError, match='checksum'):
        cert.evaluate(export, roles, CONFIG, tampered)
    with pytest.raises(ValueError, match='partition'):
        cert.evaluate(export, roles_for(export, seed=6), CONFIG, selection)


def test_evaluation_reports_counts_intervals_queries_and_costs():
    export = synthetic_export()
    roles = roles_for(export)
    result = cert.calibrate(export, roles, CONFIG, .3)
    chosen = result['selection']['selected_rule']
    assert chosen is not None
    report = cert.evaluate(export, roles, CONFIG, result['selection'])
    row = report['rules'][chosen]
    for key in ('N', 'valid_candidates', 'm', 'k', 'conditional_error', 'conditional_error_cp95', 'acceptance_rate',
                'acceptance_rate_cp95', 'queries', 'expected_cost_usd', 'usage_cost_usd', 'usage_cost_per_question',
                'usage_cost_per_accepted'):
        assert key in row
    assert row['selected_overall'] and row['N'] == sum(v == 'evaluation' for v in roles.values())
    assert len(report['descriptive_family_on_evaluation']) == 26


def test_dry_run_requires_explicit_acknowledgement(tmp_path, monkeypatch):
    path = tmp_path/'obs.json'
    atomic_json(path, synthetic_export())
    monkeypatch.setattr('sys.argv', ['x', 'calibrate', '--observations', str(path), '--output', str(tmp_path/'out'),
                                     '--config', str(Path(__file__).resolve().parents[1]/'configs'/'gpqa_risk_acceptance.json')])
    with pytest.raises(SystemExit, match='dry-run'):
        cert.main()


# ---------------------------------------------------------------- rule semantics, failures and costs

def test_ties_at_a_threshold_are_accepted_or_rejected_together():
    scored = [{'question_id': f'q{i}', 'eligible': True, 'score': .97, 'queries': 0, 'expected_cost_usd': 0.,
               'usage_cost_usd': 0.} for i in range(4)] + [{'question_id': 'q9', 'eligible': False, 'score': None,
               'queries': 0, 'expected_cost_usd': 0., 'usage_cost_usd': 0.}]
    correct = {f'q{i}': i % 2 for i in range(4)} | {'q9': 1}
    assert rc.rule_counts(scored, correct, .97)['m'] == 4
    assert rc.rule_counts(scored, correct, .9700001)['m'] == 0
    assert rc.rule_counts(scored, correct, .97)['N'] == 5


def test_invalid_candidates_and_malformed_signals_stay_in_the_denominator():
    export = synthetic_export()
    rows = list(rc.generator_rows(export, 'qwen').values())
    likelihood, cost, planner = planner_parts(rows)
    invalid, malformed = rc.generator_rows(export, 'qwen')['q0011'], rc.generator_rows(export, 'qwen')['q0007']
    a = rc.score_method('A', invalid)
    b_invalid = rc.score_method('B', invalid, likelihood=likelihood, expected_cost_usd=cost)
    b_bad = rc.score_method('B', malformed, likelihood=likelihood, expected_cost_usd=cost)
    assert not a['eligible'] and a['failure'] == 'invalid_generator' and b_invalid['queries'] == 0
    assert not b_bad['eligible'] and b_bad['failure'] == 'invalid_json' and b_bad['queries'] == 1
    assert b_bad['usage_cost_usd'] == .002 and b_bad['expected_cost_usd'] == cost
    c_bad = rc.score_method('C', {**malformed, 'raw_confidence': .9}, expected_cost_usd=cost, planner=planner)
    assert c_bad['base_action'] == 'abstain' and not c_bad['eligible'] and c_bad['failure'] == 'invalid_json'
    scored = [rc.score_method('B', r, likelihood=likelihood, expected_cost_usd=cost) for r in rows]
    counts = rc.rule_counts(scored, {r['question_id']: r['correct'] for r in rows}, 0.)
    assert counts['N'] == len(rows) and counts['m'] == len(rows) - 2


def test_costs_for_always_query_and_sequential_policies():
    export = synthetic_export()
    rows = list(rc.generator_rows(export, 'qwen').values())
    likelihood, cost, planner = planner_parts(rows)
    b = [rc.score_method('B', r, likelihood=likelihood, expected_cost_usd=cost) for r in rows]
    c = [rc.score_method('C', r, expected_cost_usd=cost, planner=planner) for r in rows]
    assert sum(s['queries'] for s in b) == sum(r['valid'] for r in rows)
    assert 0 < sum(s['queries'] for s in c) < sum(s['queries'] for s in b)
    for scored in (b, c):
        for s, r in zip(scored, rows):
            assert s['expected_cost_usd'] == pytest.approx(s['queries']*cost)
            assert s['usage_cost_usd'] == (r['signal']['usage_usd'] if s['queries'] else 0.)
    assert cost == pytest.approx(np.mean([r['signal']['usage_usd'] for r in rows if r['valid']]))


class Untouchable(dict):
    def __getitem__(self, key):
        raise AssertionError('unselected signal was read')


def test_sequential_policy_reads_only_the_selected_signal():
    export = synthetic_export()
    rows = list(rc.generator_rows(export, 'qwen').values())
    likelihood, cost, planner = planner_parts(rows)
    skipped = 0
    for r in rows:
        if not r['valid']:
            continue
        decision = planner.decide(0, r['raw_confidence'])
        if decision['action'] != 'query':
            hidden = {**r, 'signal': Untouchable()}
            result = rc.score_method('C', hidden, expected_cost_usd=cost, planner=planner)
            assert result['queries'] == 0
            skipped += 1
    assert skipped > 0


def test_a_read_beyond_the_counted_queries_is_rejected(monkeypatch):
    export = synthetic_export()
    rows = list(rc.generator_rows(export, 'qwen').values())
    likelihood, cost, planner = planner_parts(rows)

    def peeking(row, planner, expected_cost_usd, query):
        query()  # look at the signal, then report no query
        return rc.score_generator_only(row)
    monkeypatch.setattr(rc, 'score_sequential', peeking)
    with pytest.raises(AssertionError, match='outside a selected query'):
        rc.score_method('C', rows[3], expected_cost_usd=cost, planner=planner)


def test_execution_and_replay_agree_and_a_mismatch_is_detected():
    export = synthetic_export()
    rows = list(rc.generator_rows(export, 'gemini').values())
    likelihood, cost, planner = planner_parts(rows)
    for r in rows:
        scored = rc.score_method('C', r, expected_cost_usd=cost, planner=planner)
        assert rc.replay_check(r, planner, scored)
    r = next(r for r in rows if r['valid'])
    scored = rc.score_method('C', r, expected_cost_usd=cost, planner=planner)
    assert not rc.replay_check(r, planner, {**scored, 'base_action': 'flip'})


# ---------------------------------------------------------------- frozen source package, integrity, no network

def source_rows(generator, n=90):
    rows = []
    for item in synthetic_items(generator, n):
        i = int(item['question_id'][1:])
        signal = item['signal'] or {'score': .5, 'failure': None, 'usage_usd': .001, 'request_key': f'sha256:x{i}'}
        rows.append({'item_id': item['question_id'], 'partition': 'calibration', 'answer': 'A' if item['valid'] else None,
                     'prior': item['raw_confidence'], 'outcome': item['correct'],
                     'signals': {TAG: {'score': signal['score'], 'failure': signal['failure'],
                                       'estimated_usd': signal['usage_usd'], 'request_key': signal['request_key']},
                                 BINARY: {'score': 1., 'failure': None, 'estimated_usd': .001, 'request_key': f'b{i}'}}})
    return rows


def make_package(root, seeds=(3,)):
    cfg = deepcopy(CONFIG)
    cfg.update(source_config='configs/src.json', source_results='results/src', signal_trace_results='results/traces',
               seeds=list(seeds), cohorts=['all_calibration'])
    source_cfg = {**SETTINGS, 'sensitivity_incorrect_losses': [99.], 'binary_bins': 2, 'bootstrap_repeats': 20}
    atomic_json(root/'configs/src.json', source_cfg)
    files = {}
    for g in cfg['generators']:
        rows = source_rows(g)
        for seed in seeds:
            fit = raw_prior_evaluate(deepcopy(rows), [BINARY, TAG], {BINARY: 'binary', TAG: 'probability'}, source_cfg, seed)
            atomic_json(root/f'results/src/{g}/all_calibration/seed_{seed}.json', fit)
        trace = root/f'results/traces/{g}/run/seed_1'
        atomic_json(trace/'results.json', {'item_ids': [r['item_id'] for r in rows]})
        atomic_json(trace/'decisions_loss_19.json', {'always_query': [
            {'queried_tags': [TAG], 'observations': [r['signals'][TAG]]} if r['answer'] else {} for r in rows]})
    for path in sorted(root.rglob('*.json')):
        files[str(path.relative_to(root))] = file_digest(path)
    atomic_json(root/'FILE_MANIFEST.json', {'files': files})
    return cfg


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('network access attempted')
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    monkeypatch.setattr('vgx.gpqa.signal_study.runner_for', refuse)


def test_export_reproduces_frozen_fits_and_decisions_offline(tmp_path, no_network):
    cfg = make_package(tmp_path)
    export = rc.export_observations(tmp_path, cfg)
    assert digest(rc.export_observations(tmp_path, cfg)) == digest(export)
    assert export['provenance']['original_request_caches_reparsed'] is False
    assert export['provenance']['package_manifest_check']['files_checked'] == len(export['provenance']['input_sha256'])
    for g in cfg['generators']:
        scored, folds, checks = rx.score_run(export, g, 'all_calibration', 3, cfg)
        assert checks == {'pooled_refit_matches': 3, 'always_query_matches': 90, 'sequential_matches': 90,
                          'replay_matches': 90}
        assert [s['question_id'] for s in scored['A']] == export['generators'][g]['cohorts']['all_calibration']['question_ids']


def test_export_rejects_changed_sources_and_settings(tmp_path):
    cfg = make_package(tmp_path)
    path = tmp_path/'results/src/qwen/all_calibration/seed_3.json'
    value = json.loads(path.read_text())
    value['raw_priors'][0] = .5
    atomic_json(path, value)
    with pytest.raises(ValueError, match='manifest'):
        rc.export_observations(tmp_path, cfg)
    changed = deepcopy(cfg)
    changed['frozen_settings']['laplace'] = 2.
    with pytest.raises(ValueError, match='frozen setting'):
        rc.export_observations(tmp_path, changed)


def test_frozen_fold_mismatch_is_detected(tmp_path):
    cfg = make_package(tmp_path)
    export = rc.export_observations(tmp_path, cfg)
    bad = deepcopy(export)
    bad['generators']['qwen']['cohorts']['all_calibration']['seeds']['3']['folds'][0]['expected_cost_usd'] *= 1.01
    with pytest.raises(ValueError, match='refit'):
        rx.score_run(bad, 'qwen', 'all_calibration', 3, cfg)
    bad = deepcopy(export)
    bad['generators']['qwen']['cohorts']['all_calibration']['seeds']['3']['frozen_decisions']['19.0']['sequential'][0]['action'] = 'x'
    with pytest.raises(AssertionError, match='sequential'):
        rx.score_run(bad, 'qwen', 'all_calibration', 3, cfg)


def test_exploration_run_is_offline_and_refuses_a_changed_protocol(tmp_path, monkeypatch, no_network):
    cfg = make_package(tmp_path/'pkg')
    export_path, config_path = tmp_path/'obs.json', tmp_path/'cfg.json'
    atomic_json(export_path, rc.export_observations(tmp_path/'pkg', cfg))
    cfg['exploration']['bootstrap_repeats'] = 20
    atomic_json(config_path, cfg)
    argv = ['x', 'run', '--config', str(config_path), '--observations', str(export_path), '--output', str(tmp_path/'out')]
    monkeypatch.setattr('sys.argv', argv)
    rx.main()
    result = json.loads((tmp_path/'out'/'exploration.json').read_text())
    assert result['checks']['replay_matches'] == 180
    assert {r['method'] for r in result['family_rules']} == {'A', 'B', 'C'}
    cfg['exploration']['bootstrap_repeats'] = 21
    atomic_json(config_path, cfg)
    with pytest.raises(ValueError, match='protocol changed'):
        rx.main()


def test_exploratory_region_and_matched_coverage_helpers():
    points = [{'threshold': t, 'm': m, 'k': k, 'N': 100, 'acceptance_rate': m/100, 'conditional_error': k/m,
               'upper_bound_single_rule_95': 0.} for t, m, k in ((.99, 10, 2), (.98, 40, 2), (.9, 80, 9))]
    assert rx.low_risk_region(points, .05)['threshold'] == .98
    assert rx.low_risk_region(points, .01) is None
    assert rx.matched(points, .3)['threshold'] == .98 and rx.matched(points, .9) is None
