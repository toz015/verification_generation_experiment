"""Exhaustive, probability-only fixed three-verifier experiment, using cached calls.

Fit/ordering inputs are training folds only. Every ordered triple is reported;
only training-J0 selections are described as selected policies. No provider run
or generator-feedback call is available from this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
from itertools import combinations, permutations
import json
from pathlib import Path
import platform

import numpy as np

from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.generator_calibration import source, index_cache
from vgx.gpqa.matched_coverage import release_weights
from vgx.gpqa.nested_combinations import execute_public, planner_for
from vgx.gpqa.raw_prior_analysis import candidate_valid, baseline_decision, summarize_policy
from vgx.gpqa.report import mean_interval
from vgx.gpqa.score import fit_verifier_likelihood
from vgx.gpqa.sequential import seal
from vgx.gpqa.signal_study import runner_for, request_for, parse_signal


def all_orders(tags):
    if len(tags) != 5 or len(set(tags)) != 5 or any(not t.endswith(':probability') for t in tags):
        raise ValueError('five distinct probability verifiers required')
    return [tuple(order) for combo in combinations(sorted(tags), 3) for order in permutations(combo)]


def combo_id(order):
    return '+'.join(sorted(t.split(':')[0] for t in order))


def order_id(order):
    return '>'.join(t.split(':')[0] for t in order)


def table_hash(planner):
    h = hashlib.sha256()
    for table in [planner.grid, *planner.j, *planner.q]:
        h.update(np.asarray(table, dtype='<f8').tobytes())
    return h.hexdigest()


def rank_orders(arms, training_priors, cfg, loss, tags):
    """Label-free ranking once training likelihoods/costs have been fitted."""
    ranked, planners = [], {}
    for order in [*(tuple([t]) for t in sorted(tags)), *all_orders(tags)]:
        planner, costs = planner_for(order, arms, cfg, loss)
        key = order_id(order)
        planners[key] = planner, costs
        ranked.append({'order': list(order), 'order_id': key, 'combination': combo_id(order),
                       'length': len(order), 'training_model_value': float(np.interp(
                           training_priors, planner.grid, planner.j[0]).mean()),
                       'expected_cost_usd': costs, 'table_sha256': table_hash(planner)})
    ranked.sort(key=lambda r: (-r['training_model_value'], r['order']))
    return ranked, planners


def run_public(prior, valid, planner, order, costs, signals, *, force_all=False):
    """A fixed public plan sees only its queried prefix. No correctness input.

    Feedback records represent Algorithm 2 line 18's payload, not actual LLM
    private-belief updating. Frozen generators are never called again here.
    """
    reads = []
    def query(stage, tag):
        if stage != len(reads) or tag != order[stage]:
            raise AssertionError('future or out-of-order signal access')
        reads.append(tag)
        return signals[tag]
    decision = execute_public(prior, valid, planner, order, costs, query, force_all=force_all)
    if reads != list(order[:decision['verifiers_used']]):
        raise AssertionError('query accounting mismatch')
    agreement = 0
    if valid and not force_all:
        reads.clear()
        class Lazy:
            def __len__(self): return len(order)
            def __getitem__(self, index): return query(index, order[index])['score']
        replay = planner.replay(prior, Lazy())
        if any(replay[k] != decision[k] for k in ('action', 'posterior', 'verifiers_used', 'failure')):
            raise AssertionError('executor/replay disagreement')
        if len(reads) != replay['verifiers_used']:
            raise AssertionError('replay read a future signal')
        agreement = 1
    decision['feedback_events'] = [
        {'stage': k + 1, 'verifier': t, 'score': o['score'], 'delivery': 'offline_record_only'}
        for k, (t, o) in enumerate(zip(decision.get('queried_tags', []), decision.get('observations', [])))]
    return decision, agreement


def validate_fold(rows, fit, fold, tags, cfg):
    lookup = {r['item_id']: r for r in rows}
    tr, va = set(fold['train_ids']), set(fold['validation_ids'])
    if tr & va or tr | va != set(lookup):
        raise ValueError('invalid fitting/held-out partition')
    for tag in tags:
        usable = [lookup[i] for i in fold['train_ids'] if candidate_valid(lookup[i])
                  and lookup[i]['signals'][tag]['score'] is not None]
        cost_rows = [lookup[i] for i in fold['train_ids'] if candidate_valid(lookup[i])]
        arm = fold['arms'][tag]
        if set(arm['likelihood_fit_ids']) != {r['item_id'] for r in usable}:
            raise ValueError('likelihood fitting IDs differ')
        if set(arm['cost_fit_ids']) != {r['item_id'] for r in cost_rows}:
            raise ValueError('cost fitting IDs differ')
        channel = fit_verifier_likelihood([r['outcome'] for r in usable],
            [r['signals'][tag]['score'] for r in usable], bins=cfg['probability_bins'], laplace=cfg['laplace'])
        if digest(asdict(channel)) != digest(arm['likelihood']):
            raise ValueError('cached training likelihood differs from refit')
        cost = np.mean([r['signals'][tag]['estimated_usd'] for r in cost_rows])
        if not np.isclose(cost, arm['expected_cost_usd'], rtol=1e-12, atol=1e-15):
            raise ValueError('training-only expected cost differs')


def summarize(rows, decisions, raw, flash, folds, cfg, loss, seed, intervals):
    summary = summarize_policy(rows, decisions, cfg, loss)
    u = np.array(summary.pop('utility_per_item'))
    summary['conditional_error'] = (summary['wrong_released']/summary['released'] if summary['released'] else None)
    summary['query_count_histogram'] = dict(sorted(Counter(d['verifiers_used'] for d in decisions).items()))
    summary['stop_states'] = dict(Counter(f"{d['verifiers_used']}:{d['action']}" for d in decisions))
    summary['query_cost_per_question_usd'] = summary['usage_estimated_cost_usd']/len(rows)
    summary['query_cost_per_release_usd'] = (summary['usage_estimated_cost_usd']/summary['released']
                                            if summary['released'] else None)
    summary['mean_reported_posterior_on_released'] = float(np.mean(
        [d['posterior'] for d in decisions if d['action'] == 'assert'])) if summary['released'] else None
    for name, base in [('raw', raw), ('flash', flash)]:
        before = summarize_policy(rows, base, cfg, loss)
        delta = u-np.array(before['utility_per_item'])
        summary['utility_difference_vs_'+name] = float(delta.mean())
        summary['transitions_vs_'+name] = dict(Counter(
            ('kept' if a['action'] == b['action'] else 'added' if a['action'] == 'assert' else 'removed')
            + ('_correct' if r['outcome'] else '_wrong')
            for r, a, b in zip(rows, decisions, base) if a['action'] == 'assert' or b['action'] == 'assert'))
        if intervals:
            summary['paired_utility_vs_'+name] = mean_interval(delta, cfg['bootstrap_repeats'], seed)
    lookup = {r['item_id']: i for i, r in enumerate(rows)}
    weights = np.zeros(len(rows))
    for fold in folds:
        ix = [lookup[q] for q in fold['validation_ids']]
        k = sum(decisions[i]['action'] == 'assert' for i in ix)
        weights[ix] = release_weights([rows[i]['prior'] for i in ix], [candidate_valid(rows[i]) for i in ix], k)
    y = np.array([r['outcome'] for r in rows])
    matched_u = weights*np.where(y, cfg['correct_reward'], -loss)
    summary['matched_raw_expected_wrong'] = float((weights*(1-y)).sum())
    summary['utility_difference_vs_matched_raw'] = float((u-matched_u).mean())
    if intervals:
        summary['paired_utility_vs_matched_raw'] = mean_interval(u-matched_u, cfg['bootstrap_repeats'], seed)
    return summary


def evaluate(rows, fit, cfg, run_cfg, tags, output):
    output.mkdir(parents=True, exist_ok=True)
    ids = [r['item_id'] for r in rows]
    if ids != fit['item_ids'] or [r['prior'] for r in rows] != fit['raw_priors']:
        raise ValueError('frozen input mismatch')
    if [r['outcome'] for r in rows] != fit['outcomes'] or len(set(ids)) != len(ids):
        raise ValueError('frozen outcome/membership mismatch')
    valid = [candidate_valid(r) for r in rows]
    lookup = {q: i for i, q in enumerate(ids)}
    seen = Counter(q for f in fit['folds'] for q in f['validation_ids'])
    if set(seen) != set(ids) or any(n != 1 for n in seen.values()):
        raise ValueError('each question must be held out exactly once')
    for f in fit['folds']:
        validate_fold(rows, fit, f, tags, cfg)
    result = {'item_ids': ids, 'seed': fit['seed'], 'policies': {}, 'selections': {}, 'replay_agreements': 0,
              'likelihood_refit_checks': len(tags)*len(fit['folds'])}
    for loss in run_cfg['losses']:
        loss_key = str(loss)
        raw = [baseline_decision(r['prior'], v, loss, cfg['correct_reward']) for r, v in zip(rows, valid)]
        predictions = {'raw': raw, 'abstain_all': [dict(d, action='abstain') for d in raw]}
        selections = []
        for fold_id, fold in enumerate(fit['folds']):
            priors = [rows[lookup[q]]['prior'] for q in fold['train_ids'] if valid[lookup[q]]]
            ranked, planners = rank_orders(fold['arms'], priors, cfg, loss, tags)
            triples = [r for r in ranked if r['length'] == 3]
            singles = [r for r in ranked if r['length'] == 1]
            by_combo = {}
            for r in triples:
                by_combo.setdefault(r['combination'], r)
            cheapest = sorted(tags, key=lambda t: (fold['arms'][t]['expected_cost_usd'], t))
            choices = {**{'single:'+r['order_id']: (r['order'], False) for r in singles},
                       **{'triple:'+r['order_id']: (r['order'], False) for r in triples},
                       'selected_single': (singles[0]['order'], False),
                       'selected_triple': (triples[0]['order'], False),
                       'always_selected_triple': (triples[0]['order'], True),
                       'cheapest_three': (cheapest[:3], False)}
            for combo, best in by_combo.items():
                ascending = [t for t in cheapest if t in best['order']]
                choices['selected_combo:'+combo] = best['order'], False
                choices['cost_order_combo:'+combo] = ascending, False
                choices['always_combo:'+combo] = sorted(best['order']), True
            selection = seal({'fold': fold_id, 'train_ids': fold['train_ids'], 'validation_ids': fold['validation_ids'],
                'ranked_orders': ranked, 'selected_triple': triples[0], 'selected_single': singles[0],
                'best_by_combo': by_combo, 'cheapest_order': cheapest,
                'settings': {'reward': cfg['correct_reward'], 'loss': loss, 'utility_per_usd': cfg['utility_per_usd'],
                             'grid_size': cfg['grid_size']},
                'fitted_arms': {t: fold['arms'][t] for t in tags}}, 'selection_id')
            selections.append(selection)
            selected_planner = planners[triples[0]['order_id']][0]
            atomic_json(output/f'planner_loss_{loss_key}_fold_{fold_id}.json',
                        seal({'selection_id': selection['selection_id'], 'order': triples[0]['order'],
                              'planner': selected_planner.to_dict()}, 'policy_id'))
            for name, (order, force) in choices.items():
                predictions.setdefault(name, [None]*len(rows))
                planner, costs = planners[order_id(order)]
                # Equivalent aliases share decisions, not extra simulated calls.
                if name.startswith(('single:', 'triple:', 'always_combo:')) or name == 'always_selected_triple':
                    for q in fold['validation_ids']:
                        i = lookup[q]
                        d, check = run_public(rows[i]['prior'], valid[i], planner, order, costs,
                                              rows[i]['signals'], force_all=force)
                        predictions[name][i] = d
                        result['replay_agreements'] += check
                else:
                    original = ('single:' if len(order) == 1 else 'triple:')+order_id(order)
                    for q in fold['validation_ids']:
                        predictions[name][lookup[q]] = predictions[original][lookup[q]]
        flash = predictions['single:gemini_flash']
        summaries = {}
        for name, ds in predictions.items():
            if any(d is None for d in ds):
                raise ValueError('incomplete out-of-fold decisions')
            summaries[name] = summarize(rows, ds, raw, flash, fit['folds'], cfg, loss, fit['seed'],
                intervals=name in ('selected_single', 'selected_triple', 'cheapest_three', 'always_selected_triple')
                          or name.startswith('selected_combo:'))
        # Original singleton policies are a frozen reference for this new implementation.
        for tag in tags:
            old = fit['decisions'][loss_key]['sequential:'+tag]
            new = predictions['single:'+tag.split(':')[0]]
            for a, b in zip(old, new):
                if any(a[k] != b[k] for k in ('action', 'posterior', 'verifiers_used', 'expected_cost_usd',
                                             'usage_estimated_cost_usd')):
                    raise ValueError('probability singleton differs from frozen reference')
        result['policies'][loss_key] = summaries
        result['selections'][loss_key] = selections
        with gzip.open(output/f'decisions_loss_{loss_key}.json.gz', 'wt', encoding='utf-8') as f:
            json.dump({'item_ids': ids, 'decisions': predictions}, f, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))
    return result


def load_rows(source_cfg, run_cfg):
    plan, groups, labels, pilot, label_hash, bundle_id = source(Path(source_cfg['generator_results']))
    screen = json.loads(Path(source_cfg['screen_config']).read_text())
    models = {m['id']: m for m in screen['models']}
    if set(run_cfg['verifiers']) != set(models):
        raise ValueError('five frozen verifier identities required')
    index = index_cache(source_cfg['cache_roots'])
    rows, used = {}, {}
    for g in run_cfg['generators']:
        rows[g] = []
        for c in groups[g]:
            if c.partition != 'calibration': raise ValueError('development candidates only')
            row = {'item_id': c.item_id, 'partition': c.partition, 'answer': c.answer, 'prior': c.p_correct,
                   'outcome': int(c.answer is not None and 'ABCD'.index(c.answer) == labels[c.item_id]), 'signals': {}}
            if candidate_valid(row):
                for name in run_cfg['verifiers']:
                    model = models[name]; runner = runner_for(model, screen['project'])
                    req = request_for(c, model, 'probability'); key = runner.cache_key(req)
                    if key not in index: raise ValueError('missing frozen request: '+key)
                    call = index[key]
                    if (call['prompt'] != req.prompt or call['meta']['system'] != req.system
                        or call['meta']['runner_identity'] != runner.identity or call['params'] != runner.params
                        or call['meta']['resource_project'] != screen['project']
                        or call['meta']['quota_project'] != screen['project']):
                        raise ValueError('request identity/billing project mismatch')
                    est = estimate_call(call, screen['pricing'])['estimated_usd']
                    if est is None: raise ValueError('unpriced call')
                    score, failure = parse_signal(call['response'], 'probability')
                    row['signals'][name+':probability'] = {'score': score, 'failure': failure,
                                                         'estimated_usd': est, 'request_key': key}
                    used[key] = digest(call)
            rows[g].append(row)
    return rows, pilot, {'bundle_id': bundle_id, 'label_sha256': label_hash,
        'candidate_hashes': {g: digest([asdict(c) for c in groups[g]]) for g in run_cfg['generators']},
        'cache_hashes': used, 'project': screen['project'], 'models': screen['models'], 'pricing': screen['pricing']}


def analyze(config_path, output):
    config_path, output = Path(config_path), Path(output)
    run_cfg = json.loads(config_path.read_text())
    if run_cfg['new_api_requests'] != 0 or run_cfg['evaluation_labels_read'] or run_cfg['signal'] != 'probability':
        raise ValueError('offline probability-only protocol required')
    cfg = json.loads(Path(run_cfg['source_config']).read_text())
    source_cfg = dict(cfg)
    cfg.update(utility_per_usd=run_cfg['primary_utility_per_usd'], bootstrap_repeats=run_cfg['bootstrap_repeats'])
    rows, pilot, identity = load_rows(source_cfg, run_cfg)
    paths = [Path(run_cfg['source_results'])/g/c/f'seed_{s}.json'
             for g in rows for c in cfg['cohorts'] for s in cfg['seeds']]
    impl = ['probability_triples.py', 'nested_combinations.py', 'planner.py', 'raw_prior_analysis.py',
            'score.py', 'generator_calibration.py', 'signal_study.py', 'matched_coverage.py', 'report.py']
    protocol = seal({'config': run_cfg, 'source_config': source_cfg, 'frozen_sources': identity,
        'config_sha256': file_digest(config_path), 'source_config_sha256': file_digest(run_cfg['source_config']),
        'source_files': {str(p): file_digest(p) for p in paths},
        'implementation': {n: file_digest(Path(__file__).with_name(n)) for n in impl},
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                    **{p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')}},
        'new_api_requests': 0, 'evaluation_labels_read': False, 'confidence_calibration': False,
        'feedback_delivery': 'offline_record_only', 'scope': 'development-only exhaustive fixed-order comparison'}, 'analysis_id')
    path = output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('protocol changed; choose a new output directory')
    atomic_json(path, protocol)
    tags = [m+':probability' for m in run_cfg['verifiers']]
    results = {}
    for g in run_cfg['generators']:
        for cohort in cfg['cohorts']:
            subset = rows[g] if cohort == 'all_calibration' else [r for r in rows[g] if r['item_id'] not in pilot]
            for seed in cfg['seeds']:
                rel = Path(g)/cohort/f'seed_{seed}'; target = output/rel/'results.json'
                if target.exists():
                    completed = json.loads(target.read_text())
                    if completed.get('analysis_id') != protocol['analysis_id']:
                        raise ValueError('stale completed result')
                    value = completed
                else:
                    fit = json.loads((Path(run_cfg['source_results'])/(str(rel)+'.json')).read_text())
                    value = {'analysis_id': protocol['analysis_id'], **evaluate(subset, fit, cfg, run_cfg, tags, output/rel)}
                    value['output_hashes'] = {p.name: file_digest(p) for p in sorted((output/rel).iterdir()) if p.is_file()}
                    atomic_json(target, value)
                if any(file_digest(output/rel/p) != h for p, h in value['output_hashes'].items()):
                    raise ValueError('result artifact changed')
                results[str(rel)] = {k: v for k, v in value.items() if k not in ('item_ids', 'selections', 'output_hashes')}
                status = {'analysis_id': protocol['analysis_id'], 'completed_runs': len(results), 'total_runs': len(paths),
                          'last_finished': str(rel), 'status': 'running', 'new_api_requests': 0}
                atomic_json(output/'status.json', status)
                print(json.dumps(status), flush=True)
    _, _, end_identity = load_rows(source_cfg, run_cfg)
    if identity != end_identity or any(file_digest(p) != h for p, h in protocol['source_files'].items()):
        raise ValueError('frozen input changed')
    report = {'analysis_id': protocol['analysis_id'], 'results': results, 'source_integrity_unchanged': True,
              'new_api_spend_usd': 0., 'confirmed_billing_usd': None, 'evaluation_labels_read': False}
    atomic_json(output/'report.json', report)
    atomic_json(output/'status.json', {**status, 'status': 'complete'})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpqa_probability_triples.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.config, args.output)
