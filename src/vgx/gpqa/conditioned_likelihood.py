"""Offline single-verifier experiment with confidence-conditioned likelihood tables.

Pooled baseline: one training-fold table P(score bin | Y), exactly as in
raw_prior_analysis. Conditioned arm: P(score bin | Y, group) for the
generator's original-confidence group (c < threshold, c >= threshold), each
shrunk toward the same training fold's pooled table. The original confidence
selects the table once per item; value tables, Bayes updates, stopping rules,
priors and costs are unchanged.

Inputs are frozen artifacts only: raw-prior fits (splits, raw confidences,
calibration outcomes, pooled tables, expected costs) and the cached verifier
responses recorded in frozen nested-combination traces. No API transport, no
candidate regeneration and no Diamond answer keys.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.raw_prior_analysis import baseline_decision, decide_public, summarize_policy
from vgx.gpqa.report import mean_interval
from vgx.gpqa.score import VerifierLikelihood, fit_verifier_likelihood
from vgx.gpqa.sequential import seal

GROUPS = ('low', 'high')
IMPLEMENTATION = ('conditioned_likelihood.py', 'raw_prior_analysis.py', 'planner.py', 'score.py', 'report.py')


def confidence_group(confidence, threshold):
    """Group from the generator's original confidence only, never from an updated belief."""
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence):
        raise ValueError('finite original confidence required')
    return 'low' if confidence < threshold else 'high'


def bin_counts(outcomes, scores, edges):
    """Raw per-outcome bin counts using fit_verifier_likelihood's bin assignment."""
    bins = len(edges) - 1
    scores = np.asarray(scores, dtype=float)
    ids = np.clip(np.searchsorted(np.asarray(edges), scores, side='right') - 1, 0, bins - 1).astype(int)
    y = np.asarray(outcomes, dtype=int)
    return {label: np.bincount(ids[y == label], minlength=bins).astype(int).tolist() for label in (1, 0)}


def shrink(counts, pooled, tau):
    """(n_k + tau*pooled_k) / (N + tau). An empty cell returns the pooled table (fallback=True)."""
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')
    if len(counts) != len(pooled) or any(n < 0 for n in counts):
        raise ValueError('counts must be nonnegative and match the pooled bins')
    total = sum(counts)
    if total == 0:
        return tuple(float(p) for p in pooled), True
    return tuple((n + tau*p)/(total + tau) for n, p in zip(counts, pooled)), False


def fitting_rows(train):
    """Training items with a valid generator answer and a parsed verifier score."""
    return [r for r in train if r['valid'] and r['signal'] is not None and r['signal']['score'] is not None]


def fit_pooled(fitting, bins, laplace):
    return fit_verifier_likelihood([r['outcome'] for r in fitting], [r['signal']['score'] for r in fitting],
                                   bins=bins, laplace=laplace)


def group_counts(fitting, edges, threshold):
    counts = {}
    for group in GROUPS:
        members = [r for r in fitting if confidence_group(r['prior'], threshold) == group]
        counts[group] = bin_counts([r['outcome'] for r in members], [r['signal']['score'] for r in members], edges)
    return counts


def conditioned_tables(pooled, counts, tau):
    tables, fallbacks = {}, []
    for group in GROUPS:
        p1, fb1 = shrink(counts[group][1], pooled.p_bin_if_correct, tau)
        p0, fb0 = shrink(counts[group][0], pooled.p_bin_if_incorrect, tau)
        fallbacks += [f'{group}:correct'] * fb1 + [f'{group}:incorrect'] * fb0
        tables[group] = VerifierLikelihood(pooled.edges, p1, p0)
    return tables, fallbacks


def _log_probability(table, row):
    index = table.bin_index(row['signal']['score'])
    return math.log(table.p_bin_if_correct[index] if row['outcome'] else table.p_bin_if_incorrect[index])


def select_tau(fitting, grid, inner_folds, seed, threshold, bins, laplace):
    """Inner cross-validation inside one training fold; nothing outside `fitting` is read."""
    grid = sorted(float(t) for t in grid)
    cells = [2*r['outcome'] + (confidence_group(r['prior'], threshold) == 'high') for r in fitting]
    outcomes = [r['outcome'] for r in fitting]
    strata, scheme = cells, 'outcome_x_group'
    k = min(inner_folds, min(Counter(cells).values()))
    if len(set(cells)) < 4 or k < 2:
        strata, scheme = outcomes, 'outcome_only'
        k = min(inner_folds, min(Counter(outcomes).values()))
    if len(set(outcomes)) < 2 or k < 2:
        return {'tau': grid[-1], 'reason': 'insufficient_inner_strata_largest_tau', 'scheme': None,
                'inner_folds': 0, 'mean_log_likelihood': None, 'pooled_mean_log_likelihood': None}
    totals = {tau: 0. for tau in grid}
    pooled_total, scored = 0., 0
    for tr, va in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(fitting)), strata):
        inner = [fitting[i] for i in tr]
        pooled = fit_pooled(inner, bins, laplace)
        counts = group_counts(inner, pooled.edges, threshold)
        for tau in grid:
            tables, _ = conditioned_tables(pooled, counts, tau)
            totals[tau] += sum(_log_probability(tables[confidence_group(fitting[i]['prior'], threshold)], fitting[i])
                               for i in va)
        pooled_total += sum(_log_probability(pooled, fitting[i]) for i in va)
        scored += len(va)
    means = {tau: totals[tau]/scored for tau in grid}
    # Ties (to 1e-12) go to the larger tau, i.e. closer to the pooled table.
    best = max(grid, key=lambda t: (round(means[t], 12), t))
    return {'tau': best, 'reason': 'inner_cv_max_log_likelihood', 'scheme': scheme, 'inner_folds': k,
            'mean_log_likelihood': {str(t): v for t, v in means.items()},
            'pooled_mean_log_likelihood': pooled_total/scored}


def fit_fold(train, frozen_arm, settings, shrinkage, threshold, seed):
    """Fit pooled and conditioned tables from training rows only, and check them against the frozen fit."""
    fitting = fitting_rows(train)
    if [r['item_id'] for r in fitting] != frozen_arm['likelihood_fit_ids']:
        raise ValueError('training likelihood items differ from the frozen pooled fit')
    pooled = fit_pooled(fitting, settings['probability_bins'], settings['laplace'])
    frozen = VerifierLikelihood(**{k: tuple(v) for k, v in frozen_arm['likelihood'].items()})
    for mine, theirs in ((pooled.edges, frozen.edges), (pooled.p_bin_if_correct, frozen.p_bin_if_correct),
                         (pooled.p_bin_if_incorrect, frozen.p_bin_if_incorrect)):
        if not np.allclose(mine, theirs, rtol=0, atol=1e-12):
            raise ValueError('refitted pooled table differs from the frozen pooled fit')
    cost_rows = [r for r in train if r['valid']]
    if [r['item_id'] for r in cost_rows] != frozen_arm['cost_fit_ids']:
        raise ValueError('training cost items differ from the frozen cost fit')
    cost = float(np.mean([r['signal']['estimated_usd'] for r in cost_rows]))
    if not math.isclose(cost, frozen_arm['expected_cost_usd'], rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('recomputed expected cost differs from the frozen cost')
    counts = group_counts(fitting, frozen.edges, threshold)
    selection = select_tau(fitting, shrinkage['tau_grid'], shrinkage['inner_folds'], seed, threshold,
                           settings['probability_bins'], settings['laplace'])
    variants = {'conditioned': selection['tau'], **{f'conditioned_tau_{t:g}': float(t) for t in shrinkage['sensitivity_tau']}}
    tables, fallbacks = {}, {}
    for name, tau in variants.items():
        tables[name], fallbacks[name] = conditioned_tables(frozen, counts, tau)
    return {'pooled': frozen, 'expected_cost_usd': frozen_arm['expected_cost_usd'], 'counts': counts,
            'group_sizes': {g: {'correct': sum(counts[g][1]), 'incorrect': sum(counts[g][0])} for g in GROUPS},
            'tau_selection': selection, 'variants': variants, 'tables': tables, 'fallbacks': fallbacks}


def decide_conditioned(prior, valid, planners, threshold, query, *, expected_cost_usd):
    """The table is chosen from the original confidence and kept for the whole item."""
    if not valid:
        result = baseline_decision(prior, False, next(iter(planners.values())).loss, next(iter(planners.values())).reward)
        result['likelihood_group'] = None
        return result
    group = confidence_group(prior, threshold)
    result = decide_public(prior, True, planners[group], query, expected_cost_usd=expected_cost_usd)
    result['likelihood_group'] = group
    return result


def _predicted_value(decision, prior, valid, reward, loss):
    if not valid:
        return 0.
    if 'decision' in decision:
        return float(decision['decision']['value'])
    return max(0., (reward + loss)*prior - loss)


def _category(d):
    if d.get('failure') == 'invalid_generator':
        return 'invalid_generator'
    if d['verifiers_used'] == 0:
        return 'release_without_query' if d['action'] == 'assert' else 'abstain_without_query'
    return 'query_then_release' if d['action'] == 'assert' else 'query_then_abstain'


def _posteriors(decisions):
    return np.array([np.nan if d['posterior'] is None else d['posterior'] for d in decisions], dtype=float)


def _same_decision(a, b):
    same_posterior = (a['posterior'] is None and b['posterior'] is None) or (
        a['posterior'] is not None and b['posterior'] is not None
        and math.isclose(a['posterior'], b['posterior'], rel_tol=0, abs_tol=1e-12))
    return (a['action'] == b['action'] and a['verifiers_used'] == b['verifiers_used'] and same_posterior
            and math.isclose(a['expected_cost_usd'], b['expected_cost_usd'], rel_tol=0, abs_tol=1e-15)
            and math.isclose(a['usage_estimated_cost_usd'], b['usage_estimated_cost_usd'], rel_tol=0, abs_tol=1e-15))


def _brier(p, y):
    return (np.asarray(p, float) - np.asarray(y, float))**2


def evaluate(rows, fit, cfg, settings, frozen_pooled):
    """rows follow fit['item_ids']; each fold's fitting sees only that fold's training rows."""
    tag, threshold, seed = cfg['verifier'], cfg['confidence_threshold'], fit['seed']
    reward, upd, repeats = settings['correct_reward'], settings['utility_per_usd'], settings['bootstrap_repeats']
    ids = [r['item_id'] for r in rows]
    if ids != fit['item_ids'] or [r['prior'] for r in rows] != fit['raw_priors'] or [r['outcome'] for r in rows] != fit['outcomes']:
        raise ValueError('frozen item order, raw confidence or outcomes mismatch')
    lookup = {item: i for i, item in enumerate(ids)}
    losses = [settings['primary_incorrect_loss'], *settings['sensitivity_incorrect_losses']]
    names = None
    predictions = {str(loss): {} for loss in losses}
    signal_posterior = {}
    folds, replay_agreements, frozen_agreements = [], 0, 0
    for fold_id, fold in enumerate(fit['folds']):
        tr = [lookup[x] for x in fold['train_ids']]
        va = [lookup[x] for x in fold['validation_ids']]
        if set(tr) & set(va):
            raise AssertionError('training/validation overlap')
        model = fit_fold([rows[i] for i in tr], fold['arms'][tag], settings, cfg['shrinkage'], threshold, seed)
        names = ['raw_confidence', 'pooled', *model['variants']]
        cost = model['expected_cost_usd']
        folds.append({'fold': fold_id, 'n_train': len(tr), 'n_validation': len(va),
                      'tau_selection': model['tau_selection'], 'variants': model['variants'],
                      'group_sizes': model['group_sizes'], 'bin_counts': {g: {'correct': model['counts'][g][1],
                          'incorrect': model['counts'][g][0]} for g in GROUPS},
                      'expected_cost_usd': cost, 'pooled_table': asdict(model['pooled']),
                      'tables': {n: {g: asdict(t) for g, t in ts.items()} for n, ts in model['tables'].items()},
                      'fallbacks': model['fallbacks']})
        # Forced-signal posteriors: evaluation-only, every eligible held-out signal is read after decisions.
        for name in ('pooled', *model['variants']):
            signal_posterior.setdefault(name, [None]*len(rows))
        for loss in losses:
            pooled_planner = NestedPlanner([model['pooled']], [upd*cost], reward, loss, grid_size=settings['grid_size'])
            planners = {name: {g: NestedPlanner([t], [upd*cost], reward, loss, grid_size=settings['grid_size'])
                               for g, t in tables.items()} for name, tables in model['tables'].items()}
            store = predictions[str(loss)]
            for name in names:
                store.setdefault(name, [None]*len(rows))
            for i in va:
                row = rows[i]
                group = confidence_group(row['prior'], threshold) if row['valid'] else None
                raw = baseline_decision(row['prior'], row['valid'], loss, reward)
                raw['likelihood_group'] = group
                raw['predicted_value'] = _predicted_value(raw, row['prior'], row['valid'], reward, loss)
                store['raw_confidence'][i] = raw
                for name in names[1:]:
                    reads = []

                    def query(row=row, reads=reads):
                        reads.append(tag)
                        return row['signal']
                    if name == 'pooled':
                        decision = decide_public(row['prior'], row['valid'], pooled_planner, query, expected_cost_usd=cost)
                        decision['likelihood_group'] = group
                        planner = pooled_planner
                        if not _same_decision(decision, frozen_pooled[str(loss)][i]):
                            raise AssertionError('pooled policy does not reproduce the frozen raw-prior decision')
                        frozen_agreements += 1
                    else:
                        decision = decide_conditioned(row['prior'], row['valid'], planners[name], threshold, query,
                                                      expected_cost_usd=cost)
                        planner = planners[name][group] if row['valid'] else None
                    if len(reads) != decision['verifiers_used']:
                        raise AssertionError('signal read outside a selected query')
                    if decision['verifiers_used'] and decision['usage_estimated_cost_usd'] != row['signal']['estimated_usd']:
                        raise AssertionError('query cost accounting mismatch')
                    if not decision['verifiers_used'] and (decision['expected_cost_usd'] or decision['usage_estimated_cost_usd']):
                        raise AssertionError('cost charged without a query')
                    if row['valid']:
                        class LazyScore:
                            def __len__(self):
                                return 1

                            def __getitem__(self, index, query=query):
                                if index != 0:
                                    raise AssertionError('future signal')
                                return query()['score']
                        reads.clear()
                        replay = planner.replay(row['prior'], LazyScore())
                        if (replay['action'], replay['posterior'], replay['verifiers_used']) != (
                                decision['action'], decision['posterior'], decision['verifiers_used']):
                            raise AssertionError('executor / replay mismatch')
                        if len(reads) != replay['verifiers_used']:
                            raise AssertionError('replay read an unselected signal')
                        replay_agreements += 1
                    decision['predicted_value'] = _predicted_value(decision, row['prior'], row['valid'], reward, loss)
                    store[name][i] = decision
        for i in va:
            row = rows[i]
            if row['valid'] and row['signal']['score'] is not None:
                signal_posterior['pooled'][i] = model['pooled'].posterior(row['prior'], row['signal']['score'])
                for name, tables in model['tables'].items():
                    table = tables[confidence_group(row['prior'], threshold)]
                    signal_posterior[name][i] = table.posterior(row['prior'], row['signal']['score'])
    y = np.array([r['outcome'] for r in rows])
    valid = np.array([r['valid'] for r in rows])
    groups = np.array([confidence_group(r['prior'], threshold) if r['valid'] else 'invalid' for r in rows])
    eligible = [i for i, r in enumerate(rows) if r['valid'] and r['signal']['score'] is not None]
    policies = {}
    for loss in losses:
        store = predictions[str(loss)]
        if any(d is None for values in store.values() for d in values):
            raise ValueError('missing held-out decision')
        per_item = {}
        summaries = {}
        for name, decisions in store.items():
            s = summarize_policy(rows, decisions, settings, loss)
            per_item[name] = np.array(s.pop('utility_per_item'))
            released = np.array([d['action'] == 'assert' for d in decisions])
            posterior = _posteriors(decisions)
            s['mean_posterior_released'] = float(posterior[released].mean()) if released.any() else None
            s['expected_wrong_released'] = float((1 - posterior[released]).sum())
            s['mean_predicted_value'] = float(np.mean([d['predicted_value'] for d in decisions]))
            s['mean_utility_interval'] = mean_interval(per_item[name], repeats, seed)
            s['stopping'] = dict(Counter(_category(d) for d in decisions))
            s['decision_posterior_brier_matched'] = float(_brier(posterior[eligible], y[eligible]).mean())
            s['by_group'] = {}
            for g in (*GROUPS, 'invalid'):
                ix = np.flatnonzero(groups == g)
                if not len(ix):
                    continue
                sub = summarize_policy([rows[i] for i in ix], [decisions[i] for i in ix], settings, loss)
                sub.pop('utility_per_item')
                rel = released[ix]
                sub['mean_posterior_released'] = float(posterior[ix][rel].mean()) if rel.any() else None
                sub['expected_wrong_released'] = float((1 - posterior[ix][rel]).sum())
                sub['mean_predicted_value'] = float(np.mean([decisions[i]['predicted_value'] for i in ix]))
                sub['stopping'] = dict(Counter(_category(decisions[i]) for i in ix))
                s['by_group'][g] = sub
            summaries[name] = s
        base = store['pooled']
        base_released = np.array([d['action'] == 'assert' for d in base])
        base_post = _posteriors(base)
        for name, decisions in store.items():
            if name == 'pooled':
                continue
            released = np.array([d['action'] == 'assert' for d in decisions])
            post = _posteriors(decisions)
            diffs = {
                'utility': per_item[name] - per_item['pooled'],
                'released': released.astype(float) - base_released,
                'wrong_released': (released & (y == 0)).astype(float) - (base_released & (y == 0)),
                'queries': np.array([d['verifiers_used'] for d in decisions], float)
                           - np.array([d['verifiers_used'] for d in base], float)}
            paired = {k: {**mean_interval(v, repeats, seed), 'sum': float(v.sum())} for k, v in diffs.items()}
            paired['decision_posterior_brier_matched'] = mean_interval(
                _brier(post[eligible], y[eligible]) - _brier(base_post[eligible], y[eligible]), repeats, seed)
            paired['by_group'] = {}
            for g in GROUPS:
                ix = np.flatnonzero(groups == g)
                paired['by_group'][g] = {k: {**mean_interval(v[ix], repeats, seed), 'sum': float(v[ix].sum())}
                                         for k, v in diffs.items()}
            paired['stopping_transitions_from_pooled'] = {
                f'{a}->{b}': n for (a, b), n in sorted(Counter((_category(p), _category(c))
                                                              for p, c in zip(base, decisions)).items())}
            summaries[name]['paired_vs_pooled'] = paired
        policies[str(loss)] = summaries
    # Matched-eligible posterior quality after observing the verifier (independent of stopping).
    yy = y[eligible]
    forecasts = {'n_matched_eligible': len(eligible),
                 'raw_prior_brier': float(_brier([rows[i]['prior'] for i in eligible], yy).mean()), 'signal_posterior': {}}
    pooled_signal = np.array([signal_posterior['pooled'][i] for i in eligible])
    for name, values in signal_posterior.items():
        p = np.array([values[i] for i in eligible])
        entry = {'brier': float(_brier(p, yy).mean()), 'mean_posterior': float(p.mean()),
                 'by_group': {}}
        if name != 'pooled':
            entry['brier_difference_vs_pooled'] = mean_interval(_brier(p, yy) - _brier(pooled_signal, yy), repeats, seed)
        for g in GROUPS:
            ix = [k for k, i in enumerate(eligible) if groups[i] == g]
            if ix:
                entry['by_group'][g] = {'n': len(ix), 'accuracy': float(yy[ix].mean()),
                                        'mean_raw_prior': float(np.mean([rows[eligible[k]]['prior'] for k in ix])),
                                        'mean_posterior': float(p[ix].mean()), 'brier': float(_brier(p[ix], yy[ix]).mean())}
                if name != 'pooled':
                    entry['by_group'][g]['brier_difference_vs_pooled'] = mean_interval(
                        _brier(p[ix], yy[ix]) - _brier(pooled_signal[ix], yy[ix]), repeats, seed)
        forecasts['signal_posterior'][name] = entry
    return {'status': 'complete_exploratory_oof', 'seed': seed, 'n': len(rows), 'valid_generator_n': int(valid.sum()),
            'item_ids': ids, 'fold_assignment': fit['fold_assignment'], 'groups': groups.tolist(),
            'policy_names': names, 'folds': folds, 'policies': policies, 'forecasts': forecasts,
            'signal_posteriors': signal_posterior, 'decisions': predictions,
            'replay_agreements': replay_agreements, 'frozen_pooled_agreements': frozen_agreements}


def collect_signals(trace_root, generator, tag):
    """Cached verifier responses as recorded in frozen nested-combination traces, keyed by item."""
    records, files = {}, []
    for path in sorted((trace_root/generator).glob('*/seed_*/decisions_loss_*.json')):
        ids = json.loads((path.parent/'results.json').read_text())['item_ids']
        files += [path, path.parent/'results.json']
        for decisions in json.loads(path.read_text()).values():
            for item, d in zip(ids, decisions):
                for queried, obs in zip(d.get('queried_tags', []), d.get('observations', [])):
                    if queried != tag:
                        continue
                    value = {k: obs[k] for k in ('score', 'failure', 'estimated_usd', 'request_key')}
                    if item in records and records[item] != value:
                        raise ValueError('inconsistent cached verifier record for one item')
                    records[item] = value
    return records, sorted(set(files))


def load_rows(fit, signals, tag):
    """Validity follows the frozen raw-confidence decisions (invalid generator answers abstain)."""
    raw = fit['decisions'][next(iter(fit['decisions']))]['raw']
    rows = []
    for item, prior, outcome, decision in zip(fit['item_ids'], fit['raw_priors'], fit['outcomes'], raw):
        valid = decision.get('failure') != 'invalid_generator'
        if valid and item not in signals:
            raise ValueError('valid candidate without a cached verifier record')
        rows.append({'item_id': item, 'prior': prior, 'outcome': outcome, 'valid': valid,
                     'signal': signals.get(item) if valid else None})
    return rows


def verify_cache(signals, root, source_cfg, tag):
    """Optional: re-parse every used record from the original request caches (needs the full project)."""
    from vgx.common.billing import estimate_call
    from vgx.gpqa.generator_calibration import index_cache
    from vgx.gpqa.signal_study import parse_signal
    screen = json.loads((root/source_cfg['screen_config']).read_text())
    index = index_cache([str(root/p) for p in source_cfg['cache_roots']])
    hashes = {}
    for generator, records in signals.items():
        for item, record in records.items():
            call = index[record['request_key']]
            score, failure = parse_signal(call['response'], tag.split(':')[1])
            usd = estimate_call(call, screen['pricing'])['estimated_usd']
            if (score, failure) != (record['score'], record['failure']) or not math.isclose(usd, record['estimated_usd'], rel_tol=1e-12):
                raise ValueError('frozen trace record differs from the original cache')
            hashes[record['request_key']] = digest(call)
    return hashes


def analyze(config_path, root, output, *, check_cache=False):
    config_path, root, output = Path(config_path), Path(root), Path(output)
    cfg = json.loads(config_path.read_text())
    if cfg['new_api_requests'] != 0 or cfg['evaluation_labels_read'] or cfg['diamond_answer_keys_read']:
        raise ValueError('offline development-partition analysis only')
    source_cfg_path = root/cfg['source_config']
    source_cfg = json.loads(source_cfg_path.read_text())
    settings = cfg['frozen_settings']
    for key, value in settings.items():
        if source_cfg[key] != value:
            raise ValueError(f'frozen setting {key} differs from the source analysis')
    if not set(cfg['seeds']) <= set(source_cfg['seeds']) or not set(cfg['cohorts']) <= set(source_cfg['cohorts']):
        raise ValueError('seeds and cohorts must come from the source analysis')
    source_root, trace_root, tag = root/cfg['source_results'], root/cfg['signal_trace_results'], cfg['verifier']
    signals, inputs = {}, [source_cfg_path]
    for generator in cfg['generators']:
        signals[generator], files = collect_signals(trace_root, generator, tag)
        inputs += files
    fit_paths = {(g, c, s): source_root/g/c/f'seed_{s}.json'
                 for g in cfg['generators'] for c in cfg['cohorts'] for s in cfg['seeds']}
    inputs += list(fit_paths.values())
    before = {str(p.relative_to(root)): file_digest(p) for p in inputs}
    manifest_path = root/'FILE_MANIFEST.json'
    manifest_check = None
    if manifest_path.exists():
        listed = json.loads(manifest_path.read_text())['files']
        mismatched = [p for p, h in before.items() if listed.get(p) != h]
        if mismatched:
            raise ValueError(f'input files differ from the package manifest: {mismatched[:3]}')
        manifest_check = {'manifest_sha256': file_digest(manifest_path), 'files_checked': len(before)}
    cache_hashes = verify_cache(signals, root, source_cfg, tag) if check_cache else None
    protocol = seal({'config': cfg, 'config_sha256': file_digest(config_path),
                     'source_config_sha256': file_digest(source_cfg_path),
                     'input_sha256': before, 'package_manifest_check': manifest_check,
                     'signal_table_sha256': {g: digest(v) for g, v in signals.items()},
                     'cache_record_sha256': cache_hashes,
                     'implementation_sha256': {n: file_digest(Path(__file__).with_name(n)) for n in IMPLEMENTATION},
                     'package_versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scikit-learn')},
                     'new_api_requests': 0, 'evaluation_labels_read': False, 'diamond_answer_keys_read': False},
                    'analysis_id')
    path = output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('protocol changed; use a new output directory')
    atomic_json(path, protocol)  # Written before any comparison is computed.
    summary = {}
    for (generator, cohort, seed), fit_path in fit_paths.items():
        fit = json.loads(fit_path.read_text())
        rows = load_rows(fit, signals[generator], tag)
        frozen = {loss: fit['decisions'][loss]['sequential:'+tag] for loss in fit['decisions']}
        result = evaluate(rows, fit, cfg, settings, frozen)
        rel = Path(generator)/cohort/f'seed_{seed}'
        atomic_json(output/rel/'decisions.json', {'item_ids': result['item_ids'], 'groups': result['groups'],
            'fold_assignment': result['fold_assignment'], 'priors': [r['prior'] for r in rows],
            'outcomes': [r['outcome'] for r in rows], 'signals': [r['signal'] for r in rows],
            'signal_posteriors': result['signal_posteriors'], 'decisions': result['decisions']})
        body = {k: v for k, v in result.items() if k not in ('decisions', 'signal_posteriors', 'item_ids', 'groups',
                                                             'fold_assignment')}
        atomic_json(output/rel/'results.json', body)
        summary[str(rel)] = {'replay_agreements': result['replay_agreements'],
                             'frozen_pooled_agreements': result['frozen_pooled_agreements']}
        print(json.dumps({'finished': str(rel), **summary[str(rel)]}), flush=True)
    after = {str(p.relative_to(root)): file_digest(p) for p in inputs}
    if after != before:
        raise ValueError('frozen inputs changed during analysis')
    if check_cache and verify_cache(signals, root, source_cfg, tag) != cache_hashes:
        raise ValueError('cached responses changed during analysis')
    report = {'analysis_id': protocol['analysis_id'], 'runs': summary, 'source_integrity_unchanged': True,
              'new_api_spend_usd': 0, 'confirmed_billing_usd': None, 'evaluation_labels_read': False,
              'diamond_answer_keys_read': False, 'limitations': cfg['notes']}
    atomic_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpqa_conditioned_likelihood.json')
    parser.add_argument('--root', default='.', help='project or review-package root holding configs/ and results/')
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-cache', action='store_true', help='re-parse records from the original request caches')
    args = parser.parse_args()
    print(json.dumps(analyze(args.config, args.root, args.output, check_cache=args.check_cache)['analysis_id']))
