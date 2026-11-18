"""Grouped 5-fold cross-fitting of the GPQA sequential-verification method on QuantumKatas.

Reuses the GPQA implementations unchanged: per-verifier 3-bin likelihoods
(fit_verifier_likelihood), conditional-independence JointLikelihood, the exact
nested planner, prefix-only execute(), raw-prior baseline and policy summary.
For each held-out fold, everything is fitted on the other four folds only.
Labels come from the official checker; a candidate without code, a valid raw r
or an official label is invalid and the policy abstains on it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.dependence_triples import INF, ExactPlanner, JointLikelihood, all_orders, execute
from vgx.gpqa.diamond_score import boot
from vgx.gpqa.raw_prior_analysis import baseline_decision, summarize_policy
from vgx.gpqa.report import rate_interval
from vgx.gpqa.score import binary_forecast_metrics, fit_verifier_likelihood
from vgx.quantum_katas import oracle
from vgx.quantum_katas.run_pilot import Run, SignalStore


def _call_usd(cache, key, pricing):
    rows = [r for r in cache.log(key).records() if r.get('key') == key and r.get('error') is None]
    return estimate_call(rows[0], pricing)['estimated_usd'] if rows else 0.0


def load_rows(root, split, verifiers):
    """One row per planned task (no task is dropped): prior, validity, label and cost-annotated signals."""
    run = Run(root)
    if run.plan.get('split_sha256') != split['sha256']:
        raise ValueError('run was not prepared from this split')
    pricing = run.config['budget']['pricing']
    cache = run.cache()
    candidates = run.candidates.by_task()
    official = oracle.OfficialStore(run.root/'private'/'official_results.jsonl').results()
    signals = SignalStore(run.root/'verifier_signals.jsonl').index()
    rows = []
    for task in run.tasks:
        c = candidates.get(task.task_id)
        label = official.get(c.candidate_id) if c is not None else None
        sig = {}
        for v in verifiers:
            s = signals.get((c.candidate_id, v)) if c is not None and c.has_code else None
            sig[v] = {'score': s['p_correct'] if s else None, 'failure': (s or {}).get('failure', 'not_collected'),
                      'estimated_usd': _call_usd(cache, s['request_key'], pricing) if s and s.get('execution_id') else 0.0}
        rows.append({'task_id': task.task_id, 'fold': int(split['value']['task_fold'][task.task_id]),
                     'prior': c.raw_p_correct if c is not None else None,
                     'has_code': bool(c and c.has_code), 'labeled': label is not None,
                     'valid': bool(c and c.has_code and c.has_valid_confidence and label is not None),
                     'outcome': int(label['label']) if label is not None else 0,
                     'generator_usd': _call_usd(cache, c.request_key, pricing) if c is not None else 0.0,
                     'signals': sig})
    if [r['task_id'] for r in rows] != run.plan['task_ids']:
        raise ValueError('task order changed')
    return rows


def fit_fold(train, verifiers, settings):
    valid = [r for r in train if r['valid']]
    fitted = {'likelihoods': {}, 'expected_cost_usd': {}}
    for v in verifiers:
        usable = [r for r in valid if r['signals'][v]['score'] is not None]
        table = fit_verifier_likelihood([r['outcome'] for r in usable], [r['signals'][v]['score'] for r in usable],
                                        bins=settings['probability_bins'], laplace=settings['laplace'])
        fitted['likelihoods'][v] = {**asdict(table), 'fit_n': len(usable),
                                    'fit_correct_n': sum(r['outcome'] for r in usable)}
        fitted['expected_cost_usd'][v] = float(np.mean([r['signals'][v]['estimated_usd'] for r in valid]))
    return fitted


def planner_for(fitted, order, loss, settings):
    m = [{1: fitted['likelihoods'][v]['p_bin_if_correct'], 0: fitted['likelihoods'][v]['p_bin_if_incorrect']}
         for v in order]
    costs = [fitted['expected_cost_usd'][v] for v in order]
    return ExactPlanner(JointLikelihood(m, None, INF), [settings['utility_per_usd']*c for c in costs],
                        settings['correct_reward'], loss), costs


def select_order(train, fitted, verifiers, loss, settings):
    priors = np.array([r['prior'] for r in train if r['valid']])
    ranked = []
    for order in all_orders(verifiers):
        planner, costs = planner_for(fitted, order, loss, settings)
        ranked.append({'order': list(order), 'training_mean_J0': float(planner.j(0, priors).mean())})
    ranked.sort(key=lambda r: (-r['training_mean_J0'], r['order']))
    return ranked


def decide(row, planner, order, costs, edges):
    signals = {v: row['signals'][v] for v in order}
    d = execute(row['prior'] if row['valid'] else 0.0, row['valid'], planner, order, costs, signals, edges)
    # Prefix-only replay: the same decision from only the signals actually read.
    prefix = {v: row['signals'][v] for v in order[:d['verifiers_used']]}
    replay = execute(row['prior'] if row['valid'] else 0.0, row['valid'], planner, order, costs,
                     _Prefix(prefix, order), edges)
    if digest(d) != digest(replay):
        raise AssertionError('prefix-only replay disagrees')
    return d


class _Prefix(dict):
    def __init__(self, prefix, order):
        super().__init__(prefix)
        self.order = order

    def __getitem__(self, key):
        if key not in self:
            raise AssertionError('future verifier access')
        return super().__getitem__(key)


def analyze_generator(rows, cfg):
    settings, verifiers = cfg['settings'], cfg['verifiers']
    edges = tuple(np.linspace(0., 1., settings['probability_bins']+1).tolist())
    folds = sorted({r['fold'] for r in rows})
    out = {'folds': {}, 'pooled': {}}
    for loss in settings['losses']:
        key = str(int(loss))
        arms = {a: [None]*len(rows) for a in ('prior_only', 'single', 'triple')}
        forced = {a: [None]*len(rows) for a in ('single', 'triple')}
        for k in folds:
            train = [r for r in rows if r['fold'] != k]
            fitted = fit_fold(train, verifiers, settings)
            ranked = select_order(train, fitted, verifiers, loss, settings)
            triple = ranked[0]['order']
            out['folds'].setdefault(str(k), {})[key] = {
                'selected_triple': triple, 'top5_orders': ranked[:5],
                'likelihoods': fitted['likelihoods'], 'expected_cost_usd': fitted['expected_cost_usd'],
                'training_valid_n': sum(r['valid'] for r in train)}
            for name, order in (('single', [cfg['single_arm_verifier']]), ('triple', triple)):
                planner, costs = planner_for(fitted, order, loss, settings)
                for i, r in enumerate(rows):
                    if r['fold'] != k:
                        continue
                    arms[name][i] = decide(r, planner, order, costs, edges)
                    forced[name][i] = execute(r['prior'] if r['valid'] else 0.0, r['valid'], planner, order, costs,
                                              {v: r['signals'][v] for v in order}, edges, force_all=True)
            for i, r in enumerate(rows):
                if r['fold'] == k:
                    arms['prior_only'][i] = baseline_decision(r['prior'] if r['valid'] else 0.0, r['valid'], loss,
                                                              settings['correct_reward'])
        summaries, utilities = {}, {}
        for name, ds in arms.items():
            s = summarize_policy(rows, ds, settings, loss)
            utilities[name] = np.array(s.pop('utility_per_item'))
            released = [r['outcome'] for r, d in zip(rows, ds) if d['action'] == 'assert']
            s['output_error_rate'] = (s['wrong_released']/s['released']) if s['released'] else None
            s['output_error_wilson95'] = rate_interval([1-y for y in released])
            s['calls_per_task'] = s['queries']/len(rows)
            s['query_histogram'] = dict(Counter(str(d['verifiers_used']) for d in ds))
            s['mean_utility_ci95'] = boot(utilities[name], settings['bootstrap_repeats'], 20261019)['ci95']
            summaries[name] = s
        common = [i for i, r in enumerate(rows) if r['valid'] and all(not forced[a][i]['failure'] for a in forced)]
        brier = {}
        for name in ('prior_only', 'single', 'triple'):
            ps = [rows[i]['prior'] if name == 'prior_only' else forced[name][i]['posterior'] for i in common]
            brier[name] = float(np.mean([(p-rows[i]['outcome'])**2 for p, i in zip(ps, common)])) if common else None
        reps = settings['bootstrap_repeats']
        out['pooled'][key] = {
            'arms': summaries,
            'paired_utility_single_minus_prior': boot(utilities['single']-utilities['prior_only'], reps, 20261019),
            'paired_utility_triple_minus_single': boot(utilities['triple']-utilities['single'], reps, 20261019),
            'paired_utility_triple_minus_prior': boot(utilities['triple']-utilities['prior_only'], reps, 20261019),
            'always_query_brier_common': {'n': len(common), 'brier': brier}}
    valid = [r for r in rows if r['valid']]
    out['forecasts'] = {'generator_r': binary_forecast_metrics([r['outcome'] for r in valid],
                                                               [r['prior'] for r in valid]).to_dict()}
    for v in verifiers:
        pairs = [(r['outcome'], r['signals'][v]['score']) for r in valid if r['signals'][v]['score'] is not None]
        out['forecasts'][v] = binary_forecast_metrics([a for a, _ in pairs], [b for _, b in pairs]).to_dict()
    out['counts'] = {'tasks': len(rows), 'with_code': sum(r['has_code'] for r in rows),
                     'labeled': sum(r['labeled'] for r in rows), 'valid': len(valid),
                     'passed': sum(r['outcome'] for r in rows if r['labeled']),
                     'per_fold_valid': dict(Counter(str(r['fold']) for r in valid)),
                     'per_fold_pass_rate': {str(k): float(np.mean([r['outcome'] for r in valid if r['fold'] == k]))
                                            for k in folds if any(r['fold'] == k for r in valid)}}
    out['collection_usd'] = {'generator': float(sum(r['generator_usd'] for r in rows)),
                             'verifiers': {v: float(sum(r['signals'][v]['estimated_usd'] for r in rows))
                                           for v in verifiers}}
    return out


def analyze(config_path, output):
    cfg = json.loads(Path(config_path).read_text())
    output = Path(output)
    if output.exists():
        raise SystemExit('analysis output exists; refusing to overwrite')
    split = {'value': json.loads(Path(cfg['split']).read_text()), 'sha256': file_digest(cfg['split'])}
    result = {'schema': 1, 'config': cfg, 'config_sha256': file_digest(config_path), 'split_sha256': split['sha256'],
              'generators': {}}
    for name, root in cfg['runs'].items():
        result['generators'][name] = analyze_generator(load_rows(root, split, cfg['verifiers']), cfg)
    output.mkdir(parents=True)
    atomic_json(output/'analysis.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.config, args.output)
    for name, g in result['generators'].items():
        print(name, json.dumps(g['counts']))
        for loss, p in g['pooled'].items():
            print(' loss', loss, {a: (s['released'], s['wrong_released'], round(s['mean_utility'], 4), s['queries'])
                                 for a, s in p['arms'].items()})


if __name__ == '__main__':
    main()
