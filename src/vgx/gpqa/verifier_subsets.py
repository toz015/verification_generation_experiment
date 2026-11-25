"""Offline: every subset of the five cached probability verifiers.

Each of the 31 non-empty subsets is scored two ways on held-out questions:

- Brier score of the posterior after asking every verifier in the subset
  (order does not matter for this posterior);
- the Green-Laffont policy, with the subset's order picked by mean training J0
  over all of its permutations (the same rule as the triples experiment).

Splits, frozen per-verifier likelihoods, costs, the joint model, alpha
selection and exact recursion are those of dependence_triples. Priors are the
raw generator confidence and, as an offline diagnostic, the two training-fold
calibration maps. Development data only; no API calls, no evaluation labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, permutations
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json, digest, file_digest
from vgx.gpqa.dependence_triples import (INF, ExactPlanner, alpha_label, bin_of, calibrated_rows, check_fold,
                                         choose_alpha, execute, fit_joint, fit_prior_calibration, fold_marginals,
                                         load_bundle, summarize, utility)
from vgx.gpqa.raw_prior_analysis import baseline_decision, candidate_valid

PRIORS = ('raw', 'monotone_binned', 'platt')


def subset_id(subset):
    return '+'.join(t.split(':')[0] for t in sorted(subset))


def run_split(rows, fit, cfg, tags):
    ids = [r['item_id'] for r in rows]
    if ids != fit['item_ids'] or [r['outcome'] for r in rows] != fit['outcomes']:
        raise ValueError('frozen split mismatch')
    lookup = {q: i for i, q in enumerate(ids)}
    edges = tuple(np.linspace(0., 1., cfg['probability_bins']+1).tolist())
    valid = [candidate_valid(r) for r in rows]
    subsets = [c for k in range(1, len(tags)+1) for c in combinations(sorted(tags), k)]
    # Brier is compared on one common set: valid candidates with all five scores.
    common = [i for i, r in enumerate(rows) if valid[i] and all(r['signals'][t]['score'] is not None for t in tags)]
    in_common = set(common)
    losses = [float(x) for x in cfg['losses']]
    posterior = {p: {'prior_only': [None]*len(rows)} for p in PRIORS}
    decisions = {(p, loss): {} for p in PRIORS for loss in losses}
    selected = []
    for fold_id, fold in enumerate(fit['folds']):
        check_fold(rows, fold, tags, cfg)
        train_raw = [rows[lookup[q]] for q in fold['train_ids']]
        held = [lookup[q] for q in fold['validation_ids']]
        seed = cfg['inner_seed'] + 1000*fit['seed'] % 997 + fold_id
        # The alpha criterion uses outcomes and verifier bins only, so it is shared by all priors.
        alphas = {c: choose_alpha(train_raw, c, [float(a) for a in cfg['alpha_grid']] + [INF], edges, cfg, seed)[0]
                  for c in subsets if len(c) > 1}
        costs = {t: fold['arms'][t]['expected_cost_usd'] for t in tags}
        for prior in PRIORS:
            frows = rows if prior == 'raw' else calibrated_rows(rows, fit_prior_calibration(train_raw, prior)['map'])
            train = [frows[lookup[q]] for q in fold['train_ids']]
            priors = np.array([r['prior'] for r in train if candidate_valid(r)])
            for i in held:
                posterior[prior]['prior_only'][i] = frows[i]['prior'] if i in in_common else None
            for c in subsets:
                for model in (('independence',) if len(c) == 1 else ('independence', 'dependence')):
                    alpha = INF if model == 'independence' else alphas[c]
                    key = f'{model}:{subset_id(c)}'
                    joint = fit_joint(train, c, fold_marginals(fold, c), alpha, edges)
                    post = posterior[prior].setdefault(key, [None]*len(rows))
                    for i in held:
                        if i in in_common:
                            r = frows[i]
                            lr = joint.likelihood_ratio(tuple(bin_of(r['signals'][t]['score'], edges) for t in c))
                            post[i] = r['prior']*lr/(r['prior']*lr + 1-r['prior'])
                    for loss in losses:
                        def planner(o):
                            j = fit_joint(train, o, fold_marginals(fold, o), alpha, edges)
                            return ExactPlanner(j, [cfg['utility_per_usd']*costs[t] for t in o], cfg['correct_reward'], loss)
                        ranked = sorted(((float(planner(o).j(0, priors).mean()), list(o)) for o in permutations(c)),
                                        key=lambda v: (-v[0], v[1]))
                        best = tuple(ranked[0][1])
                        selected.append({'fold': fold_id, 'prior': prior, 'loss': loss, 'policy': key,
                                         'order': [t.split(':')[0] for t in best], 'alpha': alpha_label(alpha),
                                         'training_model_value': ranked[0][0]})
                        p = planner(best)
                        ds = decisions[(prior, loss)].setdefault(key, [None]*len(rows))
                        for i in held:
                            ds[i] = execute(frows[i]['prior'], valid[i], p, best, [costs[t] for t in best],
                                            frows[i]['signals'], edges)
            for loss in losses:
                for i in held:
                    decisions[(prior, loss)].setdefault('prior_only', [None]*len(rows))[i] = \
                        baseline_decision(frows[i]['prior'], valid[i], loss, cfg['correct_reward'])
                    decisions[(prior, loss)].setdefault('abstain_all', [None]*len(rows))[i] = \
                        baseline_decision(frows[i]['prior'], False, loss, cfg['correct_reward'])
    out = {'seed': fit['seed'], 'item_ids': ids, 'outcomes': [r['outcome'] for r in rows],
           'raw_priors': [r['prior'] for r in rows], 'common': common,
           'posterior': {p: {k: [v[i] for i in common] for k, v in d.items()} for p, d in posterior.items()},
           'selected': selected, 'policies': {}, 'actions': {}}
    for (prior, loss), ds in decisions.items():
        if any(d is None for v in ds.values() for d in v):
            raise ValueError('incomplete held-out decisions')
        flash = utility(rows, ds['independence:gemini_flash'], loss, cfg)
        lk = f'{prior}/{loss:g}'
        out['policies'][lk] = {}
        for name, d in ds.items():
            s = summarize(rows, d, loss, cfg)
            s['utility_difference_vs_flash'] = float((utility(rows, d, loss, cfg) - flash).mean())
            out['policies'][lk][name] = s
        out['actions'][lk] = {name: ''.join('R' if x['action'] == 'assert' else 'A' for x in d) for name, d in ds.items()}
    return out


_WORKER = {}


def _init_worker(bundle):
    _WORKER['bundle'] = load_bundle(bundle)


def _run_one(task):
    run, cfg, tags = task
    _, inputs, fits, _ = _WORKER['bundle']
    g, cohort, _ = run.split('/')
    rows = inputs['rows'][g]
    if cohort == 'remaining_calibration':
        rows = [r for r in rows if r['item_id'] not in set(inputs['pilot_ids'])]
    return run, run_split(rows, fits[run], cfg, tags)


def analyze(config_path, output, jobs=1):
    config_path, output = Path(config_path), Path(output)
    cfg = json.loads(config_path.read_text())
    if cfg['new_api_requests'] != 0 or cfg['evaluation_labels_read']:
        raise ValueError('offline, development-only protocol required')
    bundle = Path(cfg['bundle'])
    if file_digest(bundle) != cfg['bundle_sha256']:
        raise ValueError('review bundle changed')
    protocol, _, fits, _ = load_bundle(bundle)
    if protocol['analysis_id'] != cfg['source_analysis_id']:
        raise ValueError('unexpected source analysis')
    tags = [v+':probability' for v in protocol['config']['verifiers']]
    cfg['utility_per_usd'] = protocol['config']['primary_utility_per_usd']
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'config': cfg, 'config_sha256': file_digest(config_path),
                'implementation_sha256': file_digest(Path(__file__)), 'priors': PRIORS, 'runs': {}}
    with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(bundle,)) as pool:
        for run, result in pool.map(_run_one, [(run, cfg, tags) for run in sorted(fits)]):
            target = output/(run+'.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(target, result)
            manifest['runs'][run] = file_digest(target)
            print(json.dumps({'run': run}), flush=True)
    manifest['analysis_id'] = digest(manifest)
    atomic_json(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gpqa_dependence_triples.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    analyze(args.config, args.output, args.jobs)
