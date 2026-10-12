"""Tables for the dependence-aware three-verifier analysis (development data only)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

MAIN = 'all_calibration/seed_20261005'
GENERATORS = ('qwen', 'gemini')
COHORTS = ('all_calibration', 'remaining_calibration')
POLICIES = [('abstain_all', 'Abstain on all'), ('raw', 'Raw confidence (release at the bar)'),
            ('raw_ties_abstain', 'Raw confidence (abstain at the bar)'), ('single_flash', 'Single Flash'),
            ('independence:selected_triple', 'Triple, independence model'),
            ('dependence:selected_triple', 'Triple, dependence model (alpha by inner CV)'),
            ('dependence_alpha_2:selected_triple', 'Triple, dependence model, alpha 2'),
            ('dependence_alpha_8:selected_triple', 'Triple, dependence model, alpha 8'),
            ('dependence_alpha_32:selected_triple', 'Triple, dependence model, alpha 32'),
            ('independence:always_selected_triple', 'Always query all three, independence'),
            ('dependence:always_selected_triple', 'Always query all three, dependence')]


def fmt(x, d=3):
    return '' if x is None else f'{x:+.{d}f}'


def logloss(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def load(root):
    root = Path(root)
    runs = {}
    for g in GENERATORS:
        for c in COHORTS:
            for path in sorted((root/g/c).glob('seed_*.json')):
                runs[f'{g}/{c}/{path.stem}'] = json.loads(path.read_text())
    return runs


def report(root, output):
    runs, out, lines = load(root), Path(output), []
    out.mkdir(parents=True, exist_ok=True)
    losses = sorted(next(iter(runs.values()))['policies'], key=float)
    w = lines.append

    w('## Main split (seed 20261005, 249 questions)\n')
    for g in GENERATORS:
        r = runs[f'{g}/{MAIN}']
        for lk in losses:
            w(f'**{g.capitalize()}, L = {float(lk):g}**\n')
            w('| Policy | Released | Wrong | Queries | Mean utility | vs Flash | 95% interval (descriptive) |')
            w('|---|---|---|---|---|---|---|')
            for key, label in POLICIES:
                s = r['policies'][lk][key]
                ci = s.get('paired_utility_vs_flash', {}).get('ci95', {})
                w(f"| {label} | {s['released']} | {s['wrong_released']} | {s['queries']} | {fmt(s['mean_utility'])} | "
                  f"{fmt(s['utility_difference_vs_flash'])} | "
                  f"{'' if key == 'single_flash' else fmt(ci.get('low')) + ' to ' + fmt(ci.get('high'))} |")
            w('')

    w('## All splits: triple minus single Flash, mean utility per question\n')
    w('Range over five fold seeds; "> 0" counts seeds where the triple is ahead.\n')
    w('| Generator | Questions | L | Independence | > 0 | Dependence (CV alpha) | > 0 | Dependence, alpha 2 | > 0 |')
    w('|---|---|---|---|---|---|---|---|---|')
    rows_csv = []
    for g in GENERATORS:
        for c in COHORTS:
            for lk in losses:
                cells = []
                for key in ('independence:selected_triple', 'dependence:selected_triple', 'dependence_alpha_2:selected_triple'):
                    d = [runs[k]['policies'][lk][key]['utility_difference_vs_flash'] for k in runs if k.startswith(f'{g}/{c}/')]
                    cells += [f'{fmt(min(d))} to {fmt(max(d))}', f'{sum(x > 0 for x in d)}/{len(d)}']
                n = len(next(runs[k] for k in runs if k.startswith(f'{g}/{c}/'))['item_ids'])
                w(f"| {g} | {n} | {float(lk):g} | " + ' | '.join(cells) + ' |')
    w('')
    for k, r in runs.items():
        for lk in losses:
            for name, s in r['policies'][lk].items():
                rows_csv.append({'run': k, 'loss': lk, 'policy': name, **{f: s[f] for f in (
                    'n', 'released', 'wrong_released', 'queries', 'expected_cost_usd', 'usage_estimated_cost_usd',
                    'mean_utility', 'utility_difference_vs_flash', 'released_accuracy', 'mean_posterior_on_released')}})
    with open(out/'all_policies.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)

    w('## Fixed orders on the main split (diagnostic, not a selection)\n')
    w('How many of the 60 fixed ordered triples beat single Flash on held-out questions.\n')
    w('| Generator | L | Model | Orders ahead of Flash | Median vs Flash | Best vs Flash |')
    w('|---|---|---|---|---|---|')
    for g in GENERATORS:
        r = runs[f'{g}/{MAIN}']
        for lk in losses:
            for model in ('independence', 'dependence'):
                d = [s['utility_difference_vs_flash'] for k, s in r['policies'][lk].items() if k.startswith(model+':triple:')]
                w(f'| {g} | {float(lk):g} | {model} | {sum(x > 0 for x in d)}/60 | {fmt(float(np.median(d)))} | {fmt(max(d))} |')
    w('')

    w('## Chosen shrinkage (alpha) across folds, combinations and splits\n')
    w('Larger alpha means closer to independence; inf is the independence model.\n')
    for g in GENERATORS:
        counts = Counter(str(a['chosen']) for k, r in runs.items() if k.startswith(g+'/')
                         for f in r['folds'] if f['loss'] == losses[0] for a in f['alpha_by_combination'].values())
        total = sum(counts.values())
        order = sorted(counts, key=lambda a: math.inf if a == 'inf' else float(a))
        w(f"- {g}: " + ', '.join(f'{a}: {counts[a]}' for a in order) + f' (of {total})')
    w('')

    w('## Does the joint model describe the verifiers better? (held-out)\n')
    w('Mean held-out class-conditional log-likelihood of the three bins per question, averaged over the 10 '
      'combinations and all splits (higher is better).\n')
    w('| Generator | Independence | Dependence | Combinations where dependence is higher (main split) |')
    w('|---|---|---|---|')
    for g in GENERATORS:
        ind = [x['cell_loglik'] for k, r in runs.items() if k.startswith(g+'/') for x in r['calibration']['independence']]
        dep = [x['cell_loglik'] for k, r in runs.items() if k.startswith(g+'/') for x in r['calibration']['dependence']]
        main = runs[f'{g}/{MAIN}']['calibration']
        by = defaultdict(lambda: [0., 0.])
        for x in main['independence']:
            by[x['combination']][0] += x['cell_loglik']
        for x in main['dependence']:
            by[x['combination']][1] += x['cell_loglik']
        w(f'| {g} | {np.mean(ind):.4f} | {np.mean(dep):.4f} | {sum(b > a for a, b in by.values())}/10 |')
    w('')

    w('## Calibration after querying all three (main split, all 10 combinations pooled)\n')
    w('Released here means posterior at or above 0.95. Prior is the raw confidence, unchanged.\n')
    w('| Generator | Model | Mean posterior | Accuracy | Brier | Released | Wrong among released |')
    w('|---|---|---|---|---|---|---|')
    for g in GENERATORS:
        cal = runs[f'{g}/{MAIN}']['calibration']
        keep = [x['flash_likelihood_ratio'] is not None for x in cal['independence']]
        base = [x for x, k in zip(cal['independence'], keep) if k]
        dep = [x for x, k in zip(cal['dependence'], keep) if k]
        y = np.array([x['outcome'] for x in base])
        prior = np.array([x['prior'] for x in base])
        fl = np.array([x['flash_likelihood_ratio'] for x in base], dtype=float)
        flash_post = prior*fl/(prior*fl + 1-prior)
        for label, p in [('prior only', prior), ('Flash only', flash_post),
                         ('three, independence', np.array([x['posterior_all_three'] for x in base])),
                         ('three, dependence', np.array([x['posterior_all_three'] for x in dep]))]:
            rel = p >= 0.95
            w(f'| {g} | {label} | {p.mean():.3f} | {y.mean():.3f} | {np.mean((p-y)**2):.4f} | {int(rel.sum())} | '
              f'{(1-y[rel]).mean():.3f} |')
    w('')

    w('## Verifier information alone (diagnostic, training base rate as prior)\n')
    w('Held-out log loss and AUROC of base rate x likelihood ratio, ignoring the generator confidence. '
      'All splits pooled, each combination counted once per item; items with a missing Flash score dropped.\n')
    w('| Generator | Flash only: log loss | AUROC | Three, independence: log loss | AUROC | Three, dependence: log loss | AUROC |')
    w('|---|---|---|---|---|---|---|')
    for g in GENERATORS:
        rec = {'independence': [], 'dependence': []}
        for k, r in runs.items():
            if k.startswith(g+'/'):
                for m in rec:
                    rec[m] += r['calibration'][m]
        keep = [x['flash_likelihood_ratio'] is not None for x in rec['independence']]
        rec = {m: [x for x, k in zip(v, keep) if k] for m, v in rec.items()}
        y = np.array([x['outcome'] for x in rec['independence']])
        br = np.array([x['training_base_rate'] for x in rec['independence']])
        cells = []
        for lr in (np.array([x['flash_likelihood_ratio'] for x in rec['independence']], dtype=float),
                   np.array([x['likelihood_ratio'] for x in rec['independence']]),
                   np.array([x['likelihood_ratio'] for x in rec['dependence']])):
            p = br*lr/(br*lr + 1-br)
            cells += [f'{logloss(y, p):.4f}', f'{roc_auc_score(y, lr):.3f}']
        w(f'| {g} | ' + ' | '.join(cells) + ' |')
    w('')

    w('## Selected orders (dependence model, CV alpha, L = 19)\n')
    for g in GENERATORS:
        cnt = Counter(f['selected']['dependence']['order_id'] for k, r in runs.items() if k.startswith(g+'/')
                      for f in r['folds'] if f['loss'] == losses[0])
        w(f"- {g}: " + '; '.join(f'{o.replace(">", " > ")} ({n})' for o, n in cnt.most_common(5)))
    w('')

    w('## Checks\n')
    checks = Counter()
    for r in runs.values():
        checks.update(r['checks'])
    for k in sorted(checks):
        w(f'- {k}: {checks[k]}')
    (out/'TABLES.md').write_text('\n'.join(lines) + '\n')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(report(args.root, args.output))
