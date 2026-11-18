"""Tables for verifier_subsets: Brier and utility of every verifier subset."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

FLASH = 'independence:gemini_flash'


def load(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    return manifest, {run: json.loads((root/(run+'.json')).read_text()) for run in manifest['runs']}


def collect(results):
    """rows[(generator, cohort, prior, loss, policy)] -> list over seeds of metric dicts."""
    out = defaultdict(list)
    for run, r in sorted(results.items()):
        g, cohort, _ = run.split('/')
        y = np.array(r['outcomes'])[r['common']]
        for prior, posts in r['posterior'].items():
            for lk, pols in r['policies'].items():
                p, loss = lk.split('/')
                if p != prior:
                    continue
                flash_brier = float(((np.array(posts[FLASH]) - y)**2).mean())
                for name, s in pols.items():
                    m = {'seed': r['seed'], 'mean_utility': s['mean_utility'], 'released': s['released'],
                         'wrong_released': s['wrong_released'], 'utility_vs_flash': s['utility_difference_vs_flash'],
                         'queries': s['queries'], 'n': s['n']}
                    if name in posts:
                        q = np.array(posts[name])
                        m.update({'brier': float(((q - y)**2).mean()), 'brier_vs_flash': float(((q - y)**2).mean()) - flash_brier,
                                  'auroc': float(roc_auc_score(y, q)) if len(set(y)) == 2 else None,
                                  'mean_posterior_minus_accuracy': float(q.mean() - y.mean()), 'n_common': int(len(y))})
                    out[(g, cohort, prior, loss, name)].append(m)
    return out


def size(name):
    return 0 if ':' not in name else name.split(':', 1)[1].count('+') + 1


def summary(ms):
    s = {'seeds': len(ms)}
    for k in ('mean_utility', 'released', 'wrong_released', 'utility_vs_flash', 'brier', 'brier_vs_flash', 'auroc',
              'mean_posterior_minus_accuracy', 'queries'):
        v = [m[k] for m in ms if m.get(k) is not None]
        if v:
            s[k] = float(np.mean(v))
            s[k+'_min'], s[k+'_max'] = float(np.min(v)), float(np.max(v))
    s['seeds_utility_ahead_of_flash'] = sum(m['utility_vs_flash'] > 1e-12 for m in ms)
    s['seeds_brier_ahead_of_flash'] = sum(m.get('brier_vs_flash', 0.) < -1e-12 for m in ms)
    return s


def report(root, output):
    manifest, results = load(root)
    table = {k: summary(v) for k, v in collect(results).items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for s in table.values() for k in s})
    with (output/'subsets_all.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['generator', 'cohort', 'prior', 'loss', 'policy', 'model', 'verifiers', 'size', *keys])
        for (g, c, p, loss, name), s in sorted(table.items()):
            model, _, subset = name.partition(':')
            w.writerow([g, c, p, loss, name, model if subset else '', subset, size(name), *[s.get(k, '') for k in keys]])
    (output/'subsets_summary.json').write_text(json.dumps({'|'.join(k): v for k, v in table.items()}, indent=1, sort_keys=True))
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report(args.root, args.output)
