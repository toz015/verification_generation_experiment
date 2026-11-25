"""Independent rebuild of the offline multi-risk results from the logs (handbook stage F).

Rebuilds every pooled metric from decisions_*.csv.gz and labels read straight from the cache, with formulas
written here (not imported from the runner); checks Ours / Fixed mixtures against the per-rule traces, AT seed
means against the per-seed release file, that decision logs carry no label column, and m = q - p + e.
With --compare DIR, compares two runs' normalized JSON, CSV and decompressed logs.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

import vgx.baselines as B
from vgx.common.storage import atomic_json

LABEL_WORDS = ('outcome', 'label', 'correct', 'truth', 'y_true')


def labels(rows):
    return {r['item_id']: (int(r['outcome']) if r['valid'] else 0) for r in rows}


def rebuild(risk, alpha, a, cost, qprob, y, budget):
    n = len(a)
    P = float(a.sum()); E = float((a*(1-y)).sum()); Mm = float(((1-a)*y).sum()); Q = float(y.sum()); W = n-Q
    if abs(Mm/n-(Q/n-P/n+E/n)) > 1e-9:
        raise AssertionError('identity m = q - p + e fails')
    r = {'selective_error': None if P <= 1e-9 else E/P, 'fpr': None if W <= 0 else E/W, 'fnr': None if Q <= 0 else Mm/Q}[risk]
    if risk == 'selective_error':
        met = P > 1e-9 and E-alpha*P <= 1e-9
    elif risk == 'fpr':
        met = W > 0 and P > 1e-9 and E-alpha*W <= 1e-9
    else:
        met = Q > 0 and Mm-alpha*Q <= 1e-9
    c = float(np.mean(cost))
    return {'coverage': P/n, 'n_released': P, 'n_wrong_released': E, 'n_correct_deferred': Mm, 'risk': r,
            'mean_screen_cost': c, 'verifier_query_rate': float(np.mean(qprob)),
            'empirical_feasible': bool(met and c <= budget+1e-12)}


def close(x, y, tol=1e-9):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, bool) or isinstance(y, bool):
        return bool(x) == bool(y)
    return abs(float(x)-float(y)) <= tol*max(1., abs(float(y)))


def verify(cfg, out_dir):
    out_dir = Path(out_dir)
    report = {'checked_cells': 0, 'checked_fields': 0, 'mismatches': [], 'mixture_items_checked': 0, 'mixture_mismatches': 0,
              'seed_mean_items_checked': 0, 'seed_mean_mismatches': 0, 'label_columns_in_decisions': []}
    for ds in cfg['datasets']:
        data, _ = B.load(ds)
        for gen in cfg['generators']:
            res_path = out_dir/f'{ds}_{gen}.json'
            if not res_path.exists():
                report['mismatches'].append(f'{ds}:{gen} result missing'); continue
            res = json.loads(res_path.read_text())
            lab = labels(data[gen])
            traces = {}
            with gzip.open(out_dir/f'rule_traces_{ds}_{gen}.jsonl.gz', 'rt') as fh:
                for line in fh:
                    t = json.loads(line)
                    traces[(t['fold'], tuple(t['order']), tuple(t['rule']), t['task_id'])] = t
            seeded = np.load(out_dir/f'seed_releases_{ds}_{gen}.npz')
            groups = {}
            with gzip.open(out_dir/f'decisions_{ds}_{gen}.csv.gz', 'rt', newline='') as fh:
                rd = csv.DictReader(fh)
                report['label_columns_in_decisions'] += [c for c in rd.fieldnames if any(w in c.lower() for w in LABEL_WORDS)]
                for r in rd:
                    groups.setdefault((r['risk_metric'], float(r['alpha_target']), r['budget_id'], r['method']), []).append(r)
            n_test = res['n_test']
            for (risk, alpha, bid, method), rs in groups.items():
                if len(rs) != n_test:
                    report['mismatches'].append(f'{ds}:{gen}:{risk}:{alpha}:{bid}:{method} rows {len(rs)} != {n_test}'); continue
                a = np.array([float(r['release_probability']) for r in rs])
                c = np.array([float(r['expected_cost']) for r in rs])
                u = np.array([float(r['verifier_query_probability']) for r in rs])
                y = np.array([lab[r['task_id']] for r in rs], float)
                nt = {}
                for r in rs:
                    nt[int(r['fold'])] = (nt.get(int(r['fold']), (0, float(r['budget_usd'])))[0]+1, float(r['budget_usd']))
                budget = sum(k*b for k, b in nt.values())/sum(k for k, _ in nt.values())
                f = bid.split(':')[1]
                key = f'{risk}|{alpha}|{float(f)}'
                summ = res['cells'][key][method]
                mine = rebuild(risk, alpha, a, c, u, y, budget)
                report['checked_cells'] += 1
                for fld, v in mine.items():
                    report['checked_fields'] += 1
                    if not close(v, summ[fld], 1e-6 if fld in ('mean_screen_cost', 'verifier_query_rate', 'coverage', 'risk') else 1e-6):
                        report['mismatches'].append(f'{ds}:{gen}:{key}:{method}:{fld} log {v} summary {summ[fld]}')
                # mixtures against per-rule traces
                if method in ('Ours-Offline', 'Fixed-1-Mix', 'Fixed-3-Mix'):
                    for r in rs:
                        if not r['rule_ids']:
                            continue
                        rules = json.loads(r['rule_ids']); w = json.loads(r['mixture_weights'])
                        order = tuple(orders_for(out_dir, ds, gen, r['selected_config_id']))
                        report['mixture_items_checked'] += 1
                        tr = [traces.get((int(r['fold']), order, tuple(ru), r['task_id'])) for ru in rules]
                        if any(t is None for t in tr):
                            report['mixture_mismatches'] += 1; continue
                        ea = sum(wi*t['release'] for wi, t in zip(w, tr)); ec = sum(wi*t['cost'] for wi, t in zip(w, tr))
                        eu = sum(wi*(len(t['queried']) > 0) for wi, t in zip(w, tr))
                        if abs(ea-float(r['release_probability'])) > 1e-9 or abs(ec-float(r['expected_cost'])) > 1e-12 or \
                                abs(eu-float(r['verifier_query_probability'])) > 1e-9:
                            report['mixture_mismatches'] += 1
                if method in ('AT-Frozen', 'AT+Cal-Frozen'):
                    arr = seeded[f'{risk}|{alpha}|{float(f)}|{method}']
                    report['seed_mean_items_checked'] += len(rs)
                    report['seed_mean_mismatches'] += int(np.sum(np.abs(arr.mean(axis=0)-a) > 1e-9))
    report['label_columns_in_decisions'] = sorted(set(report['label_columns_in_decisions']))
    report['ok'] = not report['mismatches'] and not report['mixture_mismatches'] and not report['seed_mean_mismatches'] \
        and not report['label_columns_in_decisions']
    atomic_json(out_dir/'verify.json', report)
    print('verify:', 'ok' if report['ok'] else 'FAILED', {k: v for k, v in report.items() if k != 'mismatches'}, report['mismatches'][:10])
    return report


_ORDERS = {}


def orders_for(out_dir, ds, gen, cid):
    if (ds, gen) not in _ORDERS:
        _ORDERS[(ds, gen)] = json.loads((Path(out_dir)/f'selected_configs_{ds}_{gen}.json').read_text())
    return _ORDERS[(ds, gen)][cid]['order']


VOLATILE = {'seconds', 'started', 'finished', 'git_head', 'log_files', 'network_guard'}


def normalize(x):
    if isinstance(x, dict):
        return {k: normalize(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [normalize(v) for v in x]
    if isinstance(x, float):
        return round(x, 12)
    return x


def compare(dir1, dir2):
    d1, d2 = Path(dir1), Path(dir2)
    diffs = []
    for p in sorted(d1.glob('*_*.json')):
        if p.name.startswith(('preflight', 'verify')):
            continue
        q = d2/p.name
        if not q.exists():
            diffs.append(f'missing {p.name}'); continue
        if normalize(json.loads(p.read_text())) != normalize(json.loads(q.read_text())):
            diffs.append(f'differs {p.name}')
    for p in sorted(list(d1.glob('*.csv.gz'))+list(d1.glob('*.jsonl.gz'))):
        q = d2/p.name
        if not q.exists() or gzip.open(p).read() != gzip.open(q).read():
            diffs.append(f'differs {p.name}')
    for p in sorted(d1.glob('*.npz')):
        a, b = np.load(p), np.load(d2/p.name)
        if sorted(a.files) != sorted(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            diffs.append(f'differs {p.name}')
    print('compare:', 'identical after normalization' if not diffs else diffs)
    return diffs


if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])
