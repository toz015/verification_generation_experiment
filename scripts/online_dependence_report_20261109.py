"""Compare the independence-model and dependence-model online replays (arms A, B, C2, D).

Both grids use the same rotations, orders, rule and audit random numbers and the same 35 task groups,
so the cluster bootstrap pairs them by group. Writes comparison.json and tables.md.

  PYTHONPATH=src python scripts/online_dependence_report_20261109.py \
      --out reports/quantum_katas_online_dep_20261109
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json
from vgx.online import evaluate as ev
from vgx.online.experiment import _path, read_log
from vgx.online.views import LabelStore, load_generator

CONFIGS = {'indep': 'configs/quantum_katas_online_20261109.json',
           'dep': 'configs/quantum_katas_online_dependence_20261109.json'}
ARMS = ('A', 'B', 'C2', 'D')
KEYS = ('pub', 'risk', 'est_risk', 'brier_prior', 'brier_panel', 'decision_usd', 'total_usd')


def load(cfg, gen, q, signals, store):
    out, frac = {}, cfg['settings']['budget_fraction']
    for arm in ARMS:
        runs = []
        for rot in cfg['grid']['rotations']:
            for order in cfg['grid']['orders']:
                meta, log = read_log(_path(Path(cfg['output']), gen, rot, order, arm, 0.0 if arm == 'A' else q, frac))
                runs.append(ev.group_sums(log, store, signals, cfg['verifiers'], meta.get('hierarchy'))[0])
        out[arm] = runs
    return out


def fmt(m, k, pct=True):
    s = 100 if pct else 1
    lo, hi = m[k]['ci95']
    d = 1 if pct else 4
    return f"{m[k]['point']*s:.{d}f} [{lo*s:.{d}f}, {hi*s:.{d}f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--indep-config', default=CONFIGS[next(iter(CONFIGS))])
    ap.add_argument('--dep-config', default=CONFIGS[list(CONFIGS)[1]])
    args = ap.parse_args(argv)
    CONFIGS[list(CONFIGS)[0]], CONFIGS[list(CONFIGS)[1]] = args.indep_config, args.dep_config
    cfgs = {k: json.loads(Path(v).read_text()) for k, v in CONFIGS.items()}
    base = cfgs['dep']
    res, md = {}, ['# Independence vs dependence-aware likelihood (arms A/B/C2/D, B = 25%)', '',
                   'Cluster bootstrap over task groups (QuantumKatas: 35 groups; GPQA: items), 2000 reps, paired across models and arms. '
                   'Percent unless noted. est = model-estimated risk (mean 1 - belief over releases).', '']
    for gen in base['generators']:
        items, signals, labels = load_generator(Path(base['data_dir']), gen)
        store = LabelStore(labels)
        for q in base['grid']['q']:
            entries = {}
            for model, cfg in cfgs.items():
                for arm, runs in load(cfg, gen, q, signals, store).items():
                    entries[f'{model}:{arm}'] = runs
            b = ev.bootstrap(entries, reps=args.reps, seed=base['seeds']['bootstrap'], keys=KEYS)
            key = f'{gen}|q{q:g}'
            res[key] = b
            md += [f'## {gen}, q = {q:g}', '',
                   '| arm | model | publication | risk | est. risk | Brier prior | Brier panel | decision USD/task |',
                   '|---|---|---|---|---|---|---|---|']
            for arm in ARMS:
                for model in ('indep', 'dep'):
                    m = b['arms'][f'{model}:{arm}']
                    md.append(f"| {arm} | {model} | {fmt(m, 'pub')} | {fmt(m, 'risk')} | {m['est_risk']['point']*100:.1f} | "
                              f"{fmt(m, 'brier_prior', False)} | {fmt(m, 'brier_panel', False)} | "
                              f"{m['decision_usd']['point']*1000:.3f}e-3 |")
            md += ['', '| paired difference | publication (pp) | risk (pp) | Brier prior |', '|---|---|---|---|']
            pairs = [(f'dep:{a}', f'indep:{a}') for a in ARMS] + [('dep:B', 'dep:A'), ('dep:C2', 'dep:B'),
                                                                 ('dep:D', 'dep:B'), ('dep:C2', 'dep:D')]
            for x, y in pairs:
                d = b['diff'][f'{x}-{y}']
                md.append(f"| {x} − {y} | {fmt(d, 'pub')} | {fmt(d, 'risk')} | {fmt(d, 'brier_prior', False)} |")
            md.append('')
            print(key, {a: (round(b['arms'][a]['pub']['point'], 3), round(b['arms'][a]['risk']['point'], 3))
                        for a in b['arms']}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out/'comparison.json', res)
    (args.out/'tables.md').write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
