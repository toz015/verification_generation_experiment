"""Simulation check for the dependence-aware likelihood (synthetic data, no model calls, no real labels).

Scenario 'correlated' of vgx.online.simulate (a shared per-task nuisance tilts every verifier's bins
together given Y; phi_0 and theta_0 fitted on the labeled initialization fold). Static arm A is run under
the independence model and the dependence model, for a D_init of the real size (64) and a larger one
(512), and the realized release risk is compared with alpha = 0.05 and with the model's own estimate.

  PYTHONPATH=src python scripts/online_dependence_sim_20261109.py --seeds 20 \
      --out reports/quantum_katas_online_dep_20261109/sim_dependence.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json
from vgx.online import simulate as sim
from vgx.online.runner import Run, Settings, draws

MODELS = {'independence': Settings(), 'dependence': Settings(likelihood='dependence')}
SIZES = (64, 512)


def one(task):
    model, n_init, seed = task
    st = MODELS[model]
    src, store, I, stream = sim.setup('correlated', seed, st, n_init=n_init, n_stream=600)
    r = Run('A', 0.0, I, stream, src, draws(seed, 0, 0, len(stream)), st)
    log = r.run()
    rel = [x for x in log if x['action'] == 'release']
    wrong = sum(store.outcome(x['task_id']) == 0 for x in rel)
    return task, {'released': len(rel), 'wrong': int(wrong), 'n': len(log),
                  'est_wrong': float(sum(1-x['belief'] for x in rel)),
                  'alpha': I.info.get('joint_alpha'), 'hierarchy': I.info['hierarchy']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    todo = [(m, n, s) for m in MODELS for n in SIZES for s in range(args.seeds)]
    rows = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for (m, n, s), out in pool.map(one, todo):
            rows.setdefault(f'{m}|n_init={n}', []).append({'seed': s, **out})
    summary = {}
    for k, v in rows.items():
        R, W, E = sum(x['released'] for x in v), sum(x['wrong'] for x in v), sum(x['est_wrong'] for x in v)
        risks = [x['wrong']/x['released'] for x in v if x['released']]
        summary[k] = {'pooled_risk': W/R, 'pooled_est_risk': E/R, 'pub': R/sum(x['n'] for x in v),
                      'risk_seed_mean': float(np.mean(risks)), 'risk_seed_sd': float(np.std(risks)),
                      'seeds': len(v)}
        print(k, {a: round(b, 4) for a, b in summary[k].items()}, flush=True)
    atomic_json(args.out, {'scenario': 'correlated', 'alpha': 0.05, 'n_stream': 600, 'summary': summary, 'runs': rows})


if __name__ == '__main__':
    main()
