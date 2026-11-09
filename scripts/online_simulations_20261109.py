"""Simulation study for the online replay arms (synthetic data only), at the frozen eta0.

Four scenarios (vgx.online.simulate), all seven arms, q in {0.5, 0.1}, stream length 194 (as in the
replay) and 1000 (longer-run behaviour), 20 seeds (194) and 10 seeds (1000).

  PYTHONPATH=src python scripts/online_simulations_20261109.py --config configs/quantum_katas_online_20261109.json \
      --out reports/quantum_katas_online_20261109/sim/simulations.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vgx.online import simulate as sim
from vgx.online.runner import ARMS, Settings

SEEDS = range(2000, 2020)
SEEDS_LONG = range(2000, 2010)  # fewer seeds for the 1000-task stream


def seeds(n):
    return SEEDS if n == 194 else SEEDS_LONG


def one(task):
    scen, seed, q, n_stream, eta = task
    st = Settings(eta0=eta)
    out, store, I = sim.run_scenario(scen, seed, ARMS, q, st, n_stream=n_stream)
    res = {}
    for a, (log, r) in out.items():
        s = sim.summarize(log, store, r)
        ups = [x['theta_update'] for x in log if x['theta_update'] and 'seconds' in x['theta_update']]
        s['theta_seconds'] = float(sum(u['seconds'] for u in ups))
        s['em_iterations'] = int(sum(u.get('em_iterations', 0) for u in ups))
        s['records_processed'] = int(sum(u.get('records_processed', 0) for u in ups))
        mu = (np.array([0, .5, 1]) @ r.theta[I.audit_j, 0], np.array([0, .5, 1]) @ r.theta[I.audit_j, 1])
        s['audit_gap_end'] = float(mu[1] - mu[0])
        s['audit_gap_true'] = float(np.array([0, .5, 1]) @ (sim.TRUE_G[I.audit_j, 1] - sim.TRUE_G[I.audit_j, 0]))
        res[a] = s
    return task, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args()
    eta = json.loads(a.config.read_text())['settings']['eta0']
    tasks = [(s, seed, q, n, eta) for s in sim.SCENARIOS for q in (0.5, 0.1) for n in (194, 1000) for seed in seeds(n)]
    runs = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for task, r in ex.map(one, tasks):
            runs[str(task[:4])] = r
    table = {}
    keys = ('brier', 'brier_last_third', 'phi_err_end', 'theta_err_start', 'theta_err_end', 'pub', 'risk',
            'decision_cost', 'theta_seconds', 'em_iterations', 'records_processed', 'audit_gap_end')
    for s in sim.SCENARIOS:
        for q in (0.5, 0.1):
            for n in (194, 1000):
                for arm in ARMS:
                    vals = [runs[str((s, seed, q, n))][arm] for seed in seeds(n)]
                    row = {k: float(np.mean([v[k] for v in vals])) for k in keys}
                    row['risk_sd'] = float(np.std([v['risk'] for v in vals]))
                    row['brier_sd'] = float(np.std([v['brier'] for v in vals]))
                    table[f'{s}|q{q}|n{n}|{arm}'] = row
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'eta0': eta, 'seeds': list(SEEDS), 'table': table, 'runs': runs}, indent=1))
    for k, v in table.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items() if kk in ('brier', 'brier_last_third', 'theta_err_end', 'pub', 'risk')})


if __name__ == '__main__':
    main()
