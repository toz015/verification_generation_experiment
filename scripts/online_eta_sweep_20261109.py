"""Choose eta0 from simulations only (no real data). Writes a JSON summary.

  PYTHONPATH=src python scripts/online_eta_sweep_20261109.py --out reports/quantum_katas_online_20261109/sim/eta_sweep.json
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from vgx.online.runner import Settings
from vgx.online import simulate as sim

ETAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)  # second batch (1, 3, 10) added after the first showed phi barely moving
SCEN = ('ci_known', 'biased_conf', 'biased_lik')
ARMS = ('B', 'C1', 'C2', 'D')
SEEDS = range(1000, 1012)


def one(task):
    eta, scen, seed, q = task
    st = Settings(eta0=eta)
    out, store, I = sim.run_scenario(scen, seed, ARMS, q, st)
    return task, {a: sim.summarize(log, store, r) for a, (log, r) in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args()
    res = json.loads(a.out.read_text())['runs'] if a.out.exists() else {}
    tasks = [(e, s, seed, q) for e in ETAS for s in SCEN for seed in SEEDS for q in (0.5, 0.1) if str((e, s, seed, q)) not in res]
    with ProcessPoolExecutor(a.jobs) as ex:
        for task, r in ex.map(one, tasks):
            res[str(task)] = r
    table = {}
    for e in ETAS:
        for q in (0.5, 0.1):
            for arm in ARMS:
                for scen in SCEN:
                    vals = [res[str((e, scen, seed, q))][arm] for seed in SEEDS]
                    table[f'eta={e} q={q} {arm} {scen}'] = {k: float(np.mean([v[k] for v in vals])) for k in ('brier', 'brier_last_third', 'phi_err_end', 'theta_err_end', 'risk', 'pub')}
                vals = [res[str((e, s, seed, q))][arm] for s in SCEN for seed in SEEDS]
                table[f'eta={e} q={q} {arm}'] = {k: float(np.mean([v[k] for v in vals])) for k in ('brier', 'brier_last_third', 'phi_err_end', 'theta_err_end', 'risk', 'pub')}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'etas': ETAS, 'scenarios': SCEN, 'seeds': list(SEEDS), 'table': table, 'runs': res}, indent=1))
    for k, v in table.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})


if __name__ == '__main__':
    main()
