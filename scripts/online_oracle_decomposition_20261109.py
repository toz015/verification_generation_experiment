"""Evaluator-side oracle diagnostic: where does the static arm's excess risk come from?

Not an experimental arm, and nothing here feeds back into any learner. For each generator, rotation
and order, static arm A is replayed with the same hierarchy, costs, budget, design set and random
numbers as the real run, but with the confidence map and/or the verifier likelihood replaced by an
oracle fit on the stream's own official labels (in-sample):

  prior      phi_0 (D_init, as in the experiment)   or  phi* (Brier fit on the stream)
  likelihood theta_0 / T_0 (D_init)                  or  theta* / T* (labeled MAP fit on the stream;
                                                         the joint model keeps the D_init alpha)

If the realized risk drops to about alpha only when the likelihood is replaced, the gap comes from
estimating or transferring the likelihood from D_init; if only the prior matters, from phi_0.

  PYTHONPATH=src python scripts/online_oracle_decomposition_20261109.py --jobs 4 \
      --out reports/quantum_katas_online_dep_20261109/oracle_decomposition.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import atomic_json
from vgx.online import calibrator as cal
from vgx.online import evaluate as ev
from vgx.online import experiment as ex
from vgx.online import joint as jm
from vgx.online import likelihood as lk
from vgx.online.runner import Run, draws
from vgx.online.views import LabelStore, ScoreSource

CONFIGS = {'independence': 'configs/quantum_katas_online_20261109.json',
           'dependence': 'configs/quantum_katas_online_dependence_20261109.json'}
VARIANTS = ('phi0_theta0', 'phi0_thetaStar', 'phiStar_theta0', 'phiStar_thetaStar')


def oracle_fits(I, stream, signals, store, V):
    proc = [it for it in stream if it.processable]
    z = cal.feature([it.raw_r for it in proc])
    y = np.array([store.outcome(it.task_id) for it in proc], int)
    phi_s, _ = cal.fit_brier(z, y)
    comp = [i for i, it in enumerate(proc) if all(signals[it.task_id][v]['score'] is not None for v in V)]
    from vgx.online.views import bin_of
    bins = np.array([[bin_of(signals[proc[i].task_id][v]['score']) for v in V] for i in comp], int)
    yc = y[comp]
    g_s, _ = lk.labeled_fit(bins, yc, I.audit_j, gamma=-np.inf)  # oracle: no orientation constraint needed
    T_s = None
    if I.T0 is not None:
        hier = list(I.hierarchy)
        T_s = jm.joint_table(g_s[hier], jm.hard_cells(bins[:, hier], yc), I.joint_alpha)
    return phi_s, g_s, T_s


def one(task):
    model, gen, rot = task
    cfg = json.loads(Path(CONFIGS[model]).read_text())
    ex._worker_init(cfg)
    ex._init.cache_clear()  # the two models share a worker process; never reuse the other model's init
    items, signals, labels = ex._data(gen)
    store = LabelStore(labels)
    V = cfg['verifiers']
    I, stream = ex._init(gen, rot, cfg['settings']['budget_fraction'])
    phi_s, g_s, T_s = oracle_fits(I, stream, signals, store, V)
    out = {}
    for var in VARIANTS:
        J = I
        if 'phiStar' in var:
            J = dataclasses.replace(J, phi0=phi_s)
        if 'thetaStar' in var:
            J = dataclasses.replace(J, theta0=g_s, T0=T_s)
        runs = []
        for order in cfg['grid']['orders']:
            src = ScoreSource(signals, tuple(V))
            r = Run('A', 0.0, J, stream, src, draws(cfg['seeds']['master'], rot, order, len(stream)),
                    ex._settings(cfg, cfg['settings']['budget_fraction']))
            r.run()
            groups, _ = ev.group_sums(r.log, store, signals, V, list(I.hierarchy))
            runs.append(groups)
        out[var] = runs
    return task, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    gens = json.loads(Path(CONFIGS['independence']).read_text())['generators']
    todo = [(m, g, r) for m in CONFIGS for g in gens for r in range(5)]
    acc = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for (m, g, r), out in pool.map(one, todo):
            for var, runs in out.items():
                acc.setdefault((m, g), {}).setdefault(var, []).extend(runs)
    res = {'note': 'evaluator-side oracle diagnostic (stream labels, in-sample); not an experimental arm', 'results': {}}
    for (m, g), by_var in acc.items():
        b = ev.bootstrap(by_var, reps=args.reps, seed=20261109, keys=('pub', 'risk', 'est_risk', 'brier_panel', 'total_usd'))
        res['results'][f'{m}|{g}'] = {v: {k: b['arms'][v][k] for k in ('pub', 'risk', 'est_risk', 'brier_panel', 'counts')}
                                      for v in by_var}
        print(m, g, {v: (round(b['arms'][v]['pub']['point'], 3), round(b['arms'][v]['risk']['point'], 3),
                         [round(x, 3) for x in b['arms'][v]['risk']['ci95']]) for v in by_var}, flush=True)
    atomic_json(args.out, res)


if __name__ == '__main__':
    main()
