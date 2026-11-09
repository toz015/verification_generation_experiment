"""Grid driver for the online replay on cached QuantumKatas outputs (no model calls).

  PYTHONPATH=src python -m vgx.online.experiment run --config configs/quantum_katas_online_20261109.json --jobs 4
  PYTHONPATH=src python -m vgx.online.experiment timing --config configs/quantum_katas_online_20261109.json
  PYTHONPATH=src python -m vgx.online.experiment summarize --config configs/quantum_katas_online_20261109.json

`run` skips runs whose log already exists (resume). Each run log is results/<out>/runs/<gen>/r<rot>/o<order>/
<arm>_q<q>_B<frac>.jsonl.gz; initialization records go to init/<gen>_r<rot>_B<frac>.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np

from vgx.common.storage import atomic_json, file_digest
from vgx.online import evaluate as ev
from vgx.online.runner import Run, Settings, draws, initialize, roles
from vgx.online.views import InitLabels, LabelGate, LabelStore, ScoreSource, load_generator

_CFG = None


def _settings(cfg, frac, eta=None):
    s = cfg['settings']
    return Settings(alpha=s['alpha'], budget_fraction=frac, eta0=s['eta0'] if eta is None else eta, gamma=s['gamma'],
                    em_max_iter=s['em_max_iter'], em_tol=s['em_tol'], hierarchy_size=s['hierarchy_size'])


@lru_cache(maxsize=None)
def _data(gen):
    items, signals, labels = load_generator(Path(_CFG['data_dir']), gen)
    return items, signals, labels


@lru_cache(maxsize=None)
def _init(gen, rot, frac):
    items, signals, labels = _data(gen)
    src, store = ScoreSource(signals, tuple(_CFG['verifiers'])), LabelStore(labels)
    init, design, stream = roles(items, rot)
    I = initialize(init, design, src, InitLabels(store, [i.task_id for i in init]), _settings(_CFG, frac))
    return I, stream


def _worker_init(cfg):
    global _CFG
    _CFG = cfg
    os.environ.setdefault('OMP_NUM_THREADS', '1')


def _path(out, gen, rot, order, arm, q, frac, eta=None):
    tag = '' if eta is None else f'_eta{eta:g}'
    return out/'runs'/gen/f'r{rot}'/f'o{order}'/f'{arm}_q{q:g}_B{frac:g}{tag}.jsonl.gz'


def run_one(task):
    gen, rot, order, arm, q, frac, eta = task
    out = Path(_CFG['output'])
    path = _path(out, gen, rot, order, arm, q, frac, eta)
    if path.exists():
        return task, 'exists'
    items, signals, labels = _data(gen)
    I, stream = _init(gen, rot, frac)
    ipath = out/'init'/f'{gen}_r{rot}_B{frac:g}.json'
    if not ipath.exists():
        atomic_json(ipath, {**I.info, 'stream_task_ids': [s.task_id for s in stream]})
    src, store = ScoreSource(signals, tuple(_CFG['verifiers'])), LabelStore(labels)
    rand = draws(_CFG['seeds']['master'], rot, order, len(stream))
    t0 = time.perf_counter()
    r = Run(arm, q, I, stream, src, rand, _settings(_CFG, frac, eta), LabelGate(store) if arm == 'D' else None)
    r.run()
    wall = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with gzip.GzipFile(tmp, 'wb', mtime=0) as f:
        f.write(json.dumps({'meta': {'gen': gen, 'rotation': rot, 'order': order, 'arm': arm, 'q': q,
                                     'budget_fraction': frac, 'eta0': _settings(_CFG, frac, eta).eta0, 'wall_seconds': wall,
                                     'initial_design_seconds': r.design_seconds_initial,
                                     'final_phi': r.phi.tolist(), 'final_theta': r.theta.tolist()}}).encode() + b'\n')
        for rec in r.log:
            f.write(json.dumps(rec).encode() + b'\n')
    tmp.rename(path)
    return task, 'done'


def tasks(cfg):
    g = cfg['grid']
    out = []
    for frac in g['budget_fractions']:
        qs = g['q'] if frac == cfg['settings']['budget_fraction'] else g['q_for_budget_sensitivity']
        for gen in cfg['generators']:
            for rot in g['rotations']:
                for order in g['orders']:
                    out.append((gen, rot, order, 'A', 0.0, frac, None))
                    for arm in g['arms']:
                        if arm == 'A':
                            continue
                        for q in qs:
                            out.append((gen, rot, order, arm, q, frac, None))
    es = g.get('eta_sensitivity')
    if es:
        frac = cfg['settings']['budget_fraction']
        for eta in es['eta0']:
            for gen in cfg['generators']:
                for rot in g['rotations']:
                    for order in g['orders']:
                        for arm in es['arms']:
                            for q in es['q']:
                                out.append((gen, rot, order, arm, q, frac, eta))
    return out


def cmd_run(cfg, jobs):
    todo = tasks(cfg)
    t0 = time.time()
    with ProcessPoolExecutor(jobs, initializer=_worker_init, initargs=(cfg,)) as ex:
        for i, (task, status) in enumerate(ex.map(run_one, todo, chunksize=1)):
            if i % 50 == 0:
                print(f'{i+1}/{len(todo)} {task} {status} {time.time()-t0:.0f}s', flush=True)
    print('all runs present', len(todo))


def read_log(path):
    with gzip.open(path, 'rt') as f:
        meta = json.loads(f.readline())['meta']
        return meta, [json.loads(line) for line in f]


def run_diagnostics(meta, log):
    pu = [x['phi_update'] for x in log if x['phi_update']]
    tu = [x['theta_update'] for x in log if x['theta_update']]
    upd = [u for u in tu if 'seconds' in u]
    zs = [x['zhat'] for x in log if x.get('zhat') is not None]
    return {'audits': sum(x['audited'] for x in log),
            'phi_updates': sum(u['status'] in ('ok', 'projected') for u in pu),
            'phi_projected': sum(u['status'] == 'projected' for u in pu),
            'phi_skipped': sum(u['status'].startswith('skipped') or u['status'].startswith('nonfinite') for u in pu),
            'theta_updates': len(upd), 'theta_skipped': sum(u['status'].startswith('skipped') for u in tu),
            'em_iterations': sum(u.get('em_iterations', 0) for u in upd),
            'em_not_converged': sum(u.get('stop_reason') not in (None, 'converged') for u in upd),
            'em_monotone_violations': sum(u.get('monotone_violations', 0) for u in upd),
            'records_processed': sum(u.get('records_processed', 0) for u in upd),
            'theta_seconds': sum(u['seconds'] for u in upd),
            'redesign_seconds': sum(x.get('redesign_seconds', 0.) for x in log),
            'redesigns': sum('redesign_seconds' in x for x in log),
            'design_not_converged': sum(x.get('redesign_status') not in (None, 'converged') for x in log),
            'constraint_active': sum(bool(u.get('constraint_active')) for u in upd),
            'zhat_abs_max': max(map(abs, zs)) if zs else None, 'zhat_mean': float(np.mean(zs)) if zs else None,
            'failures': sum(1 for x in log if x.get('reason') not in (None, 'not_processable')),
            'final_phi': meta['final_phi'],
            'final_audit_gap': None,
            'state_bytes_end': next((x['state_bytes'] for x in reversed(log) if 'state_bytes' in x), None),
            'log_bytes': sum(len(json.dumps(x)) for x in log)}


def cmd_timing(cfg, reps_orders=(0,)):
    """Single process, fixed environment; the first run is warm-up and discarded."""
    _worker_init(cfg)
    frac = cfg['settings']['budget_fraction']
    out = {}
    gen0 = cfg['generators'][0]
    I, stream = _init(gen0, 0, frac)
    items, signals, labels = _data(gen0)
    src = ScoreSource(signals, tuple(cfg['verifiers']))
    Run('C1', .5, I, stream, src, draws(cfg['seeds']['master'], 0, 0, len(stream)), _settings(cfg, frac)).run()  # warm-up
    for arm in ('C1', 'C2', 'C1F', 'C2F', 'B'):
        rows = []
        for gen in cfg['generators']:
            items, signals, labels = _data(gen)
            for rot in cfg['grid']['rotations']:
                I, stream = _init(gen, rot, frac)
                for order in reps_orders:
                    src = ScoreSource(signals, tuple(cfg['verifiers']))
                    r = Run(arm, .5, I, stream, src, draws(cfg['seeds']['master'], rot, order, len(stream)), _settings(cfg, frac))
                    t0 = time.perf_counter()
                    r.run()
                    wall = time.perf_counter() - t0
                    d = run_diagnostics({'final_phi': r.phi.tolist()}, r.log)
                    per = [x['theta_update']['seconds'] for x in r.log if x['theta_update'] and 'seconds' in x['theta_update']]
                    rows.append({'gen': gen, 'rot': rot, 'order': order, 'wall': wall, **{k: d[k] for k in (
                        'theta_updates', 'em_iterations', 'records_processed', 'theta_seconds', 'redesign_seconds',
                        'redesigns', 'state_bytes_end', 'log_bytes')}, 'theta_seconds_per_update_p50':
                        float(np.median(per)) if per else None, 'theta_seconds_per_update_max': max(per) if per else None})
        out[arm] = rows
    env = {'cpu_count': os.cpu_count(), 'processes': 1, 'omp_threads': os.environ.get('OMP_NUM_THREADS'),
           'numpy': np.__version__, 'warmup_runs_discarded': 1}
    atomic_json(Path(cfg['output'])/'timing.json', {'environment': env, 'runs': out})
    for arm, rows in out.items():
        print(arm, {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ('wall', 'theta_updates', 'em_iterations',
              'records_processed', 'theta_seconds', 'redesign_seconds')})


def cmd_summarize(cfg):
    out = Path(cfg['output'])
    _worker_init(cfg)
    res = {'config_sha256': cfg.get('_sha256'), 'by_setting': {}}
    curves = {}
    for frac in cfg['grid']['budget_fractions']:
        qs = cfg['grid']['q'] if frac == cfg['settings']['budget_fraction'] else cfg['grid']['q_for_budget_sensitivity']
        for gen in cfg['generators']:
            items, signals, labels = _data(gen)
            store = LabelStore(labels)
            for q in qs:
                entries, diags, per_rot = {}, {}, {}
                for arm in cfg['grid']['arms']:
                    qq = 0.0 if arm == 'A' else q
                    entries[arm], diags[arm] = [], []
                    for rot in cfg['grid']['rotations']:
                        for order in cfg['grid']['orders']:
                            meta, log = read_log(_path(out, gen, rot, order, arm, qq, frac))
                            groups, rows = ev.group_sums(log, store, signals, cfg['verifiers'])
                            entries[arm].append(groups)
                            diags[arm].append(run_diagnostics(meta, log))
                            per_rot.setdefault(arm, {}).setdefault(rot, []).append(groups)
                            c = curves.setdefault(f'{gen}|q{q:g}|B{frac:g}|{arm}', {'brier_prior': [], 'released': [],
                                                                                   'wrong': [], 'audits': []})
                            c['brier_prior'].append([r.get('brier_prior') for r in rows])
                            c['released'].append([r['release'] for r in rows])
                            c['wrong'].append([r['wrong'] for r in rows])
                boot = ev.bootstrap(entries, reps=cfg['bootstrap_reps'], seed=cfg['seeds']['bootstrap'])
                rot_metrics = {a: {str(r): ev.metrics(ev.pooled(v)) for r, v in d.items()} for a, d in per_rot.items()}
                order_spread = {}
                for a in cfg['grid']['arms']:
                    ms = [ev.metrics(ev.pooled([g])) for g in entries[a]]
                    order_spread[a] = {k: {'sd_over_runs': float(np.nanstd([m[k] for m in ms])),
                                           'min': float(np.nanmin([m[k] for m in ms])),
                                           'max': float(np.nanmax([m[k] for m in ms]))}
                                       for k in ('pub', 'risk', 'brier_prior', 'total_usd')}
                diag = {a: {k: (float(np.mean([d[k] for d in v])) if all(isinstance(d[k], (int, float)) for d in v) else None)
                            for k in v[0] if k not in ('final_phi', 'state_bytes_end', 'final_audit_gap')} for a, v in diags.items()}
                for a, v in diags.items():
                    diag[a]['state_bytes_end_mean'] = {k: float(np.mean([d['state_bytes_end'][k] for d in v if d['state_bytes_end']]))
                                                       for k in ('c1_history', 'c2_stats')} if any(d['state_bytes_end'] for d in v) else None
                    diag[a]['final_phi_mean'] = np.mean([d['final_phi'] for d in v], axis=0).tolist()
                res['by_setting'][f'{gen}|q{q:g}|B{frac:g}'] = {'bootstrap': boot, 'per_rotation': rot_metrics,
                                                                'order_spread': order_spread, 'diagnostics': diag}
                print(gen, q, frac, {a: (round(boot['arms'][a]['pub']['point'], 3), round(boot['arms'][a]['risk']['point'], 3),
                                         round(boot['arms'][a]['brier_prior']['point'], 4)) for a in boot['arms']}, flush=True)
    atomic_json(out/'summary.json', res)
    # learning curves: mean over runs of per-round values at each stream position
    cv = {}
    for k, c in curves.items():
        L = max(len(run) for run in c['released'])  # stream lengths differ by rotation (191-195); pad with nan
        cv[k] = {m: np.nanmean(np.array([[np.nan if v is None else float(v) for v in run] + [np.nan]*(L-len(run))
                                         for run in c[m]], float), axis=0).tolist() for m in ('brier_prior', 'released', 'wrong')}
    atomic_json(out/'curves.json', cv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('run', 'timing', 'summarize'))
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    cfg['_sha256'] = file_digest(a.config)
    Path(cfg['output']).mkdir(parents=True, exist_ok=True)
    if a.command == 'run':
        cmd_run(cfg, a.jobs)
    elif a.command == 'timing':
        _worker_init(cfg)
        cmd_timing(cfg)
    else:
        cmd_summarize(cfg)


if __name__ == '__main__':
    main()
