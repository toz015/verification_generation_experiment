"""Replay AT and AT+CC on cached QuantumKatas candidates (no API calls).

  python -m vgx.selective_auditing_runs tune        # development: GPQA calibration partition, writes the frozen config
  python -m vgx.selective_auditing_runs simulate    # synthetic streams under the frozen config
  python -m vgx.selective_auditing_runs smoke       # small engineering check on QuantumKatas (10 seeds, one alpha)
  python -m vgx.selective_auditing_runs run         # full QuantumKatas runs under the frozen config

Development data never include QuantumKatas labels: hyperparameters are chosen on the 249 GPQA calibration
(Main minus Diamond) questions of the same three generators, then frozen in CONFIG before any QuantumKatas run.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

from vgx import selective_auditing as sa
from vgx.common.storage import atomic_json, file_digest

CONFIG = Path('configs/selective_auditing_qk_20261110.json')
OUT = Path('results/selective_auditing_qk_20261110')
QK_CONFIG = 'configs/quantum_katas_crossfit_3gen_20261019.json'
GENS = ('qwen3', 'sonnet55', 'gptoss')
ALPHAS = (0.05, 0.10, 0.20, 0.30)
# The first search (eta up to 0.2, gamma up to 0.2) had no feasible point on the development streams (249
# questions each); it was widened once, on development data only, before freezing. Both grids are recorded.
TUNE_FIRST = {'eta': (0.005, 0.01, 0.02, 0.05, 0.1, 0.2), 'gamma': (0.05, 0.1, 0.2)}
TUNE = {'eta': (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0), 'gamma': (0.05, 0.1, 0.2, 0.3, 0.5, 0.7), 'beta': (0.1, 0.5, 1.0, 2.0, 5.0),
        'grid': {'step0.1': [round(0.1*i, 10) for i in range(11)], 'step0.05': [round(0.05*i, 10) for i in range(21)]}}
TUNE_DEFAULTS = {'beta': 1.0, 'grid': 'step0.1'}
TUNE_SEEDS = 50
TUNE_SLACK = 0.01  # mean over generators of (leak rate - alpha) allowed during selection, per method and alpha


# ----------------------------------------------------------------------------- streams

def qk_streams():
    """Per generator: the evaluable stream in plan order, plus the excluded rounds with their reasons."""
    from vgx.quantum_katas import crossfit
    cfg = json.loads(Path(QK_CONFIG).read_text())
    split = {'value': json.loads(Path(cfg['split']).read_text()), 'sha256': file_digest(cfg['split'])}
    out = {}
    for gen in GENS:
        rows = crossfit.load_rows(cfg['runs'][gen], split, [])
        keep, excluded = [], []
        for i, r in enumerate(rows):
            if not r['has_code']:
                excluded.append((i, r['task_id'], 'no extractable code or confidence (end-to-end failure)'))
            elif r['prior'] is None or not 0 <= r['prior'] <= 1:
                excluded.append((i, r['task_id'], 'missing or invalid confidence'))
            elif not r['labeled']:
                excluded.append((i, r['task_id'], 'no official checker label'))
            else:
                keep.append(i)
        out[gen] = {'task_ids': [rows[i]['task_id'] for i in keep], 'fold': [rows[i]['fold'] for i in keep],
                    'confidence': np.array([rows[i]['prior'] for i in keep], float),
                    'errors': np.array([1-rows[i]['outcome'] for i in keep], int),
                    'excluded': excluded, 'all_task_ids': [r['task_id'] for r in rows]}
    return out


def gpqa_dev_streams():
    import vgx.baselines as B
    data, _ = B.load('gpqa')
    out = {}
    for gen in GENS:
        rows = [r for r in data[gen] if r['fold'] == 1 and r['valid']]
        out[gen] = {'confidence': np.array([r['prior'] for r in rows], float),
                    'errors': np.array([1-r['outcome'] for r in rows], int)}
    return out


def simulated_streams(T=2000, seed=20261110):
    rng = np.random.default_rng(seed)
    c = rng.beta(4, 2, T)
    out = {'informative': (c, (rng.random(T) > c).astype(int))}
    c = rng.random(T)
    out['uninformative'] = (c, (rng.random(T) < 0.3).astype(int))
    c = rng.beta(8, 1.5, T)
    out['overconfident'] = (c, (rng.random(T) > c**6).astype(int))
    c = rng.beta(6, 1, T)
    out['rare_errors'] = (c, (rng.random(T) < 0.02+0.1*(1-c)).astype(int))
    c = rng.random(T)
    e = np.where(np.arange(T) < T//2, rng.random(T) > c, rng.random(T) < c).astype(int)
    out['shift'] = (c, e)
    return out


# ----------------------------------------------------------------------------- one replay

def make(method, alpha, hp, runs):
    if method == 'AT':
        return sa.AdditiveThresholding(alpha, hp['gamma'], hp['eta'], runs)
    if method == 'AT+CC':
        return sa.CalibratedThresholding(alpha, hp['gamma'], hp['eta'], hp['beta'], TUNE['grid'][hp['grid']], runs)
    return sa.RandomAudit(alpha, runs)


def replay(method, alpha, hp, confidence, errors, u):
    m = make(method, alpha, hp, u.shape[0])
    oracle = sa.Oracle(errors, u.shape[0])
    tr = sa.run(m, confidence, oracle, u)
    if not np.array_equal(oracle.revealed, tr.audited):
        raise AssertionError('reveals and audits disagree')
    return tr, sa.metrics(tr.audited, tr.pi, errors, tr.score)


def summary(mt, hp=None):
    """Ratios of expected counts (the paper's Err(T)) and the distribution of per-run ratios, kept apart."""
    E, T = mt['E'], mt['T']
    M, A = mt['M'].astype(float), mt['A'].astype(float)
    q = lambda x: None if np.all(np.isnan(x)) else [float(v) for v in np.nanpercentile(x, [5, 50, 95])]
    out = {'T': T, 'E': E, 'runs': len(M),
           'mean_counts': {k: float(np.mean(mt[k])) for k in ('M', 'A', 'audits', 'mandatory_audits', 'spot_audits')},
           'ratio_of_means': {'leak_error_rate': None if E == 0 else float(M.mean()/E),
                              'selective_risk': None if A.mean() == 0 else float(M.mean()/A.mean()),
                              'total_unaudited_error_rate': float(M.mean()/T),
                              'auto_accept_rate': float(A.mean()/T), 'audit_rate': float(mt['audits'].mean()/T),
                              'raw_error_rate': E/T},
           'per_run': {k: {'mean': None if np.all(np.isnan(mt[k])) else float(np.nanmean(mt[k])), 'q05_50_95': q(mt[k]),
                           'na_runs': int(np.isnan(mt[k]).sum())}
                       for k in ('leak_error_rate', 'selective_risk', 'total_unaudited_error_rate', 'auto_accept_rate', 'audit_rate')}}
    if 'brier' in mt:
        out['brier'] = float(np.mean(mt['brier']))
    if hp is not None and E:
        out['theory_gap_bound'] = (1/hp['eta']+1/hp['gamma'])/E
    return out


# ----------------------------------------------------------------------------- development tuning

def evaluate(streams, hp, methods, seeds, order_seed=20261109):
    rows = []
    for gen, s in streams.items():
        perm = np.random.default_rng(order_seed).permutation(len(s['errors']))
        c, e = s['confidence'][perm], s['errors'][perm]
        u = sa.uniforms(seeds, len(e))
        for a in ALPHAS:
            if hp['gamma'] > 1-a:
                continue
            for m in methods:
                _, mt = replay(m, a, hp, c, e, u)
                rows.append({'gen': gen, 'alpha': a, 'method': m, 'leak': float(mt['M'].mean()/mt['E']),
                             'audit_rate': float(mt['audits'].mean()/mt['T'])})
    return rows


def score(rows, methods):
    excess = max(np.mean([r['leak']-r['alpha'] for r in rows if r['method'] == m and r['alpha'] == a])
                 for m in methods for a in ALPHAS)
    return excess, float(np.mean([r['audit_rate'] for r in rows]))


def tune():
    streams = gpqa_dev_streams()
    seeds = list(range(TUNE_SEEDS))
    stage1 = []
    for eta in TUNE['eta']:
        for gamma in TUNE['gamma']:
            hp = {'eta': eta, 'gamma': gamma, **TUNE_DEFAULTS}
            rows = evaluate(streams, hp, ('AT', 'AT+CC'), seeds)
            excess, audit = score(rows, ('AT', 'AT+CC'))
            stage1.append({'hp': hp, 'max_mean_excess': excess, 'mean_audit_rate': audit})
            print('stage1', hp, round(excess, 4), round(audit, 4), flush=True)
    pick = lambda xs: min(xs, key=lambda r: (r['max_mean_excess'] > TUNE_SLACK, r['mean_audit_rate'] if r['max_mean_excess'] <= TUNE_SLACK else r['max_mean_excess']))
    best1 = pick(stage1)
    stage2 = []
    for beta in TUNE['beta']:
        for grid in TUNE['grid']:
            hp = {**best1['hp'], 'beta': beta, 'grid': grid}
            rows = evaluate(streams, hp, ('AT+CC',), seeds)
            excess, audit = score(rows, ('AT+CC',))
            stage2.append({'hp': hp, 'max_mean_excess': excess, 'mean_audit_rate': audit})
            print('stage2', hp, round(excess, 4), round(audit, 4), flush=True)
    best2 = pick(stage2)
    config = {
        'schema': 1, 'kind': 'selective_auditing_baselines',
        'paper': 'Gibbs, Tibshirani, Han, Zrnic. Reliable AI-Assisted Decisions via Selective Auditing. arXiv:2610.31662v1',
        'status': 'frozen before any QuantumKatas replay; hyperparameters are this implementation\'s choices (the paper reports none)',
        'hyperparameters': {'AT': {'eta': best1['hp']['eta'], 'gamma': best1['hp']['gamma'], 'tau0': 0.5},
                            'AT+CC': {'eta': best1['hp']['eta'], 'gamma': best1['hp']['gamma'], 'tau0': 0.5,
                                      'beta': best2['hp']['beta'], 'grid': best2['hp']['grid'],
                                      'grid_points': TUNE['grid'][best2['hp']['grid']], 'w0': 'uniform over k+1'}},
        'alphas_leak': list(ALPHAS),
        'selection': {'data': 'GPQA calibration partition (249 questions, Main minus Diamond), raw p_correct and answer-key labels of the frozen candidates; no QuantumKatas labels',
                      'seeds': TUNE_SEEDS, 'order_seed': 20261109, 'search': {k: list(v) for k, v in TUNE.items()},
                      'first_search_without_feasible_point': {k: list(v) for k, v in TUNE_FIRST.items()},
                      'rule': f'stage 1: (eta, gamma) shared by AT and AT+CC with beta={TUNE_DEFAULTS["beta"]}, grid={TUNE_DEFAULTS["grid"]}; stage 2: (beta, grid) for AT+CC. Feasible: for every method and alpha, the mean over generators of (leak rate - alpha) <= {TUNE_SLACK}. Among feasible, least mean audit rate; if none, least excess.',
                      'stage1': stage1, 'stage2': stage2},
        'streams': {'generators': list(GENS), 'evaluable': 'extractable code, confidence in [0, 1] and an official checker label; others excluded and counted',
                    'orders': {'primary': 'seeded permutation of the evaluable stream, seed 20261110',
                               'plan': 'the run plan order (tasks grouped by kata)',
                               'random': 'seeded permutations, seeds 20261111..20261130'},
                    'audit_seeds': 'run r uses numpy default_rng(1000 + r); 100 runs per order; shared by all methods (common random numbers)',
                    'end_to_end_sensitivity': 'primary order over all 322 tasks with end-to-end failures entered as confidence 0 and error 1'},
    }
    atomic_json(CONFIG, config)
    print(json.dumps(config['hyperparameters'], indent=1))


def frozen():
    cfg = json.loads(CONFIG.read_text())
    return cfg, {'AT': cfg['hyperparameters']['AT'], 'AT+CC': cfg['hyperparameters']['AT+CC'],
                 'random': cfg['hyperparameters']['AT']}


# ----------------------------------------------------------------------------- simulations

def simulate():
    _, hps = frozen()
    out = {}
    seeds = list(range(1000, 1200))
    for name, (c, e) in simulated_streams().items():
        u = sa.uniforms(seeds, len(e))
        for a in (0.1, 0.2):
            for m in ('AT', 'AT+CC', 'random'):
                tr, mt = replay(m, a, hps[m], c, e, u)
                out[f'{name}|{a}|{m}'] = {**summary(mt, hps[m]), 'seconds': tr.seconds}
                r = out[f'{name}|{a}|{m}']['ratio_of_means']
                print(name, a, m, 'E', mt['E'], 'leak', round(r['leak_error_rate'], 3), 'audit', round(r['audit_rate'], 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/'simulations.json', out)


# ----------------------------------------------------------------------------- QuantumKatas

def orders(n, plan_index):
    out = {'primary': np.random.default_rng(20261110).permutation(n), 'plan': np.arange(n)}
    for i in range(20):
        out[f'random_{i+1:02d}'] = np.random.default_rng(20261111+i).permutation(n)
    return out


def qk(n_seeds=100, alphas=ALPHAS, which=None, save_traces=True, tag='full'):
    cfg, hps = frozen()
    streams = qk_streams()
    seeds = [1000+r for r in range(n_seeds)]
    res = {'config_sha256': file_digest(CONFIG), 'seeds': seeds, 'generators': {}}
    traces = {}
    for gen, s in streams.items():
        c0, e0 = s['confidence'], s['errors']
        g = {'evaluable': len(e0), 'excluded': [{'task_id': t, 'reason': why} for _, t, why in s['excluded']],
             'raw_errors': int(e0.sum()), 'orders': {}}
        for oname, perm in orders(len(e0), None).items():
            if which and oname not in which:
                continue
            c, e = c0[perm], e0[perm]
            u = sa.uniforms(seeds, len(e))
            cell = {}
            for a in alphas:
                per = {}
                for m in ('AT', 'AT+CC', 'random'):
                    tr, mt = replay(m, a, hps[m], c, e, u)
                    per[m] = {**summary(mt, hps[m]), 'seconds_per_run': tr.seconds/len(seeds), 'state_bytes_per_run': tr.state_bytes//len(seeds)}
                    per[m]['_mt'] = mt
                    if oname == 'primary' and save_traces and m != 'random':
                        traces[f'{gen}|{a}|{m}'] = tr
                d = {k: per['AT+CC']['_mt'][k]-per['AT']['_mt'][k] for k in ('audit_rate', 'leak_error_rate', 'auto_accept_rate')}
                per['paired_cc_minus_at'] = {k: {'mean': float(np.nanmean(v)), 'q025_975': [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]}
                                             for k, v in d.items()}
                thr, aud = sa.oracle_threshold(c, e, a)
                mt = sa.metrics(aud[None, :], np.ones((1, len(e))), e)
                per['oracle_threshold'] = {'threshold': None if np.isinf(thr) else float(thr), **summary(mt)['ratio_of_means']}
                for m in ('AT', 'AT+CC', 'random'):
                    per[m].pop('_mt')
                cell[str(a)] = per
                if oname in ('primary', 'plan'):
                    print(gen, oname, a, {m: (round(per[m]['ratio_of_means']['leak_error_rate'], 3), round(per[m]['ratio_of_means']['audit_rate'], 3)) for m in ('AT', 'AT+CC', 'random')}, flush=True)
            g['orders'][oname] = cell
        res['generators'][gen] = g
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/f'{tag}.json', res)
    if traces:
        arrays = {}
        for key, tr in traces.items():
            k = key.replace('|', '__').replace('+', 'p')
            arrays.update({f'{k}__pi': tr.pi.astype(np.float32), f'{k}__audited': tr.audited,
                           f'{k}__score': tr.score.astype(np.float32), f'{k}__tau_before': tr.tau_before.astype(np.float32),
                           f'{k}__tau_after': tr.tau_after.astype(np.float32)})
        for gen, s in streams.items():
            perm = orders(len(s['errors']), None)['primary']
            arrays[f'{gen}__task_ids'] = np.array(s['task_ids'])[perm]
            arrays[f'{gen}__confidence'] = s['confidence'][perm]
            arrays[f'{gen}__errors_for_evaluator'] = s['errors'][perm]
        np.savez_compressed(OUT/f'traces_primary_{tag}.npz', **arrays)
    return res


def sensitivity(n_seeds=100):
    """Pre-declared alternative hyperparameters on the primary order (config: hyperparameter_sensitivity)."""
    cfg, _ = frozen()
    streams = qk_streams()
    seeds = [1000+r for r in range(n_seeds)]
    res = {}
    for name, hp in cfg['hyperparameter_sensitivity']['sets'].items():
        for gen, s in streams.items():
            perm = orders(len(s['errors']), None)['primary']
            c, e = s['confidence'][perm], s['errors'][perm]
            u = sa.uniforms(seeds, len(e))
            res.setdefault(name, {})[gen] = {str(a): {m: summary(replay(m, a, hp, c, e, u)[1], hp) for m in ('AT', 'AT+CC')}
                                             for a in ALPHAS if hp['gamma'] <= 1-a}
    atomic_json(OUT/'hyperparameter_sensitivity.json', res)


def end_to_end(n_seeds=100):
    """Sensitivity: all 322 tasks, end-to-end failures entered as confidence 0 and error 1 (primary order)."""
    _, hps = frozen()
    from vgx.quantum_katas import crossfit
    cfg = json.loads(Path(QK_CONFIG).read_text())
    split = {'value': json.loads(Path(cfg['split']).read_text()), 'sha256': file_digest(cfg['split'])}
    seeds = [1000+r for r in range(n_seeds)]
    res = {}
    for gen in GENS:
        rows = crossfit.load_rows(cfg['runs'][gen], split, [])
        c0 = np.array([r['prior'] if r['valid'] else 0.0 for r in rows], float)
        e0 = np.array([1-r['outcome'] if r['valid'] else 1 for r in rows], int)
        perm = np.random.default_rng(20261110).permutation(len(e0))
        c, e = c0[perm], e0[perm]
        u = sa.uniforms(seeds, len(e))
        res[gen] = {str(a): {m: summary(replay(m, a, hps[m], c, e, u)[1], hps[m]) for m in ('AT', 'AT+CC', 'random')} for a in ALPHAS}
    atomic_json(OUT/'end_to_end.json', res)


def main():
    cmd = sys.argv[1]
    if cmd == 'tune':
        tune()
    elif cmd == 'simulate':
        simulate()
    elif cmd == 'smoke':
        t = time.perf_counter()
        qk(n_seeds=10, alphas=(0.1,), which=('primary',), save_traces=False, tag='smoke')
        print('smoke seconds', round(time.perf_counter()-t, 2))
    elif cmd == 'run':
        qk()
        end_to_end()
        sensitivity()


if __name__ == '__main__':
    main()
