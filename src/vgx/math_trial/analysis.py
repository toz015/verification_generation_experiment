"""MATH-500 sequential-verification analysis: 5-fold cross-fitting with the unchanged GPQA method.

Same likelihoods (3 bins, Laplace), conditional independence, exact planner, raw-r prior and arms
as the QuantumKatas analysis (crossfit_extended); the calibrated-prior variant (appendix) is the
QuantumKatas monotone map fitted on training folds only. Labels come from the pinned official
Omni-MATH-Rule grader (graded.json). A reply without a parsed answer, a valid p_correct or a
grade is invalid: the policy abstains on it and asks no verifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vgx.common.billing import estimate_call
from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.sequential import unseal
from vgx.quantum_katas import crossfit_calibrated, crossfit_extended

GRID = (1.0, 1.5, 7/3, 4.0, 17/3, 9.0, 14.0, 19.0, 97/3, 49.0, 99.0)


def _ledger(path):
    return json.loads((Path(path)/'cache'/'budget.json').read_text())['requests']


def load_rows(gen_dir, name, signals, ver_ledger, split, verifiers):
    gen_dir = Path(gen_dir)
    graded = json.loads((gen_dir/'graded.json').read_text()); unseal(graded, 'grade_id')
    bundle = json.loads((gen_dir/'frozen_candidates.json').read_text()); unseal(bundle, 'bundle_id')
    gl = _ledger(gen_dir)
    labels = {r['item_id']: r for r in graded['rows']}
    sig = {r['item_id']: r['signals'] for r in signals['rows'] if r['generator'] == name}
    rows = []
    for c in bundle['rows']:
        g = labels[c['item_id']]; has = c.get('answer') is not None
        s = sig.get(c['item_id'], {})
        rows.append({'task_id': c['item_id'], 'fold': int(split['item_fold'][c['item_id']]), 'prior': c.get('p_correct'),
                     'has_code': has, 'labeled': g['grade_status'] in ('graded', 'timeout', 'grader_error'),
                     'valid': bool(has and c.get('p_correct') is not None and g['grade_status'] == 'graded'),
                     'outcome': int(g['correct']),
                     'generator_usd': (gl.get(c.get('request_key') or '', {}).get('estimated_usd') or 0.0),
                     'signals': {v: {'score': (s.get(v) or {}).get('score'), 'failure': (s.get(v) or {}).get('failure', 'not_collected'),
                                     'estimated_usd': (ver_ledger.get((s.get(v) or {}).get('request_key') or '', {}).get('estimated_usd') or 0.0)}
                                 for v in verifiers}})
    return rows


def analyze(cfg_path, output):
    cfg = json.loads(Path(cfg_path).read_text()); output = Path(output)
    split = json.loads(Path(cfg['split']).read_text())
    signals = json.loads((Path(cfg['verification'])/'signals.json').read_text()); unseal(signals, 'signals_id')
    vl = _ledger(cfg['verification'])
    out = {'schema': 1, 'config_sha256': file_digest(cfg_path), 'split_sha256': file_digest(cfg['split']),
           'signals_id': signals['signals_id'], 'grid': list(GRID), 'generators': {}}
    for name, d in cfg['generators'].items():
        rows = load_rows(d, name, signals, vl, split, cfg['verifiers'])
        res = {'curves': {}, 'stopping': {}}
        cal = crossfit_calibrated.calibrator(rows)
        folds19 = None
        for kind, transform in (('raw', None), ('calibrated', cal)):
            curve = []
            for loss in GRID:
                arms, folds = crossfit_extended.arms_for(rows, cfg, loss, transform)
                summ, util = {}, {}
                for a, ds in arms.items():
                    summ[a], util[a] = crossfit_extended._summary(rows, ds, cfg['settings'], loss)
                curve.append({'t': loss/(1+loss), 'loss': loss, 'arms': {a: {k: s[k] for k in (
                    'released', 'wrong_released', 'output_error_rate', 'coverage', 'calls_per_task', 'measured_usd_per_task',
                    'mean_utility', 'mean_utility_ci95', 'query_histogram')} for a, s in summ.items()}})
                if loss == 19.0:
                    from vgx.gpqa.diamond_score import boot
                    reps = cfg['settings']['bootstrap_repeats']
                    res['stopping'][kind] = {f'{a}_minus_{b}': (lambda x: {'mean': x['mean'], 'ci95': x['ci95']})(
                        boot(util[a]-util[b], reps, 20261021)) for a, b in (
                        ('adaptive_triple', 'prior_only'), ('adaptive_triple', 'fixed_triple'), ('adaptive_triple', 'fixed_five'))}
                    if kind == 'raw':
                        folds19 = folds
                        res['selected_triples_L19'] = {k: f['triple'] for k, f in folds.items()}
            res['curves'][kind] = curve
        res['verifiers'] = crossfit_extended.verifier_section(rows, cfg, {'19': folds19})
        valid = [r for r in rows if r['valid']]
        res['counts'] = {'n': len(rows), 'with_answer': sum(r['has_code'] for r in rows), 'valid': len(valid),
                         'correct_all': sum(r['outcome'] for r in rows), 'correct_valid': sum(r['outcome'] for r in valid),
                         'mean_prior_valid': float(np.mean([r['prior'] for r in valid])),
                         'share_prior_ge_0.95': float(np.mean([r['prior'] >= .95 for r in valid])),
                         'brier_prior_valid': float(np.mean([(r['prior']-r['outcome'])**2 for r in valid]))}
        res['usd'] = {'generator': float(sum(r['generator_usd'] for r in rows)),
                      'verifiers_all_calls': {v: float(sum(r['signals'][v]['estimated_usd'] for r in rows)) for v in cfg['verifiers']}}
        out['generators'][name] = res
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.config, a.output); print('wrote', a.output)


if __name__ == '__main__':
    main()
