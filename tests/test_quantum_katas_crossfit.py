"""Cross-fitting uses only training folds and pools held-out decisions over every task."""
import json
from pathlib import Path

import numpy as np

from vgx.quantum_katas import crossfit

CFG = json.loads(Path('configs/quantum_katas_crossfit_20261019.json').read_text())


def synthetic_rows(n=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = int(rng.random() < 0.6)
        valid = i % 17 != 0
        sig = {v: {'score': float(np.clip((0.75 if y else 0.3) + rng.normal(0, 0.2), 0, 1)) if i % 11 else None,
                   'failure': None if i % 11 else 'invalid_json', 'estimated_usd': 0.0005}
               for v in CFG['verifiers']}
        rows.append({'task_id': f't{i}', 'fold': i % 5, 'prior': float(rng.uniform(0.5, 1.0)) if valid else None,
                     'has_code': True, 'labeled': True, 'valid': valid, 'outcome': y, 'generator_usd': 0.001,
                     'signals': sig})
    return rows


def test_crossfit_pools_every_task_and_fits_out_of_fold(monkeypatch):
    rows = synthetic_rows()
    seen = []
    original = crossfit.fit_fold

    def spy(train, verifiers, settings):
        seen.append({r['fold'] for r in train})
        return original(train, verifiers, settings)
    monkeypatch.setattr(crossfit, 'fit_fold', spy)
    cfg = {**CFG, 'settings': {**CFG['settings'], 'bootstrap_repeats': 50}}
    out = crossfit.analyze_generator(rows, cfg)
    assert len(seen) == 5 * len(cfg['settings']['losses'])
    assert all(len(folds) == 4 for folds in seen)
    for loss, pooled in out['pooled'].items():
        for arm in pooled['arms'].values():
            assert arm['n'] == len(rows)
        assert pooled['arms']['prior_only']['queries'] == 0
    for k, by_loss in out['folds'].items():
        for fit in by_loss.values():
            assert len(fit['selected_triple']) == 3
    assert out['counts']['valid'] == sum(r['valid'] for r in rows)


def test_extended_stopping_arms_are_consistent():
    from vgx.quantum_katas import crossfit_extended
    rows = synthetic_rows(120, seed=1)
    cfg = {**CFG, 'settings': {**CFG['settings'], 'bootstrap_repeats': 50}}
    st, folds = crossfit_extended.stopping(rows, cfg, 19.0)
    arms = st['arms']
    valid = sum(r['valid'] for r in rows)
    assert arms['fixed_five']['queries'] == 5 * valid and arms['fixed_triple']['queries'] == 3 * valid
    assert arms['adaptive_triple']['queries'] <= arms['fixed_triple']['queries']
    assert arms['prior_only']['queries'] == 0 and len(folds) == 5
    ref = crossfit.analyze_generator(rows, cfg)['pooled']['19']['arms']['triple']
    assert (ref['released'], ref['queries']) == (arms['adaptive_triple']['released'], arms['adaptive_triple']['queries'])


def test_calibrated_prior_uses_training_folds_only():
    from vgx.quantum_katas import crossfit_calibrated
    rows = synthetic_rows(100, seed=2)
    cal = crossfit_calibrated.calibrator(rows)
    out = cal(0, rows)
    flipped = [dict(r, outcome=1-r['outcome']) if r['fold'] == 0 else r for r in rows]
    again = crossfit_calibrated.calibrator(flipped)(0, flipped)
    assert [r['prior'] for r in out] == [r['prior'] for r in again]  # fold-0 labels never used for fold 0
    assert all(r['raw_prior'] == s['prior'] for r, s in zip(out, rows) if s['valid'])
