import json

import numpy as np

from vgx.gpqa.diamond_cost import decisions, score
from vgx.gpqa.sequential import seal

VS = ['a', 'b', 'c', 'd', 'e']


def plan(rng):
    lik = {v+':probability': {'p_bin_if_correct': list(rng.dirichlet([1, 1, 4])),
                              'p_bin_if_incorrect': list(rng.dirichlet([3, 1, 2]))} for v in VS}
    fit = {'likelihoods': lik, 'expected_cost_usd': {v+':probability': 0.001*(i+1) for i, v in enumerate(VS)},
           'adaptive_policies': {'raw/19': {'independence': {'selected_order': ['c', 'a', 'e']}},
                                 'raw/99': {'independence': {'selected_order': ['b', 'c', 'd']}}}}
    return seal({'settings': {'correct_reward': 1.0, 'losses': [19.0, 99.0], 'probability_bins': 3,
                              'utility_per_usd': 10.0}, 'generators': {'g': fit}}, 'plan_id')


def signals(rng, n=40):
    rows = []
    for i in range(n):
        sig = {v+':probability': {'score': float(rng.uniform()), 'estimated_usd': 0.002} for v in VS}
        rows.append({'item_id': str(i), 'answer': 'A' if i else None, 'prior': float(rng.uniform(0.85, 1.0)),
                     'signals': sig})
    rows[5]['signals']['e:probability']['score'] = None
    return seal({'rows': {'g': rows}}, 'signals_id')


def test_fixed_arms_ask_everything_and_threshold_the_full_posterior():
    rng = np.random.default_rng(0)
    p, s = plan(rng), signals(rng)
    lik = p['generators']['g']['likelihoods']
    ds = decisions(p, s)['g']['19']['arms']
    rows = s['rows']['g']
    for d, r in zip(ds['fixed_five'], rows):
        if r['answer'] is None:
            assert d['verifiers_used'] == 0 and d['action'] == 'abstain'
            continue
        assert d['verifiers_used'] == 5
        if r is rows[5]:
            assert d['action'] == 'abstain' and d['failure'] == 'missing_verifier_score'
            continue
        lr = np.prod([lik[v+':probability']['p_bin_if_correct'][min(int(r['signals'][v+':probability']['score']*3), 2)]
                      / lik[v+':probability']['p_bin_if_incorrect'][min(int(r['signals'][v+':probability']['score']*3), 2)]
                      for v in VS])
        b = r['prior']*lr/(r['prior']*lr + 1 - r['prior'])
        assert d['action'] == ('assert' if b >= 0.95 else 'abstain')
        assert np.isclose(d['expected_cost_usd'], 0.015) and np.isclose(d['usage_estimated_cost_usd'], 0.01)
    assert all(d['verifiers_used'] == 3 for d, r in zip(ds['fixed_triple'], rows) if r['answer'])
    assert all(d['verifiers_used'] <= 3 for d in ds['adaptive'])


def test_score_writes_aggregates_only(tmp_path):
    rng = np.random.default_rng(1)
    p, s = plan(rng), signals(rng)
    flags = seal({'signals_id': s['signals_id'],
                  'outcomes': {'g': {r['item_id']: (int(rng.uniform() < 0.8) if r['answer'] else None)
                                     for r in s['rows']['g']}}}, 'outcomes_id')
    paths = {}
    for name, value in (('plan', p), ('signals', s), ('outcomes', flags)):
        paths[name] = tmp_path/f'{name}.json'
        paths[name].write_text(json.dumps(value))
    r = score(paths['plan'], paths['signals'], paths['outcomes'], tmp_path/'result.json', repeats=200)
    arms = r['policies']['g']['19']['arms']
    assert set(arms) == {'prior_only', 'adaptive', 'fixed_triple', 'fixed_five'}
    assert arms['fixed_five']['calls'] == 5*39 and arms['prior_only']['calls'] == 0
    assert 'item_id' not in json.dumps(r) and r['brier']['g']['n'] == 38
