"""Evaluator: scores replay logs with the official labels. Nothing here feeds back into any learner.

Per round (stream task t, quantities fixed before the round's update):
  brier_prior   (p_hat_t - Y_t)^2, p_hat_t = r_phi_t(r_t)
  brier_panel   (b_full - Y_t)^2, b_full = posterior from p_hat_t and the full cached panel under theta_t
                (evaluates theta_t; the learner never sees the full panel unless the round was audited)
  label_ll      log prod_j theta_t[j, Y_t, A_tj] on the full panel (true class-conditional fit of theta_t)
  (dependence-model logs carry joint_start: the hierarchy enters through the joint table T_t and the
  other verifiers through their marginals; see vgx.online.joint.panel_logs)
  release, wrong release, decision cost, audit cost, belief at release (model-estimated error 1 - b)

Intervals: cluster bootstrap over the 35 task groups. A resampled group brings every one of its
(rotation, order) entries, so repeated orders of the same tasks never count as new samples.
"""
from __future__ import annotations

import numpy as np

from vgx.online import joint as jm
from vgx.online import likelihood as lk
from vgx.online.views import bin_of

SUMS = ('n', 'proc', 'released', 'wrong', 'brier_prior', 'brier_panel', 'n_panel', 'label_ll', 'est_wrong',
        'decision_usd', 'audit_usd', 'audits', 'extreme')


def round_eval(rec, labels, signals, verifiers, hier=None):
    y = labels.outcome(rec['task_id'])
    out = {'y': y, 'release': rec['action'] == 'release'}
    out['wrong'] = out['release'] and y == 0
    if rec['p_hat'] is not None:
        out['brier_prior'] = (rec['p_hat'] - y)**2
        out['extreme'] = rec['p_hat'] < .01 or rec['p_hat'] > .99
        sig = signals[rec['task_id']]
        if all(sig[v]['score'] is not None for v in verifiers):
            bins = np.array([[bin_of(sig[v]['score']) for v in verifiers]])
            g = np.array(rec['theta_start'])
            if rec.get('joint_start') is not None:
                if hier is None:
                    raise ValueError('dependence-model log needs the hierarchy')
                T = np.array(rec['joint_start'])
                b = float(jm.posterior([rec['p_hat']], bins, g, T, hier)[0])
                out['label_ll'] = float(jm.label_loglik(g, T, hier, bins, [y])[0])
            else:
                b = float(lk.posterior([rec['p_hat']], bins, g)[0])
                j = np.arange(len(verifiers))
                out['label_ll'] = float(np.log(g[j, y, bins[0]]).sum())
            out['brier_panel'] = (b - y)**2
    if out['release'] and rec.get('belief') is not None:
        out['est_wrong'] = 1 - rec['belief']
    return out


def group_sums(log, labels, signals, verifiers, hier=None):
    """Per-group sums of the additive quantities, plus per-round evaluation rows."""
    groups, rows = {}, []
    for rec in log:
        e = round_eval(rec, labels, signals, verifiers, hier)
        rows.append(e)
        s = groups.setdefault(rec['group'], dict.fromkeys(SUMS, 0.))
        s['n'] += 1
        s['proc'] += rec['p_hat'] is not None
        s['released'] += e['release']
        s['wrong'] += e['wrong']
        s['brier_prior'] += e.get('brier_prior', 0.)
        s['extreme'] += e.get('extreme', False)
        if 'brier_panel' in e:
            s['brier_panel'] += e['brier_panel']
            s['label_ll'] += e['label_ll']
            s['n_panel'] += 1
        s['est_wrong'] += e.get('est_wrong', 0.)
        s['decision_usd'] += rec['decision_cost_usd']
        s['audit_usd'] += rec['audit_cost_usd']
        s['audits'] += rec['audited']
    return groups, rows


def metrics(tot):
    f = lambda a, b: tot[a]/tot[b] if tot[b] else float('nan')
    return {'pub': f('released', 'n'), 'risk': f('wrong', 'released'), 'brier_prior': f('brier_prior', 'proc'),
            'brier_panel': f('brier_panel', 'n_panel'), 'label_ll': f('label_ll', 'n_panel'),
            'est_risk': f('est_wrong', 'released'), 'risk_gap': (f('wrong', 'released') - f('est_wrong', 'released')),
            'decision_usd': f('decision_usd', 'n'), 'audit_usd': f('audit_usd', 'n'),
            'total_usd': (tot['decision_usd'] + tot['audit_usd'])/tot['n'] if tot['n'] else float('nan'),
            'extreme_rate': f('extreme', 'proc'), 'released': tot['released'], 'wrong': tot['wrong'], 'n': tot['n']}


def pooled(entries):
    """entries: list of per-run group-sum dicts. Pools sums over runs and groups."""
    tot = dict.fromkeys(SUMS, 0.)
    for g in entries:
        for s in g.values():
            for k in SUMS:
                tot[k] += s[k]
    return tot


def bootstrap(entries_by_arm, reps=2000, seed=20261109, keys=('pub', 'risk', 'brier_prior', 'brier_panel', 'total_usd')):
    """entries_by_arm[arm] = list of per-run group-sum dicts (paired across arms by list position).

    Returns per-arm metric CIs and paired differences arm - reference for every ordered pair."""
    arms = list(entries_by_arm)
    groups = sorted({g for runs in entries_by_arm.values() for run in runs for g in run})
    # per arm, per group: summed over runs
    per = {a: {g: dict.fromkeys(SUMS, 0.) for g in groups} for a in arms}
    for a in arms:
        for run in entries_by_arm[a]:
            for g, s in run.items():
                for k in SUMS:
                    per[a][g][k] += s[k]
    mat = {a: np.array([[per[a][g][k] for k in SUMS] for g in groups]) for a in arms}
    rng = np.random.default_rng(seed)
    draws = {a: [] for a in arms}
    for _ in range(reps):
        idx = rng.integers(0, len(groups), len(groups))
        for a in arms:
            tot = dict(zip(SUMS, mat[a][idx].sum(axis=0)))
            m = metrics(tot)
            draws[a].append([m[k] for k in keys])
    out = {'arms': {}, 'diff': {}}
    for a in arms:
        d = np.array(draws[a])
        point = metrics(dict(zip(SUMS, mat[a].sum(axis=0))))
        out['arms'][a] = {k: {'point': point[k], 'ci95': np.nanpercentile(d[:, i], [2.5, 97.5]).tolist()}
                          for i, k in enumerate(keys)}
        out['arms'][a]['counts'] = {k: point[k] for k in ('released', 'wrong', 'n')}
    for a in arms:
        for b in arms:
            if a == b:
                continue
            dd = np.array(draws[a]) - np.array(draws[b])
            pa = metrics(dict(zip(SUMS, mat[a].sum(axis=0))))
            pb = metrics(dict(zip(SUMS, mat[b].sum(axis=0))))
            out['diff'][f'{a}-{b}'] = {k: {'point': pa[k]-pb[k], 'ci95': np.nanpercentile(dd[:, i], [2.5, 97.5]).tolist()}
                                      for i, k in enumerate(keys)}
    return out
