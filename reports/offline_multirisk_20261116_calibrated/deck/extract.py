"""Numbers for the PPT deck, all from the calibrated and raw result files (no API calls)."""
import json
import os
import sys

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from vgx.offline_multirisk import BASELINES, METHODS, RISKS
from vgx import offline_multirisk_report as R

CFG = os.environ.get('CFG', 'configs/offline_multirisk_20261116_calibrated.json')
cfg = json.load(open(CFG))
res = R.load(cfg, cfg['output_dir'])
res_r = R.load(cfg, cfg['output_dir'].replace('_calibrated', '_raw'))


def rows_I(res):
    rows = [r for r in R.rows_of(res) if R.in_main(r['dataset'], r['generator'], r['risk_metric'])]
    return rows, R.index(rows)


rows, I = rows_I(res)
rows_r, I_r = rows_I(res_r)
name = lambda m: 'Cal-threshold' if m == 'Raw-threshold' else ('SSV' if m == 'SSV-Frozen' else m.replace('-Mix', ''))
out = {'methods': [name(m) for m in METHODS], 'baselines': [name(b) for b in BASELINES], 'alpha': cfg['alpha_target'],
       'alpha_main': cfg['alpha_main'], 'risk_name': R.RISK_NAME}


def sig(I, rows):
    d = {}
    for risk in RISKS:
        S = R.sig_counts(I, rows, risk)
        ex = {name(m): sum(R.over(I[k]) for k in I if k[2] == risk and k[5] == m) for m in METHODS}
        nc = {name(m): sum(1 for k in I if k[2] == risk and k[5] == m) for m in METHODS}
        bud = {name(m): sum(not I[k]['budget_met'] for k in I if k[2] == risk and k[5] == m) for m in METHODS}
        d[risk] = {'n': S[BASELINES[0]]['n'], 'nb': [S[b]['n'] for b in BASELINES], 'ours': [S[b]['ours'] for b in BASELINES], 'base': [S[b]['base'] for b in BASELINES],
                   'per': {name(b): S[b]['per'] for b in BASELINES}, 'overrun': ex, 'budget_over': bud, 'n_cond': nc}
    return d


out['sig_cal'] = sig(I, rows)
out['sig_raw'] = sig(I_r, rows_r)

# mean objective vs alpha at f = 1, over the dataset x generator pairs in the main comparison; also per dataset
curves = {}
for risk in RISKS:
    c = {}
    for m in METHODS:
        c[name(m)] = [float(np.mean([R.objective(I[k], risk) for k in I if k[2] == risk and k[5] == m and k[3] == a and k[4] == 1.0]))
                      for a in cfg['alpha_target'][risk]]
    curves[risk] = c
out['curves_f1'] = curves
perds = {}
for risk in RISKS:
    perds[risk] = {}
    for ds, dn in R.DS:
        ks = [k for k in I if k[0] == ds and k[2] == risk]
        if not ks:
            continue
        perds[risk][dn] = {name(m): [float(np.mean([R.objective(I[k], risk) for k in ks if k[5] == m and k[3] == a and k[4] == 1.0]))
                                     for a in cfg['alpha_target'][risk]] for m in METHODS}
out['curves_f1_ds'] = perds

# main alpha table at f = 1 per dataset x generator
main = {}
for risk in RISKS:
    a = cfg['alpha_main'][risk]
    tab = []
    for ds, dn in R.DS:
        for g, gn in R.GEN.items():
            if not R.in_main(ds, g, risk):
                continue
            cells = []
            for m in METHODS:
                x = I.get((ds, g, risk, a, 1.0, m))
                cells.append(None if x is None else [R.objective(x, risk), x['risk'], bool(x['empirical_feasible'])])
            tab.append([f'{dn} · {gn}', cells])
    main[risk] = tab
out['main_f1'] = main

# alpha examples (calibrated p, no verifier)
ex = []
for ds, dn in R.DS:
    for g, gn in R.GEN.items():
        fo = res[(ds, g)]['folds']
        b = [x for f in fo for x in f['test_prior_eligible']]
        n_inel = sum(f['n_test'] - f['test_eligible'] for f in fo)
        cells = []
        for risk in RISKS:
            if R.excluded(ds, g, risk):
                cells.append('不测'); continue
            t, sh = R.threshold_example(b, n_inel, risk, cfg['alpha_main'][risk])
            if t is None:
                cells.append('一道都不放行')
            elif sh * (len(b) + n_inel) >= len(b):
                cells.append(f'全部放行（{100*sh:.0f}%）')
            else:
                cells.append(f'p ≥ {t:.2f}，放行 {100*sh:.0f}%')
        ex.append([f'{dn} · {gn}', cells])
out['examples'] = ex

# input compare
ic = {}
for risk in RISKS:
    ic[risk] = {}
    for m in METHODS:
        ks = [k for k in I if k[2] == risk and k[5] == m and k in I_r]
        ic[risk][name(m)] = [float(np.mean([R.objective(I[k], risk) for k in ks])), float(np.mean([R.objective(I_r[k], risk) for k in ks])),
                             sum(R.over(I[k]) for k in ks), sum(R.over(I_r[k]) for k in ks), len(ks)]
out['input_compare'] = ic

# spending diagnostics per risk (f > 0)
sp = {}
for risk in RISKS:
    sp[risk] = {}
    for m in METHODS:
        ks = [k for k in I if k[2] == risk and k[5] == m and k[4] > 0]
        sp[risk][name(m)] = {'budget_use': float(np.mean([I[k]['mean_screen_cost'] / I[k]['budget_usd'] for k in ks])),
                             'query_rate': float(np.mean([I[k].get('verifier_query_rate') or 0 for k in ks])),
                             'fallback': float(np.mean([I[k]['fallback_share'] for k in ks])),
                             'coverage': float(np.mean([I[k]['coverage'] for k in ks])),
                             'e': float(np.mean([I[k]['joint_wrong_release_rate'] for k in ks]))}
out['spend'] = sp

# FNR: both-feasible comparison against Fixed-1
P = R.both_feasible_perf(I, rows, 'fnr')
out['fnr_both'] = {name(b): {k: P[b][k] for k in ('n', 'median', 'ours_better_ci', 'baseline_better_ci')} for b in BASELINES}
out['cref'] = {dn: [min(f['c_ref_usd'] * 1e4 for g in R.GEN for f in res[(ds, g)]['folds']),
                    max(f['c_ref_usd'] * 1e4 for g in R.GEN for f in res[(ds, g)]['folds'])] for ds, dn in R.DS}
out['fpr_excl'] = {gn: int(sum(f['test_wrong'] for f in res[('math500', g)]['folds'])) for g, gn in R.GEN.items()}
out['n_cond'] = len({(r['dataset'], r['generator'], r['risk_metric'], r['alpha_target'], r['budget_fraction']) for r in rows})
v = json.load(open(cfg['output_dir'] + '/verify.json'))
out['verify'] = {'cells': v['checked_cells'], 'ok': v['ok'], 'fields': v['checked_fields']}
vr = json.load(open(cfg['output_dir'].replace('_calibrated', '_raw') + '/verify.json'))
out['verify_raw'] = {'cells': vr['checked_cells'], 'ok': vr['ok']}
out['code'] = next(iter(res.values()))['code_hash']
bb = {}
for risk in RISKS:
    bb[risk] = {}
    for b in ('Fixed-1-Mix', 'Fixed-3-Mix'):
        c = {}
        for r in rows:
            if r['risk_metric'] != risk or r['method'] != 'Ours-Offline':
                continue
            lo, hi = r['paired'][b]['ci95']
            good = (lo > 0) if risk != 'fnr' else (hi < 0)
            bad = (hi < 0) if risk != 'fnr' else (lo > 0)
            x = c.setdefault(str(r['budget_fraction']), [0, 0, 0]); x[0] += 1; x[1] += good; x[2] += bad
        bb[risk][name(b)] = c
out['by_budget'] = bb
out['variant'] = 'pool' if cfg['ours'].get('fixed_panels_in_pool') else 'orig'
out['cfg_path'] = CFG
out['out_dir'] = cfg['output_dir']
out['report_dir'] = cfg['report_dir']
if out['variant'] == 'pool':
    share = {}
    for (ds, g), r in res.items():
        confs = json.load(open(f"{cfg['output_dir']}/selected_configs_{ds}_{g}.json")); confs = confs.get('configs', confs)
        for key, cell in r['cells'].items():
            risk, _, f = key.split('|')
            if float(f) == 0 or not R.in_main(ds, g, risk):
                continue
            grp = 'lo' if float(f) <= 0.25 else 'hi'
            for ids in cell['Ours-Offline']['config_ids']:
                for cid in ids:
                    x = share.setdefault(risk, {}).setdefault(grp, [0, 0, 0, 0])
                    x[3] += 1
                    if cid is None or str(cid).startswith('fallback'):
                        continue
                    c = confs[cid]
                    if c.get('pool') == 'fixed_panel':
                        x[0 if len(c['order']) == 1 else 1] += 1
                    else:
                        x[2] += 1
    out['pool_share'] = share
# FNR spending per dataset (f > 0)
sd = {}
for ds, dn in R.DS:
    sd[dn] = {}
    for m in ('Ours-Offline', 'Fixed-1-Mix', 'Fixed-3-Mix'):
        ks = [k for k in I if k[0] == ds and k[2] == 'fnr' and k[5] == m and k[4] > 0]
        sd[dn][name(m)] = {'ask': float(np.mean([I[k].get('verifier_query_rate') or 0 for k in ks])),
                           'spend': float(np.mean([I[k]['mean_screen_cost']/I[k]['budget_usd'] for k in ks])),
                           'fnr': float(np.mean([I[k]['fnr'] for k in ks])),
                           'fallback': float(np.mean([I[k]['fallback_share'] for k in ks]))}
out['fnr_spend_ds'] = sd
sg = {}
for r in rows:
    if r['method'] != 'Ours-Offline':
        continue
    dn = dict(R.DS)[r['dataset']]
    for b in BASELINES:
        if b not in r['paired']:
            continue
        lo, hi = r['paired'][b]['ci95']
        good = (lo > 0) if r['risk_metric'] != 'fnr' else (hi < 0); bad = (hi < 0) if r['risk_metric'] != 'fnr' else (lo > 0)
        x = sg.setdefault(dn, {}).setdefault(r['risk_metric'], {}).setdefault(name(b), [0, 0, 0, 0, 0, 0, 0])
        k = 3 if r['budget_fraction'] <= 0.25 else 5
        x[0] += good; x[1] += bad; x[2] += 1; x[k] += good; x[k+1] += bad
out['sig_ds'] = sg
# typical comparison: main alpha, f = 0.5, mean over generators per dataset; cost and risk curves
TF = 0.5
typ, ccur, rcur = {}, {}, {}
for risk in RISKS:
    am = cfg['alpha_main'][risk]
    typ[risk] = {}
    for ds, dn in R.DS:
        d = {name(m): float(np.mean(v)) for m in METHODS for v in [[R.objective(I[k], risk) for k in I if k[0] == ds and k[2] == risk and k[3] == am and k[4] == TF and k[5] == m]] if v}
        if d:
            typ[risk][dn] = d
    ccur[risk], rcur[risk] = {}, {}
    for m in METHODS:
        cc = []
        for f in cfg['budget_fractions']:
            v = [I[k]['mean_screen_cost']/I[(k[0], k[1], risk, am, 1.0, m)]['budget_usd'] for k in I if k[2] == risk and k[3] == am and k[4] == f and k[5] == m]
            cc.append(float(np.mean(v)) if v else None)
        if any(x for x in cc if x):
            ccur[risk][name(m)] = cc
        rr = []
        for a in cfg['alpha_target'][risk]:
            v = [I[k]['risk'] for k in I if k[2] == risk and k[3] == a and k[4] == TF and k[5] == m and I[k]['risk'] is not None]
            rr.append(float(np.mean(v)) if v else None)
        if any(x is not None for x in rr):
            rcur[risk][name(m)] = rr
out['budget_fractions'] = cfg['budget_fractions']
out['typ'], out['cost_curve'], out['risk_curve'], out['typ_f'] = typ, ccur, rcur, TF
out['n_pairs'] = {risk: len({(r['dataset'], r['generator']) for r in rows if r['risk_metric'] == risk}) for risk in RISKS}
# the variants tried on top of the original version
var = {}
for tag, label in (('20261116_calibrated', '原版本'), ('20261117_pool_calibrated', '加固定 panel'), ('20261117_shrink_calibrated', '收缩校准'),
                   ('20261117_gamma_calibrated', 'verifier 打折'), ('20261117_shrinkgamma_calibrated', '收缩 + 打折')):
    c2 = json.load(open(f'configs/offline_multirisk_{tag}.json'))
    r2 = R.load(c2, c2['output_dir'])
    rw, Iv = rows_I(r2)
    d = {}
    for risk in RISKS:
        Sv = R.sig_counts(Iv, rw, risk)
        d[risk] = {'f1': [Sv['Fixed-1-Mix']['ours'], Sv['Fixed-1-Mix']['base']], 'f3': [Sv['Fixed-3-Mix']['ours'], Sv['Fixed-3-Mix']['base']],
                   'over': sum(R.over(Iv[k]) for k in Iv if k[2] == risk and k[5] == 'Ours-Offline')}
    good = bad = 0
    for r in rw:
        if r['risk_metric'] == 'selective_error' and r['method'] == 'Ours-Offline' and r['budget_fraction'] == 1.0:
            lo, hi = r['paired']['Fixed-3-Mix']['ci95']; good += lo > 0; bad += hi < 0
    d['sel_f1_vs_f3'] = [good, bad]
    var[label] = d
out['variants'] = var
def scrub(o):
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if not (isinstance(v, float) and v != v)}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    return o
out = scrub(out)
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('sig_cal', 'spend', 'fnr_both', 'cref', 'n_cond', 'verify')}, ensure_ascii=False)[:6000])
