"""Data for the graph-first deck: per dataset x generator panels (alpha curves, risk curves, cost curves),
mean release rate per budget, accuracy per pair (for the FNR note). Cache-only result files."""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
from vgx import offline_multirisk_report as R
from vgx.offline_multirisk import METHODS

CFG = 'configs/offline_multirisk_20261118_calibrated.json'
cfg = json.load(open(CFG))
res = R.load(cfg, cfg['output_dir'])
rows = [r for r in R.rows_of(res) if R.in_main(r['dataset'], r['generator'], r['risk_metric'])]
I = R.index(rows)
name = lambda m: 'Cal-threshold' if m == 'Raw-threshold' else ('SSV' if m == 'SSV-Frozen' else m.replace('-Mix', ''))
MAIN = [m for m in METHODS if m != 'Self-consistency']
shown = cfg['budget_fractions_shown']
out = {'methods': [name(m) for m in MAIN], 'alpha': cfg['alpha_target'], 'alpha_main': cfg['alpha_main'], 'shown': shown,
       'fracs': cfg['budget_fractions'], 'risks': ['selective_error', 'fpr'], 'panels': {}, 'meanbar': {}, 'cost': {}}
for risk in out['risks']:
    al = cfg['alpha_target'][risk]
    pan = []
    for ds, dn in R.DS:
        for g, gn in R.GEN.items():
            if not R.in_main(ds, g, risk):
                continue
            p = {'label': f'{dn} · {gn}', 'obj': {}, 'risk': {}, 'cost': {}}
            for f in shown:
                p['obj'][str(f)] = {name(m): [None if x is None else float(R.objective(x, risk)) for a in al for x in [I.get((ds, g, risk, a, f, m))]]
                                    for m in MAIN if any((ds, g, risk, a, f, m) in I for a in al)}
                p['risk'][str(f)] = {name(m): [None if x is None or x['risk'] is None else float(x['risk']) for a in al for x in [I.get((ds, g, risk, a, f, m))]]
                                     for m in MAIN if any((ds, g, risk, a, f, m) in I for a in al)}
            am = cfg['alpha_main'][risk]
            cref = I[(ds, g, risk, am, 1.0, 'Ours-Offline')]['budget_usd']
            p['cost'] = {name(m): [float(I[(ds, g, risk, am, float(f), m)]['mean_screen_cost']/cref) if (ds, g, risk, am, float(f), m) in I else None
                                   for f in cfg['budget_fractions']] for m in MAIN if (ds, g, risk, am, 1.0, m) in I}
            pan.append(p)
    out['panels'][risk] = pan
    out['meanbar'][risk] = {str(f): {name(m): float(np.mean(v)) for m in MAIN
                                     for v in [[R.objective(I[k], risk) for k in I if k[2] == risk and k[4] == f and k[5] == m]] if v}
                            for f in shown}
acc = []
for ds, dn in R.DS:
    for g, gn in R.GEN.items():
        x = I[(ds, g, 'fnr', cfg['alpha_main']['fnr'], 1.0, 'Release-eligible')] if (ds, g, 'fnr', cfg['alpha_main']['fnr'], 1.0, 'Release-eligible') in I else None
        r = res[(ds, g)]
        nt = r['n_test']; corr = sum(f['test_correct'] for f in r['folds'])
        acc.append([f'{dn} · {gn}', corr/nt])
out['accuracy'] = acc
win = []
for r in rows:
    if r['risk_metric'] != 'fpr' or r['method'] != 'Ours-Offline':
        continue
    k = (r['dataset'], r['generator'], 'fpr', r['alpha_target'], r['budget_fraction'])
    lo, hi = r['paired']['Fixed-1-Mix']['ci95']
    if hi < 0:
        o, f1 = I[k+('Ours-Offline',)], I[k+('Fixed-1-Mix',)]
        win.append({'pair': f"{dict(R.DS)[r['dataset']]} · {R.GEN[r['generator']]}", 'alpha': r['alpha_target'],
                    'f1_risk': f1['risk'], 'ours_risk': o['risk']})
out['fpr_f1'] = {'n': len(win), 'f1_over': sum(w['f1_risk'] > w['alpha']+1e-12 for w in win),
                 'ours_over': sum(w['ours_risk'] > w['alpha']+1e-12 for w in win),
                 'f1_ratio': float(np.mean([w['f1_risk']/w['alpha'] for w in win])), 'ours_ratio': float(np.mean([w['ours_risk']/w['alpha'] for w in win])),
                 'f1_ex': float(np.mean([w['f1_risk']-w['alpha'] for w in win])), 'ours_ex': float(np.mean([w['ours_risk']-w['alpha'] for w in win])),
                 'by_pair': {p: sum(w['pair'] == p for w in win) for p in sorted({w['pair'] for w in win})}}
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False)
print(json.dumps(out['meanbar'], ensure_ascii=False)[:800]); print(acc)
