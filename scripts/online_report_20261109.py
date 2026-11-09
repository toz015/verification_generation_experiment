"""Tables and figures for the online replay report (reads results/quantum_katas_online_20261109).

  PYTHONPATH=src python scripts/online_report_20261109.py --config configs/quantum_katas_online_20261109.json \
      --out reports/quantum_katas_online_20261109
Writes key_results.json, tables.md and figures/*.png. Labels are used only for evaluation.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from vgx.online import evaluate as ev
from vgx.online.experiment import _path, read_log, run_diagnostics
from vgx.online.views import LabelStore, bin_of, load_generator

MAIN = ('A', 'B', 'C1', 'C2', 'D')
ALL = ('A', 'B', 'C1', 'C2', 'D', 'C1F', 'C2F')
DIFFS = ('B-A', 'C1-B', 'C2-B', 'C1-C2', 'C1-D', 'C2-D', 'C1F-A', 'C2F-A', 'D-A')
COLORS = {'A': '#52514e', 'B': '#2a78d6', 'C1': '#eb6834', 'C2': '#1baf7a', 'D': '#6250d6'}


def ci(d, scale=1., fmt='{:.3f}'):
    return f"{fmt.format(d['point']*scale)} [{fmt.format(d['ci95'][0]*scale)}, {fmt.format(d['ci95'][1]*scale)}]"


def theta_quality(cfg, gen, signals, labels, out, frac, q, eta=None, arms=ALL):
    """Q4: internal C1 objective per record vs true class-conditional fit and audit-verifier gap."""
    V = cfg['verifiers']
    res = {}
    codes = np.array([0, .5, 1])
    for arm in arms:
        if arm == 'A':
            continue
        lab, internal, gap0, gap1, first_jump = [], [], [], [], []
        for rot in cfg['grid']['rotations']:
            info = json.loads((out/'init'/f'{gen}_r{rot}_B{frac:g}.json').read_text())
            aj = V.index(info['audit_verifier'])
            for o in cfg['grid']['orders']:
                meta, log = read_log(_path(out, gen, rot, o, arm, q, frac, eta))
                st = [x for x in log if x['p_hat'] is not None and all(signals[x['task_id']][v]['score'] is not None for v in V)]
                B = np.array([[bin_of(signals[x['task_id']][v]['score']) for v in V] for x in st])
                y = np.array([labels[x['task_id']] or 0 for x in st])
                j = np.arange(len(V))[None, :]
                g0, g1 = np.array(log[0]['theta_start']), np.array(meta['final_theta'])
                lab.append(float(np.log(g1[j, y[:, None], B]).sum(1).mean() - np.log(g0[j, y[:, None], B]).sum(1).mean()))
                gap0.append(float(codes @ (g0[aj, 1]-g0[aj, 0])))
                gap1.append(float(codes @ (g1[aj, 1]-g1[aj, 0])))
                ups = [x['theta_update'] for x in log if x['theta_update'] and 'objective_end' in x['theta_update']]
                if ups:
                    internal.append(ups[-1]['objective_end']/ups[-1]['records'] - ups[0]['objective_start']/ups[0]['records'])
                firsts = [x for x in log if x['theta_update'] and x['theta_update'].get('status') not in ('skipped_incomplete_panel',)]
                if firsts:
                    k = log.index(firsts[0])
                    nxt = np.array(log[k+1]['theta_start']) if k+1 < len(log) else g1
                    first_jump.append(float(np.abs(nxt - np.array(firsts[0]['theta_start'])).max()))
        res[arm] = {'labeled_loglik_change_per_task': float(np.mean(lab)), 'labeled_loglik_change_sd': float(np.std(lab)),
                    'internal_objective_change_per_record': float(np.mean(internal)) if internal else None,
                    'audit_gap_start': float(np.mean(gap0)), 'audit_gap_end': float(np.mean(gap1)),
                    'audit_gap_end_min': float(np.min(gap1)), 'first_update_max_abs_change': float(np.mean(first_jump)) if first_jump else None}
    return res


def conditional_dependence(cfg, items, signals, labels):
    V = cfg['verifiers']
    rows = [(labels[i.task_id], [bin_of(signals[i.task_id][v]['score']) for v in V]) for i in items
            if i.processable and labels[i.task_id] is not None and all(signals[i.task_id][v]['score'] is not None for v in V)]
    out = {}
    for y in (0, 1):
        X = np.array([b for yy, b in rows if yy == y], float)
        with np.errstate(invalid='ignore', divide='ignore'):
            C = np.corrcoef(X.T)
        out[str(y)] = {'n': len(X), 'mean_pairwise_corr': float(np.nanmean(C[np.triu_indices(len(V), 1)])),
                       'max_pairwise_corr': float(np.nanmax(C[np.triu_indices(len(V), 1)]))}
    return out


def eta_sensitivity(cfg, out, gen, signals, labels):
    es = cfg['grid']['eta_sensitivity']
    frac = cfg['settings']['budget_fraction']
    store = LabelStore(labels)
    res = {}
    for eta in [None, *es['eta0']]:
        entries = {}
        for arm in ('A', *es['arms']):
            entries[arm] = []
            for rot in cfg['grid']['rotations']:
                for o in cfg['grid']['orders']:
                    if arm == 'A':
                        p = _path(out, gen, rot, o, 'A', 0.0, frac)
                    else:
                        p = _path(out, gen, rot, o, arm, es['q'][0], frac, eta)
                    meta, log = read_log(p)
                    entries[arm].append(ev.group_sums(log, store, signals, cfg['verifiers'])[0])
        b = ev.bootstrap(entries, reps=cfg['bootstrap_reps'], seed=cfg['seeds']['bootstrap'])
        res[str(cfg['settings']['eta0'] if eta is None else eta)] = b
    return res


def curves_figure(curves, cfg, path):
    gens = cfg['generators']
    fig, axes = plt.subplots(2, len(gens), figsize=(4.2*len(gens), 6.4), sharex=True)
    for c, gen in enumerate(gens):
        for arm in MAIN:
            k = f'{gen}|q0.5|B{cfg["settings"]["budget_fraction"]:g}|{arm}'
            cv = curves[k]
            rel, wr = np.nan_to_num(np.array(cv['released'])), np.nan_to_num(np.array(cv['wrong']))
            t = np.arange(1, len(rel)+1)
            cr = np.cumsum(wr)/np.maximum(np.cumsum(rel), 1e-9)
            br = np.array(cv['brier_prior'])
            w = 25
            roll = np.convolve(np.nan_to_num(br), np.ones(w)/w, mode='valid')
            axes[0, c].plot(t, cr, color=COLORS[arm], lw=2, label=arm)
            axes[1, c].plot(np.arange(w, len(br)+1), roll, color=COLORS[arm], lw=2, label=arm)
        axes[0, c].axhline(cfg['settings']['alpha'], color='#8a8984', lw=1, ls='--')
        axes[0, c].set_title(gen, fontsize=11)
        axes[1, c].set_xlabel('stream position t')
        for ax in axes[:, c]:
            ax.grid(alpha=.25, lw=.6)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel('cumulative error rate among released')
    axes[1, 0].set_ylabel('Brier of p_hat (rolling 25)')
    axes[0, -1].legend(frameon=False, fontsize=9, loc='lower right')
    fig.suptitle('Learning curves, q = 0.5 (mean over 5 rotations x 5 orders; dashed: alpha = 0.05)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tradeoff_figure(summary, cfg, path):
    gens = cfg['generators']
    fig, axes = plt.subplots(1, len(gens), figsize=(4.2*len(gens), 3.8))
    markers = {0.5: 'o', 0.25: 's', 0.1: '^'}
    for c, gen in enumerate(gens):
        ax = axes[c]
        for q in cfg['grid']['q']:
            b = summary[f'{gen}|q{q:g}|B{cfg["settings"]["budget_fraction"]:g}']['bootstrap']['arms']
            for arm in MAIN:
                if arm == 'A' and q != 0.5:
                    continue
                x, y = b[arm]['pub']['point'], b[arm]['risk']['point']
                ax.scatter(x, y, s=46, marker=markers[q], color=COLORS[arm], edgecolor='white', linewidth=1.2, zorder=3,
                           label=f'{arm}' if q == 0.5 else None)
        ax.axhline(cfg['settings']['alpha'], color='#8a8984', lw=1, ls='--')
        ax.set_title(gen, fontsize=11)
        ax.set_xlabel('publication rate (stream)')
        ax.grid(alpha=.25, lw=.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('realized error among released')
    axes[-1].legend(frameon=False, fontsize=9, title='arm (o q=.5, s .25, ^ .1)', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    res_dir = Path(cfg['output'])
    summary = json.loads((res_dir/'summary.json').read_text())['by_setting']
    curves = json.loads((res_dir/'curves.json').read_text())
    timing = json.loads((res_dir/'timing.json').read_text()) if (res_dir/'timing.json').exists() else None
    frac = cfg['settings']['budget_fraction']
    key = {'theta_quality': {}, 'dependence': {}, 'eta_sensitivity': {}, 'init': {}}
    L = []
    for gen in cfg['generators']:
        items, signals, labels = load_generator(Path(cfg['data_dir']), gen)
        key['dependence'][gen] = conditional_dependence(cfg, items, signals, labels)
        key['theta_quality'][gen] = theta_quality(cfg, gen, signals, labels, res_dir, frac, 0.5)
        key['eta_sensitivity'][gen] = eta_sensitivity(cfg, res_dir, gen, signals, labels)
        key['init'][gen] = [json.loads((res_dir/'init'/f'{gen}_r{r}_B{frac:g}.json').read_text()) for r in cfg['grid']['rotations']]
    (a.out/'figures').mkdir(parents=True, exist_ok=True)
    curves_figure(curves, cfg, a.out/'figures'/'learning_curves_q0.5.png')
    tradeoff_figure(summary, cfg, a.out/'figures'/'risk_publication.png')
    # ---- tables ----
    L.append('# Online replay tables (generated by scripts/online_report_20261109.py)\n')
    L.append('Intervals: 95% cluster bootstrap over the 35 task groups, pooling 5 rotations x 5 orders.\n')
    for gen in cfg['generators']:
        L.append(f'\n## {gen}\n')
        L.append('\n### Initialization by rotation\n')
        L.append('| rotation | phi0 (a, b) | audit verifier | hierarchy | budget USD/task | init correct / panels |')
        L.append('|---|---|---|---|---|---|')
        for r, inf in enumerate(key['init'][gen]):
            L.append(f"| {r} | ({inf['phi0'][0]:.2f}, {inf['phi0'][1]:.2f}) | {inf['audit_verifier']} | {' > '.join(inf['hierarchy'])} | "
                     f"{inf['budget_usd']:.5f} | {inf['init_correct']} / {inf['init_complete_panels']} |")
        for q in cfg['grid']['q']:
            b = summary[f'{gen}|q{q:g}|B{frac:g}']
            L.append(f'\n### q = {q}\n')
            L.append('| arm | publication | realized risk | model-estimated risk | Brier p_hat | Brier full panel | decision USD/task (x1e-3) | audit USD/task (x1e-3) | released / wrong (summed over 25 runs) |')
            L.append('|---|---|---|---|---|---|---|---|---|')
            for arm in ALL:
                x = b['bootstrap']['arms'][arm]
                pr = b['per_rotation'][arm]
                est = np.nansum([pr[r]['est_risk']*pr[r]['released'] for r in pr])/max(sum(pr[r]['released'] for r in pr), 1)
                dec = np.mean([pr[r]['decision_usd'] for r in pr])*1e3
                aud = np.mean([pr[r]['audit_usd'] for r in pr])*1e3
                L.append(f"| {arm} | {ci(x['pub'])} | {ci(x['risk'])} | {est:.3f} | {ci(x['brier_prior'], fmt='{:.4f}')} | "
                         f"{ci(x['brier_panel'], fmt='{:.4f}')} | {dec:.3f} | {aud:.3f} | {int(x['counts']['released'])} / {int(x['counts']['wrong'])} |")
            L.append('\n| paired difference | publication | realized risk | Brier p_hat | Brier full panel |')
            L.append('|---|---|---|---|---|')
            for d in DIFFS:
                x = b['bootstrap']['diff'][d]
                L.append(f"| {d} | {ci(x['pub'])} | {ci(x['risk'])} | {ci(x['brier_prior'], fmt='{:+.4f}')} | {ci(x['brier_panel'], fmt='{:+.4f}')} |")
            if q == 0.5:
                L.append('\n| arm | realized risk by rotation 0-4 | publication by rotation 0-4 | risk sd over 25 runs |')
                L.append('|---|---|---|---|')
                for arm in ALL:
                    pr = b['per_rotation'][arm]
                    risks = ', '.join(f"{pr[r]['risk']:.3f}" for r in sorted(pr))
                    pubs = ', '.join(f"{pr[r]['pub']:.3f}" for r in sorted(pr))
                    L.append(f"| {arm} | {risks} | {pubs} | {b['order_spread'][arm]['risk']['sd_over_runs']:.3f} |")
                L.append('\n| arm | theta updates | EM iterations | E-step records | theta update s | redesign s | C1 history bytes | C2 stats bytes | max abs Z-hat |')
                L.append('|---|---|---|---|---|---|---|---|---|')
                for arm in ALL:
                    dg = b['diagnostics'][arm]
                    sb = dg.get('state_bytes_end_mean') or {'c1_history': 0, 'c2_stats': 0}
                    za = dg.get('zhat_abs_max')
                    L.append(f"| {arm} | {dg['theta_updates']:.1f} | {dg['em_iterations']:.0f} | {dg['records_processed']:.0f} | "
                             f"{dg['theta_seconds']:.3f} | {dg['redesign_seconds']:.2f} | {sb['c1_history']:.0f} | {sb['c2_stats']:.0f} | "
                             f"{'' if za is None else f'{za:.2f}'} |")
        L.append('\n### Likelihood: internal objective vs true fit (q = 0.5, mean over 25 runs)\n')
        L.append('| arm | C1 internal objective change / record | true class-conditional log-lik change / stream task | audit gap start -> end (min end) | first-update max abs change in theta |')
        L.append('|---|---|---|---|---|')
        for arm, v in key['theta_quality'][gen].items():
            io = v['internal_objective_change_per_record']
            fj = v['first_update_max_abs_change']
            L.append(f"| {arm} | {'' if io is None else f'{io:+.3f}'} | {v['labeled_loglik_change_per_task']:+.3f} (sd {v['labeled_loglik_change_sd']:.3f}) | "
                     f"{v['audit_gap_start']:.3f} -> {v['audit_gap_end']:.3f} ({v['audit_gap_end_min']:.3f}) | {'' if fj is None else f'{fj:.3f}'} |")
        dep = key['dependence'][gen]
        L.append(f"\nWithin-class correlation of verifier bins (all processable labeled tasks): Y=0 n={dep['0']['n']} mean {dep['0']['mean_pairwise_corr']:.2f} (max {dep['0']['max_pairwise_corr']:.2f}); "
                 f"Y=1 n={dep['1']['n']} mean {dep['1']['mean_pairwise_corr']:.2f} (max {dep['1']['max_pairwise_corr']:.2f}).\n")
        L.append('\n### eta0 sensitivity (q = 0.5; A is the shared static baseline)\n')
        L.append('| eta0 | arm | publication | realized risk | Brier p_hat | Brier p_hat minus A |')
        L.append('|---|---|---|---|---|---|')
        for eta, bb in key['eta_sensitivity'][gen].items():
            for arm in cfg['grid']['eta_sensitivity']['arms']:
                x = bb['arms'][arm]
                d = bb['diff'][f'{arm}-A']
                L.append(f"| {eta} | {arm} | {x['pub']['point']:.3f} | {ci(x['risk'])} | {x['brier_prior']['point']:.4f} | {ci(d['brier_prior'], fmt='{:+.4f}')} |")
    if timing:
        L.append('\n## Controlled timing pass (single process, warm-up discarded, q = 0.5, order 0, 15 runs per arm)\n')
        L.append(f"Environment: {json.dumps(timing['environment'])}\n")
        L.append('| arm | wall s / run | theta updates | EM iterations | E-step records | theta update s (total) | median s / update | max s / update | redesign s (total) |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for arm, rows in timing['runs'].items():
            m = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else float('nan')
            L.append(f"| {arm} | {m('wall'):.2f} | {m('theta_updates'):.1f} | {m('em_iterations'):.0f} | {m('records_processed'):.0f} | "
                     f"{m('theta_seconds'):.4f} | {m('theta_seconds_per_update_p50'):.5f} | {m('theta_seconds_per_update_max'):.5f} | {m('redesign_seconds'):.2f} |")
    (a.out/'tables.md').write_text('\n'.join(L) + '\n')
    (a.out/'key_results.json').write_text(json.dumps(key, indent=1, default=float))
    print('\n'.join(L[:60]))


if __name__ == '__main__':
    main()
