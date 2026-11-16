"""HTML report and CSV tables of the offline multi-risk comparison. Every number and count in the text is
computed here from the result files; nothing is hard-coded. Output: <report_dir>/results.html plus CSVs."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from vgx.common.storage import file_digest
from vgx.offline_multirisk import BASELINES, METHODS, REFERENCES, RISKS

DS = [('gpqa', 'GPQA'), ('math500', 'MATH-500'), ('qkatas', 'QuantumKatas')]
GEN = {'qwen3': 'Qwen3', 'gptoss': 'gpt-oss', 'sonnet55': 'Sonnet 5.5'}
RISK_NAME = {'selective_error': '放行错误率', 'fpr': 'FPR', 'fnr': 'FNR'}
ALPHA_DEF = {'selective_error': 'α 限制放行错误率：放行的答案里错的比例不超过 α（e ≤ α p）',
             'fpr': 'α 限制 FPR：所有错误答案里被放行的比例不超过 α（e ≤ α (1 − q)）',
             'fnr': 'α 限制 FNR：所有正确答案里被拦下的比例不超过 α（m ≤ α q）'}
RISK_DEF = {'selective_error': '放行的答案里错的比例', 'fpr': '所有错误答案里被放行的比例', 'fnr': '所有正确答案里被拦下的比例'}
GOAL = {'selective_error': '放行率（越高越好）', 'fpr': '放行率（越高越好）', 'fnr': '错误放行占全部题的比例 e（越低越好）'}
COLORS = {'Ours-Offline': 's1', 'AT-Frozen': 's2', 'AT+Cal-Frozen': 's3', 'Raw-threshold': 's4', 'Fixed-1-Mix': 's5', 'Fixed-3-Mix': 's6'}
# Dataset/generator pairs not evaluated for a risk; the report drops them everywhere (tables, charts, CSVs).
# FPR's denominator is the number of wrong answers, and MATH-500 has too few of them for every generator.
EXCLUDE_MAIN = {'fpr': {('math500', g) for g in ('qwen3', 'gptoss', 'sonnet55')}}


def in_main(ds, gen, risk):
    return (ds, gen) not in EXCLUDE_MAIN.get(risk, ())


def excluded(ds, gen, risk):
    return not in_main(ds, gen, risk)


FIELDS = ['dataset', 'generator', 'risk_metric', 'alpha_target', 'budget_fraction', 'method', 'n_tasks', 'n_eligible', 'n_correct',
          'n_wrong', 'n_released', 'n_wrong_released', 'n_correct_deferred', 'coverage', 'release_error_rate', 'fpr', 'fnr',
          'fpr_eligible_subset', 'joint_wrong_release_rate', 'correct_retention_rate', 'mean_screen_cost', 'budget_usd',
          'cost_excess_usd', 'cost_excess_relative', 'weak_verifier_calls_per_task', 'verifier_query_rate', 'risk', 'risk_excess',
          'risk_estimable', 'risk_met', 'budget_met', 'empirical_feasible', 'zero_release', 'fallback_used', 'fallback_share',
          'selection_status', 'fold_budget_exceeded', 'seed_risk_exceeded_share']


def pct(x, d=1):
    return '—' if x is None else f'{100*x:.{d}f}%'


def fmt(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'


def load(cfg, out_dir):
    res = {}
    for ds, _ in DS:
        for gen in GEN:
            p = Path(out_dir)/f'{ds}_{gen}.json'
            if p.exists():
                res[(ds, gen)] = json.loads(p.read_text())
    return res


def rows_of(res):
    out = []
    for (ds, gen), r in res.items():
        for key, cell in r['cells'].items():
            risk, a, f = key.split('|')
            for m, x in cell.items():
                out.append({'dataset': ds, 'generator': gen, 'risk_metric': risk, 'alpha_target': float(a), 'budget_fraction': float(f),
                            'method': m, **x})
    return out


def index(rows):
    return {(r['dataset'], r['generator'], r['risk_metric'], r['alpha_target'], r['budget_fraction'], r['method']): r for r in rows}


def write_csv(rows, path):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([json.dumps(r.get(k)) if isinstance(r.get(k), (dict, list)) else r.get(k) for k in FIELDS])


# ----------------------------------------------------------------------------- comparisons

def pair_status(o, b, risk):
    """both / ours / baseline / neither / not_evaluable for one condition."""
    if not o['risk_estimable'] and not b['risk_estimable'] and risk != 'selective_error':
        return 'not_evaluable'
    if risk != 'selective_error' and (o['n_wrong' if risk == 'fpr' else 'n_correct'] <= 0):
        return 'not_evaluable'
    fo, fb = o['empirical_feasible'], b['empirical_feasible']
    return 'both' if fo and fb else 'ours' if fo else 'baseline' if fb else 'neither'


STATUS_ORDER = [('both', '双方达标'), ('ours', '仅我们达标'), ('baseline', '仅对照达标'), ('neither', '双方未达标'), ('not_evaluable', '不可评价')]


def feasibility(I, rows, cfg, risk, exclude_fallback=False):
    out = {b: {k: 0 for k, _ in STATUS_ORDER} for b in BASELINES}
    excluded = {b: 0 for b in BASELINES}
    for r in rows:
        if r['risk_metric'] != risk or r['method'] != 'Ours-Offline':
            continue
        for b in BASELINES:
            x = I[(r['dataset'], r['generator'], risk, r['alpha_target'], r['budget_fraction'], b)]
            if exclude_fallback and (r['fallback_used'] or x['fallback_used']):
                excluded[b] += 1
                continue
            out[b][pair_status(r, x, risk)] += 1
    return out, excluded


def both_feasible_perf(I, rows, risk):
    out = {}
    for b in BASELINES:
        d, pos, neg, n = [], 0, 0, 0
        per_ds = {}
        for r in rows:
            if r['risk_metric'] != risk or r['method'] != 'Ours-Offline':
                continue
            x = I[(r['dataset'], r['generator'], risk, r['alpha_target'], r['budget_fraction'], b)]
            if pair_status(r, x, risk) != 'both':
                continue
            p = r['paired'][b]
            n += 1
            d.append(p['point'])
            lo, hi = p['ci95']
            good = (lo > 0) if risk != 'fnr' else (hi < 0)
            bad = (hi < 0) if risk != 'fnr' else (lo > 0)
            pos += good; neg += bad
            s = per_ds.setdefault(r['dataset'], [0, 0, 0])
            s[0] += 1; s[1] += good; s[2] += bad
        out[b] = {'n': n, 'median': float(np.median(d)) if d else None, 'mean': float(np.mean(d)) if d else None,
                  'ours_better_ci': pos, 'baseline_better_ci': neg, 'per_dataset': per_ds}
    return out


# ----------------------------------------------------------------------------- charts

def panel(title, series, xs, xticks, yr, yticks, yfmt, diag=None, hline=None):
    W, H, L, R_, T, Bm = 300, 210, 46, 10, 10, 28
    pw, ph = W-L-R_, H-T-Bm
    x0, x1 = xs; y0, y1 = yr
    sx = lambda v: L+(min(max(v, x0), x1)-x0)/(x1-x0 or 1)*pw
    sy = lambda v: T+ph-(min(max(v, y0), y1)-y0)/(y1-y0 or 1)*ph
    g = []
    for v in yticks:
        g.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/><text class="tk" x="{L-5}" y="{sy(v)+4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    for v, lab in xticks:
        g.append(f'<text class="tk" x="{sx(v):.1f}" y="{T+ph+16}" text-anchor="middle">{lab}</text>')
    if diag:
        g.append(f'<polyline class="ideal" points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in diag)}" fill="none"/>')
    if hline is not None:
        g.append(f'<line class="ideal" x1="{L}" x2="{L+pw}" y1="{sy(hline):.1f}" y2="{sy(hline):.1f}"/>')
    for m in reversed(METHODS):
        pts = series.get(m) or []
        if not pts:
            continue
        col = f'var(--{COLORS[m]})'
        main = m == 'Ours-Offline'
        g.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, *_ in pts)}" fill="none" stroke="{col}" stroke-width="{2.6 if main else 1.6}" stroke-linejoin="round"/>')
        for x, y, tip, ok in pts:
            fill = col if ok else 'var(--surface)'
            g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{4.5 if main else 3.6}" fill="{fill}" stroke="{col}" stroke-width="1.6"><title>{html.escape(m+"："+tip)}</title></circle>')
    return f'<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{"".join(g)}</svg></figure>'


def tip(x, risk):
    s = f'放行 {pct(x["coverage"])}，{RISK_NAME[risk]} {pct(x["risk"])}，每题花费 {x["mean_screen_cost"]*1e4:.1f}×10⁻⁴ 美元'
    if not x['empirical_feasible']:
        s += '（未达标：' + ('风险超标' if not x['risk_met'] else '') + ('、' if not x['risk_met'] and not x['budget_met'] else '') + \
             ('预算超标' if not x['budget_met'] else '') + ('无放行' if x['risk_met'] and x['budget_met'] else '') + '）'
    return s


def objective(x, risk):
    return x['coverage'] if risk != 'fnr' else x['joint_wrong_release_rate']


def chart_alpha(I, cfg, risk, frac):
    alphas = cfg['alpha_target'][risk]
    out = ''
    for ds, dname in DS:
        figs = ''
        ys = [objective(I[(ds, g, risk, a, frac, m)], risk) for g in GEN for a in alphas for m in METHODS if (ds, g, risk, a, frac, m) in I]
        ymax = 1.0 if risk != 'fnr' else max(.05, np.ceil(max(ys+[0])*20)/20)
        for gen in GEN:
            if not in_main(ds, gen, risk):
                continue
            series = {}
            for m in METHODS:
                series[m] = [(i, objective(x, risk), f'α = {a:g}：'+tip(x, risk), x['empirical_feasible'])
                             for i, a in enumerate(alphas) for x in [I.get((ds, gen, risk, a, frac, m))] if x]
            yt = (0, .5, 1) if risk != 'fnr' else (0, ymax/2, ymax)
            figs += panel(f'{dname} · {GEN[gen]}', series, (-.3, len(alphas)-.7), [(i, f'{a:g}') for i, a in enumerate(alphas)], (0, ymax), yt, lambda v: f'{v:.0%}' if risk != 'fnr' else f'{v:.1%}')
        out += f'<div class="row">{figs}</div>'
    return out


def chart_budget(I, cfg, risk):
    a = cfg['alpha_main'][risk]
    fr = cfg['budget_fractions']
    out = ''
    for ds, dname in DS:
        figs = ''
        ys = [objective(I[(ds, g, risk, a, f, m)], risk) for g in GEN for f in fr for m in METHODS if (ds, g, risk, a, f, m) in I]
        ymax = 1.0 if risk != 'fnr' else max(.05, np.ceil(max(ys+[0])*20)/20)
        for gen in GEN:
            if not in_main(ds, gen, risk):
                continue
            series = {m: [(f, objective(x, risk), f'预算 {f:g}×c_ref：'+tip(x, risk), x['empirical_feasible'])
                          for f in fr for x in [I.get((ds, gen, risk, a, f, m))] if x] for m in METHODS}
            yt = (0, .5, 1) if risk != 'fnr' else (0, ymax/2, ymax)
            figs += panel(f'{dname} · {GEN[gen]}', series, (0, 1), [(f, f'{f:g}') for f in (0, .25, .5, .75, 1)], (0, ymax), yt, lambda v: f'{v:.0%}' if risk != 'fnr' else f'{v:.1%}')
        out += f'<div class="row">{figs}</div>'
    return out


def spaced(vals, span, gap=.08):
    """Tick labels at the given values, skipping any that would sit too close to the previous one."""
    out = []
    for v in sorted(vals, reverse=True):
        if not out or out[-1][0]-v >= gap*span:
            out.append((v, f'{v:g}'))
    return out[::-1]


def chart_risk(I, cfg, risk, frac):
    alphas = cfg['alpha_target'][risk]
    top = max(alphas)*1.6
    out = ''
    for ds, dname in DS:
        figs = ''
        for gen in GEN:
            if not in_main(ds, gen, risk):
                continue
            series = {m: [(a, x['risk'], f'α = {a:g}：'+tip(x, risk), x['empirical_feasible'])
                          for a in alphas for x in [I.get((ds, gen, risk, a, frac, m))] if x and x['risk'] is not None] for m in METHODS}
            figs += panel(f'{dname} · {GEN[gen]}', series, (0, max(alphas)*1.05), spaced(alphas, max(alphas)*1.05), (0, top),
                          (0, max(alphas)/2, max(alphas), top), lambda v: f'{v:.0%}', diag=[(0, 0), (max(alphas)*1.05, max(alphas)*1.05)])
        out += f'<div class="row">{figs}</div>'
    return out


def chart_cost(I, cfg, risk):
    a = cfg['alpha_main'][risk]
    fr = cfg['budget_fractions']
    out = ''
    for ds, dname in DS:
        figs = ''
        for gen in GEN:
            if not in_main(ds, gen, risk):
                continue
            bmax = max(I[(ds, gen, risk, a, f, 'Ours-Offline')]['budget_usd'] for f in fr)*1e4
            cmax = max([I[(ds, gen, risk, a, f, m)]['mean_screen_cost']*1e4 for f in fr for m in METHODS]+[bmax])*1.05
            series = {m: [(I[(ds, gen, risk, a, f, m)]['budget_usd']*1e4, I[(ds, gen, risk, a, f, m)]['mean_screen_cost']*1e4,
                           f'预算 {f:g}×c_ref：'+tip(I[(ds, gen, risk, a, f, m)], risk), I[(ds, gen, risk, a, f, m)]['empirical_feasible'])
                          for f in fr] for m in METHODS}
            step = next(v for v in (5, 10, 20, 50, 100) if cmax/v <= 5)
            ticks = [v for v in np.arange(0, cmax, step)]
            figs += panel(f'{dname} · {GEN[gen]}', series, (0, bmax*1.05), [(v, f'{v:g}') for v in np.arange(0, bmax*1.05, step)],
                          (0, cmax), ticks, lambda v: f'{v:g}', diag=[(0, 0), (bmax*1.05, bmax*1.05)])
        out += f'<div class="row">{figs}</div>'
    return out


# ----------------------------------------------------------------------------- tables

def table(head, body, num_from=1):
    th = ''.join(f'<th{" class=n" if i >= num_from else ""}>{h}</th>' for i, h in enumerate(head))
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def tr(cells, num_from=1):
    return '<tr>'+''.join(f'<{"th scope=row" if i == 0 else "td"}{" class=n" if i >= num_from else ""}>{c}</{"th" if i == 0 else "td"}>' for i, c in enumerate(cells))+'</tr>'


def feas_table(I, rows, cfg, risk):
    f, _ = feasibility(I, rows, cfg, risk)
    fx, ex = feasibility(I, rows, cfg, risk, exclude_fallback=True)
    body = ''.join(tr([b]+[f[b][k] for k, _ in STATUS_ORDER]+[f'{fx[b]["both"]} / {fx[b]["ours"]} / {fx[b]["baseline"]} / {fx[b]["neither"]}（去掉 {ex[b]}）']) for b in BASELINES)
    return table(['对照方法']+[n for _, n in STATUS_ORDER]+['去掉用了回退的条件后：双方 / 仅我们 / 仅对照 / 都没'], body)


def method_feasible_table(I, cfg, risk):
    body = ''
    tot = len(cfg['alpha_target'][risk])*len(cfg['budget_fractions'])
    for ds, dname in DS:
        for gen in GEN:
            if not in_main(ds, gen, risk):
                continue
            cells = [f'{dname} · {GEN[gen]}']
            for m in METHODS:
                xs = [I[(ds, gen, risk, a, f, m)] for a in cfg['alpha_target'][risk] for f in cfg['budget_fractions']]
                cells.append(f'{sum(x["empirical_feasible"] for x in xs)} / {tot}')
            body += tr(cells)
    return table(['数据集 · generator']+list(METHODS), body)


def perf_table(I, rows, risk):
    P = both_feasible_perf(I, rows, risk)
    unit = '个百分点' if risk != 'fnr' else '个百分点（e）'
    body = ''.join(tr([b, P[b]['n'], '—' if P[b]['median'] is None else f'{100*P[b]["median"]:+.1f}',
                       P[b]['ours_better_ci'], P[b]['baseline_better_ci'],
                       ' · '.join(f'{dict(DS)[d]} {v[0]}（{v[1]}/{v[2]}）' for d, v in P[b]['per_dataset'].items()) or '—']) for b in BASELINES)
    head = ['对照方法', '双方达标的条件数', f'差值中位数（我们减对照，{unit}）', '区间显示我们更好', '区间显示对照更好', '按数据集：条件数（我们更好/对照更好）']
    return table(head, body)


def main_alpha_table(I, cfg, risk, keep=in_main):
    a = cfg['alpha_main'][risk]
    body = ''
    for ds, dname in DS:
        for gen in GEN:
            if not keep(ds, gen, risk):
                continue
            for f in cfg['budget_fractions_shown']:
                cells = [f'{dname} · {GEN[gen]} · B = {f:g}×c_ref']
                for m in METHODS:
                    x = I[(ds, gen, risk, a, f, m)]
                    mark = '' if x['empirical_feasible'] else ' ✕'
                    cells.append(f'{pct(objective(x, risk))} / {pct(x["risk"])}{mark}')
                body += tr(cells)
    return table(['条件']+list(METHODS), body)


def diagnostics(res, I, cfg):
    # class denominators per fold
    body = ''
    for ds, dname in DS:
        for gen in GEN:
            r = res[(ds, gen)]
            vw = [f['validation_wrong'] for f in r['folds']]; vc = [f['validation_correct'] for f in r['folds']]
            tw = [f['test_wrong'] for f in r['folds']]; tc = [f['test_correct'] for f in r['folds']]
            el = sum(f['test_eligible'] for f in r['folds'])/r['n_test']
            body += tr([f'{dname} · {GEN[gen]}', f'{min(vw)}–{max(vw)}', f'{min(vc)}–{max(vc)}', f'{min(tw)}–{max(tw)}', f'{min(tc)}–{max(tc)}', pct(el)])
    denom = table(['数据集 · generator', '验证：错误数（每折）', '验证：正确数（每折）', '测试：错误数（每折）', '测试：正确数（每折）', '测试：有答案的比例'], body)
    # selection status and fallback share per method and risk
    body = ''
    for risk in RISKS:
        for m in METHODS:
            xs = [x for k, x in I.items() if k[2] == risk and k[5] == m]
            st = {}
            for x in xs:
                for s, c in x['selection_status'].items():
                    st[s] = st.get(s, 0)+c
            tot = sum(st.values())
            body += tr([f'{RISK_NAME[risk]} · {m}', tot, st.get('selected', 0), st.get('no_feasible_candidate', 0), st.get('risk_not_estimable', 0),
                        pct(np.mean([x['fallback_share'] for x in xs]))])
    sel = table(['风险 · 方法', '选择次数（折×种子）', '正常选中', '无可行候选', '风险不可估计', '用了回退的比例'], body)
    # duplicate working points: adjacent conditions choosing the same config
    body = ''
    for risk in RISKS:
        for m in ('Ours-Offline', 'Fixed-1-Mix', 'Fixed-3-Mix', 'Raw-threshold'):
            keys = [k for k in I if k[2] == risk and k[5] == m]
            ids = [json.dumps(I[k]['config_ids']) for k in keys]
            body += tr([f'{RISK_NAME[risk]} · {m}', len(ids), len(set(ids))])
    dup = table(['风险 · 方法', '条件数', '不同的冻结策略数'], body)
    # seed stability of AT
    body = ''
    for risk in RISKS:
        for m in ('AT-Frozen', 'AT+Cal-Frozen'):
            xs = [x for k, x in I.items() if k[2] == risk and k[5] == m]
            pooled_ex = sum(not x['risk_met'] for x in xs)
            maj = sum((x.get('seed_risk_exceeded_share') or 0) > .5 for x in xs)
            sd = np.mean([np.std([c for c in x['per_seed']['coverage']]) for x in xs if 'per_seed' in x])
            body += tr([f'{RISK_NAME[risk]} · {m}', len(xs), pooled_ex, maj, pct(sd, 2)])
    seed = table(['风险 · 方法', '条件数', '种子平均后风险超标', '多数种子风险超标', '放行率在种子间的标准差（平均）'], body)
    # mixture draws vs expectation
    body = ''
    for risk in RISKS:
        for m in ('Ours-Offline', 'Fixed-1-Mix', 'Fixed-3-Mix'):
            xs = [x for k, x in I.items() if k[2] == risk and k[5] == m and 'sampled_draws' in x]
            gap = [abs(np.mean(x['sampled_draws']['coverage'])-x['coverage']) for x in xs]
            flip = [np.mean([f != x['empirical_feasible'] for f in x['sampled_draws']['empirical_feasible']]) for x in xs]
            body += tr([f'{RISK_NAME[risk]} · {m}', len(xs), pct(np.mean(gap), 2), pct(np.mean(flip))])
    draws = table(['风险 · 方法', '条件数', '20 次抽签平均放行率与期望的平均差', '单次抽签的达标结论与期望不同的比例'], body)
    return denom, sel, dup, seed, draws


def brier_table(res):
    body = ''
    for ds, dname in DS:
        for gen in GEN:
            b = res[(ds, gen)]['brier']
            ac = b['at_cal'].get('selective_error', {})
            body += tr([f'{dname} · {GEN[gen]}', b['raw']['n'], fmt(b['raw']['brier']), fmt(b['ours_prior']['brier']),
                        fmt(ac.get('brier_selected_task_seeds')), fmt(ac.get('raw_on_same_pairs')), fmt(ac.get('ours_prior_on_same_pairs')),
                        f'{ac.get("status_counts", {}).get("selected", 0)}/{sum(ac.get("status_counts", {}).values())}'])
    return table(['数据集 · generator', '有答案的测试题', '原始 p', '我们的校准先验', 'AT+Cal 选中校准（题×种子）', '同一批题×种子：原始 p', '同一批：我们的先验', 'AT+Cal 有选中的折×种子'], body)


def threshold_example(b, n_inel, risk, alpha):
    """Lowest calibrated p that still gets released when the calibrated p is read as the chance the answer is right,
    no verifier is asked and questions are released from the highest p down: (threshold, share of all questions released)."""
    b = np.sort(np.asarray(b, float))[::-1]
    n = len(b)+n_inel
    if risk == 'fnr':
        q = b.sum()
        blocked = np.cumsum(b[::-1])  # block from the lowest p up; questions with no answer are blocked anyway
        j = int(np.sum(blocked <= alpha*q+1e-12))
        k = len(b)-j
    else:
        wrong = np.cumsum(1-b)
        cap = alpha*np.arange(1, len(b)+1) if risk == 'selective_error' else alpha*(float((1-b).sum())+n_inel)
        ok = np.nonzero(wrong <= cap+1e-12)[0]
        k = int(ok[-1])+1 if len(ok) else 0
    return (float(b[k-1]) if k else None), k/n


def alpha_examples(res, cfg):
    """Rows: dataset x generator; columns: each risk at its main alpha."""
    body = ''
    for ds, dname in DS:
        for g in GEN:
            fo = res[(ds, g)]['folds']
            b = [x for f in fo for x in f['test_prior_eligible']]
            n_inel = sum(f['n_test']-f['test_eligible'] for f in fo)
            cells = [f'{dname} · {GEN[g]}']
            for risk in RISKS:
                a = cfg['alpha_main'][risk]
                if excluded(ds, g, risk):
                    cells.append('不测'); continue
                t, sh = threshold_example(b, n_inel, risk, a)
                if t is None:
                    cells.append('一道都不放行')
                elif sh*(len(b)+n_inel) >= len(b):
                    cells.append(f'有答案的全放行（{100*sh:.0f}%）')
                else:
                    cells.append(f'p ≥ {t:.2f}，放行 {100*sh:.0f}%')
            body += tr(cells, num_from=9)
    head = ['数据集 · generator']+[f'{RISK_NAME[r]} α = {cfg["alpha_main"][r]:g}' for r in RISKS]
    return table(head, body, num_from=9)


def input_compare(I_cal, I_raw, cfg):
    """Each method under the two inputs: mean objective and overrun count over the same conditions."""
    body = ''
    for risk in RISKS:
        for m in METHODS:
            ks = [k for k in I_cal if k[2] == risk and k[5] == m and k in I_raw]
            oc = np.mean([objective(I_cal[k], risk) for k in ks]); orw = np.mean([objective(I_raw[k], risk) for k in ks])
            name = 'Cal-threshold / Raw-threshold' if m == 'Raw-threshold' else m
            body += tr([f'{RISK_NAME[risk]} · {name}', len(ks), pct(oc), pct(orw), sum(over(I_cal[k]) for k in ks), sum(over(I_raw[k]) for k in ks)])
    return table(['风险 · 方法', '条件数', '校准 p：平均放行率（FNR 为平均 e）', '原始 p：同左', '校准 p：风险超标条件数', '原始 p：风险超标条件数'], body)


def over(x):
    """Risk measured and above the target (zero release is counted separately, not as an overrun)."""
    return x['risk_excess'] is not None and x['risk_excess'] > 1e-12


def sig_counts(I, rows, risk, keep=in_main):
    """Previous version's comparison: over all conditions, paired-interval counts on the main objective."""
    out = {}
    for b in BASELINES:
        c = {'n': 0, 'ours': 0, 'base': 0, 'per': {}}
        for r in rows:
            if r['risk_metric'] != risk or r['method'] != 'Ours-Offline' or not keep(r['dataset'], r['generator'], risk):
                continue
            lo, hi = r['paired'][b]['ci95']
            good = (lo > 0) if risk != 'fnr' else (hi < 0)
            bad = (hi < 0) if risk != 'fnr' else (lo > 0)
            c['n'] += 1; c['ours'] += good; c['base'] += bad
            s = c['per'].setdefault(r['dataset'], [0, 0, 0]); s[0] += 1; s[1] += good; s[2] += bad
        out[b] = c
    return out


def sig_table(I, rows, risk, keep=in_main):
    S = sig_counts(I, rows, risk, keep)
    word = '放行显著更多' if risk != 'fnr' else '错误放行显著更少'
    body = ''.join(tr([b, S[b]['n'], S[b]['ours'], S[b]['base'], S[b]['n']-S[b]['ours']-S[b]['base'],
                       ' · '.join(f'{dict(DS)[d]} {v[1]}/{v[2]}（共 {v[0]}）' for d, v in S[b]['per'].items())]) for b in BASELINES)
    return table(['对照方法', '条件数', f'我们{word}', f'对照{word}', '区间含 0', '按数据集：我们/对照'], body)


def exceed_table(I, cfg, risk, keep=in_main):
    body = ''
    keys = [k for k in I if k[2] == risk and keep(k[0], k[1], risk)]
    for m in METHODS:
        xs = [I[k] for k in keys if k[5] == m]
        per = ' · '.join(f'{dname} {sum(over(I[k]) for k in keys if k[5] == m and k[0] == ds)}' for ds, dname in DS if any(k[0] == ds for k in keys))
        body += tr([m, len(xs), sum(over(x) for x in xs), sum(not x['budget_met'] for x in xs),
                    sum(x['zero_release'] for x in xs), per])
    return table(['方法', '条件数', f'测试集{RISK_NAME[risk]}超标', '预算超标', '零放行', '按数据集：风险超标'], body)


def headline(I, rows, cfg):
    out = []
    for risk in RISKS:
        S = sig_counts(I, rows, risk)
        n = S[BASELINES[0]]['n']
        word = '放行显著更多' if risk != 'fnr' else '错误放行显著更少'
        parts = '；'.join(f'对 {b} {S[b]["ours"]}:{S[b]["base"]}' for b in BASELINES)
        ex = {m: sum(over(I[k]) for k in I if k[2] == risk and k[5] == m and in_main(k[0], k[1], risk)) for m in METHODS}
        exs = '，'.join(f'{m} {ex[m]}' for m in METHODS)
        note = '，不含错误答案太少的 MATH-500' if EXCLUDE_MAIN.get(risk) else ''
        out.append(f'<li><b>{RISK_NAME[risk]}</b>（{n} 个条件{note}）：我们{word}的条件数比对照{word}的条件数为 {parts}。'
                   f'测试集{RISK_NAME[risk]}超标的条件数：{exs}。</li>')
    return '<ul>'+''.join(out)+'</ul>'


def exclusion_note(res, risk):
    pairs = sorted(EXCLUDE_MAIN.get(risk, ()), key=lambda p: (p[0], list(GEN).index(p[1])))
    if not pairs:
        return ''
    ds = pairs[0][0]
    counts = '、'.join(f'{GEN[g]} {sum(f["test_wrong"] for f in res[(d, g)]["folds"])} 道' for d, g in pairs)
    return (f'<p class="meta">FPR 不测 {dict(DS)[ds]}：FPR 的分母是错误答案的数量，而 {dict(DS)[ds]} 的错误答案太少'
            f'（{len(res[pairs[0]]["folds"])} 折测试合计：{counts}，没给答案的也算在内），一两道题就能让 FPR 变化很多。</p>')


def report(cfg, out_dir, report_dir):
    res = load(cfg, out_dir)
    missing = [f'{ds}:{g}' for ds, _ in DS for g in GEN if (ds, g) not in res]
    if missing:
        raise SystemExit(f'missing results: {missing}')
    rows = [r for r in rows_of(res) if in_main(r['dataset'], r['generator'], r['risk_metric'])]
    I = index(rows)
    report_dir = Path(report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, report_dir/'all_conditions.csv')
    write_csv(rows, Path(out_dir)/'summary.csv')
    tpl = Path(__file__).with_name('offline_multirisk_template.html').read_text()
    mode = cfg.get('confidence_input', 'mixed')
    ren = (lambda h: h.replace('Raw-threshold', 'Cal-threshold')) if mode == 'calibrated' else (lambda h: h)
    raw_dir = Path(str(out_dir).replace('_calibrated', '_raw'))
    rawp = ''
    if mode == 'calibrated' and raw_dir != Path(out_dir) and all((raw_dir/f'{ds}_{g}.json').exists() for ds, _ in DS for g in GEN):
        res_r = load(cfg, raw_dir)
        rows_r = [r for r in rows_of(res_r) if in_main(r['dataset'], r['generator'], r['risk_metric'])]
        I_r = index(rows_r)
        rawp = ('<section id="rawp">\n  <h2>5. 对照：所有方法都用原始 p_correct</h2>\n'
                '  <p>同样的数据、划分、α 和预算，只是所有方法都读原始 p_correct：Ours 和 Fixed 不做 isotonic 校准，直接把原始 p 当先验'
                '（verifier 的 likelihood 不变）；AT、AT+Cal 和阈值方法也用原始 p。下表先把每个方法两种输入的结果放在一起，再给出原始 p 下的比较。</p>\n'
                '  <h3>同一方法：校准 p 和原始 p</h3>\n  ' + input_compare(I, I_r, cfg))
        for k, risk in enumerate(RISKS, 1):
            rawp += (f'\n  <h3>5.{k} {RISK_NAME[risk]}：我们和每个对照的比较（原始 p，{sig_counts(I_r, rows_r, risk)[BASELINES[0]]["n"]} 个条件）</h3>\n  '
                     + sig_table(I_r, rows_r, risk) + f'\n  <h3>5.{k} {RISK_NAME[risk]}：超标情况（原始 p）</h3>\n  ' + exceed_table(I_r, cfg, risk))
        rawp += '\n</section>'
    denom, sel, dup, seed, draws = diagnostics(res, I, cfg)
    verify = json.loads((Path(out_dir)/'verify.json').read_text()) if (Path(out_dir)/'verify.json').exists() else None
    pre = json.loads((Path(out_dir)/'preflight.json').read_text()) if (Path(out_dir)/'preflight.json').exists() else {}
    n_cond = len({(r['dataset'], r['generator'], r['risk_metric'], r['alpha_target'], r['budget_fraction']) for r in rows})
    sec = {}
    for risk in RISKS:
        sec[risk] = {'SIG': sig_table(I, rows, risk), 'EXCEED': exceed_table(I, cfg, risk), 'FEAS': feas_table(I, rows, cfg, risk), 'METHOD': method_feasible_table(I, cfg, risk), 'PERF': perf_table(I, rows, risk),
                     'MAIN': main_alpha_table(I, cfg, risk),
                     'ALPHA': ''.join(f'<div class="bud" data-b="{f:g}"{"" if f == 1 else " hidden"}>{chart_alpha(I, cfg, risk, f)}</div>' for f in cfg['budget_fractions_shown']),
                     'RISK': ''.join(f'<div class="bud" data-b="{f:g}"{"" if f == 1 else " hidden"}>{chart_risk(I, cfg, risk, f)}</div>' for f in cfg['budget_fractions_shown']),
                     'COST': chart_cost(I, cfg, risk)}
    legend = '<div class="legend">'+''.join(f'<span><i style="background:var(--{COLORS[m]})"></i>{m}</span>' for m in METHODS)+'<span>实心点 = 风险和预算都达标，空心点 = 未达标</span></div>'
    blocks = ''
    for k, risk in enumerate(RISKS, 1):
        s = sec[risk]
        btn = ''.join(f'<button type="button" data-b="{f:g}"{" aria-pressed=true" if f == 1 else " aria-pressed=false"}>{f:g}×c_ref</button>' for f in cfg['budget_fractions_shown'])
        blocks += f'''
<section id="risk-{risk}">
  <h2>4.{k} {RISK_NAME[risk]}约束（{RISK_DEF[risk]}）</h2>
  <p>{ALPHA_DEF[risk]}，在测试集上检查。α 取 {", ".join(f"{a:g}" for a in cfg["alpha_target"][risk])}；主要展示 α = {cfg["alpha_main"][risk]:g}。目标：{GOAL[risk]}。</p>
  {exclusion_note(res, risk)}
  <h3>我们和每个对照的比较（{sig_counts(I, rows, risk)[BASELINES[0]]["n"]} 个条件，配对 bootstrap 区间）</h3>
  {s["SIG"]}
  <h3>超标情况（单独列出，不影响上表）</h3>
  {s["EXCEED"]}
  <h3>α = {cfg["alpha_main"][risk]:g} 时的结果（{GOAL[risk].split("（")[0]} / 实际{RISK_NAME[risk]}，✕ 表示风险或预算未达标）</h3>
  {s["MAIN"]}
  <h3>α 和表现（固定预算）</h3>
  <p class="meta">横轴：目标 α；纵轴：{'放行率' if risk != 'fnr' else '错误放行 e（占全部题的比例）'}。按钮切换预算。</p>
  {legend}
  <div class="seg" role="group" aria-label="预算">{btn}</div>
  {s["ALPHA"]}
  <h3>实际{RISK_NAME[risk]}和目标 α</h3>
  <p class="meta">横轴：目标 α；纵轴：测试集上实际的 {RISK_NAME[risk]}。虚线是实际 = 目标，点在虚线上方就是超标。预算跟上面的按钮一致。</p>
  {s["RISK"]}
  <h3>实际花费和预算（α = {cfg["alpha_main"][risk]:g}）</h3>
  <p class="meta">横轴：预算；纵轴：测试集上实际的每题平均花费；单位都是 10⁻⁴ 美元/题。虚线是花费 = 预算，点在虚线上方就是超预算。</p>
  {s["COST"]}
</section>'''
    appendix = ''
    for k, risk in enumerate(RISKS, 1):
        s = sec[risk]
        appendix += f'''
  <h3>A.{k} {RISK_NAME[risk]}：经验可行状态</h3>
  {s["FEAS"]}
  <h3>A.{k} {RISK_NAME[risk]}：各方法同时达标的条件数</h3>
  {s["METHOD"]}
  <h3>A.{k} {RISK_NAME[risk]}：双方都达标时的差值</h3>
  {s["PERF"]}'''
    cref = {ds: [f['c_ref_usd']*1e4 for g in GEN for f in res[(ds, g)]['folds']] for ds, _ in DS}
    vtxt = ('没有找到重建结果' if verify is None else
            f'从逐题日志独立重算了 {verify["checked_cells"]} 个条件×方法、{verify["checked_fields"]} 个字段，'
            f'{"全部一致" if verify["ok"] else "有不一致：" + "；".join(verify["mismatches"][:5])}；'
            f'混合策略逐题核对 {verify["mixture_items_checked"]} 次，不一致 {verify["mixture_mismatches"]} 次；'
            f'AT 种子平均核对 {verify["seed_mean_items_checked"]} 题次，不一致 {verify["seed_mean_mismatches"]} 次；决策日志里没有标签列。')
    net = sum(len(r.get('network_attempts', [])) for r in res.values())
    fill = {'{{HEADLINE}}': ren(headline(I, rows, cfg)), '{{LEGEND}}': '', '{{BLOCKS}}': ren(blocks), '{{APPENDIX}}': ren(appendix), '{{DENOM}}': denom, '{{SEL}}': ren(sel),
            '{{DUP}}': ren(dup), '{{SEED}}': seed, '{{DRAWS}}': draws, '{{RAWP}}': rawp, '{{EXAMPLES}}': alpha_examples(res, cfg), '{{BRIER}}': brier_table(res), '{{NCOND}}': str(n_cond),
            '{{NROWS}}': str(len(rows)), '{{VERIFY}}': vtxt, '{{NET}}': f'{net} 次网络请求尝试（全部被拦截）' if net else '没有任何网络请求尝试',
            '{{CREF}}': '；'.join(f'{dict(DS)[ds]} {min(v):.0f} 到 {max(v):.0f}' for ds, v in cref.items()),
            '{{CODE}}': next(iter(res.values()))['code_hash'], '{{PRE}}': '通过' if not pre.get('problems') else '；'.join(pre['problems'][:5])}
    for k, v in fill.items():
        tpl = tpl.replace(k, v)
    (report_dir/'results.html').write_text(tpl)
    print('wrote', report_dir/'results.html', file_digest(report_dir/'results.html'))
