"""Tables and page for the offline frozen comparison: results/offline_at_frozen_20261111/summary.csv and
reports/offline_at_frozen_20261111/results.html."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

SRC = Path('results/offline_at_frozen_20261111')
OUT = Path('reports/offline_at_frozen_20261111/results.html')
DS = [('gpqa', 'GPQA'), ('math500', 'MATH-500'), ('qkatas', 'QuantumKatas')]
GEN = {'qwen3': 'Qwen3', 'gptoss': 'gpt-oss', 'sonnet55': 'Sonnet 5.5'}
MAIN = ['AT-Frozen', 'AT+Cal-Frozen', 'Ours-Offline']
SUPP = ['Raw-threshold', 'AT-Frozen (calibrated confidence)', 'AT-Frozen (gamma_test=gamma_fit)', 'AT+Cal-Frozen (gamma_test=gamma_fit)', 'Ours-Offline (raw prior)']
FRACS = ['frac:0', 'frac:0.1', 'frac:0.25', 'frac:0.5', 'frac:0.75', 'frac:1.0']
COLORS = {'AT-Frozen': 'var(--s2)', 'AT+Cal-Frozen': 'var(--s3)', 'Ours-Offline': 'var(--s1)', 'Raw-threshold': 'var(--s4)'}


def pooled_budget(G, bkey):
    kind, v = bkey.split(':')
    v = float(v)
    if kind == 'usd':
        return v
    sizes = [s['test'] for s in G['splits']]
    return float(np.average([f['c_ref']*v for f in G['folds']], weights=sizes))


def summarize(runs, alpha, budget):
    """Mean over seeds of counts and rates; the share of seeds over the risk target or budget."""
    f = lambda k: [r[k] for r in runs if r[k] is not None]
    mean = lambda k: None if not f(k) else float(np.mean(f(k)))
    rel = np.array([r['n_released'] for r in runs]); wr = np.array([r['n_wrong_released'] for r in runs])
    rer = [w/n for n, w in zip(rel, wr) if n > 1e-9]
    out = {'n_tasks': runs[0]['n_tasks'], 'seeds': len(runs), 'n_released': float(rel.mean()), 'n_released_range': [float(rel.min()), float(rel.max())],
           'n_wrong_released': float(wr.mean()), 'release_error_rate': None if rel.mean() < 1e-9 else float(wr.mean()/rel.mean()),
           'release_error_rate_runs_mean': None if not rer else float(np.mean(rer)),
           'coverage': mean('coverage'), 'mean_screen_cost': mean('mean_screen_cost'),
           'weak_verifier_calls_per_task': mean('weak_verifier_calls_per_task'), 'verifier_query_rate': mean('verifier_query_rate'),
           'defer_rate': mean('defer_rate'), 'error_capture_rate': mean('error_capture_rate'),
           'error_share_in_deferred': mean('error_share_in_deferred'), 'correct_deferral_rate': mean('correct_deferral_rate'),
           'external_labels_acquired': 0,
           'risk_target_exceeded': float(np.mean([(w/n > alpha) if n > 1e-9 else False for n, w in zip(rel, wr)])),
           'budget_exceeded': float(np.mean([r['mean_screen_cost'] > budget*(1+1e-9)+1e-12 for r in runs])),
           'infeasible_folds': float(np.mean([r['infeasible_folds'] for r in runs])),
           'no_release': float(np.mean(rel < 1e-9))}
    br = [b for r in runs for b in r['brier'] if b is not None]
    out['brier'] = None if not br else float(np.mean(br))
    if 'sampled_runs' in runs[0]:
        s = runs[0]['sampled_runs']
        out['sampled_n_released'] = [float(min(s['n_released'])), float(max(s['n_released']))]
        out['sampled_n_wrong_released'] = [float(min(s['n_wrong_released'])), float(max(s['n_wrong_released']))]
    return out


def load():
    rows, res = [], {}
    for ds, name in DS:
        p = SRC/f'{ds}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        res[ds] = d
        for gen, G in d['generators'].items():
            for a, row in G['cells'].items():
                for bkey, cell in row.items():
                    B = pooled_budget(G, bkey)
                    for m, x in cell.items():
                        if m == 'paired_vs_ours':
                            continue
                        s = summarize(x['runs'], float(a), B)
                        pv = cell.get('paired_vs_ours', {}).get(m)
                        rows.append({'dataset': ds, 'generator': gen, 'alpha_target': float(a), 'budget': bkey, 'budget_usd': B, 'method': m, **s,
                                     'ours_minus_coverage': None if pv is None else pv['coverage_diff'],
                                     'ours_minus_release_error': None if pv is None else pv['release_error_diff']})
    return rows, res


def pct(v, d=1):
    return '—' if v is None else f'{v*100:.{d}f}%'


def main_table(rows, ds, alpha, bkeys, methods):
    r = [x for x in rows if x['dataset'] == ds and x['alpha_target'] == alpha and x['budget'] in bkeys and x['method'] in methods]
    body = ''
    for gen in GEN:
        for bk in bkeys:
            for m in methods:
                x = next((y for y in r if y['generator'] == gen and y['budget'] == bk and y['method'] == m), None)
                if x is None:
                    continue
                flag = []
                if x['risk_target_exceeded'] > 0:
                    flag.append(f'风险超标 {x["risk_target_exceeded"]:.0%}' if x['seeds'] > 1 else '风险超标')
                if x['budget_exceeded'] > 0:
                    flag.append('预算超限')
                if x['no_release'] == 1:
                    flag.append('全部弃答')
                elif x['infeasible_folds'] > 0:
                    flag.append(f'{x["infeasible_folds"]:.1f} 折无可行解')
                rng = f'<br><small>{x["n_released_range"][0]:.0f}–{x["n_released_range"][1]:.0f}</small>' if x['seeds'] > 1 else ''
                calls = '—' if x['weak_verifier_calls_per_task'] is None else f'{x["weak_verifier_calls_per_task"]:.2f}'
                body += (f'<tr><td>{GEN[gen]}</td><td class="n">{bk.split(":")[1]}{"×c_ref" if bk.startswith("frac") else " USD"}</td><td>{html.escape(m)}</td>'
                         f'<td class="n">{x["n_tasks"]}</td><td class="n">{x["n_released"]:.1f}{rng}</td><td class="n">{x["n_wrong_released"]:.1f}</td>'
                         f'<td class="n">{pct(x["release_error_rate"])}</td><td class="n">{pct(x["coverage"])}</td>'
                         f'<td class="n">{x["mean_screen_cost"]*1e4:.2f}</td><td class="n">{calls}</td><td>{"；".join(flag)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Generator</th><th class="n">预算 B</th><th>方法</th><th class="n">n</th>'
            '<th class="n">放行数</th><th class="n">放行中错的</th><th class="n">放行错误率</th><th class="n">放行率</th>'
            '<th class="n">每题筛选成本（10⁻⁴ USD）</th><th class="n">每题 verifier 调用</th><th>超标与不可行</th></tr></thead><tbody>'
            + body+'</tbody></table></div>')


def capture_table(rows, ds, alpha, bkey):
    r = [x for x in rows if x['dataset'] == ds and x['alpha_target'] == alpha and x['budget'] == bkey and x['method'] in MAIN]
    body = ''
    for gen in GEN:
        for m in MAIN:
            x = next((y for y in r if y['generator'] == gen and y['method'] == m), None)
            if x is None:
                continue
            body += (f'<tr><td>{GEN[gen]}</td><td>{m}</td><td class="n">{pct(x["defer_rate"])}</td><td class="n">{pct(x["error_capture_rate"])}</td>'
                     f'<td class="n">{pct(x["error_share_in_deferred"])}</td><td class="n">{pct(x["correct_deferral_rate"])}</td>'
                     f'<td class="n">{pct(x["verifier_query_rate"])}</td><td class="n">0</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Generator</th><th>方法</th><th class="n">拦截率</th><th class="n">错误拦截率</th>'
            '<th class="n">拦截集合中错误比例</th><th class="n">正确答案误拦率</th><th class="n">verifier 查询率</th><th class="n">外部标签获取数</th></tr></thead><tbody>'
            + body+'</tbody></table></div>')


def service_table(rows, alpha, bkey, ratios=(1, 5, 20)):
    body = ''
    for ds, name in DS:
        for gen in GEN:
            for m in MAIN:
                x = next((y for y in rows if y['dataset'] == ds and y['generator'] == gen and y['alpha_target'] == alpha and y['budget'] == bkey and y['method'] == m), None)
                if x is None:
                    continue
                cref = x['budget_usd']/float(bkey.split(':')[1]) if bkey.startswith('frac') and float(bkey.split(':')[1]) else None
                cells = ''.join(f'<td class="n">{"—" if cref is None else f"{(x["mean_screen_cost"]+x["defer_rate"]*r*cref)*1e4:.1f}"}</td>' for r in ratios)
                body += f'<tr><td>{name}</td><td>{GEN[gen]}</td><td>{m}</td><td class="n">{x["defer_rate"]*x["n_tasks"]:.1f}</td>{cells}</tr>'
    head = ''.join(f'<th class="n">c_A = {r}×c_ref</th>' for r in ratios)
    return ('<div class="scroll"><table><thead><tr><th>数据集</th><th>Generator</th><th>方法</th><th class="n">送审次数（= 拦截数）</th>'
            + head+'</tr></thead><tbody>'+body+'</tbody></table></div>')


def overruns(rows):
    body = ''
    for m in MAIN+SUPP:
        r = [x for x in rows if x['method'] == m]
        if not r:
            continue
        body += (f'<tr><td>{html.escape(m)}</td><td class="n">{len(r)}</td><td class="n">{sum(x["risk_target_exceeded"] > 0.5 for x in r)}</td>'
                 f'<td class="n">{sum(x["budget_exceeded"] > 0.5 for x in r)}</td><td class="n">{sum(x["no_release"] == 1 for x in r)}</td>'
                 f'<td class="n">{sum(x["infeasible_folds"] > 0 for x in r)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>方法</th><th class="n">格数</th><th class="n">测试放行错误率超过 α</th>'
            '<th class="n">测试成本超过 B</th><th class="n">全部弃答</th><th class="n">至少一折无可行解</th></tr></thead><tbody>'+body+'</tbody></table></div>')


def chart(rows, ds, gen, alpha, kind):
    W, H, L, R_, T, Bm = 300, 200, 44, 10, 10, 28
    pw, ph = W-L-R_, H-T-Bm
    g = []
    if kind == 'budget':
        xs = FRACS
        sx = lambda i: L+i/(len(xs)-1)*pw
        sy = lambda v: T+ph-v*ph
        for v in (0, .5, 1):
            g.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/><text class="tk" x="{L-5}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0%}</text>')
        for i, b in enumerate(xs):
            g.append(f'<text class="tk" x="{sx(i):.1f}" y="{H-6}" text-anchor="middle">{b.split(":")[1]}</text>')
        for m in MAIN+['Raw-threshold']:
            pts = [(i, next((y for y in rows if y['dataset'] == ds and y['generator'] == gen and y['alpha_target'] == alpha and y['budget'] == b and y['method'] == m), None)) for i, b in enumerate(xs)]
            pts = [(i, x) for i, x in pts if x]
            g.append(f'<polyline points="{" ".join(f"{sx(i):.1f},{sy(x["coverage"]):.1f}" for i, x in pts)}" fill="none" stroke="{COLORS[m]}" stroke-width="{3 if m == "Ours-Offline" else 1.6}"/>')
            for i, x in pts:
                over = x['risk_target_exceeded'] > 0.5
                g.append(f'<circle cx="{sx(i):.1f}" cy="{sy(x["coverage"]):.1f}" r="4" fill="{"var(--surface)" if over else COLORS[m]}" stroke="{COLORS[m]}" stroke-width="2"><title>{html.escape(m)}，B = {xs[i].split(":")[1]}×c_ref：放行率 {pct(x["coverage"])}，放行错误率 {pct(x["release_error_rate"])}{"（超标）" if over else ""}</title></circle>')
    else:
        sx = lambda v: L+v*pw
        sy = lambda v: T+ph-v*ph
        for v in (0, .5, 1):
            g.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/><text class="tk" x="{L-5}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0%}</text>')
            g.append(f'<text class="tk" x="{sx(v):.1f}" y="{H-6}" text-anchor="middle">{v:.0%}</text>')
        for m in MAIN:
            for x in [y for y in rows if y['dataset'] == ds and y['generator'] == gen and y['method'] == m and y['budget'] in FRACS]:
                if x['error_capture_rate'] is None:
                    continue
                g.append(f'<circle cx="{sx(x["defer_rate"]):.1f}" cy="{sy(x["error_capture_rate"]):.1f}" r="3" fill="{COLORS[m]}" opacity=".8"><title>{html.escape(m)}，α = {x["alpha_target"]:.0%}，B = {x["budget"].split(":")[1]}：拦截率 {pct(x["defer_rate"])}，错误拦截率 {pct(x["error_capture_rate"])}</title></circle>')
    return f'<figure><figcaption>{GEN[gen]}</figcaption><svg viewBox="0 0 {W} {H}" role="img" aria-label="{GEN[gen]}">{"".join(g)}</svg></figure>'


def brier_table(rows):
    body = ''
    for ds, name in DS:
        for gen in GEN:
            r = {m: next((y for y in rows if y['dataset'] == ds and y['generator'] == gen and y['alpha_target'] == 0.05 and y['budget'] == 'frac:1.0' and y['method'] == m), None)
                 for m in ('AT-Frozen', 'AT+Cal-Frozen', 'Ours-Offline', 'Ours-Offline (raw prior)')}
            if not r['AT-Frozen']:
                continue
            body += (f'<tr><td>{name}</td><td>{GEN[gen]}</td>' + ''.join(f'<td class="n">{"—" if x is None or x["brier"] is None else f"{x["brier"]:.3f}"}</td>' for x in r.values()) + '</tr>')
    return ('<div class="scroll"><table><thead><tr><th>数据集</th><th>Generator</th><th class="n">原始置信度（AT 用）</th><th class="n">AT+Cal 冻结校准分数</th>'
            '<th class="n">Ours 校准先验（isotonic）</th><th class="n">Ours 原始先验</th></tr></thead><tbody>'+body+'</tbody></table></div>')


LINES = [('Ours-Offline', '我们的方法（Ours-Offline）', 'var(--s1)'), ('AT-Frozen', 'AT-Frozen', 'var(--s2)'),
         ('AT+Cal-Frozen', 'AT+Cal-Frozen', 'var(--s3)'), ('Raw-threshold', '原始置信度阈值', 'var(--s4)')]
ALPHAS = [0.01, 0.05, 0.10]
TOP = 'frac:1.0'


def cell(rows, ds, gen, alpha, bkey, m):
    return next((y for y in rows if y['dataset'] == ds and y['generator'] == gen and y['alpha_target'] == alpha and y['budget'] == bkey and y['method'] == m), None)


def panel(gen, series, xs, xticks, ymax, yticks, diag=None, xfmt=None):
    W, H, L, R_, T, Bm = 300, 210, 44, 10, 10, 28
    pw, ph = W-L-R_, H-T-Bm
    x0, x1 = xs
    sx = lambda v: L+(min(max(v, x0), x1)-x0)/(x1-x0)*pw
    sy = lambda v: T+ph-min(v, ymax)/ymax*ph
    g = []
    for v in yticks:
        g.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/><text class="tk" x="{L-5}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0%}</text>')
    for v, lab in xticks:
        g.append(f'<text class="tk" x="{sx(v):.1f}" y="{H-6}" text-anchor="middle">{lab}</text>')
    if diag:
        g.append(f'<polyline class="ideal" points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in diag)}" fill="none"/>')
    for key, name, col in reversed(LINES):
        pts = series.get(key) or []
        if not pts:
            continue
        g.append(f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in pts)}" fill="none" stroke="{col}" stroke-width="{3 if key == "Ours-Offline" else 1.6}"/>')
        for x, y, tip in pts:
            g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{4.5 if key == "Ours-Offline" else 3.5}" fill="{col}" stroke="var(--surface)" stroke-width="1.5"><title>{html.escape(name+"，"+tip)}</title></circle>')
    return f'<figure><figcaption>{GEN[gen]}</figcaption><svg viewBox="0 0 {W} {H}" role="img" aria-label="{GEN[gen]}">{"".join(g)}</svg></figure>'


def grid(rows, kind):
    out = ''
    for ds, name in DS:
        figs = ''
        for gen in GEN:
            series = {}
            if kind == 'cost':
                a = 0.01 if ds == 'math500' else 0.05
                xmax = max([(cell(rows, ds, gen, a, b, m) or {}).get('mean_screen_cost') or 0 for b in FRACS for m, _, _ in LINES]+[1e-5])*1e4*1.1
                step = next(v for v in (1, 2, 5, 10, 20) if xmax/v <= 6)
                for m, _, _ in LINES:
                    pts = []
                    for b in FRACS:
                        x = cell(rows, ds, gen, a, b, m)
                        if x:
                            pts.append((x['mean_screen_cost']*1e4, x['coverage'], f'预算 {b.split(":")[1]}×c_ref：放行 {pct(x["coverage"])}，每题花费 {x["mean_screen_cost"]*1e4:.1f}'))
                    series[m] = sorted(pts)
                figs += panel(gen, series, (0, xmax), [(v, f'{v:g}') for v in np.arange(0, xmax, step)], 1.0, (0, .5, 1))
                continue
            for m, _, _ in LINES:
                pts = []
                for i, a in enumerate(ALPHAS):
                    x = cell(rows, ds, gen, a, TOP, m)
                    if not x:
                        continue
                    if kind == 'actual':
                        if x['release_error_rate'] is not None:
                            pts.append((x['release_error_rate'], x['coverage'], f'α = {a:.0%}：实际放行错误率 {pct(x["release_error_rate"])}，放行 {pct(x["coverage"])}'))
                    elif kind == 'alpha':
                        pts.append((i, x['coverage'], f'α = {a:.0%}：放行 {pct(x["coverage"])}'))
                    elif kind == 'fpr':
                        if x['error_capture_rate'] is not None:
                            pts.append((i, 1-x['error_capture_rate'], f'α = {a:.0%}：错答案被放行的比例 {pct(1-x["error_capture_rate"])}'))
                    elif x['release_error_rate'] is not None:
                        pts.append((i, x['release_error_rate'], f'α = {a:.0%}：实际放行错误率 {pct(x["release_error_rate"])}'))
                series[m] = sorted(pts) if kind == 'actual' else pts
            if kind == 'actual':
                figs += panel(gen, series, (0, .2), [(v, f'{v:.0%}') for v in (0, .05, .1, .15, .2)], 1.0, (0, .5, 1))
            elif kind == 'alpha':
                figs += panel(gen, series, (0, 2), [(i, f'{a:.0%}') for i, a in enumerate(ALPHAS)], 1.0, (0, .5, 1))
            elif kind == 'fpr':
                figs += panel(gen, series, (0, 2), [(i, f'{a:.0%}') for i, a in enumerate(ALPHAS)], .6, (0, .3, .6))
            else:
                figs += panel(gen, series, (0, 2), [(i, f'{a:.0%}') for i, a in enumerate(ALPHAS)], .2, (0, .1, .2),
                              diag=[(i, a) for i, a in enumerate(ALPHAS)])
        out += f'<h3>{name}</h3><div class="row">{figs}</div>'
    return out


def method_table(rows, kind):
    out = ''
    for ds, name in DS:
        al = [0.01, 0.05] if ds == 'math500' else [0.05, 0.10]
        head = ''.join(f'<th class="n">{GEN[g]} α={a:.0%}</th>' for g in GEN for a in al)
        body = ''
        for m, label, _ in LINES:
            cells = ''
            for g in GEN:
                for a in al:
                    x = cell(rows, ds, g, a, TOP, m)
                    if x is None:
                        cells += '<td class="n">—</td>'
                    elif kind == 'rate':
                        star = '*' if x['risk_target_exceeded'] > 0.5 else ''
                        cells += f'<td class="n">{pct(x["coverage"])}<br><small>错误率 {pct(x["release_error_rate"])}{star}</small></td>'
                    else:
                        cells += f'<td class="n">{x["n_released"]:.1f} / {x["n_wrong_released"]:.1f}</td>'
            body += f'<tr><th scope="row">{label}</th>{cells}</tr>'
        out += f'<h3>{name}</h3><div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    return out


def significance(rows):
    body = ''
    for m in ['AT-Frozen', 'AT+Cal-Frozen', 'Raw-threshold', 'AT-Frozen (calibrated confidence)', 'Ours-Offline (raw prior)']:
        cells = ''
        for ds, _ in DS:
            r = [x for x in rows if x['dataset'] == ds and x['method'] == m and x['budget'].startswith('frac') and x['ours_minus_coverage']]
            more = sum(x['ours_minus_coverage'][1] > 0 for x in r); less = sum(x['ours_minus_coverage'][2] < 0 for x in r)
            cells += f'<td class="n">{more} / {less}（共 {len(r)}）</td>'
        body += f'<tr><td>{html.escape(m)}</td>{cells}</tr>'
    return ('<div class="scroll"><table><thead><tr><th>和 Ours 比</th>' + ''.join(f'<th class="n">{n}：Ours 显著更多 / 更少</th>' for _, n in DS)
            + '</tr></thead><tbody>'+body+'</tbody></table></div>')


def main():
    rows, res = load()
    SRC.mkdir(parents=True, exist_ok=True)
    keys = ['dataset', 'generator', 'alpha_target', 'budget', 'budget_usd', 'method', 'n_tasks', 'seeds', 'n_released', 'n_wrong_released',
            'release_error_rate', 'release_error_rate_runs_mean', 'coverage', 'mean_screen_cost', 'weak_verifier_calls_per_task', 'verifier_query_rate',
            'defer_rate', 'error_capture_rate', 'error_share_in_deferred', 'correct_deferral_rate', 'external_labels_acquired',
            'risk_target_exceeded', 'budget_exceeded', 'infeasible_folds', 'no_release', 'brier', 'ours_minus_coverage', 'ours_minus_release_error']
    with open(SRC/'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    capture = ''.join(f'<h3>{n}</h3>'+capture_table(rows, ds, 0.05, TOP) for ds, n in DS)
    supp = ''.join(f'<h3>{n}</h3>'+main_table(rows, ds, 0.05, [TOP], SUPP) for ds, n in DS)
    legend = ''.join(f'<span><i style="background:{c}"></i>{n}</span>' for _, n, c in LINES)
    page = Path(__file__).with_name('offline_frozen_template.html').read_text()
    for k, v in {'{{LEGEND}}': legend, '{{ACTUAL}}': grid(rows, 'actual'), '{{ALPHA}}': grid(rows, 'alpha'), '{{CONTROL}}': grid(rows, 'control'), '{{FPR}}': grid(rows, 'fpr'),
                 '{{COST}}': grid(rows, 'cost'), '{{RATE}}': method_table(rows, 'rate'), '{{COUNT}}': method_table(rows, 'count'),
                 '{{SIG}}': significance(rows), '{{OVERRUNS}}': overruns(rows), '{{CAPTURE}}': capture, '{{SUPP}}': supp,
                 '{{BRIER}}': brier_table(rows), '{{SERVICE}}': service_table(rows, 0.05, TOP)}.items():
        page = page.replace(k, v)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print('wrote', OUT, len(rows), 'rows')


if __name__ == '__main__':
    main()
