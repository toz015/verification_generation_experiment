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
    sections = ''
    for ds, name in DS:
        if ds not in res:
            continue
        sections += (f'<section><h2>{name}</h2><p class="meta">α = 5%。B 以 Fit 上 5 个 verifier 全面板的平均成本 c_ref 为单位。AT 类方法是 20 个训练种子的平均，放行数下面是种子间范围；「风险超标 x%」是超过 α 的种子比例。</p>'
                     + main_table(rows, ds, 0.05, ['frac:0', 'frac:0.25', 'frac:1.0'], MAIN)
                     + '<h3>放行率对预算（α = 5%；空心点表示测试放行错误率超过 α）</h3><div class="row">'
                     + ''.join(chart(rows, ds, g, 0.05, 'budget') for g in GEN) + '</div>'
                     + '<h3>错误拦截率对拦截率（全部 α 和预算）</h3><div class="row">'
                     + ''.join(chart(rows, ds, g, 0.05, 'capture') for g in GEN) + '</div>'
                     + '<h3>送审选择质量（α = 5%，B = 1×c_ref）</h3>' + capture_table(rows, ds, 0.05, 'frac:1.0')
                     + '<h3>补充对照（α = 5%，B = 1×c_ref）</h3>' + main_table(rows, ds, 0.05, ['frac:1.0'], SUPP)
                     + '</section>')
    page = (TEMPLATE.replace('{{SECTIONS}}', sections).replace('{{OVERRUNS}}', overruns(rows))
            .replace('{{SUMMARY}}', SUMMARY).replace('{{SERVICE}}', service_table(rows, 0.05, 'frac:1.0')).replace('{{BRIER}}', brier_table(rows)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print('wrote', OUT, len(rows), 'rows')


SUMMARY = '''<section>
  <h2>结论</h2>
  <ul>
    <li><b>有 verifier 预算时，Ours-Offline 放行更多。</b>在以 c_ref 为单位的 6 档预算下（含 B = 0，各数据集 54 个 α × 预算 × generator 格子，配对 bootstrap，QuantumKatas 按 kata 组重采样），Ours 比 AT-Frozen 显著放行更多的格子是 GPQA 44、MATH-500 25、QuantumKatas 38；比 AT+Cal-Frozen 是 50、26、39；比直接在 Validation 上选原始置信度阈值是 30、24、26，但原始阈值也有 9、6、17 格显著更多。例如 α = 5%、B = 1×c_ref 时，QuantumKatas 的 Qwen3：Ours 放行 115 题，原始阈值 29 题，AT-Frozen 12 题。</li>
    <li><b>没有 verifier 预算（B = 0）时，Ours 和原始置信度阈值基本一样；AT 两种方法更差。</b>它们的阈值是按论文的漏审比例更新学出来的，没有针对放行错误率；AT+Cal-Frozen 在 207 个格子里有 75 个找不到可行工作点，只能全部弃答。</li>
    <li><b>放行错误率常常超过目标，Ours 最多。</b>测试放行错误率超过 α 的格子：Ours 103 / 207，原始阈值 81，AT-Frozen 63，AT+Cal-Frozen 24（另有 75 格全弃答）。α = 5% 时 Ours 典型为 6% 到 8%，例如 QuantumKatas Qwen3 7.2%、GPQA Qwen3 7.8%。Validation 只有 50 到 100 题，在其上挑放行最多的工作点会偏乐观。</li>
    <li><b>只有 Ours 会超预算，且只在 GPQA。</b>23 个格子测试成本超过 B，都是 GPQA（Validation 仅 50 题）；AT 类方法不调用 verifier，成本为 0。</li>
    <li><b>测试时保留随机送审（γ_test = γ_fit）只会按比例少放 10%，</b>放行错误率不变，所以主表不依赖这个设定。</li>
    <li><b>MATH-500 的 Qwen3 和 gpt-oss 几乎全部放行，</b>方法间没有差别；差别集中在 Sonnet。</li>
  </ul>
</section>'''

TEMPLATE = '''<title>离线冻结对照实验</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column, wide tables scroll in their own box, small multiples per generator. */
:root {
  --bg: #f6f7f5; --surface: #ffffff; --fg: #1d2522; --muted: #5b6662; --line: #dbe0dd;
  --s1: #1F5FA8; --s2: #B4581F; --s3: #1E8A6E; --s4: #8E4FA8;
  --display: "Noto Serif SC", "Songti SC", serif; --body: "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; --s4: #A873C8; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; --s4: #A873C8; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.65; }
main { max-width: 1160px; margin: 0 auto; padding-block: 36px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 30px; min-width: 0; }
section { display: grid; gap: 10px; min-width: 0; }
h1 { font-family: var(--display); font-size: 28px; line-height: 1.3; margin: 0; text-wrap: balance; }
h2 { font-size: 19px; margin: 0; } h3 { font-size: 15px; margin: 10px 0 0; }
p { margin: 0; max-width: 76ch; } .meta { color: var(--muted); font-size: 13px; }
ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 80ch; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 9px; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: top; }
thead th { color: var(--muted); font-weight: 500; }
.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; } thead .n { font-family: var(--body); }
small { color: var(--muted); }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-width: 0; }
figcaption { font-size: 13px; font-weight: 700; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--line); stroke-width: 1; }
.tk { font: 10.5px var(--mono); fill: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); }
.legend i { display: inline-block; width: 16px; height: 4px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
code { font-family: var(--mono); font-size: 13px; }
</style>
<main>
<header style="display:grid;gap:8px">
  <h1>离线冻结对照实验：AT-Frozen、AT+Cal-Frozen、Ours-Offline</h1>
  <p class="meta">按 docs/offline-at-frozen-experiment-protocol-20261110.md 实施。AT 两种方法是论文在线算法的 offline adaptation，论文的在线风险保证不适用于冻结版本。只用缓存，没有新增 API 调用。</p>
</header>
{{SUMMARY}}
<section>
  <h2>设置</h2>
  <ul>
    <li><b>统一任务：</b>只决定是否放行原始候选；「拦截」即不放行，包括弃答或送审，测试时不获取任何外部标签（外部标签获取数恒为 0）。verifier 查询单独计数，查询后仍可能放行。</li>
    <li><b>划分：</b>GPQA 的 249 道 calibration 题按种子 20261111 分为 Fit 199 / Validation 50，Diamond 196 道作 Test；MATH-500 和 QuantumKatas 的五折中，测试折 k、验证折 (k+1) mod 5，其余三折 Fit。三种方法用同一安排，选定工作点后不再合并 Fit 和 Validation 重训。</li>
    <li><b>工作点选择（三种方法统一）：</b>在 Validation 上，从满足 Σ a(e − α) ≤ 0 且平均成本 ≤ B 的候选中选放行最多者；平局时成本低、规则简单、序号小者优先。没有非零放行的可行候选时记为 no_nontrivial_feasible_policy，部署全弃答。</li>
    <li><b>AT-Frozen：</b>在 Fit 上按种子打乱顺序跑一遍论文更新（τ₁ = 0.5，γ_fit = 0.1，α_at × η 网格），冻结 τ；测试时 γ_test = 0，放行条件为 C ≥ τ。<b>AT+Cal-Frozen：</b>同上并按 Algorithm 1 学习 21 点网格上的指数权重（β 网格），冻结 w 和 τ，放行条件为 S(C) ≥ τ。各 20 个训练种子。</li>
    <li><b>Ours-Offline：</b>在 Fit 上拟合 isotonic 校准（无 Laplace 的单调 Brier 最优拟合）和 verifier 似然（5 档，Laplace 1）；在 Validation 的报告置信度上按模型规划（不看标签）生成混合路由，风险定义为 Σw(e − α_plan p) ≤ 0；verifier 顺序先按不看标签的规划放行率从 60 种里取前 3。候选为 3 个顺序 × α_plan 6 档 × B_plan 6 档，再按统一规则在 Validation 上选。</li>
    <li><b>成本：</b>主表只计筛选决策成本，即实际读取的 verifier 调用（含返回格式错误但已付费的调用）；generator 自带的置信度不计费。候选生成成本和标签采集成本是共同成本，单列不计入 B。</li>
  </ul>
</section>
{{SECTIONS}}
<section>
  <h2>风险与预算超限汇总</h2>
  <p class="meta">全部数据集、generator、α ∈ {1%, 5%, 10%} 和预算的格子。AT 类方法以过半种子超标计为超标。</p>
  {{OVERRUNS}}
</section>
<section>
  <h2>校准诊断（Test 上的 Brier）</h2>
  {{BRIER}}
</section>
<section>
  <h2>外部审计服务情景（补充）</h2>
  <p class="meta">假设所有被拦截的答案都转交同一种外部审计服务，每次花费 c_A。没有可信单价，因此按 c_A = 1、5、20 倍 c_ref 做情景估算（单位 10⁻⁴ USD/题，含筛选成本）。工作点沿用主表，没有在新成本下重新选择。</p>
  {{SERVICE}}
</section>
<section>
  <h2>复现</h2>
  <p><code>python -m vgx.offline_frozen config</code>，<code>python -m pytest tests/test_offline_frozen.py tests/test_selective_auditing.py</code>，<code>python -m vgx.offline_frozen run</code>，<code>python -m vgx.offline_frozen_report</code>。逐题决策日志在 results/offline_at_frozen_20261111/*_decisions.jsonl.gz，全部格子在 summary.csv。</p>
</section>
</main>
'''

if __name__ == '__main__':
    main()
