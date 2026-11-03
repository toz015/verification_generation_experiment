"""Curves page for publication under FPR and budget targets on GPQA, MATH-500 and QuantumKatas.

Reads results/gpqa_fpr_routing_20261102/baselines*.json and writes
reports/gpqa_fpr_routing_20261102/results.html.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

SRC = Path('results/gpqa_fpr_routing_20261102')
FDR = len(sys.argv) > 1 and sys.argv[1] == 'fdr'
OUT = Path(f'reports/gpqa_fpr_routing_20261102/results{"_fdr" if FDR else ""}.html')
DATASETS = [('gpqa', 'baselines_gpqa_fdr' if FDR else 'baselines', 'GPQA', '在 249 道 calibration 题上选参数，在 196 道 Diamond 题上评估。'),
            ('math500', 'baselines_math500'+('_fdr' if FDR else ''), 'MATH-500', '5 折：每折用其余 4 折选参数，5 折测试结果合并，共 500 题。'),
            ('qkatas', 'baselines_qkatas'+('_fdr' if FDR else ''), 'QuantumKatas', '5 折：同上，共 322 题。')]
GEN = {'qwen3': 'Qwen3-235B', 'gptoss': 'gpt-oss-120b', 'sonnet55': 'Sonnet 5.5'}
LINES = [('ours', '我们的方法', 1), ('confidence_only', '只看置信度', 2), ('fixed_3', '固定问 3 个', 3),
         ('weaver_5', 'Weaver 5 个', 4), ('sc_vote', 'Self-consistency 投票', 5), ('ssv_model', 'SSV', 6)]
TABLE = [('ours', '我们的方法'), ('ours_raw', '我们的方法（原始 p_correct）'), ('confidence_only', '只看置信度'), ('fixed_1', '固定问 1 个'),
         ('fixed_2', '固定问 2 个'), ('fixed_3', '固定问 3 个'), ('weaver_3', 'Weaver 3 个'), ('weaver_5', 'Weaver 5 个'),
         ('sc_vote', 'Self-consistency 投票'), ('ssv_model', 'SSV')]
SHARE = ('1/12', '1/6', '1/3', '1/2', '1')
W, H, L, Bt, T, R = 300, 220, 42, 30, 10, 10


def frame(xs, ymax, ticks):
    pw, ph = W-L-R, H-T-Bt
    sx = lambda i: L+pw*i/(len(xs)-1)
    sy = lambda v: T+ph-min(v, ymax)/ymax*ph
    g = []
    for t in ticks:
        g.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        g.append(f'<text class="tk" x="{L-6}" y="{sy(t)+4:.1f}" text-anchor="end">{t*100:.0f}%</text>')
    for i, x in enumerate(xs):
        g.append(f'<text class="tk" x="{sx(i):.1f}" y="{T+ph+16}" text-anchor="middle">{x}</text>')
    return g, sx, sy


def chart(title, xs, ymax, ticks, series, diagonal=None):
    g, sx, sy = frame(xs, ymax, ticks)
    if diagonal:
        d = ' '.join(f'{sx(i):.1f},{sy(v):.1f}' for i, v in enumerate(diagonal))
        g.append(f'<polyline class="ideal" points="{d}" fill="none"/>')
    for key, name, slot, vals in reversed(series):
        if not any(v is not None for v in vals):
            continue
        w = 3 if key == 'ours' else 1.5
        pts = [(i, v) for i, v in enumerate(vals) if v is not None]
        g.append(f'<polyline points="{" ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in pts)}" fill="none" stroke="var(--s{slot})" stroke-width="{w}"/>')
        for i, v in pts:
            r = 4.5 if key == 'ours' else 3.5
            tip = html.escape(f'{name}，{xs[i]}：{v:.1%}')
            g.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="{r}" fill="var(--s{slot})" stroke="var(--surface)" stroke-width="1.5"><title>{tip}</title></circle>')
    return f'<figure><figcaption>{title}</figcaption><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{"".join(g)}</svg></figure>'


def rel_err(x):
    return None if x['released'] < 1 else x['wrong_released']/x['released']


def target(x):
    return rel_err(x) if FDR else x['fpr']


def row(d, kind):
    alphas, budgets = d['alphas'], d['budgets']
    out = ''
    for gen in ('qwen3', 'gptoss', 'sonnet55'):
        c = d['generators'][gen]
        get = lambda a, b, m, f: c[f'{a}|{b}'][m][f] if m in c[f'{a}|{b}'] else None
        if kind == 'pub':
            s = [(m, n, k, [get(a, budgets[3], m, 'pub') for a in alphas]) for m, n, k in LINES]
            out += chart(GEN[gen], [f'{a:.0%}' for a in alphas], 1., (0, .25, .5, .75, 1.), s)
        elif kind in ('target', 'other'):
            on_target = kind == 'target'
            f = target if on_target else (lambda x: x['fpr'] if FDR else rel_err(x))
            val = lambda a, m: f(c[f'{a}|{budgets[3]}'][m]) if m in c[f'{a}|{budgets[3]}'] else None
            s = [(m, n, k, [val(a, m) for a in alphas]) for m, n, k in LINES]
            out += chart(GEN[gen], [f'{a:.0%}' for a in alphas], .4, (0, .1, .2, .3, .4), s, diagonal=list(alphas) if on_target else None)
        else:
            s = [(m, n, k, [get(.05, b, m, 'pub') for b in budgets]) for m, n, k in LINES]
            out += chart(GEN[gen], list(SHARE), 1., (0, .25, .5, .75, 1.), s)
    return f'<div class="row">{out}</div>'


def table(d):
    b = d['budgets'][3]
    gens = ('qwen3', 'gptoss', 'sonnet55')
    head = ''.join(f'<th class="n">{GEN[g]} α=5%</th><th class="n">{GEN[g]} α=10%</th>' for g in gens)
    body = ''
    for m, name in TABLE:
        cells = ''
        for g in gens:
            for a in (.05, .1):
                x = d['generators'][g][f'{a}|{b}'].get(m)
                cells += '<td class="n">—</td>' if x is None else (
                    f'<td class="n">{x["released"]:.0f} / {x["wrong_released"]:.1f}{"*" if (target(x) or 0) > a else ""}<br><small>FPR {x["fpr"]:.1%}，放行错误率 {"—" if rel_err(x) is None else f"{rel_err(x):.1%}"}</small></td>')
        body += f'<tr><th scope="row">{name}</th>{cells}</tr>'
    return f'<div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def counts(d):
    tot, over, more, less = {}, {}, {}, {}
    for g, c in d['generators'].items():
        for k, cell in c.items():
            a = float(k.split('|')[0])
            for m, x in cell.items():
                tot[m] = tot.get(m, 0)+1
                over[m] = over.get(m, 0)+((target(x) or 0) > a)
                q = x.get('ours_minus_pub')
                if q:
                    more[m] = more.get(m, 0)+(q['ci95'][0] > 0)
                    less[m] = less.get(m, 0)+(q['ci95'][1] < 0)
    return tot, over, more, less


def summary(res):
    rows = ''
    for m, name in TABLE:
        cells = ''
        for ds, _, _, _ in DATASETS:
            tot, over, more, less = counts(res[ds])
            cells += '<td class="n">—</td><td class="n">—</td>' if m not in tot else (
                f'<td class="n">{"参照" if m not in more else f"{more[m]} / {less[m]}"}</td><td class="n">{over[m]} / {tot[m]}</td>')
        rows += f'<tr><th scope="row">{name}</th>{cells}</tr>'
    head = ''.join(f'<th class="n">{n}：我们放行显著更多 / 它放行显著更多</th><th class="n">{n}：{"放行错误率" if FDR else "FPR"} 超 α</th>' for _, _, n, _ in DATASETS)
    return f'<div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def main():
    res = {ds: json.loads((SRC/f'{f}.json').read_text()) for ds, f, _, _ in DATASETS}
    pub = ''.join(f'<h3>{n}</h3><p class="meta">{s}</p>{row(res[ds], "pub")}' for ds, _, n, s in DATASETS)
    fpr = ''.join(f'<h3>{n}</h3>{row(res[ds], "target")}' for ds, _, n, _ in DATASETS)
    err = ''.join(f'<h3>{n}</h3>{row(res[ds], "other")}' for ds, _, n, _ in DATASETS)
    bud = ''.join(f'<h3>{n}</h3>{row(res[ds], "bud")}' for ds, _, n, _ in DATASETS)
    tab = ''.join(f'<h3>{n}</h3>{table(res[ds])}' for ds, _, n, _ in DATASETS)
    legend = ''.join(f'<span><i style="background:var(--s{k})"></i>{n}</span>' for _, n, k in LINES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE_FDR if FDR else TEMPLATE
    OUT.write_text(page.replace('{{PUB}}', pub).replace('{{FPR}}', fpr).replace('{{BUD}}', bud).replace('{{ERR}}', err).replace('{{TAB}}', tab)
                   .replace('{{SUM}}', summary(res)).replace('{{LEGEND}}', legend))
    print('wrote', OUT)


TEMPLATE = '''<title>FPR 目标下的放行曲线</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; each dataset is a row of three small multiples, one per generator. */
:root {
  --bg: #f6f8f7; --surface: #ffffff; --fg: #1c2522; --muted: #5a6661; --line: #dbe1de;
  --s1: #1F5FA8; --s2: #B4581F; --s3: #1E8A6E; --s4: #8E4FA8; --s5: #9A7B12; --s6: #B8336A;
  --display: "Noto Serif SC", "Songti SC", serif; --body: "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; --s4: #A873C8; --s5: #A38C26; --s6: #D9648F; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; --s4: #A873C8; --s5: #A38C26; --s6: #D9648F; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.65; }
main { max-width: 1120px; margin: 0 auto; padding-block: 36px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 30px; min-width: 0; }
section { display: grid; gap: 10px; min-width: 0; }
h1 { font-family: var(--display); font-size: 28px; line-height: 1.3; margin: 0; text-wrap: balance; }
h2 { font-size: 18px; margin: 0; } h3 { font-size: 15px; margin: 10px 0 0; }
p { margin: 0; max-width: 72ch; } .meta { color: var(--muted); font-size: 13px; }
ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 76ch; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 12px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-width: 0; }
figcaption { font-size: 13px; font-weight: 700; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--line); stroke-width: 1; }
.ideal { stroke: var(--muted); stroke-width: 1.25; stroke-dasharray: 5 4; }
.tk { font: 10.5px var(--mono); fill: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); }
.legend i { display: inline-block; width: 16px; height: 4px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: top; }
thead th { color: var(--muted); font-weight: 500; }
tbody th { font-weight: 500; }
.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; } thead .n { font-family: var(--body); }
small { color: var(--muted); }
</style>
<main>
<header style="display:grid;gap:8px">
  <h1>FPR 目标下的放行曲线</h1>
  <p class="meta">给定 FPR 上限 α 和每题预算 B，谁放行最多。放行比例 = 放行数 / 全部题数；FPR = 放行的错答案 / 全部错答案。</p>
</header>
<section>
  <h2>怎么比的</h2>
  <p>每个方法都走同样的流程：在训练部分用真实标签（样本外）选参数，要求 FPR 不超过 α、平均花费不超过 B，放行尽量多，然后固定参数，到测试部分评估。规则是随机混合的，所以放行数是期望值。Self-consistency 的 10 次采样算进花费，放行的是投票答案，它的 FPR 按投票答案算。QuantumKatas 的代码无法投票，所以没有这一项。</p>
</section>
<section>
  <h2>主要结果</h2>
  <ul>
    <li>放行数：三个数据集上，我们在大多数设定下都放行显著更多（见下表）。主要例外是 GPQA 的 Sonnet 在小 α 时，以及 MATH-500 的 Qwen3 上的 Weaver。</li>
    <li>FPR：<b>只有 GPQA 守住了目标。</b>我们在 MATH-500 有 54 / 75 格超过 α，在 QuantumKatas 有 65 / 75 格超过 α。QuantumKatas 多数只是略高（α = 5% 时平均 5.3% 到 6.7%）。MATH-500 的 Qwen3 超得最多（α = 5% 时平均 15.2%），它只有 16 道错题。</li>
    <li>放行更多不等于完全赢：在这两个数据集上，我们超 α 的格数比固定问 3 个更多（54 对 35，65 对 40），所以放行的优势有一部分是用更高的 FPR 换来的。</li>
  </ul>
  {{SUM}}
  <p class="meta">每个方法有 75 格（3 个 generator × 5 个 α × 5 个预算）。「我们放行显著更多」是其中我们的放行比例显著高于它的格数，「它放行显著更多」反过来，按配对 bootstrap 区间是否不含 0 判断，剩下的格数是差别不显著。「FPR 超 α」是它自己在测试上超过目标的格数。</p>
</section>
<div class="legend">{{LEGEND}}<span>虚线：FPR = α</span></div>
<section>
  <h2>放行比例对 α</h2>
  <p class="meta">预算 B 取固定问 3 个花费的一半。横轴 α 均匀排列，不是线性刻度。</p>
  {{PUB}}
</section>
<section>
  <h2>测试 FPR 对 α</h2>
  <p class="meta">同样的预算。点在虚线以上表示超过目标。纵轴上限 40%。</p>
  {{FPR}}
</section>
<section>
  <h2>放行答案的错误率对 α</h2>
  <p class="meta">放行的答案里有多少是错的，即 1 − 放行答案的正确率。同样的预算。放行不到 1 题的点不画。</p>
  {{ERR}}
</section>
<section>
  <h2>放行比例对预算</h2>
  <p class="meta">α = 5%。横轴是预算占固定问 3 个花费的比例。</p>
  {{BUD}}
</section>
<section>
  <h2>各方法的数字</h2>
  <p class="meta">预算同上。格子是放行 / 错，* 表示 FPR 超过 α。</p>
  {{TAB}}
</section>
</main>
'''

TEMPLATE_FDR = '''<title>放行错误率目标下的验证结果</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; each dataset is a row of three small multiples, one per generator. */
:root {
  --bg: #f6f8f7; --surface: #ffffff; --fg: #1c2522; --muted: #5a6661; --line: #dbe1de;
  --s1: #1F5FA8; --s2: #B4581F; --s3: #1E8A6E; --s4: #8E4FA8; --s5: #9A7B12; --s6: #B8336A;
  --display: "Noto Serif SC", "Songti SC", serif; --body: "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; --s4: #A873C8; --s5: #A38C26; --s6: #D9648F; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; --s4: #A873C8; --s5: #A38C26; --s6: #D9648F; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.65; }
main { max-width: 1120px; margin: 0 auto; padding-block: 36px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 30px; min-width: 0; }
section { display: grid; gap: 10px; min-width: 0; }
h1 { font-family: var(--display); font-size: 28px; line-height: 1.3; margin: 0; text-wrap: balance; }
h2 { font-size: 18px; margin: 0; } h3 { font-size: 15px; margin: 10px 0 0; }
p { margin: 0; max-width: 72ch; } .meta { color: var(--muted); font-size: 13px; }
ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 76ch; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 12px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-width: 0; }
figcaption { font-size: 13px; font-weight: 700; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--line); stroke-width: 1; }
.ideal { stroke: var(--muted); stroke-width: 1.25; stroke-dasharray: 5 4; }
.tk { font: 10.5px var(--mono); fill: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); }
.legend i { display: inline-block; width: 16px; height: 4px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: top; }
thead th { color: var(--muted); font-weight: 500; }
tbody th { font-weight: 500; }
.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; } thead .n { font-family: var(--body); }
small { color: var(--muted); }
</style>
<main>
<header style="display:grid;gap:8px">
  <h1>放行错误率目标下的验证结果</h1>
  <p class="meta">GPQA、MATH-500、QuantumKatas 三个数据集，三个 generator。约束改为 FPR 的版本见另一页。</p>
</header>

<section>
  <h2>1. 任务和数据</h2>
  <p>每道题，generator 给出一个固定答案，并报告它认为这个答案对的概率 p_correct。系统要决定放行这个答案还是弃权。决定之前，可以花钱请 verifier 给这个答案打分。目标是在错误和花费都受控的前提下，放行尽量多的答案。</p>
  <div class="scroll"><table><thead><tr><th>数据集</th><th>题目</th><th>怎么判对</th><th>训练 / 测试</th></tr></thead><tbody>
    <tr><th scope="row">GPQA</th><td>445 道研究生水平的四选一科学题</td><td>和标准答案比对</td><td>249 道 calibration 题（Main 去掉 Diamond）用来选参数，196 道 Diamond 题测试</td></tr>
    <tr><th scope="row">MATH-500</th><td>500 道竞赛数学题</td><td>固定版本的 math_equal 判等价</td><td>5 折：每折用其余 4 折选参数，测试结果合并</td></tr>
    <tr><th scope="row">QuantumKatas</th><td>322 道量子编程题（Qiskit）</td><td>官方测试程序</td><td>5 折，同上</td></tr>
  </tbody></table></div>
  <h3>Generator</h3>
  <p class="meta">表里是 generator 的固定答案在全部题目上的正确率，也就是不做筛选、全部放行时的正确率。没给答案的题算错。Gemini 3.8 Flash 没有用作 generator，因为它和 Gemini 系的 verifier 太相似。</p>
  <div class="scroll"><table><thead><tr><th>Generator</th><th>模型</th><th class="n">GPQA</th><th class="n">MATH-500</th><th class="n">QuantumKatas</th></tr></thead><tbody>
    <tr><th scope="row">Qwen3</th><td>qwen3-235b-a22b-instruct-2507</td><td class="n">63.4%</td><td class="n">96.8%</td><td class="n">54.7%</td></tr>
    <tr><th scope="row">gpt-oss</th><td>gpt-oss-120b（reasoning medium）</td><td class="n">66.5%</td><td class="n">79.8%</td><td class="n">60.6%</td></tr>
    <tr><th scope="row">Sonnet 5.5</th><td>claude-sonnet-5-5</td><td class="n">89.2%</td><td class="n">87.4%</td><td class="n">89.1%</td></tr>
  </tbody></table></div>
  <h3>Verifier</h3>
  <p class="meta">5 个 verifier 来自三家公司。每个都给出「这个答案是对的」的概率，按 0.2 一档分成 5 档使用。表里是每次调用的平均花费（美元）。</p>
  <div class="scroll"><table><thead><tr><th>名称</th><th>模型</th><th class="n">GPQA</th><th class="n">MATH-500</th><th class="n">QuantumKatas</th></tr></thead><tbody>
    <tr><th scope="row">mistral_small</th><td>mistral-small-2503</td><td class="n">$0.00004</td><td class="n">$0.00003</td><td class="n">$0.00011</td></tr>
    <tr><th scope="row">mistral_medium</th><td>mistral-medium-3</td><td class="n">$0.00015</td><td class="n">$0.00012</td><td class="n">$0.00043</td></tr>
    <tr><th scope="row">gemini_lite</th><td>gemini-3.5-flash-lite</td><td class="n">$0.0018</td><td class="n">$0.00066</td><td class="n">$0.00034</td></tr>
    <tr><th scope="row">gemini_flash</th><td>gemini-3.7-flash</td><td class="n">$0.0019</td><td class="n">$0.0012</td><td class="n">$0.0022</td></tr>
    <tr><th scope="row">claude_haiku</th><td>claude-haiku-4-5</td><td class="n">$0.0028</td><td class="n">$0.00047</td><td class="n">$0.0014</td></tr>
  </tbody></table></div>
  <h3>怎么选 verifier</h3>
  <p>5 个是候选池，每道题最多问其中 3 个，按固定顺序问。对每个 generator（MATH-500 和 QuantumKatas 是每一折），在训练部分试遍全部 60 种「选哪 3 个、按什么顺序」，挑在整个 α、预算网格上平均放行最多的那一组，测试时固定不变。这个标准同时考虑准不准和贵不贵，所以便宜的 mistral_small 常被放在前面。GPQA 选出的是：Qwen3 用 mistral_small → mistral_medium → gemini_flash；gpt-oss 用 gemini_flash → claude_haiku → mistral_medium；Sonnet 用 gemini_flash → mistral_small → claude_haiku。另外两个数据集每折选的略有不同。</p>
</section>

<section>
  <h2>2. 为什么校准、怎么校准</h2>
  <p><b>为什么：</b>generator 自报的置信度不准，而且偏差方向因模型而异。比如 GPQA 上 Qwen3 报 95% 以上的 120 题实际只有 78% 对，Sonnet 则偏保守。我们的方法从这个置信度出发，用 verifier 分数做贝叶斯更新，起点必须是一个真实的概率，更新才有意义。</p>
  <p><b>怎么做：</b>只用 p_correct 一个输入，在训练部分的带标签题目上拟合一个单调函数（isotonic 回归，等价于最小化 Brier 损失），把报的置信度映射成实际正确率，拟合完就固定。测试部分不参与拟合。校准前后的对比见可靠性图那一页。</p>
  <p><b>要说明的是：</b>因为放行门槛最后也是用训练标签选的，会自动补偿置信度的偏差，所以校准对最终放行数的影响不大。表里「原始 p_correct」那一行是不校准的对照。</p>
</section>

<section>
  <h2>3. 我们的方法</h2>
  <p>每道题从校准后的置信度出发，按选定的顺序逐个问 verifier，每问一个就用贝叶斯公式更新「答案是对的」的概率，并决定现在放行、现在弃权，还是再问下一个。这个决定由动态规划算出，它权衡继续问的花费和可能带来的信息。几条这样的规则按比例混合使用，混合比例由线性规划决定，目标是放行最多，约束是放行错误率 ≤ α、平均花费 ≤ B。规则满不满足约束，是在训练部分用真实标签、样本外地数出来的。</p>
</section>

<section>
  <h2>4. Baseline，以及各自用来说明什么</h2>
  <div class="scroll"><table><thead><tr><th>方法</th><th>做法</th><th>和它比，说明我们的什么</th></tr></thead><tbody>
    <tr><th scope="row">只看置信度</th><td>只用校准后的置信度设门槛，不问 verifier</td><td>verifier 本身值不值得花钱</td></tr>
    <tr><th scope="row">固定问 1 / 2 / 3 个</th><td>和我们相同的更新和放行规则，但每题都问满，不提前停</td><td>自适应地决定问不问、问几个，能省多少钱、多放多少</td></tr>
    <tr><th scope="row">Weaver（3 个 / 5 个）</th><td>不看标签，用 EM 估计各 verifier 的可靠度，再合并成后验</td><td>用标签估计 verifier 可靠度的价值</td></tr>
    <tr><th scope="row">Self-consistency 投票</th><td>多采 10 个答案投票，用多数答案的平均置信度设门槛</td><td>同样的花费，花在 verifier 上是否比多采样更划算</td></tr>
    <tr><th scope="row">SSV</th><td>一个弱 verifier 加一个强 verifier，在线调整接受和拒绝两个阈值</td><td>和已有的弱强级联验证方法比</td></tr>
  </tbody></table></div>
  <p class="meta">所有方法都走同样的流程：在训练部分用真实标签选参数，满足同样的 α 和预算，然后到测试部分评估。规则是随机混合的，放行数是期望值，会有小数。Self-consistency 的 10 次采样计入花费；QuantumKatas 的代码没法投票，所以没有这一项。</p>
</section>

<section>
  <h2>5. 怎么判断方法好坏</h2>
  <p>在同一个 α 和预算下比较：<b>放行比例越高越好</b>，前提是测试集上的放行错误率不超过 α（下面错误率图的虚线），平均花费不超过预算。放行比例相同的话，错误率更低的更好。不同 α 之间不能直接比，α 越大本来就放得越多。</p>
</section>

<section>
  <h2>6. 结果</h2>
  <ul>
    <li><b>放行更多：</b>三个数据集上，我们在绝大多数设定下放行显著更多。75 格里，没有一个 baseline 显著多于我们超过 5 格（见下表）。</li>
    <li><b>错误率：</b>α ≥ 5% 时，我们的放行错误率基本贴着目标，比如 α = 5% 时 QuantumKatas 平均 5.5% 到 6.9%，GPQA 1.6% 到 5.7%。α = 1% 到 2% 时放行很少，几道错题就能让错误率跳很高，这一段不稳定。</li>
    <li><b>全部放行也可能是最优：</b>generator 本身的正确率已经高于 1 − α 时就会这样。比如 MATH-500 的 Qwen3 整体 96.8% 对，α ≥ 5% 时我们和只看置信度一样放行 495 题，verifier 用不上。</li>
    <li><b>GPQA 的 Qwen3 在 α = 5% 时几乎不放行（4 题），这是数据决定的。</b>在 249 道 calibration 题上，即使用全部 5 个 verifier 更新后再按概率从高到低排，排在最前面的 20 题也有 15% 是错的，前 50 题有 8% 是错的，找不到一段能做到 95% 正确的前排。gpt-oss 的前 105 题就能做到。原因是 Qwen3 的置信度区分不开对错，校准后最高一档也只有 84% 对；verifier 之间又常常一起错。两者叠加，最有把握的那批答案里也混着错题。所以算法在训练数据上看不到能达标的放行方式，就选择弃权。α = 10% 时它能放行 111 题。</li>
  </ul>
  {{SUM}}
  <p class="meta">每个方法有 75 格（3 个 generator × 5 个 α × 5 个预算）。「我们放行显著更多」是其中我们的放行比例显著高于它的格数，「它放行显著更多」反过来，按配对 bootstrap 区间是否不含 0 判断，剩下的格数是差别不显著。「放行错误率超 α」是它自己在测试上超过目标的格数。</p>
</section>
<div class="legend">{{LEGEND}}<span>虚线：FPR = α</span></div>
<section>
  <h2>6.1 放行比例对 α</h2>
  <p class="meta">预算 B 取固定问 3 个花费的一半。横轴 α 均匀排列，不是线性刻度。</p>
  {{PUB}}
</section>
<section>
  <h2>6.2 测试放行错误率对 α</h2>
  <p class="meta">同样的预算。点在虚线以上表示超过目标。放行不到 1 题的点不画。纵轴上限 40%。</p>
  {{FPR}}
</section>
<section>
  <h2>6.3 测试 FPR 对 α（参考）</h2>
  <p class="meta">错答案里有多少被放行了。这一版不约束它，只作参考。</p>
  {{ERR}}
</section>
<section>
  <h2>6.4 放行比例对预算</h2>
  <p class="meta">α = 5%。横轴是预算占固定问 3 个花费的比例。</p>
  {{BUD}}
</section>
<section>
  <h2>附录：各方法的数字</h2>
  <p class="meta">预算同上。格子是放行 / 错，* 表示放行错误率超过 α。</p>
  {{TAB}}
</section>
</main>
'''

if __name__ == '__main__':
    main()
