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
FDR = 'fdr' in sys.argv[1:]
# single: one deployed rule per method (main results); the mixture runs move to an appendix table.
SINGLE = False  # main results follow Algorithm 2 (mixture); single-rule runs are an appendix
SUFFIX = ('_fdr' if FDR else '')+('_single' if SINGLE else '')
OUT = Path(f'reports/gpqa_fpr_routing_20261102/results{"_fdr" if FDR else ""}.html')
MAIN_B = -1 if SINGLE else 3
DATASETS = [('gpqa', 'baselines_gpqa'+SUFFIX if FDR else 'baselines', 'GPQA', '在 249 道 calibration 题上选参数，在 196 道 Diamond 题上评估。'),
            ('math500', 'baselines_math500'+SUFFIX, 'MATH-500', '5 折：每折用其余 4 折选参数，5 折测试结果合并，共 500 题。'),
            ('qkatas', 'baselines_qkatas'+SUFFIX, 'QuantumKatas', '5 折：同上，共 322 题。')]
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


def tradeoff(gen, c, alphas, b):
    """Release rate against the realized error among released answers; one polyline per method across alpha."""
    xmax = .25
    pw, ph = W-L-R, H-T-Bt
    sx = lambda v: L+min(v, xmax)/xmax*pw
    sy = lambda v: T+ph-v*ph
    g = []
    for t in (0, .25, .5, .75, 1.):
        g.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        g.append(f'<text class="tk" x="{L-6}" y="{sy(t)+4:.1f}" text-anchor="end">{t:.0%}</text>')
    for t in (0, .05, .1, .15, .2, .25):
        g.append(f'<text class="tk" x="{sx(t):.1f}" y="{T+ph+16}" text-anchor="middle">{t:.0%}</text>')
    for key, name, slot in reversed(LINES):
        pts = []
        for a in alphas:
            x = c[f'{a}|{b}'].get(key)
            if x is not None:
                pts.append((a, rel_err(x) or 0., x['pub']))
        if not pts:
            continue
        w = 3 if key == 'ours' else 1.5
        g.append(f'<polyline points="{" ".join(f"{sx(e):.1f},{sy(p):.1f}" for _, e, p in pts)}" fill="none" stroke="var(--s{slot})" stroke-width="{w}"/>')
        for a, e, p in pts:
            tip = html.escape(f'{name}，α = {a:.0%}：放行 {p:.1%}，实际错误率 {e:.1%}')
            g.append(f'<circle cx="{sx(e):.1f}" cy="{sy(p):.1f}" r="{4.5 if key == "ours" else 3.5}" fill="var(--s{slot})" stroke="var(--surface)" stroke-width="1.5"><title>{tip}</title></circle>')
    return f'<figure><figcaption>{GEN[gen]}</figcaption><svg viewBox="0 0 {W} {H}" role="img" aria-label="{GEN[gen]} 放行比例对实际错误率">{"".join(g)}</svg></figure>'


def row(d, kind):
    alphas, budgets = d['alphas'], d['budgets']
    out = ''
    for gen in ('qwen3', 'gptoss', 'sonnet55'):
        c = d['generators'][gen]
        get = lambda a, b, m, f: c[f'{a}|{b}'][m][f] if m in c[f'{a}|{b}'] else None
        if kind == 'trade':
            out += tradeoff(gen, c, alphas, budgets[-1])
            continue
        if kind == 'pub':
            s = [(m, n, k, [get(a, budgets[MAIN_B], m, 'pub') for a in alphas]) for m, n, k in LINES]
            out += chart(GEN[gen], [f'{a:.0%}' for a in alphas], 1., (0, .25, .5, .75, 1.), s)
        elif kind in ('target', 'other'):
            on_target = kind == 'target'
            f = target if on_target else (lambda x: x['fpr'] if FDR else rel_err(x))
            val = lambda a, m: f(c[f'{a}|{budgets[MAIN_B]}'][m]) if m in c[f'{a}|{budgets[MAIN_B]}'] else None
            s = [(m, n, k, [val(a, m) for a in alphas]) for m, n, k in LINES]
            out += chart(GEN[gen], [f'{a:.0%}' for a in alphas], .4, (0, .1, .2, .3, .4), s, diagonal=list(alphas) if on_target else None)
        else:
            s = [(m, n, k, [get(.05, b, m, 'pub') for b in budgets]) for m, n, k in LINES]
            out += chart(GEN[gen], list(SHARE)+(['不限'] if len(budgets) > len(SHARE) else []), 1., (0, .25, .5, .75, 1.), s)
    return f'<div class="row">{out}</div>'


def table(d, alphas=(.05, .1), kind='count'):
    """kind='rate': release rate and error among released; kind='count': released / wrong."""
    b = d['budgets'][MAIN_B]
    gens = ('qwen3', 'gptoss', 'sonnet55')
    head = ''.join(''.join(f'<th class="n">{GEN[g]} α={a:.0%}</th>' for a in alphas) for g in gens)
    body = ''
    for m, name in TABLE:
        cells = ''
        for g in gens:
            for a in alphas:
                x = d['generators'][g][f'{a}|{b}'].get(m)
                star = '*' if x is not None and (target(x) or 0) > a else ''
                if x is None:
                    cells += '<td class="n">—</td>'
                elif kind == 'rate':
                    e = rel_err(x)
                    cells += f'<td class="n">{x["pub"]:.1%}<br><small>错误率 {"—" if e is None else f"{e:.1%}"}{star}</small></td>'
                else:
                    cells += f'<td class="n">{x["released"]:.1f} / {x["wrong_released"]:.1f}{star}</td>'
        body += f'<tr><th scope="row">{name}</th>{cells}</tr>'
    return f'<div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def budget_table(d, a):
    """Release rate at one alpha across the budget grid; * marks error among released above alpha."""
    gens = ('qwen3', 'gptoss', 'sonnet55')
    money = lambda b: f'{b*1e4:.2g}' if b < 1 else '不限'
    labels = [money(b) for b in d['budgets']]
    head1 = ''.join(f'<th class="n" colspan="{len(labels)}">{GEN[g]}</th>' for g in gens)
    head2 = ''.join(f'<th class="n">{l}</th>' for _ in gens for l in labels)
    body = ''
    for m, name in TABLE:
        cells = ''
        for g in gens:
            for b in d['budgets']:
                x = d['generators'][g][f'{a}|{b}'].get(m)
                cells += '<td class="n">—</td>' if x is None else (
                    f'<td class="n">放行 {x["pub"]:.0%}{"*" if (target(x) or 0) > a else ""}<br><small>花费 {x["usd"]*1e4:.2g}</small></td>')
        body += f'<tr><th scope="row">{name}</th>{cells}</tr>'
    return f'<div class="scroll"><table><thead><tr><th></th>{head1}</tr><tr><th>每题预算（×10⁻⁴ 美元）</th>{head2}</tr></thead><tbody>{body}</tbody></table></div>'


def counts(d, only_main=False):
    tot, over, more, less = {}, {}, {}, {}
    for g, c in d['generators'].items():
        for k, cell in c.items():
            if only_main and float(k.split('|')[1]) != d['budgets'][MAIN_B]:
                continue
            a = float(k.split('|')[0])
            for m, x in cell.items():
                tot[m] = tot.get(m, 0)+1
                over[m] = over.get(m, 0)+((target(x) or 0) > a)
                q = x.get('ours_minus_pub')
                if q:
                    more[m] = more.get(m, 0)+(q['ci95'][0] > 0)
                    less[m] = less.get(m, 0)+(q['ci95'][1] < 0)
    return tot, over, more, less


def summary(res, only_main=False):
    rows = ''
    for m, name in TABLE:
        cells = ''
        for ds, _, _, _ in DATASETS:
            tot, over, more, less = counts(res[ds], only_main)
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
    trade = ''.join(f'<h3>{n}</h3>{row(res[ds], "trade")}' for ds, _, n, _ in DATASETS)
    bud = ''.join(f'<h3>{n}</h3>{row(res[ds], "bud")}' for ds, _, n, _ in DATASETS)
    al = lambda ds: (.01, .02, .05) if FDR and ds == 'math500' else (.05, .1)
    tab = ''.join(f'<h3>{n}</h3>{table(res[ds], al(ds), "rate")}' for ds, _, n, _ in DATASETS)
    cnt = ''.join(f'<h3>{n}</h3>{table(res[ds], al(ds), "count")}' for ds, _, n, _ in DATASETS)
    if FDR:
        bud = ''.join(f'<h3>{n}</h3>{budget_table(res[ds], .02 if ds == "math500" else .05)}' for ds, _, n, _ in DATASETS)
    legend = ''.join(f'<span><i style="background:var(--s{k})"></i>{n}</span>' for _, n, k in LINES)
    mix = ''
    if FDR:
        rm = {ds: json.loads((SRC/f'{f}_single.json').read_text()) for ds, f, _, _ in DATASETS}
        MAIN = globals()['MAIN_B']
        globals()['MAIN_B'] = -1
        mix = summary(rm, True)
        globals()['MAIN_B'] = MAIN
    OUT.parent.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE_FDR if FDR else TEMPLATE
    OUT.write_text(page.replace('{{PUB}}', pub).replace('{{FPR}}', fpr).replace('{{BUD}}', bud).replace('{{ERR}}', err).replace('{{TAB}}', tab).replace('{{CNT}}', cnt)
                   .replace('{{SUM}}', summary(res, SINGLE)).replace('{{MIX}}', mix).replace('{{TRADE}}', trade).replace('{{LEGEND}}', legend))
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
  <p class="meta">5 个 verifier 来自三家公司。每个都给出「这个答案是对的」的概率，按 0.2 一档分成 5 档使用。表里是每次调用的平均花费，单位 10⁻⁴ 美元。</p>
  <div class="scroll"><table><thead><tr><th>名称</th><th>模型</th><th class="n">GPQA</th><th class="n">MATH-500</th><th class="n">QuantumKatas</th></tr></thead><tbody>
    <tr><th scope="row">mistral_small</th><td>mistral-small-2503</td><td class="n">0.4</td><td class="n">0.3</td><td class="n">1.1</td></tr>
    <tr><th scope="row">mistral_medium</th><td>mistral-medium-3</td><td class="n">1.5</td><td class="n">1.2</td><td class="n">4.3</td></tr>
    <tr><th scope="row">gemini_lite</th><td>gemini-3.5-flash-lite</td><td class="n">18</td><td class="n">6.6</td><td class="n">3.4</td></tr>
    <tr><th scope="row">gemini_flash</th><td>gemini-3.7-flash</td><td class="n">19</td><td class="n">12</td><td class="n">22</td></tr>
    <tr><th scope="row">claude_haiku</th><td>claude-haiku-4-5</td><td class="n">28</td><td class="n">4.7</td><td class="n">14</td></tr>
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
  <p><b>为什么要混合，以及表里的小数：</b>单独一条规则往往卡不准 α，比如严一点的规则错误率 4%、松一点的 6%。按比例混着用（比如 60% 用严的、40% 用松的），平均下来正好卡在 α，能多放一些题。部署时每道题按这个比例抽签决定用哪条规则，所以同一道题是否放行带有随机性。测试集上我们没有真的抽签，而是把每道题被放行的概率加起来，也就是反复抽签的平均结果。所以表里的放行数和错误数都是期望值，会有小数。混合还有一个作用：花费高的做法可以只用在一部分题上，其余题弃权，从而满足预算。只用一条规则、结果为整数的版本见附录。</p>
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
    <li><b>同样的实际错误率下，我们放得更多或持平（见 6.1）。</b>比如 QuantumKatas 的 Qwen3，实际错误率约 5% 时我们放行 37%，固定问 3 个放行 26%；约 11% 时是 48% 对 34%。GPQA 的 gpt-oss 在实际错误率 8.7% 时，我们放行 66%，固定问 3 个 57%。GPQA 的 Qwen3 两者基本重合。所以多放的部分不是靠多犯错换来的。</li>
    <li><b>放行更多：</b>三个数据集上，我们在绝大多数设定下放行显著更多。75 格里，没有一个 baseline 显著多于我们超过 5 格（见附录的汇总表）。</li>
    <li><b>错误率：</b>α ≥ 5% 时，我们的放行错误率基本贴着目标，比如 α = 5% 时 QuantumKatas 平均 5.5% 到 6.9%，GPQA 1.6% 到 5.7%。α = 1% 到 2% 时放行很少，几道错题就能让错误率跳很高，这一段不稳定。</li>
    <li><b>MATH-500 要看小 α：</b>Qwen3 和 gpt-oss 给出答案的题已经超过 95% 对，α ≥ 5% 时直接全部放行，方法之间没有差别，所以 MATH-500 的表改成 α = 1%、2%、5%。在 α = 1% 和 2% 时，我们在三个 generator 上都比所有 baseline 放行显著更多，只有 gpt-oss 在 α = 2% 时 SSV 更多（187 对 154）。但这一段错题容许得很少，放行错误率偏高：gpt-oss 在 α = 1% 时是 7.6%，Qwen3 是 3.5%，Sonnet 是 1.3%。</li>
    <li><b>全部放行也可能是最优：</b>generator 本身的正确率已经高于 1 − α 时就会这样。比如 MATH-500 的 Qwen3 整体 96.8% 对，α ≥ 5% 时我们和只看置信度一样放行 495 题，verifier 用不上。</li>
    <li><b>GPQA 的 Qwen3 在 α = 5% 时几乎不放行（4 题），这是数据决定的。</b>在 249 道 calibration 题上，即使用全部 5 个 verifier 更新后再按概率从高到低排，排在最前面的 20 题也有 15% 是错的，前 50 题有 8% 是错的，找不到一段能做到 95% 正确的前排。gpt-oss 的前 105 题就能做到。原因是 Qwen3 的置信度区分不开对错，校准后最高一档也只有 84% 对；verifier 之间又常常一起错。两者叠加，最有把握的那批答案里也混着错题。所以算法在训练数据上看不到能达标的放行方式，就选择弃权。α = 10% 时它能放行 111 题。</li>
  </ul>
</section>
<div class="legend">{{LEGEND}}<span>虚线：实际错误率 = 目标 α</span></div>
<section>
  <h2>6.1 同样的实际错误率下谁放得多</h2>
  <p class="meta">横轴是测试集上放行答案的实际错误率，纵轴是放行比例，每个方法的 5 个点对应 α = 1%、2%、5%、10%、20%，连成一条线。线越靠左上越好：同样的实际错误率下放得更多。这张图不看 α 定了多少，只看实际结果，所以「放得多是不是只是因为错得多」可以直接看出来。预算取最高一档（每题问 3 个 verifier 的平均花费），尽量减少预算对 baseline 的限制。放行 0 题的点画在原点，横轴超过 25% 的点画在右边界。</p>
  {{TRADE}}
</section>
<section>
  <h2>6.2 放行比例对 α</h2>
  <p class="meta">预算 B 取固定问 3 个花费的一半。横轴 α 均匀排列，不是线性刻度。</p>
  {{PUB}}
</section>
<section>
  <h2>6.3 测试放行错误率对 α</h2>
  <p class="meta">同样的预算。点在虚线以上表示超过目标。放行不到 1 题的点不画。纵轴上限 40%。</p>
  {{FPR}}
</section>
<section>
  <h2>6.4 测试 FPR 对 α（参考）</h2>
  <p class="meta">错答案里有多少被放行了。这一版不约束它，只作参考。</p>
  {{ERR}}
</section>
<section>
  <h2>6.5 放行比例对预算</h2>
  <p class="meta">表里是测试集上的放行比例，* 表示放行错误率超过 α。α 取 5%；MATH-500 取 2%，因为 5% 时大多数方法都全部放行。金额单位都是 10⁻⁴ 美元。表头是每题平均最多能花的钱；每格上面是放行比例，下面是这个方法在测试集上实际每题平均花了多少钱。作为参照，GPQA 上问一次 verifier 约 0.4（mistral_small）到 28（claude_haiku）。花费固定的 baseline 在低预算下只能对一部分题问 verifier，其余弃权，所以放得少。</p>
  {{BUD}}
</section>
<section>
  <h2>6.6 各方法的放行比例和放行错误率</h2>
  <p class="meta">每格上面是放行比例（放行数 / 全部题数），下面是放行错误率（放行里错的比例），* 表示超过 α。MATH-500 用 α = 1%、2%、5%，原因见第 6 节。</p>
  {{TAB}}
</section>
<section>
  <h2>6.7 各方法的放行数和错误数</h2>
  <p class="meta">和 6.6 对应的具体数字：放行数 / 其中错的数。规则是按比例混合的，这些是期望值，所以有小数。</p>
  {{CNT}}
</section>
<section>
  <h2>附录：汇总表</h2>
  <p class="meta">每个 baseline 和我们比 75 次（3 个 generator × 5 个 α × 5 个预算）。「我们放行显著更多 / 它放行显著更多」是其中我们的放行比例显著高于它、它显著高于我们的次数，按配对 bootstrap 区间是否不含 0 判断，剩下的是差别不显著。「放行错误率超 α」是这个方法自己在测试集上超过目标的次数。</p>
  {{SUM}}
</section>
<section>
  <h2>附录：每个方法只用一条规则的结果</h2>
  <p class="meta">不混合，训练后只保留满足约束、放行最多的那一条规则，每道题结果确定，放行数和错误数是整数。这时花费固定的 baseline（比如每题都问满 3 个）一旦超出预算就只能全部弃权，所以这里只在「不限预算」下汇总，每个数据集每个方法 15 格（3 个 generator × 5 个 α）。不限预算时，我们和问满 verifier 的方法放行数相近，但每题花费低得多。</p>
  {{MIX}}
</section>
</main>
'''

if __name__ == '__main__':
    main()
