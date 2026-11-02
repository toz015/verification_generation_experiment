"""Reliability of generator confidence on the 196 GPQA test items, before and after r_phi (no API calls).

Page: reports/gpqa_reporting_trial_20261102/reliability.html. r_phi is the isotonic fit of
reporting_trial, on the 249 calibration items only. Each point is one confidence bin: mean stated
probability against the observed share correct, with a Wilson 95% interval.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

import vgx.baselines as B
from vgx.gpqa.reporting_trial import isotonic, recalibrated

OUT = Path('reports/gpqa_reporting_trial_20261102')
EDGES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001)
LABEL = {'qwen3': 'Qwen3-235B', 'gptoss': 'gpt-oss-120b', 'sonnet55': 'Sonnet 5.5'}
SERIES = {'raw': ('原始 p_correct', 'var(--raw)'), 'calibrated': ('校准后 r_φ', 'var(--cal)')}


def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    c, h = (p+z*z/(2*n))/d, z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0., c-h), min(1., c+h)


def reliability(rows, idx):
    v = [i for i in idx if rows[i]['valid']]
    p = np.array([rows[i]['prior'] for i in v]); y = np.array([rows[i]['outcome'] for i in v], float)
    b = np.digitize(p, EDGES)-1
    pts = []
    for k in range(len(EDGES)-1):
        m = b == k
        if m.any():
            lo, hi = wilson(float(y[m].sum()), int(m.sum()))
            pts.append({'bin': k, 'conf': float(p[m].mean()), 'acc': float(y[m].mean()), 'n': int(m.sum()), 'lo': lo, 'hi': hi})
    return {'points': pts, 'ece': sum(q['n']*abs(q['acc']-q['conf']) for q in pts)/len(v),
            'brier': float(((p-y)**2).mean()), 'n': len(v)}


def panel(gen, res):
    W, H, L, B_, T, R = 340, 330, 44, 40, 10, 12
    pw, ph = W-L-R, H-T-B_
    sx = lambda v: L+v*pw
    sy = lambda v: T+ph-v*ph
    g = []
    for t in (0, .25, .5, .75, 1):
        g.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        g.append(f'<text class="tk" x="{L-7}" y="{sy(t)+4:.1f}" text-anchor="end">{int(t*100)}%</text>')
        g.append(f'<text class="tk" x="{sx(t):.1f}" y="{T+ph+17}" text-anchor="middle">{int(t*100)}%</text>')
    g.append(f'<line class="ideal" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}"/>')
    g.append(f'<text class="ax" x="{L+pw/2}" y="{H-4}" text-anchor="middle">模型给出的置信度</text>')
    g.append(f'<text class="ax" transform="translate(11 {T+ph/2}) rotate(-90)" text-anchor="middle">实际答对比例</text>')
    for key, (name, color) in SERIES.items():
        pts = res[key]['points']
        d = ' '.join(f'{sx(p["conf"]):.1f},{sy(p["acc"]):.1f}' for p in pts)
        g.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for p in pts:
            x, y = sx(p['conf']), sy(p['acc'])
            r = 4+math.sqrt(p['n'])*.55
            g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(p["lo"]):.1f}" y2="{sy(p["hi"]):.1f}" stroke="{color}" stroke-width="1.5" opacity=".45"/>')
            tip = f'{name}：平均置信度 {p["conf"]:.0%}，实际答对 {p["acc"]:.0%}，{p["n"]} 题，区间 {p["lo"]:.0%} 到 {p["hi"]:.0%}'
            g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="var(--surface)" stroke-width="2"><title>{html.escape(tip)}</title></circle>')
    stat = ''.join(f'<span><i style="background:{c}"></i>{n}：ECE {res[k]["ece"]:.3f}，Brier {res[k]["brier"]:.3f}</span>' for k, (n, c) in SERIES.items())
    return (f'<figure><figcaption>{LABEL[gen]}</figcaption><div class="stat">{stat}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{LABEL[gen]} 校准前后的可靠性图">{"".join(g)}</svg></figure>')


def rows_html(results):
    out = ''
    for gen, res in results.items():
        for key, (name, _) in SERIES.items():
            for p in res[key]['points']:
                out += (f'<tr><td>{LABEL[gen]}</td><td>{name}</td><td class="n">{p["n"]}</td><td class="n">{p["conf"]:.1%}</td>'
                        f'<td class="n">{p["acc"]:.1%}</td><td class="n">{p["lo"]:.1%} 到 {p["hi"]:.1%}</td></tr>')
    return out


def main():
    data, _ = B.load('gpqa')
    results = {}
    for gen, rows in data.items():
        (train, test), = B.splits('gpqa', rows)
        results[gen] = {'raw': reliability(rows, test), 'calibrated': reliability(recalibrated(rows, train, isotonic)[0], test)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'reliability.json').write_text(json.dumps(results, indent=1))
    (OUT/'reliability.html').write_text(TEMPLATE.replace('{{PANELS}}', ''.join(panel(g, r) for g, r in results.items()))
                                        .replace('{{ROWS}}', rows_html(results)))
    print('wrote', OUT/'reliability.html')


TEMPLATE = '''<title>GPQA 置信度校准</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column, then three equal small multiples that stack on a phone. */
:root {
  --bg: #f6f8f7; --surface: #ffffff; --fg: #1c2522; --muted: #5a6661; --line: #dbe1de;
  --raw: #B4581F; --cal: #2360A5;
  --display: "Noto Serif SC", "Songti SC", serif; --body: "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --raw: #DE8F55; --cal: #62A6EE; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --raw: #DE8F55; --cal: #62A6EE; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.65; }
main { max-width: 1120px; margin: 0 auto; padding-block: 36px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 28px; }
h1 { font-family: var(--display); font-size: 28px; line-height: 1.3; margin: 0; text-wrap: balance; }
h2 { font-size: 17px; margin: 0; }
.lede { color: var(--muted); max-width: 68ch; margin: 8px 0 0; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px; display: grid; gap: 8px; min-width: 0; }
figcaption { font-weight: 700; }
.stat { display: grid; gap: 2px; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.stat i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
svg { width: 100%; height: auto; display: block; }
svg circle { cursor: help; }
.grid { stroke: var(--line); stroke-width: 1; }
.ideal { stroke: var(--muted); stroke-width: 1.25; stroke-dasharray: 5 4; }
.tk { font: 11px var(--mono); fill: var(--muted); }
.ax { font: 12px var(--body); fill: var(--muted); }
ul { margin: 8px 0 0; padding-left: 20px; display: grid; gap: 6px; max-width: 76ch; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; }
.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
th.n { font-family: var(--body); }
</style>
<main>
<header>
  <h1>GPQA 置信度校准前后</h1>
  <p class="lede">196 道测试题。点落在虚线上，表示模型说多少就对多少。点在虚线下方是过度自信，在上方是过于保守。圆越大题越多，竖线是 95% 区间。r_φ 只用 249 道 calibration 题拟合。</p>
</header>
<section class="panels">{{PANELS}}</section>
<section>
  <h2>怎么读</h2>
  <ul>
    <li><b>Qwen3：</b>原始置信度严重过高。报 95% 以上的 120 题实际只有 78% 答对。校准后点贴近虚线，ECE 从 0.285 降到 0.062，但最高只到 84%。</li>
    <li><b>gpt-oss：</b>原始置信度在 90% 以下偏高，校准后 ECE 从 0.146 降到 0.081。低置信度那一组校准后反而偏低（平均 24%，实际 35%）。</li>
    <li><b>Sonnet 5.5：</b>原始置信度偏保守，校准后往上提，ECE 从 0.081 降到 0.029，Brier 不变。</li>
    <li><b>校准不增加信息：</b>它只修正数值。isotonic 的两端可能到 0 或 1，比如 Sonnet 有 2 道题校准后是 0%，实际都答对了。这两道题数太少，只能看作样本量的限制。</li>
    <li><b>ECE 的算法：</b>按 0–50%、50–70%、70–80%、80–90%、90–95%、95–100% 分组，对每组的 |实际 − 置信度| 按题数加权平均。分组方式会影响它的数值，所以同时给出 Brier。</li>
  </ul>
</section>
<section>
  <h2>各组数字</h2>
  <div class="scroll"><table><thead><tr><th>Generator</th><th>版本</th><th class="n">题数</th><th class="n">平均置信度</th><th class="n">实际答对</th><th class="n">95% 区间</th></tr></thead><tbody>{{ROWS}}</tbody></table></div>
</section>
</main>
'''

if __name__ == '__main__':
    main()
