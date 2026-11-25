"""Readable HTML page for results/baselines_20261027/{gpqa,math500,qkatas}.json (no computation of its own)."""
from __future__ import annotations

import html
import json
from pathlib import Path

SRC = Path('results/baselines_20261027')
OUT = Path('reports/baselines_20261027/results.html')
DATASETS = [('gpqa', 'GPQA', '在 249 道校准题上拟合，在 196 道 Diamond 题上评测'),
            ('math500', 'MATH-500', '500 题，5 折交叉拟合（按学科×难度分层）'),
            ('qkatas', 'QuantumKatas', '322 题，5 折交叉拟合（按 kata 分组）')]
GEN = {'qwen3': 'Qwen3-235B', 'gptoss': 'gpt-oss-120b', 'sonnet55': 'Sonnet 5.5'}
MAIN = [('generator_only', '只用 generator'), ('fixed_1', '固定问 1 个 verifier'), ('fixed_2', '固定问 2 个'),
        ('fixed_3', '固定问 3 个'), ('ours', '我们的方法'), ('sc_vote', 'Self-consistency（多数投票）'),
        ('weaver_3', 'Weaver（3 个 verifier）'), ('weaver_5', 'Weaver（5 个 verifier）'),
        ('ssv_model', 'SSV（强 verifier 用模型）')]
APPENDIX = [('fixed_5', '固定问 5 个'), ('self_consistency', 'Self-consistency（一致比例当分数）'),
            ('weaver_5_noprior', 'Weaver（5 个，不用 generator 置信度）'), ('weaver_5_fit', 'Weaver（5 个，不用置信度，训练部分选阈值）'),
            ('weaver_5_binary', 'Weaver（5 个，不用置信度，0.5 二值化）'),
            ('ssv_oracle', 'SSV（强 verifier 用真实标签）')]
CURVES = [('generator_only', '只用 generator', 'var(--c1)'), ('sc_vote', 'Self-consistency（多数投票）', 'var(--c2)'),
          ('weaver_5', 'Weaver（5 个）', 'var(--c3)')]


def num(x, d=2):
    return '—' if x is None else f'{x:.{d}f}'.replace('-', '−')


def cell(arms, diff, key):
    a = arms.get(key)
    if a is None:
        return '<td class="na">不适用</td>'
    cls = ''
    extra = f'<span class="q">oracle {a["oracle_queries_per_task"]:.2f}/题</span>' if 'oracle_queries_per_task' in a else ''
    return (f'<td class="c{cls}{" ours" if key == "ours" else ""}"><span class="rw">{a["released"]} / {a["wrong_released"]}</span>'
            f'<span class="u">{num(a["mean_utility"], 3)}</span><span class="usd">${a["expected_usd_per_task"]:.4f}</span>{extra}</td>')


def table(r, rows):
    gens = list(r['generators'])
    head = ''.join(f'<th>{GEN[g]}<span class="sub">{r["generators"][g]["correct_test"]} / {r["generators"][g]["n_test"]} 题答对</span></th>' for g in gens)
    body = ''
    for key, label in rows:
        body += f'<tr><th scope="row">{label}</th>' + ''.join(cell(r['generators'][g]['arms'], r['generators'][g]['ours_minus'], key) for g in gens) + '</tr>'
    return f'<div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def chart(v, gen):
    W, H, L, B, T, R = 300, 210, 40, 30, 12, 10
    pw, ph = W-L-R, H-T-B
    ymax = 0.3
    sx = lambda c: L + c*pw
    sy = lambda e: T + ph - min(e, ymax)/ymax*ph
    g = [f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" class="ax"/>', f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" class="ax"/>']
    for t in (0, .25, .5, .75, 1):
        g.append(f'<text x="{sx(t):.1f}" y="{T+ph+14}" class="tk" text-anchor="middle">{int(t*100)}%</text>')
    for e in (0, .1, .2, .3):
        g.append(f'<line x1="{L}" y1="{sy(e):.1f}" x2="{L+pw}" y2="{sy(e):.1f}" class="grid"/>')
        g.append(f'<text x="{L-5}" y="{sy(e)+3:.1f}" class="tk" text-anchor="end">{int(e*100)}%</text>')
    g.append(f'<line x1="{L}" y1="{sy(.05):.1f}" x2="{L+pw}" y2="{sy(.05):.1f}" class="target"/>')
    for key, _, color in CURVES:
        pts = v['risk_coverage'].get(key)
        if pts:
            d = ' '.join(f'{sx(c):.1f},{sy(e):.1f}' for c, e in pts)
            g.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    a = v['arms']['ours']
    if a['released']:
        g.append(f'<circle cx="{sx(a["coverage"]):.1f}" cy="{sy(a["output_error_rate"]):.1f}" r="4.5" fill="var(--accent)" stroke="var(--bg)" stroke-width="1.5"/>')
    return (f'<figure><figcaption>{GEN[gen]}</figcaption><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{GEN[gen]} 错误率对覆盖率">{"".join(g)}</svg></figure>')


def main():
    res = {ds: json.loads((SRC/f'{ds}.json').read_text()) for ds, _, _ in DATASETS}
    sections, appendix, charts = [], [], []
    for ds, name, split in DATASETS:
        r = res[ds]
        sections.append(f'<section id="{ds}"><h2>{name}</h2><p class="meta">{split}</p>{table(r, MAIN)}</section>')
        appendix.append(f'<h3>{name}</h3>{table(r, APPENDIX)}')
        charts.append(f'<h3>{name}</h3><div class="charts">' + ''.join(chart(v, g) for g, v in r['generators'].items()) + '</div>')
    legend = ''.join(f'<span><i style="background:{c}"></i>{html.escape(l)}</span>' for _, l, c in CURVES)
    page = TEMPLATE.replace('{{SECTIONS}}', ''.join(sections)).replace('{{APPENDIX}}', ''.join(appendix)) \
        .replace('{{CHARTS}}', ''.join(charts)).replace('{{LEGEND}}', legend)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print('wrote', OUT)


TEMPLATE = '''<title>Baseline 对比结果</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; wide result tables scroll inside their own box. */
:root {
  --bg: #f7f8f6; --surface: #ffffff; --fg: #1d2421; --muted: #5d6863; --line: #d9dfdb;
  --accent: #b5462b; --good-bg: #e3f1e6; --good-fg: #1f6b39; --bad-bg: #f8e3df; --bad-fg: #9a2f1c;
  --c1: #8a948f; --c2: #2f6f9f; --c3: #8b5fb0; --c4: #c08a1e;
  --display: "Noto Serif SC", "Songti SC", serif; --body: "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #98a59f; --line: #2f3935;
  --accent: #ec7a5c; --good-bg: #183323; --good-fg: #8fd6a4; --bad-bg: #3a1f19; --bad-fg: #f2a08b;
  --c1: #9aa5a0; --c2: #6fb0df; --c3: #b994dc; --c4: #e2b453; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #98a59f; --line: #2f3935;
  --accent: #ec7a5c; --good-bg: #183323; --good-fg: #8fd6a4; --bad-bg: #3a1f19; --bad-fg: #f2a08b;
  --c1: #9aa5a0; --c2: #6fb0df; --c3: #b994dc; --c4: #e2b453; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.65; }
main { max-width: 1040px; margin: 0 auto; padding-block: 40px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 36px; }
h1, h2 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; }
h1 { font-size: 30px; line-height: 1.3; }
h2 { font-size: 22px; }
h3 { font-size: 16px; margin: 18px 0 8px; }
p { margin: 0; max-width: 68ch; }
.lede { color: var(--muted); }
section { display: grid; gap: 12px; min-width: 0; }
.meta { color: var(--muted); font-size: 13px; }
.findings { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 18px 22px; }
.findings ul, .notes ul { margin: 8px 0 0; padding-left: 20px; display: grid; gap: 6px; max-width: 72ch; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
thead th { font-size: 13px; font-weight: 700; white-space: nowrap; }
thead .sub { display: block; font-weight: 400; color: var(--muted); font-size: 12px; }
tbody th { font-weight: 500; white-space: nowrap; font-size: 14px; }
tbody tr:last-child > * { border-bottom: 0; }
td.c { font-family: var(--mono); font-size: 13px; line-height: 1.45; white-space: nowrap; }
td.c span { display: block; }
td .rw { font-weight: 500; }
td .usd, td .q { color: var(--muted); font-size: 12px; }
td.ours { box-shadow: inset 3px 0 0 var(--accent); }
td.better { background: var(--good-bg); } td.better .u { color: var(--good-fg); font-weight: 500; }
td.worse { background: var(--bad-bg); } td.worse .u { color: var(--bad-fg); font-weight: 500; }
td.na { color: var(--muted); font-size: 13px; }
.key { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: var(--muted); }
.key b { font-family: var(--mono); font-weight: 500; color: var(--fg); }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -1px; margin-right: 5px; }
.charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px 6px; min-width: 0; }
figcaption { font-size: 13px; font-weight: 500; }
svg { width: 100%; height: auto; display: block; }
svg .ax { stroke: var(--muted); stroke-width: 1; } svg .grid { stroke: var(--line); stroke-width: 1; }
svg .target { stroke: var(--accent); stroke-width: 1; stroke-dasharray: 4 3; }
svg .tk { fill: var(--muted); font-size: 10px; font-family: var(--mono); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); }
.legend i { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 6px; }
.legend .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); }
.notes { font-size: 14px; }
.notes dl { margin: 8px 0 0; display: grid; gap: 4px; max-width: 76ch; }
.notes dt { font-weight: 700; margin-top: 10px; }
.notes dd { margin: 0; }
.notes .ex { margin: 4px 0 0; padding: 6px 10px; border-left: 3px solid var(--line); background: var(--surface); color: var(--muted); font-size: 13px; }
code { font-family: var(--mono); font-size: 13px; }
</style>
<main>
<header style="display:grid;gap:10px">
  <h1>Baseline 对比结果</h1>
  <p class="lede">除多数投票的 self-consistency 会自己选答案外，所有方法面对的都是同一件事：主实验里每个 generator 给出的那个固定答案，放行还是弃权。数据划分、目标正确率 95%（L = 19）、效用的算法都和我们的方法相同，凡是要拟合的东西都只用训练部分。</p>
</header>

<section class="notes">
  <h2>各个方法怎么做的</h2>
  <p class="meta">例子里的数字是示意，用来说明决策过程，不是某道真实题目。</p>
  <dl>
    <dt>只用 generator</dt>
    <dd>generator 自己报的 p_correct 不低于 95% 就放行，否则弃权，不问任何 verifier。
      <p class="ex">例：generator 答 B，报 p_correct = 0.97，放行。另一题报 0.90，弃权。</p></dd>
    <dt>固定问 k 个 verifier</dt>
    <dd>用和我们相同的后验更新和放行规则，但每题都固定问所选 3 个 verifier 里的前 k 个，不提前停止（附录 A 还有固定问全部 5 个）。
      <p class="ex">例：先验 0.90。k = 2 时，第 1 个 verifier 给高分，后验升到 0.97；规则上已经可以放行，但仍要问第 2 个并为它付钱，第 2 个也给高分，后验 0.99，放行。</p></dd>
    <dt>我们的方法</dt>
    <dd>同样的后验更新，但每问一个 verifier 之后都判断：现在放行、弃权，还是再问一个更划算。
      <p class="ex">例：同一题，第 1 个 verifier 给高分后后验 0.97，再问的期望收益抵不上花费，停下放行，省下后面两个 verifier 的钱。</p></dd>
    <dt>Self-consistency（多数投票）</dt>
    <dd>用 generator 的 10 个样本投票选出最终答案（GPQA 比字母，MATH 用同一个 grader 判等价；平票时选 p_correct 总和更高的那组）。置信度不改，取投给多数答案的那几个样本自报 p_correct 的平均值，再按「只用 generator」的规则放行。所以这一行的答案可能和固定答案不同，对错按投票出来的答案算。花费是这 10 次采样。代码没法投票，QuantumKatas 上不做。附录 A 的旧版本把「10 个样本里和固定答案一致的比例」当分数。
      <p class="ex">例：10 个样本里 7 个选 B、3 个选 C，答案取 B。这 7 个样本报的 p_correct 平均 0.96，放行；如果平均只有 0.92，弃权。</p></dd>
    <dt>Weaver</dt>
    <dd>Weaver 的弱监督标签模型：每个 verifier 的分数按我们方法的三档（0–1/3、1/3–2/3、2/3–1）离散化，假设各 verifier 在答案对或错时相互独立。各 verifier 在对、错两种情况下落在每档的概率不用标签，用 EM 在全部题上估计。和我们的方法一样，generator 报的 p_correct 是每题的先验，再用这些 verifier 的似然比做贝叶斯更新，后验 ≥ 95% 就放行。所以它和我们的区别只有两点：verifier 的似然不看标签估计（我们在训练部分用标签估），以及固定问 3 个或 5 个、不提前停止。附录 A 有不用 generator 置信度（先验取训练部分的正确率）的版本，以及它改用训练部分选阈值、按 0.5 二值化的版本。
      <p class="ex">例：generator 报 p_correct = 0.90，先验赔率 9。EM 估出 verifier A 给高分的似然比是 3，B 给中档的似然比是 0.8。两者都问过后，赔率 9 × 3 × 0.8 = 21.6，后验 0.956，≥ 95%，放行。</p></dd>
    <dt>SSV</dt>
    <dd>照搬论文代码：一个弱 verifier 加一个强 verifier，弱 verifier 的分数高于接受阈值就放行，低于拒绝阈值就弃权，中间的交给强 verifier 判。另外按一定概率抽查本来直接放行或弃权的题，用强 verifier 的结论在线调整两个阈值（α 设成 5%，其余参数用默认值）。先在训练部分热身，再在评测部分继续在线更新。弱 verifier 用训练部分上单独效果最好的那个；强 verifier 在主表里用其余 4 个中 AUROC 最高的那个，附录 A 用真实标签。
      <p class="ex">例：当前接受阈值 0.9、拒绝阈值 0.3。弱 verifier 给 0.85，在两者之间，于是问强 verifier；强 verifier 判对，放行，并按这次结果微调阈值。</p></dd>
    <dt>AT（selective auditing 的阈值更新）</dt>
    <dd>来自 Gibbs 等人的 selective auditing 论文：只看 generator 自报的 p_correct（记为 C），维护一个阈值 τ，从 0.5 开始。每来一道题，C &lt; τ 就一定送去核对；C ≥ τ 也以概率 γ 抽查。被核对的题会看到对错，按 τ ← τ + η(1 − π − α)·e / π 更新阈值（π 是这道题被核对的概率，e = 1 表示答错）：高置信度却被抽查出错时 τ 明显升高，低置信度的错题让 τ 略降。我们的版本在训练部分把这个流程跑一遍，然后把 τ 冻结；测试时不再抽查，C ≥ τ 就放行，否则弃权，不问 verifier。超参数（η、γ）在 GPQA 校准题上选定后固定，40 个配置 × 20 个随机种子，按目标风险在验证部分挑配置。
      <p class="ex">例：α = 0.1、η = 0.1、γ = 0.5，当前 τ = 0.80。一道题 C = 0.85 ≥ τ，被抽中核对（π = 0.5），发现答错，τ 变成 0.80 + 0.1 × (1 − 0.5 − 0.1) / 0.5 = 0.88。之后 C = 0.85 的题就不再放行。训练部分跑完冻结 τ = 0.88，测试时 C = 0.90 放行，C = 0.85 弃权。</p></dd>
    <dt>AT+Cal（论文的 Algorithm 1）</dt>
    <dd>在 AT 上加一层在线校准：先在 0 到 1 之间取 21 个阈值点 θ，每个点有一个权重 w，把 C 换成校准分数 S = Σ w·1{C ≥ θ}，再用 S 代替 C 走和 AT 一样的阈值和核对规则。每次核对看到对错后，按 S 和真实结果的差，用指数梯度更新各点的权重，让 S 更接近真实正确率。同样在训练部分跑完后冻结 τ 和权重，测试时 S ≥ τ 放行，否则弃权。160 个配置 × 20 个随机种子。
      <p class="ex">例：C = 0.80，权重里有 70% 落在不超过 0.80 的阈值点上，所以 S = 0.70；冻结后的 τ = 0.75，S &lt; τ，弃权。如果 C = 0.95，S = 0.90 ≥ τ，放行。</p></dd>

  </dl>
</section>

<section class="notes">
  <h2>各个指标怎么算的</h2>
  <p class="meta">离线多风险实验里新加的三种风险指标，和本页的效用一起说明。例子统一用：100 道题，80 道答对、20 道答错；某个方法放行 70 道，其中 4 道是错的。</p>
  <dl>
    <dt>放行率 p</dt>
    <dd>放行的题占全部题的比例。放行错误率和 FPR 两种约束下，这是要尽量提高的目标。
      <p class="ex">例：放行 70 道，p = 70 / 100 = 70%。</p></dd>
    <dt>错误放行 e</dt>
    <dd>放行了但答错的题占全部题的比例。FNR 约束下，这是要尽量降低的目标。
      <p class="ex">例：放行的 70 道里有 4 道错，e = 4 / 100 = 4%。</p></dd>
    <dt>放行错误率（e / p）</dt>
    <dd>放行的答案里，错的比例。约束是它不超过 α。允许的错误放行随放行量一起增加，所以放行 p 很高、本来大多正确的题，几乎不消耗额度。
      <p class="ex">例：4 / 70 = 5.7%。α = 0.05 时超标，α = 0.1 时达标。</p></dd>
    <dt>FPR（e / (1 − q)）</dt>
    <dd>全部错误答案里，被放行的比例（q 是正确答案的比例）。约束是它不超过 α。允许放出的错题是一个固定的数，取决于错题总数，和放行多少无关，所以要多放行就得把错题一道道挑出来。错题很少时（比如 MATH-500）这个比例很不稳，所以 MATH-500 不测 FPR。
      <p class="ex">例：20 道错题里放出 4 道，FPR = 4 / 20 = 20%。α = 0.1 时最多只能放出 2 道错题。</p></dd>
    <dt>FNR（m / q）</dt>
    <dd>全部正确答案里，被拦下的比例（m 是被拦下的正确答案占全部题的比例）。约束是它不超过 α，目标改成错误放行 e 越低越好。正确答案占多数时，这个约束只允许拦下 p 最低的一小段，各方法都要放行大部分题，区分度低，所以只作说明，不作比较。
      <p class="ex">例：80 道正确答案里放行了 66 道、拦下 14 道，FNR = 14 / 80 = 17.5%。α = 0.05 时最多只能拦下 4 道正确答案。</p></dd>
    <dt>效用（本页各表使用）</dt>
    <dd>按题平均：答对放行 +1，答错放行 −19，弃权 0，再减去 10 × 每题花费（美元）。−19 对应目标正确率 95%（L = 19）。
      <p class="ex">例：66 道对、4 道错、花费每题 $0.002：(66 − 19 × 4) / 100 − 10 × 0.002 = −0.12。</p></dd>
  </dl>
</section>

<div class="findings">
  <strong>主要结论</strong>
  <ul>
    <li>和「固定问同样 3 个 verifier」比，我们的方法接受数和错误数基本不变，效用在 9 组里有 8 组显著更高（QuantumKatas 的 Qwen 不显著）。差别来自提前停止省下的钱。</li>
    <li>和「固定只问 1 个或 2 个」比，结果有输有赢。固定问更少显著更好的有：MATH 的 Sonnet（1 个），MATH 的 gpt-oss（2 个）。我们显著更好的有：GPQA 的 Sonnet、MATH 的 gpt-oss、QuantumKatas 的 Qwen（都是对比 1 个）。</li>
    <li>Self-consistency（多数投票，置信度用 generator 自己报的）在 GPQA 和 MATH 的 6 组里没有一组显著好过我们，Qwen 和 Sonnet 上显著更差。Qwen 是因为它自报的置信度偏高（GPQA 上放行 34 个，错 17 个）；Sonnet 主要差在多采样 10 次的花费。</li>
    <li>Weaver（5 个 verifier）用 generator 置信度当先验、按后验 ≥ 95% 放行时，9 组里没有一组好过我们，7 组显著更差（GPQA 和 QuantumKatas 的 Sonnet 不显著）。它放得更多、错得也更多，例如 GPQA 的 Qwen 放行 150 个，错 30 个；我们放行 122 个，错 10 个。它和我们的结构相同，差别在于 verifier 的似然不看标签：EM 假设各 verifier 互相独立，但它们常常一起错，于是后验被高估。这说明在训练部分用标签估计似然是必要的。附录 A 里不用置信度的版本，在 QuantumKatas 的 Qwen 上显著好过我们（那一组我们放行的答案有 16% 是错的），其余各组不显著或更差。</li>
    <li>SSV 用模型当强 verifier 时，9 组里有 6 组点估计不如我们（4 组显著），只有 MATH 的 Sonnet 显著更好（差 0.017）。用真实标签当强 verifier 的原版 SSV 会直接查看一部分题的答案，放在附录 A。</li>
  </ul>
</div>

<div class="key">
  <span>每格三行：<b>接受数 / 其中答错</b>，<b>平均效用</b>，<b>每题预期花费</b>。效用按题平均：答对放行 +1，答错放行 −19，弃权 0，再减去 10 × 每题花费（美元）。</span>
</div>

{{SECTIONS}}



<section>
  <h2>附录 A：其余变体</h2>
  <p class="meta">固定问全部 5 个、旧版 self-consistency、训练部分选阈值或按 0.5 二值化的 Weaver，以及用真实标签当强 verifier 的 SSV（括注里是每题平均查看真实标签的次数，这部分成本没有计入效用）。</p>
  {{APPENDIX}}
</section>

<section>
  <h2>附录 B：错误率对覆盖率</h2>
  <p class="meta">只看评测部分：把有效答案按各方法的分数从高到低放行，横轴是放行比例，纵轴是放行里答错的比例。虚线是 5% 的目标，圆点是我们的方法在 L = 19 时的位置。</p>
  <div class="legend">{{LEGEND}}<span><i class="dot"></i>我们的方法</span></div>
  {{CHARTS}}
</section>
</main>
'''

if __name__ == '__main__':
    main()
