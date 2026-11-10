"""Results page for the AT / AT+CC replay on QuantumKatas: reports/selective_auditing_qk_20261110/results.html."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

SRC = Path('results/selective_auditing_qk_20261110')
OUR = Path('results/gpqa_fpr_routing_20261102')
OUT = Path('reports/selective_auditing_qk_20261110/results.html')
GEN = {'qwen3': 'Qwen3', 'gptoss': 'gpt-oss', 'sonnet55': 'Sonnet 5.5'}
METHODS = [('AT', 'AT', 'var(--s1)'), ('AT+CC', 'AT+CC', 'var(--s2)'), ('random', '随机审计', 'var(--s3)')]
pct = lambda v, d=1: '—' if v is None else f'{v*100:.{d}f}%'


def main_table(full):
    rows = ''
    for gen in GEN:
        g = full['generators'][gen]
        for a, cell in g['orders']['primary'].items():
            for i, (m, name, _) in enumerate(METHODS+[('oracle_threshold', '事后最优阈值', '')]):
                x = cell[m]
                if m == 'oracle_threshold':
                    r, per, mand, spot, brier = x, None, None, None, None
                else:
                    r, per = x['ratio_of_means'], x['per_run']['leak_error_rate']['q05_50_95']
                    mand, spot, brier = x['mean_counts']['mandatory_audits'], x['mean_counts']['spot_audits'], x.get('brier')
                head = f'<th scope="row" rowspan="4">{GEN[gen]}<br><small>E = {g["raw_errors"]} / {g["evaluable"]}</small></th><td rowspan="4" class="n">{float(a):.0%}</td>' if i == 0 and a == '0.05' else (f'<td rowspan="4" class="n">{float(a):.0%}</td>' if i == 0 else '')
                if i == 0 and a == '0.05':
                    head = head.replace('rowspan="4">' + GEN[gen], f'rowspan="{4*len(g["orders"]["primary"])}">' + GEN[gen], 1)
                rows += (f'<tr class="{"first" if i == 0 else ""}">{head}<td>{name}</td><td class="n">{pct(r["leak_error_rate"])}</td>'
                         f'<td class="n">{"—" if per is None else pct(per[0], 0) + " – " + pct(per[2], 0)}</td>'
                         f'<td class="n">{pct(r["selective_risk"])}</td><td class="n">{pct(r["auto_accept_rate"])}</td>'
                         f'<td class="n">{pct(r["audit_rate"])}</td><td class="n">{"—" if mand is None else f"{mand:.0f} / {spot:.0f}"}</td>'
                         f'<td class="n">{"—" if brier is None else f"{brier:.3f}"}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Generator</th><th class="n">α_leak</th><th>方法</th><th class="n">漏审错误比例 M/E</th>'
            '<th class="n">单次运行 5%–95%</th><th class="n">selective risk M/A</th><th class="n">自动接受 A/T</th><th class="n">审计 /T</th>'
            '<th class="n">必审 / 抽查（次）</th><th class="n">Brier</th></tr></thead><tbody>' + rows + '</tbody></table></div>')


def paired_table(full):
    rows = ''
    for gen in GEN:
        for a, cell in full['generators'][gen]['orders']['primary'].items():
            p = cell['paired_cc_minus_at']
            f = lambda k: f'{p[k]["mean"]*100:+.1f}（{p[k]["q025_975"][0]*100:+.1f} 到 {p[k]["q025_975"][1]*100:+.1f}）'
            rows += f'<tr><td>{GEN[gen]}</td><td class="n">{float(a):.0%}</td><td class="n">{f("audit_rate")}</td><td class="n">{f("leak_error_rate")}</td></tr>'
    return ('<div class="scroll"><table><thead><tr><th>Generator</th><th class="n">α_leak</th><th class="n">审计率差（百分点）</th>'
            '<th class="n">漏审错误比例差（百分点）</th></tr></thead><tbody>' + rows + '</tbody></table></div>')


def curves(npz, a='0.1'):
    W, H, L, R, T, B = 320, 200, 44, 10, 10, 28
    pw, ph = W-L-R, H-T-B
    figs = ''
    for kind, title, ymax in (('leak', '累计漏审错误比例', .3), ('audit', '累计审计率', 1.0)):
        out = ''
        for gen in GEN:
            e = npz[f'{gen}__errors_for_evaluator'].astype(float)
            n = len(e)
            sx = lambda i: L+i/(n-1)*pw
            sy = lambda v: T+ph-min(v, ymax)/ymax*ph
            g = [f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/><text class="tk" x="{L-5}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0%}</text>'
                 for v in (np.linspace(0, ymax, 4))]
            g.append(f'<text class="tk" x="{L}" y="{H-6}">1</text><text class="tk" x="{L+pw}" y="{H-6}" text-anchor="end">{n}</text>')
            if kind == 'leak':
                g.append(f'<line class="ideal" x1="{L}" x2="{L+pw}" y1="{sy(float(a)):.1f}" y2="{sy(float(a)):.1f}"/>')
            for m, key, col in (('AT', 'AT', 'var(--s1)'), ('AT+CC', 'ATpCC', 'var(--s2)')):
                H_ = npz[f'{gen}__{a}__{key}__audited'].astype(float)
                if kind == 'leak':
                    cm = ((1-H_)*e[None, :]).cumsum(axis=1).mean(axis=0)
                    ce = np.maximum(e.cumsum(), 1)
                    y = cm/ce
                    start = int(np.argmax(e.cumsum() >= 5))
                else:
                    y = H_.cumsum(axis=1).mean(axis=0)/np.arange(1, n+1)
                    start = 0
                pts = ' '.join(f'{sx(i):.1f},{sy(v):.1f}' for i, v in enumerate(y) if i >= start)
                g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
            out += f'<figure><figcaption>{GEN[gen]}</figcaption><svg viewBox="0 0 {W} {H}" role="img" aria-label="{GEN[gen]} {title}">{"".join(g)}</svg></figure>'
        figs += f'<h3>{title}（α_leak = {float(a):.0%}，主题序，100 次的平均）</h3><div class="row">{out}</div>'
    return figs


def stability(full, sens, e2e):
    rows = ''
    for gen in GEN:
        for a in ('0.1', '0.2'):
            for m in ('AT', 'AT+CC'):
                L = [full['generators'][gen]['orders'][o][a][m]['ratio_of_means'] for o in full['generators'][gen]['orders']]
                lk, au = [x['leak_error_rate'] for x in L], [x['audit_rate'] for x in L]
                s1 = sens['lower_spot_check'][gen].get(a, {}).get(m)
                s2 = sens['smaller_steps'][gen].get(a, {}).get(m)
                ee = e2e[gen][a][m]['ratio_of_means']
                f = lambda x: '—' if x is None else f'{pct(x["ratio_of_means"]["leak_error_rate"])} / {pct(x["ratio_of_means"]["audit_rate"])}'
                rows += (f'<tr><td>{GEN[gen]}</td><td class="n">{float(a):.0%}</td><td>{m}</td><td class="n">{pct(min(lk))} – {pct(max(lk))}</td>'
                         f'<td class="n">{pct(min(au))} – {pct(max(au))}</td><td class="n">{f(s1)}</td><td class="n">{f(s2)}</td>'
                         f'<td class="n">{pct(ee["leak_error_rate"])} / {pct(ee["audit_rate"])}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Generator</th><th class="n">α_leak</th><th>方法</th><th class="n">22 种题序：漏审比例范围</th>'
            '<th class="n">22 种题序：审计率范围</th><th class="n">γ = 0.2：漏审 / 审计</th><th class="n">η = 0.2, γ = 0.1：漏审 / 审计</th>'
            '<th class="n">含失败题：漏审 / 审计</th></tr></thead><tbody>' + rows + '</tbody></table></div>')


def simulations(sim):
    rows = ''
    names = {'informative': '置信度有区分力', 'uninformative': '置信度无信息', 'overconfident': '严重过度自信',
             'rare_errors': '错误稀少', 'shift': '中途分布变化'}
    for s, n in names.items():
        for a in ('0.1', '0.2'):
            cells = ''.join(f'<td class="n">{pct(sim[f"{s}|{a}|{m}"]["ratio_of_means"]["leak_error_rate"])} / {pct(sim[f"{s}|{a}|{m}"]["ratio_of_means"]["audit_rate"])}</td>'
                            for m in ('AT', 'AT+CC', 'random'))
            rows += f'<tr><td>{n}<br><small>E = {sim[f"{s}|{a}|AT"]["E"]} / 2000</small></td><td class="n">{float(a):.0%}</td>{cells}</tr>'
    return ('<div class="scroll"><table><thead><tr><th>模拟数据流</th><th class="n">α_leak</th><th class="n">AT：漏审 / 审计</th>'
            '<th class="n">AT+CC：漏审 / 审计</th><th class="n">随机审计：漏审 / 审计</th></tr></thead><tbody>' + rows + '</tbody></table></div>')


def cross(full):
    ours = {k: json.loads((OUR/f'{f}.json').read_text()) for k, f in (('fpr', 'baselines_qkatas'), ('sel', 'baselines_qkatas_fdr'))}
    rows = ''
    for gen in GEN:
        for a in ('0.05', '0.1', '0.2'):
            at = full['generators'][gen]['orders']['primary'][a]
            o = {}
            for k, d in ours.items():
                b = d['budgets'][-1]
                x = d['generators'][gen][f'{float(a)}|{b}']['ours']
                o[k] = (x['pub'], x['fpr'], x['wrong_released']/x['released'] if x['released'] >= 1 else None, x['usd']*1e4)
            r1, r2 = at['AT']['ratio_of_means'], at['AT+CC']['ratio_of_means']
            rows += (f'<tr><td>{GEN[gen]}</td><td class="n">{float(a):.0%}</td>'
                     f'<td class="n">{pct(r1["auto_accept_rate"])} / {pct(r1["leak_error_rate"])} / {pct(r1["selective_risk"])}</td>'
                     f'<td class="n">{pct(r2["auto_accept_rate"])} / {pct(r2["leak_error_rate"])} / {pct(r2["selective_risk"])}</td>'
                     f'<td class="n">{pct(o["fpr"][0])} / {pct(o["fpr"][1])} / {pct(o["fpr"][2])}<br><small>verifier {o["fpr"][3]:.1f}</small></td>'
                     f'<td class="n">{pct(o["sel"][0])} / {pct(o["sel"][1])} / {pct(o["sel"][2])}<br><small>verifier {o["sel"][3]:.1f}</small></td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Generator</th><th class="n">α</th><th class="n">AT：自动接受 / M/E / M/A</th>'
            '<th class="n">AT+CC：同左</th><th class="n">我们（约束 FPR = α）：放行 / FPR / 放行错误率</th>'
            '<th class="n">我们（约束放行错误率 = α）：同左</th></tr></thead><tbody>' + rows + '</tbody></table></div>')


def main():
    full = json.loads((SRC/'full.json').read_text())
    sens = json.loads((SRC/'hyperparameter_sensitivity.json').read_text())
    e2e = json.loads((SRC/'end_to_end.json').read_text())
    sim = json.loads((SRC/'simulations.json').read_text())
    cfg = json.loads(Path('configs/selective_auditing_qk_20261110.json').read_text())
    npz = np.load(SRC/'traces_primary_full.npz')
    hp = cfg['hyperparameters']
    sec = max(full['generators'][g]['orders']['primary'][a][m]['seconds_per_run'] for g in GEN for a in full['generators'][g]['orders']['primary'] for m in ('AT', 'AT+CC'))
    page = (TEMPLATE.replace('{{MAIN}}', main_table(full)).replace('{{PAIRED}}', paired_table(full)).replace('{{CURVES}}', curves(npz))
            .replace('{{STAB}}', stability(full, sens, e2e)).replace('{{SIM}}', simulations(sim)).replace('{{CROSS}}', cross(full))
            .replace('{{ETA}}', f'{hp["AT"]["eta"]:g}').replace('{{GAMMA}}', f'{hp["AT"]["gamma"]:g}').replace('{{BETA}}', f'{hp["AT+CC"]["beta"]:g}')
            .replace('{{K}}', str(len(hp['AT+CC']['grid_points']))).replace('{{SEC}}', f'{sec*1e3:.2f}'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print('wrote', OUT)


TEMPLATE = '''<title>AT 与 AT+CC 复现</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; wide tables scroll in their own box; curves as small multiples. */
:root {
  --bg: #f6f7f5; --surface: #ffffff; --fg: #1d2522; --muted: #5b6662; --line: #dbe0dd;
  --s1: #1F5FA8; --s2: #B4581F; --s3: #1E8A6E;
  --display: "Noto Serif SC", "Songti SC", serif; --body: "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --surface: #1c2320; --fg: #e4ebe7; --muted: #9aa7a1; --line: #303a36;
  --s1: #4F97E3; --s2: #C8743A; --s3: #2FA383; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.65; }
main { max-width: 1120px; margin: 0 auto; padding-block: 36px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 30px; min-width: 0; }
section { display: grid; gap: 10px; min-width: 0; }
h1 { font-family: var(--display); font-size: 28px; line-height: 1.3; margin: 0; text-wrap: balance; }
h2 { font-size: 18px; margin: 0; } h3 { font-size: 15px; margin: 8px 0 0; }
p { margin: 0; max-width: 74ch; } .meta { color: var(--muted); font-size: 13px; }
ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 78ch; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 9px; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: top; }
thead th { color: var(--muted); font-weight: 500; }
tr.first td { border-top: 1px solid var(--line); }
.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; } thead .n { font-family: var(--body); }
small { color: var(--muted); }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-width: 0; }
figcaption { font-size: 13px; font-weight: 700; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--line); stroke-width: 1; } .ideal { stroke: var(--muted); stroke-width: 1.25; stroke-dasharray: 5 4; }
.tk { font: 10.5px var(--mono); fill: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); }
.legend i { display: inline-block; width: 16px; height: 4px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
</style>
<main>
<header style="display:grid;gap:8px">
  <h1>AT 与 AT+CC 在 QuantumKatas 上的复现</h1>
  <p class="meta">论文：Gibbs, Tibshirani, Han, Zrnic, Reliable AI-Assisted Decisions via Selective Auditing（arXiv:2610.31662v1）。在已有缓存数据上追加的实验，不是预注册主实验。没有新增 API 调用。</p>
</header>

<section>
  <h2>结论</h2>
  <ul>
    <li><b>漏审错误比例控制住了。</b>主题序下，三个 generator、四个 α_leak 的漏审错误比例（多次审计随机性的计数期望之比）都在目标附近，例如 α_leak = 10% 时为 9.9% 到 11.0%。但单次运行的波动很大，Sonnet 只有 35 道错题，单次运行的 5%–95% 区间可以从 0% 到 20% 以上。</li>
    <li><b>省下的审计很少。</b>审计率只比随机审计（每题以 1 − α 的概率审）低几个百分点，比如 α_leak = 10% 时为 81% 到 87%，随机审计为 90%；而事后最优阈值只要 56% 到 76%。原因是数据流短：为了让理论误差项变小，开发数据选出的 γ = {{GAMMA}}，即高置信度题也要抽查一半。</li>
    <li><b>在线校准改善了 Brier，但没有明显减少审计。</b>AT+CC 的 Brier 比原始置信度低（Qwen3 0.32 → 0.23，gpt-oss 0.26 → 0.19），但和 AT 的配对审计率差都在 ±5 个百分点内，区间都包含 0。</li>
    <li><b>自动接受集合里的错误率很高。</b>因为论文约束的是「漏掉的错误占全部错误的比例」，Qwen3 和 gpt-oss 自动接受的答案里有 21% 到 38% 是错的；这和我们约束的「放行答案的错误率」是两个不同的指标。</li>
    <li><b>和我们的方法只能作跨协议参照。</b>AT 每次审计都会纠正错误、只靠在线审计拿标签；我们的方法不纠正、用约 258 道带标签训练题和付费 verifier。下面的对照表逐项列出了差异。</li>
  </ul>
</section>

<section>
  <h2>设置</h2>
  <ul>
    <li><b>语义：</b>H = 0 自动接受原始候选，不读取标签；H = 1 转交审计，读取官方 checker 标签。候选和原始标签都不改；不声称生成了修正答案。学习器只能通过受控 oracle 在审计时拿到标签。</li>
    <li><b>数据：</b>Qwen3、gpt-oss、Sonnet 5.5 的固定候选和原始 p_correct。可评估题数 310 / 308 / 322，排除 12 / 14 / 0 道无代码、无置信度的失败题（另做含失败题的敏感性分析）。</li>
    <li><b>超参数</b>（论文没有给出具体数值，这里是我们的选择）：在 GPQA 的 249 道 calibration 题上选定后冻结，η = {{ETA}}，γ = {{GAMMA}}，AT+CC 的 β = {{BETA}}，网格 {{K}} 个点（步长 0.05），τ₁ = 0.5，w₁ 均匀。第一轮搜索（η ≤ 0.2，γ ≤ 0.2）没有可行解，所以在开发数据上扩大过一次搜索范围。</li>
    <li><b>随机性：</b>每种题序 100 个审计随机种子，所有方法共用同一组随机数，但各自按自己的审计概率决定是否审计。题序：主题序、按 kata 分组的原始顺序，以及另外 20 种随机题序。种子和题序不是新增的独立题目。</li>
    <li><b>成本：</b>每次 H = 1 算一次逻辑审计，主成本指标是审计次数和审计比例；置信度来自原生成调用，不重复计费；新增 API 支出为 0；本地更新每次运行最多 {{SEC}} 毫秒。没有给审计定美元单价。</li>
  </ul>
</section>

<section>
  <h2>主结果（主题序，100 次审计随机性）</h2>
  <p class="meta">「漏审错误比例」是计数期望之比 E[M] / E[E]，即论文的 Err(T)；「单次运行 5%–95%」是每次运行比例的分布。selective risk = M / A，即自动接受答案里的错误率。必审 / 抽查是两类审计的平均次数。Brier 由评估器在全部题目上计算：AT 用原始置信度，AT+CC 用本轮更新前的校准分数。事后最优阈值使用全部标签，不可部署，只作参照。理论误差项 (1/η + 1/γ) / E：Qwen3 2.2%，gpt-oss 2.7%，Sonnet 8.6%，这是期望指标的界，不保证单次运行。</p>
  {{MAIN}}
</section>

<section>
  <h2>AT+CC 减 AT 的配对差异</h2>
  <p class="meta">同一审计种子下两者之差，括号里是 100 个种子的 2.5%–97.5% 分位数。只反映审计随机性，不是对题目抽样的置信区间。</p>
  {{PAIRED}}
</section>

<section>
  <h2>学习曲线</h2>
  <div class="legend"><span><i style="background:var(--s1)"></i>AT</span><span><i style="background:var(--s2)"></i>AT+CC</span><span>虚线：α_leak</span></div>
  <p class="meta">漏审曲线从累计出现 5 道错题后开始画，之前分母太小。</p>
  {{CURVES}}
</section>

<section>
  <h2>稳定性：题序、超参数和失败题</h2>
  <p class="meta">两组备选超参数在 QuantumKatas 运行前就已声明，只作敏感性分析，没有用来挑主结果。它们没有带来明显更少的审计；η、γ 更小时，错题少的 Sonnet 漏审比例明显超过 α（α_leak = 5% 时为 22%）。</p>
  {{STAB}}
</section>

<section>
  <h2>模拟数据</h2>
  <p class="meta">每个数据流 2000 轮，200 个审计种子，冻结的超参数。随机审计的漏审比例理论上等于 α，可作解析检查。</p>
  {{SIM}}
</section>

<section>
  <h2>和我们方法的跨协议参照（不是排名）</h2>
  <p class="meta">把我们的「不放行」对应到「审计」后：我们的 FPR 对应论文的漏审错误比例 M/E，我们的放行错误率对应 selective risk M/A。我们的结果取不限预算一档，按 Algorithm 2 混合，T 含失败题（322）；花费单位 10⁻⁴ 美元/题。</p>
  {{CROSS}}
  <ul>
    <li><b>标签权限不同：</b>AT 只在审计时在线拿到标签；我们在每折约 258 道训练题上用全部标签拟合似然和选规则。</li>
    <li><b>动作语义不同：</b>AT 审计后由 auditor 纠正错误，错误不会发布；我们的「不放行」是弃权，不纠正。</li>
    <li><b>信号不同：</b>AT 只用 generator 的置信度；我们还付费调用 verifier。</li>
    <li><b>时机不同：</b>AT 在发布前审计；我们原方案的学习审计发生在发布或弃权决定固定之后。</li>
  </ul>
  <p>如果要做同协议比较，需要先定义 oracle-assisted 版本：两者用同一个审计 oracle，同样只在审计时拿标签，「审计」统一为纠正后发布。这需要改动我们方法的反馈设定，所以这里只给出建议，没有执行。</p>
</section>

<section>
  <h2>和论文原算法的差异</h2>
  <ul>
    <li>超参数是我们在开发数据上选的；论文没有给出数值。</li>
    <li>数据流只有 308 到 322 轮，远短于论文实验；错题少时理论界很宽。</li>
    <li>置信度是 generator 自报的 p_correct，取值很粗（每个 generator 18 到 25 种不同值）。</li>
    <li>auditor 是缓存的官方 checker 标签，没有生成修正答案。</li>
    <li>α_leak 取 5%、10%、20%、30%（论文图里是 10% 到 30%）。</li>
  </ul>
</section>
</main>
'''

if __name__ == '__main__':
    main()
