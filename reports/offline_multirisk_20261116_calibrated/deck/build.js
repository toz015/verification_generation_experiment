// Offline multi-risk deck. Every number comes from data.json (extract.py, calibrated + raw result files).
const pptxgen = require('pptxgenjs');
const path = require('path');
const fs = require('fs');
const SK = process.env.PPTX_SKILL;
const { applyTheme } = require(path.join(SK, 'scripts/apply_theme.js'));
const D = JSON.parse(fs.readFileSync(process.env.DATA || path.join(__dirname, 'data.json'), 'utf8'));
const POOL = D.variant === 'pool';
const VNAME = POOL ? '加固定 panel 版' : '原版本';
const OUT = process.argv[2];

const THEME = {
  name: 'Multi-risk',
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  colors: {
    dk1: '1B2A34', lt1: 'FFFFFF', dk2: '55646D', lt2: 'EEF2F4',
    accent1: '2A78D6', accent2: 'EB6834', accent3: '1BAF7A', accent4: 'EDA100', accent5: 'E87BA4', accent6: '008300',
    hlink: '2A78D6', folHlink: '6B5BA8',
  },
};
const HEX = THEME.colors;
const SERIES = ['2A78D6', 'EB6834', '1BAF7A', 'EDA100', 'E87BA4', '008300', '6B5BA8', '8A6D3B', '4A9BB0'];
const MCOL = (m) => SERIES[D.methods.indexOf(m)];
const GRAY = '9AA5AB';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = `离线多风险实验（${VNAME}）`;
pres.author = 'tony';
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;

pres.defineSlideMaster({
  title: 'TITLE',
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.1, w: 11.7, h: 1.3, fontSize: 44, bold: true, color: C.background1, valign: 'bottom', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 3.6, w: 11.7, h: 1.4, fontSize: 20, color: 'CADBE6', valign: 'top', align: 'left', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: HEX.lt1 },
  margin: [0.4, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.35, w: W - 2 * M, h: 0.8, fontSize: 30, bold: true, color: C.text1, valign: 'middle', align: 'left', margin: 0 }, text: '' } },
    { text: { text: `离线多风险实验 · ${VNAME}`, options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: 'right' },
});

const pct = (v, d = 0) => `${(100 * v).toFixed(d)}%`;
const RN = { selective_error: '放行错误率', fpr: 'FPR', fnr: 'FNR' };
const RISKS = ['selective_error', 'fpr', 'fnr'];
const S = D.sig_cal, SR = D.sig_raw;
const BI = (b) => D.baselines.indexOf(b);
let nShape = 0;
let SN = 0;
const CHARTS = [];
const nm = (s) => `${s}-${++nShape}`;

function slide(title, section, notes) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: section });
  SN += 1;
  s.addText(title, { placeholder: 'title' });
  if (notes) s.addNotes(notes);
  return s;
}
function text(s, t, o) { s.addText(t, Object.assign({ isTextBox: true, objectName: nm('text'), fontSize: 16, color: C.text1, margin: 0, valign: 'top' }, o)); }
function card(s, x, y, w, h, fill) { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { type: 'none' }, objectName: nm('card') }); }
function pill(s, label, x, y, w) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.36, rectRadius: 0.18, fill: { color: C.accent1, transparency: 86 }, line: { type: 'none' }, objectName: nm('pill') });
  text(s, label, { x, y, w, h: 0.36, fontSize: 13, bold: true, color: C.accent1, align: 'center', valign: 'middle' });
}
function legendGrid(s, items, x, y, perRow, w) {
  const cw = w / perRow;
  items.forEach(([label, col], i) => {
    const cx = x + (i % perRow) * cw, cy = y + Math.floor(i / perRow) * 0.34;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy + 0.08, w: 0.32, h: 0.16, rectRadius: 0.05, fill: { color: col }, line: { type: 'none' }, objectName: nm('legend') });
    text(s, label, { x: cx + 0.42, y: cy, w: cw - 0.5, h: 0.32, fontSize: 13, color: C.text2, valign: 'middle' });
  });
}
function legendRow(s, items, x, y) {
  let cx = x;
  for (const [label, col] of items) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: y + 0.08, w: 0.32, h: 0.16, rectRadius: 0.05, fill: { color: col }, line: { type: 'none' }, objectName: nm('legend') });
    const w = 0.22 * label.length + 0.5;
    text(s, label, { x: cx + 0.42, y, w, h: 0.32, fontSize: 13, color: C.text2, valign: 'middle' });
    cx += 0.42 + w + 0.2;
  }
}
const TH = (t, o) => ({ text: t, options: Object.assign({ bold: true, color: C.text2, fill: { color: C.background2 } }, o) });
const TD = (t, o) => ({ text: t, options: Object.assign({}, o) });
function table(s, rows, o) {
  s.addTable(rows, Object.assign({ x: M, y: 1.45, w: W - 2 * M, fontSize: 14, color: C.text1, border: { type: 'solid', pt: 0.75, color: 'D5DCE0' }, margin: [0.05, 0.1, 0.05, 0.1], valign: 'middle', objectName: nm('table') }, o));
}
const AXIS = {
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt', legendFontFace: '+mn-lt', titleFontFace: '+mn-lt', dataLabelFontFace: '+mn-lt',
  valGridLine: { color: 'E1E6E9', size: 0.75 }, catGridLine: { style: 'none' }, catAxisLineShow: true, valAxisLineShow: false,
  catAxisTitleFontFace: '+mn-lt', valAxisTitleFontFace: '+mn-lt', catAxisTitleColor: HEX.dk2, valAxisTitleColor: HEX.dk2, catAxisTitleFontSize: 12, valAxisTitleFontSize: 12,
};

// ---------------------------------------------------------------- 1. cover
pres.addSection({ title: '概览' });
{
  const s = pres.addSlide({ masterName: 'TITLE', sectionTitle: '概览' });
  SN += 1;
  s.addText('离线多风险实验', { placeholder: 'title' });
  s.addText([
    { text: '放行错误率、FPR、FNR 三种约束下的按需验证比较', options: { breakLine: true } },
    { text: POOL ? '加固定 panel 版：Ours 的候选池里加入「每题问同一组 verifier」的固定策略' : '原版本：Ours 只在自适应策略里挑选', options: { breakLine: true, fontSize: 18, color: 'FFFFFF', bold: true } },
    { text: 'GPQA · MATH-500 · QuantumKatas ｜ Qwen3 · gpt-oss · Sonnet 5.5 ｜ 只用缓存', options: { fontSize: 16, color: '9FB4C2' } },
  ], { placeholder: 'body' });
  text(s, '2026-11-16', { x: 0.8, y: 6.4, w: 4, h: 0.4, fontSize: 14, color: '9FB4C2' });
  s.addNotes('主比较里所有方法都读校准后的 p_correct；第「原始 p 对照」页是全部读原始 p 的重跑。');
}

// ---------------------------------------------------------------- 2. summary
{
  const s = slide(POOL ? '结论：三种约束都领先，但约一半 FPR / FNR 条件选中固定策略' : '结论：前两种约束领先，FNR 输给固定面板', '概览',
    '每个数字是条件数：配对 bootstrap 95% 区间显示我们更好 : 显示对照更好。区间含 0 的条件不计。');
  const cards = [
    ['selective_error', '放行率越高越好', '放行错误率', ['Cal-threshold', 'Fixed-1'], '多数条件下放行更多'],
    ['fpr', '放行率越高越好', 'FPR', ['Cal-threshold', 'Fixed-1'], POOL ? '领先阈值方法和 Fixed-1' : '领先阈值方法；预算多时不如 Fixed'],
    ['fnr', '错误放行越少越好', 'FNR', ['Cal-threshold', 'Fixed-1'], POOL ? '错误放行少于 Fixed；超标变多' : '输给 Fixed：问 verifier 太少'],
  ];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cards.forEach(([risk, goal, label, bs, take], i) => {
    const x = M + i * (cw + 0.3), y = 1.5;
    card(s, x, y, cw, 4.7);
    pill(s, label, x + 0.3, y + 0.3, 1.6);
    text(s, `${S[risk].n} 个条件`, { x: x + 2.05, y: y + 0.3, w: cw - 2.3, h: 0.36, fontSize: 13, color: C.text2, valign: 'middle' });
    text(s, `目标：${goal}`, { x: x + 0.3, y: y + 3.35, w: cw - 0.6, h: 0.36, fontSize: 13, color: C.text2 });
    bs.forEach((b, j) => {
      const yy = y + 0.95 + j * 1.35;
      text(s, `对 ${b}`, { x: x + 0.3, y: yy, w: cw - 0.6, h: 0.35, fontSize: 14, color: C.text2 });
      text(s, [
        { text: String(S[risk].ours[BI(b)]), options: { color: C.accent1, bold: true } },
        { text: ' : ', options: { color: C.text2 } },
        { text: String(S[risk].base[BI(b)]), options: { color: C.text1, bold: true } },
      ], { x: x + 0.3, y: yy + 0.35, w: cw - 0.6, h: 0.8, fontSize: 40, fontFace: 'Arial' });
    });
    text(s, take, { x: x + 0.3, y: y + 3.8, w: cw - 0.6, h: 0.7, fontSize: 16, bold: true });
  });
  text(s, [
    { text: '蓝色', options: { color: C.accent1, bold: true } },
    { text: ` = 我们显著更好的条件数，黑色 = 对照显著更好。一个条件 = 一组（数据集 · generator，α，预算 f）：放行错误率 ${D.n_pairs.selective_error} 组 × ${D.alpha.selective_error.length} 个 α × 6 档预算 = ${S.selective_error.n}；FPR ${D.n_pairs.fpr} 组（不含 MATH-500）× ${D.alpha.fpr.length} × 6 = ${S.fpr.n}；FNR ${D.n_pairs.fnr} × ${D.alpha.fnr.length} × 6 = ${S.fnr.n}。下一页按数据集拆开。` },
  ], { x: M, y: 6.4, w: W - 2 * M, h: 0.45, fontSize: 13, color: C.text2 });
}

// ---------------------------------------------------------------- 2b. per dataset
{
  const s = slide(POOL ? '按数据集看：每个数据集对每个对照的结果' : '按数据集看：放行错误率处处领先，FNR 在 GPQA 和 MATH-500 输给 Fixed', '概览',
    '每格是「我们显著更好 : 对照显著更好」的条件数（FNR 下更好指错误放行更少），括号里拆成低预算 f ≤ 0.25（含 f = 0）和高预算 f ≥ 0.5。每个数据集有 3 个 generator，所以条件数 = 3 × α 个数 × 6 档预算。AT 列是 AT-Frozen，AT+Cal 和它接近。');
  const SG = D.sig_ds;
  const cell = (x, split) => !x ? '—' : split ? `${x[0]}:${x[1]}（${x[3]}:${x[4]} / ${x[5]}:${x[6]}）` : `${x[0]}:${x[1]}`;
  const EXTRA = ['Weaver', 'Self-consistency', 'SSV'].filter((m) => D.baselines.includes(m));
  const rows = [[TH('数据集 · 风险'), TH('条件数', { align: 'right' }), TH('对 AT', { align: 'right' }), TH('对 Cal-thr', { align: 'right' }),
    TH('对 Fixed-1（低 / 高）', { align: 'right' }), TH('对 Fixed-3（低 / 高）', { align: 'right' }),
    ...EXTRA.map((m) => TH(`对 ${m === 'Self-consistency' ? 'SC' : m}`, { align: 'right' }))]];
  Object.keys(SG).forEach((ds) => RISKS.forEach((r) => {
    const x = SG[ds][r];
    if (!x) return;
    rows.push([TD(`${ds} · ${RN[r]}`, { bold: true }), TD(String(x['AT-Frozen'][2]), { align: 'right', fontFace: 'Arial' }),
      TD(cell(x['AT-Frozen'], false), { align: 'right', fontFace: 'Arial' }), TD(cell(x['Cal-threshold'], false), { align: 'right', fontFace: 'Arial' }),
      TD(cell(x['Fixed-1'], true), { align: 'right', fontFace: 'Arial' }), TD(cell(x['Fixed-3'], true), { align: 'right', fontFace: 'Arial' }),
      ...EXTRA.map((m) => TD(cell(x[m], false), { align: 'right', fontFace: 'Arial' }))]);
  }));
  const cw0 = EXTRA.length ? [2.55, 0.75, 0.95, 1.0, 1.95, 1.95, ...EXTRA.map(() => (W - 2 * M - 9.15) / EXTRA.length)] : [2.9, 1.0, 1.25, 1.9, 2.55, 2.533];
  table(s, rows, { y: 1.35, colW: cw0, fontSize: EXTRA.length ? 12 : 13, rowH: 0.4 });
  if (!POOL) {
    text(s, [
      { text: 'GPQA：', options: { bold: true } }, { text: '放行错误率全面领先（只有低预算对 Cal-threshold 略输）；FPR 低预算赢 Fixed、高预算输；FNR 输给 Fixed。', options: { breakLine: true } },
      { text: 'MATH-500：', options: { bold: true } }, { text: '放行错误率对所有对照大比分领先；FNR 对 Fixed 一次没赢（错答案太少，Fixed 每题都问更容易抓到）；不测 FPR。', options: { breakLine: true } },
      { text: 'QuantumKatas：', options: { bold: true } }, { text: '放行错误率对 Fixed 几乎全胜；FPR 低预算赢、高预算输；FNR 和 Fixed 大致打平。', options: { breakLine: !!EXTRA.length } },
      ...(EXTRA.length ? [{ text: '新加的对照：', options: { bold: true } }, { text: `Weaver 和 SSV 在放行错误率下几乎全输给我们；Weaver 在 FNR 下和我们接近。Self-consistency 不是 verifier 方法（靠多生成 10 个样本），放在附录。` }] : []),
    ], { x: M, y: 5.15, w: W - 2 * M, h: 1.75, fontSize: 12, paraSpaceAfter: 3 });
  }
}

// ---------------------------------------------------------------- 3. task and data
pres.addSection({ title: '设定' });
{
  const s = slide('任务：每道题决定放行还是拦下', '设定',
    '策略不生成也不修改答案。测试部分只评估，标签只给评估器用。');
  const steps = [
    ['1', 'generator 给出答案和 p_correct', '一个固定答案，加上它自报的正确概率'],
    ['2', '可以花钱问 verifier', '缓存里 5 个 verifier 的打分，每问一个花一次钱'],
    ['3', '放行，或者拦下', '不改答案；没给答案的题按官方规则算错，直接拦下'],
  ];
  steps.forEach(([n, h, d], i) => {
    const y = 1.6 + i * 1.5;
    s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.7, h: 0.7, fill: { color: C.accent1 }, line: { type: 'none' }, objectName: nm('step') });
    text(s, n, { x: M, y, w: 0.7, h: 0.7, fontSize: 22, bold: true, color: C.background1, align: 'center', valign: 'middle', fontFace: 'Arial' });
    text(s, h, { x: M + 0.95, y: y - 0.02, w: 4.6, h: 0.42, fontSize: 18, bold: true });
    text(s, d, { x: M + 0.95, y: y + 0.42, w: 4.6, h: 0.7, fontSize: 14, color: C.text2 });
  });
  const x0 = 6.6;
  text(s, '数据和划分（所有方法：拟合部分训练，验证部分选工作点，测试部分冻结评估）', { x: x0, y: 1.5, w: W - M - x0, h: 0.6, fontSize: 14, color: C.text2 });
  table(s, [
    [TH('数据集'), TH('拟合 / 验证 / 测试')],
    [TD('GPQA', { bold: true }), TD('249 道 calibration 题分成拟合 199、验证 50；196 道 Diamond 题测试')],
    [TD('MATH-500', { bold: true }), TD('5 折：一折测试，下一折验证，其余 3 折拟合；5 次测试合起来是全部 500 题')],
    [TD('QuantumKatas', { bold: true }), TD('322 题，5 折，同上；按合并后的 kata 组划分，同组不跨部分')],
  ], { x: x0, y: 2.2, w: W - M - x0, colW: [1.7, W - M - x0 - 1.7] });
  text(s, 'Generator：Qwen3、gpt-oss、Sonnet 5.5。Verifier：gemini_flash、gemini_lite、mistral_small、mistral_medium、claude_haiku。',
    { x: x0, y: 5.4, w: W - M - x0, h: 0.8, fontSize: 14, color: C.text2 });
}

// ---------------------------------------------------------------- 4. three risks
{
  const s = slide('三种风险约束：α 限制的东西不一样', '设定',
    'FNR 的目标换成最小化错误放行：如果还最大化放行，全部放行就让 FNR = 0，问不出筛选的问题。');
  const a = D.alpha, am = D.alpha_main;
  const rng = (r) => `${a[r][0]} 到 ${a[r][a[r].length - 1]}（主要展示 ${am[r]}）`;
  table(s, [
    [TH('风险'), TH('定义'), TH('α 限制什么'), TH('优化目标'), TH('α 取值')],
    [TD('放行错误率', { bold: true }), TD('e / p'), TD('放行的答案里，错的比例 ≤ α'), TD('放行率 p 越高越好'), TD(rng('selective_error'))],
    [TD('FPR', { bold: true }), TD('e / (1 − q)'), TD('所有错误答案里，被放行的比例 ≤ α'), TD('放行率 p 越高越好'), TD(rng('fpr'))],
    [TD('FNR', { bold: true }), TD('m / q'), TD('所有正确答案里，被拦下的比例 ≤ α'), TD('错误放行 e 越低越好'), TD(rng('fnr'))],
  ], { y: 1.5, colW: [1.5, 1.3, 3.75, 2.3, 3.283], fontSize: 15, rowH: 0.55 });
  const defs = [['p', '放行的题占全部题的比例'], ['e', '放行了但答案错的题，占全部题的比例'], ['m', '拦下了但答案对的题，占全部题的比例'], ['q', '答案正确的题占全部题的比例']];
  card(s, M, 4.1, W - 2 * M, 2.55);
  text(s, '记号', { x: M + 0.3, y: 4.3, w: 3, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
  defs.forEach(([k, v], i) => {
    const x = M + 0.3 + (i % 2) * 6.0, y = 4.75 + Math.floor(i / 2) * 0.5;
    text(s, [{ text: k, options: { bold: true, italic: true, fontFace: 'Cambria' } }, { text: `  ${v}` }], { x, y, w: 5.8, h: 0.4, fontSize: 15 });
  });
  text(s, '同一个 α 在三种风险下严格程度不同，不能横向比较。FNR 如果也最大化放行，全部放行就让 FNR = 0，所以它的目标是错误放行最少。',
    { x: M + 0.3, y: 5.85, w: W - 2 * M - 0.6, h: 0.65, fontSize: 14, color: C.text2 });
}

// ---------------------------------------------------------------- 5. budget
{
  const s = slide('预算 B = f × c_ref：c_ref 是问满 5 个 verifier 的花费', '设定',
    'c_ref 只用拟合部分算，不看验证和测试。f = 0 就是不问 verifier。');
  text(s, 'c_ref：一道题把 5 个 verifier 全部问一遍的花费，在拟合部分有答案的题上取平均；每个数据集 × generator × 折单独算。测试集上每题平均 verifier 花费不能超过 B。',
    { x: M, y: 1.45, w: W - 2 * M, h: 0.85, fontSize: 16 });
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  Object.entries(D.cref).forEach(([ds, [lo, hi]], i) => {
    const x = M + i * (cw + 0.3), y = 2.6;
    card(s, x, y, cw, 2.2);
    text(s, ds, { x: x + 0.3, y: y + 0.25, w: cw - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
    text(s, `${lo.toFixed(0)}–${hi.toFixed(0)}`, { x: x + 0.3, y: y + 0.7, w: cw - 0.6, h: 0.9, fontSize: 48, bold: true, color: C.accent1, fontFace: 'Arial' });
    text(s, '×10⁻⁴ 美元 / 题（c_ref 范围）', { x: x + 0.3, y: y + 1.6, w: cw - 0.6, h: 0.4, fontSize: 13, color: C.text2 });
  });
  text(s, '预算比例 f', { x: M, y: 5.2, w: 3, h: 0.4, fontSize: 16, bold: true });
  ['0', '0.1', '0.25', '0.5', '0.75', '1'].forEach((f, i) => {
    const x = M + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.7, w: 0.95, h: 0.5, rectRadius: 0.25, fill: { color: C.background2 }, line: { type: 'none' }, objectName: nm('f') });
    text(s, f, { x, y: 5.7, w: 0.95, h: 0.5, fontSize: 16, bold: true, align: 'center', valign: 'middle', fontFace: 'Arial' });
  });
  text(s, 'f = 1：平均每题最多花掉「问满 5 个」的钱；f = 0.5 是一半；f = 0 是不问 verifier。',
    { x: M + 7.2, y: 5.6, w: W - 2 * M - 7.2, h: 0.75, fontSize: 14, color: C.text2 });
}

// ---------------------------------------------------------------- 6. alpha examples
{
  const s = slide('直观例子：只看校准 p，多高才能放行', '设定',
    '不问 verifier，把校准后的 p_correct 当作答对的概率，从 p 最高的题往下放行，直到再放一道就违反约束。测试题各折合起来。实际方法会问 verifier 更新 p，所以门槛只是参考。');
  const am = D.alpha_main;
  const rows = [[TH('数据集 · generator'), TH(`放行错误率 α = ${am.selective_error}`), TH(`FPR α = ${am.fpr}`), TH(`FNR α = ${am.fnr}`)]];
  D.examples.forEach(([lab, cells]) => rows.push([TD(lab, { bold: true }), ...cells.map((c) => TD(c, { color: c === '不测' ? C.text2 : C.text1 }))]));
  table(s, rows, { y: 1.45, colW: [3.4, 2.9, 2.9, 2.933], fontSize: 14, rowH: 0.42 });
  text(s, '「全部放行」：把所有给出答案的题都放行也满足约束，不需要门槛；括号里是给出答案的题占全部题目的比例。MATH-500 不测 FPR（错误答案太少）。',
    { x: M, y: 6.35, w: W - 2 * M, h: 0.45, fontSize: 12, color: C.text2 });
}

// ---------------------------------------------------------------- 7. methods
pres.addSection({ title: '方法' });
{
  const s = slide(D.methods.length > 6 ? `${['', '', '', '', '', '', '六', '七', '八', '九'][D.methods.length]}个方法：五个会在测试时问 verifier` : '六个方法：只有三个会在测试时问 verifier', '方法',
    '所有方法读同一个校准后的 p_correct：isotonic 只在拟合部分拟合，再用到验证和测试部分。');
  const many = D.methods.length > 6;
  const rowsM = [
    [TH('方法'), TH('测试时问 verifier'), TH('怎么决定放行')],
    [TD('Ours-Offline', { bold: true, color: C.accent1 }), TD('按需，最多 3 个'), TD('校准先验 + likelihood；每步决定放行、拦下或再问（Bellman + LP 混合）')],
    [TD('AT-Frozen', { bold: true }), TD('不问'), TD('selective auditing 的阈值更新；拟合部分学好后冻结')],
    [TD('AT+Cal-Frozen', { bold: true }), TD('不问'), TD('论文 Algorithm 1：阈值 + 校准权重；同样冻结')],
    [TD('Cal-threshold', { bold: true }), TD('不问'), TD('直接对校准后的 p 取阈值，验证部分选')],
    [TD('Fixed-1', { bold: true }), TD('每题固定 1 个'), TD('同样的校准和 likelihood，后验过阈值放行；预算不够时随机抽题问')],
    [TD('Fixed-3', { bold: true }), TD('每题固定 3 个'), TD('同上，10 种组合，不提前停')],
    ...(D.methods.includes('Weaver') ? [
      [TD('Weaver', { bold: true }), TD('每题 5 个'), TD('5 个 verifier 的弱监督标签模型（EM），后验过阈值放行')],
      ...(D.methods.includes('Self-consistency') ? [[TD('Self-consistency', { bold: true }), TD('不问'), TD('10 个样本与答案一致的比例过阈值；只有 GPQA、MATH-500')]] : []),
      [TD('SSV', { bold: true }), TD('先 1 个，必要时 2 个'), TD('弱 verifier 分高放行、分低拦下，中间问强 verifier；只比放行错误率')]] : []),
  ];
  table(s, rowsM, { y: 1.35, colW: [2.1, 2.1, 7.933], fontSize: many ? 12 : 14, rowH: many ? 0.4 : 0.62 });
  text(s, '所有方法读同一个校准后的 p_correct。Fixed 和我们用同样的校准、likelihood 和花费模型，只是不按需决定问谁、问几个。Weaver 和 SSV 也和「全放行 / 全拦下」混合以满足预算。Self-consistency 靠多生成样本、不是 verifier 方法，放在附录。',
    { x: M, y: many ? 6.45 : 6.05, w: W - 2 * M, h: 0.45, fontSize: 11, color: C.text2 });
}

// ---------------------------------------------------------------- 8. evaluation
{
  const s = slide('怎么比较：验证部分选工作点，测试部分数显著条件', '方法',
    '区间没有做多重比较校正；相邻 α 和预算常选到同一策略，条件之间不独立，条件数只是描述。');
  const boxes = [['拟合', '校准 p_correct，学 likelihood 和阈值'], ['验证', '用标签和回放花费，选满足约束、目标最好的工作点'], ['测试', '冻结策略，只评估']];
  const bw = 3.5;
  boxes.forEach(([h, d], i) => {
    const x = M + i * (bw + 0.8);
    card(s, x, 1.5, bw, 1.6);
    text(s, h, { x: x + 0.25, y: 1.65, w: bw - 0.5, h: 0.45, fontSize: 20, bold: true, color: C.accent1 });
    text(s, d, { x: x + 0.25, y: 2.15, w: bw - 0.5, h: 0.85, fontSize: 14 });
    if (i < 2) s.addShape(pres.shapes.CHEVRON, { x: x + bw + 0.25, y: 2.1, w: 0.3, h: 0.4, fill: { color: GRAY }, line: { type: 'none' }, objectName: nm('arrow') });
  });
  const items = [
    ['主比较', `一个条件 = 一组（数据集 · generator，α，预算 f），比较我们和对照的目标：放行率，FNR 下是错误放行 e。放行错误率 ${S.selective_error.n}、FPR ${S.fpr.n}、FNR ${S.fnr.n}，共 ${D.n_cond} 个条件。`],
    ['显著', '2000 次配对 bootstrap，按测试折分层，QuantumKatas 按 kata 组重抽；95% 区间不含 0 才算一方显著更好。'],
    ['超标', '测试集风险超过 α、或花费超过预算的情况单独列出，不影响主比较。'],
    ['回退', '验证部分没有可行候选时：放行错误率和 FPR 全拦下，FNR 放行所有有答案的题。'],
  ];
  items.forEach(([h, d], i) => {
    const y = 3.5 + i * 0.8;
    text(s, h, { x: M, y, w: 1.3, h: 0.6, fontSize: 16, bold: true });
    text(s, d, { x: M + 1.4, y, w: W - 2 * M - 1.4, h: 0.7, fontSize: 15, color: C.text1 });
  });
}

const BB = (risk, b, fs) => { const c = D.by_budget[risk][b]; let o = 0, x = 0; fs.forEach((f) => { o += c[f][1]; x += c[f][2]; }); return `${o}:${x}`; };
// ---------------------------------------------------------------- 9. main comparison chart
pres.addSection({ title: '结果' });
{
  const s = slide('主比较：显著更好的条件数', '结果',
    '每个条形是条件数。FNR 下「更好」指错误放行更少。区间含 0 的条件不计入任何一边。');
  legendRow(s, [['我们显著更好', SERIES[0]], ['对照显著更好', GRAY]], M, 1.3);
  const cw = (W - 2 * M - 2 * 0.25) / 3;
  RISKS.forEach((risk, i) => {
    const x = M + i * (cw + 0.25);
    text(s, `${RN[risk]}`, { x, y: 1.85, w: cw, h: 0.4, fontSize: 16, bold: true, align: 'center' });
    text(s, `${S[risk].n} 个条件`, { x, y: 2.25, w: cw, h: 0.3, fontSize: 13, color: C.text2, align: 'center' });
    const keep = D.baselines.map((b, j) => j).filter((j) => !S[risk].nb || S[risk].nb[j] > 0);
    CHARTS.push({ slide: SN, kind: 'hbar', x, y: 2.6, w: cw, h: 4.25, categories: keep.map((j) => D.baselines[j]), max: 230, gap: 50,
      series: [{ name: '我们显著更好', values: keep.map((j) => S[risk].ours[j]), color: SERIES[0] }, { name: '对照显著更好', values: keep.map((j) => S[risk].base[j]), color: GRAY }],
      labelFmt: '0' });
  });
}

{
  const s = slide(POOL ? '按预算拆开：低预算优势更大，FNR 各预算都不输 Fixed' : '按预算拆开：和 Fixed-3 比，优势在低预算', '结果',
    'f = 0 时 Fixed 不能问 verifier，只能在全放行和全拦下之间混合，所以我们大比分领先。预算一多，Fixed-3 能把 3 个 verifier 问满，前两种约束下追平或反超。');
  const fs = ['0.0', '0.1', '0.25', '0.5', '0.75', '1.0'];
  const head = [TH('预算 f')];
  const BBS = POOL ? ['Fixed-1', 'Fixed-3'] : ['Fixed-3'];
  RISKS.forEach((r) => BBS.forEach((b) => head.push(TH(`${RN[r]} 对 ${b}`, { align: 'right' }))));
  const rows = [head];
  fs.forEach((f) => {
    const r = [TD(String(+f), { bold: true })];
    RISKS.forEach((risk) => BBS.forEach((b) => {
      const c = D.by_budget[risk][b][f];
      r.push(TD(`${c[1]} : ${c[2]}`, { align: 'right', fontFace: 'Arial' }));
    }));
    rows.push(r);
  });
  table(s, rows, { y: 1.5, colW: POOL ? [1.333, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8] : [2.533, 3.2, 3.2, 3.2], fontSize: 15, rowH: 0.5 });
  text(s, [
    { text: '每格是「我们显著更好 : 对照显著更好」的条件数（FNR 下更好指错误放行更少）。', options: { breakLine: true } },
    { text: !POOL ? `放行错误率：f ≤ 0.5 时 ${BB('selective_error', 'Fixed-3', ['0.0', '0.1', '0.25', '0.5'])}，f = 1 时 ${BB('selective_error', 'Fixed-3', ['1.0'])}。FPR：f ≤ 0.25 时 ${BB('fpr', 'Fixed-3', ['0.0', '0.1', '0.25'])}，f ≥ 0.5 时 ${BB('fpr', 'Fixed-3', ['0.5', '0.75', '1.0'])}。FNR：f > 0 时 ${BB('fnr', 'Fixed-3', ['0.1', '0.25', '0.5', '0.75', '1.0'])}，各预算都是 Fixed-3 更好。` : POOL ? `放行错误率和 FPR：低预算时几乎全胜；f = 1 时对 Fixed-3 是 ${BB('selective_error', 'Fixed-3', ['1.0'])}（放行错误率）和 ${BB('fpr', 'Fixed-3', ['1.0'])}（FPR）。FNR：f > 0 时对 Fixed-1 是 ${BB('fnr', 'Fixed-1', ['0.1', '0.25', '0.5', '0.75', '1.0'])}，对 Fixed-3 是 ${BB('fnr', 'Fixed-3', ['0.1', '0.25', '0.5', '0.75', '1.0'])}。` : '放行错误率和 FPR：低预算时我们几乎全胜，预算 ≥ 0.5 或 0.75 后 Fixed-3 追平或反超。FNR：f > 0 时各预算都是 Fixed 更好。' },
  ], { x: M, y: 5.5, w: W - 2 * M, h: 1.2, fontSize: 15 });
}

// ---------------------------------------------------------------- 10-12. per-risk curves
function curveSlide(risk, title, notes, side) {
  const s = slide(title, '结果', notes);
  const alphas = D.alpha[risk].map(String);
  const cur = D.curves_f1[risk];
  const isFnr = risk === 'fnr';
  const npairs = risk === 'fpr' ? 6 : 9;
  pill(s, RN[risk], M, 1.35, 1.6);
  text(s, `B = 1 × c_ref；${npairs} 个数据集 · generator 的平均`, { x: M + 1.8, y: 1.35, w: 6, h: 0.36, fontSize: 13, color: C.text2, valign: 'middle' });
  const vals = D.methods.flatMap((m) => cur[m].map((v) => 100 * v));
  const top = isFnr ? Math.ceil(Math.max(...vals) / 5) * 5 : 100;
  CHARTS.push({ slide: SN, kind: 'line', x: M, y: 1.85, w: 7.9, h: 4.95, categories: alphas, min: 0, max: top, unit: isFnr ? 5 : 20,
    series: D.methods.map((m, i) => ({ name: m, values: cur[m].map((v) => +(100 * v).toFixed(1)), color: SERIES[i], bold: m === 'Ours-Offline' })),
    catTitle: '目标 α', valTitle: isFnr ? '错误放行 e（占全部题，越低越好）' : '放行率（越高越好）', valFmt: '0"%"', legend: true });
  side(s);
}
function sidePanel(s, risk, head, lines) {
  const x0 = 8.85, w = W - M - x0;
  card(s, x0, 1.35, w, 5.45);
  text(s, head, { x: x0 + 0.25, y: 1.55, w: w - 0.5, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  const am = D.alpha_main[risk];
  const k = D.alpha[risk].indexOf(am);
  const cur = D.curves_f1[risk];
  const order = D.methods.slice().sort((a, b) => risk === 'fnr' ? cur[a][k] - cur[b][k] : cur[b][k] - cur[a][k]);
  order.forEach((m, i) => {
    const y = 2.05 + i * 0.42;
    s.addShape(pres.shapes.OVAL, { x: x0 + 0.25, y: y + 0.12, w: 0.14, h: 0.14, fill: { color: SERIES[D.methods.indexOf(m)] }, line: { type: 'none' }, objectName: nm('dot') });
    text(s, m, { x: x0 + 0.5, y, w: 2.1, h: 0.38, fontSize: 14, bold: m === 'Ours-Offline', valign: 'middle' });
    text(s, pct(cur[m][k], 1), { x: x0 + 2.6, y, w: w - 2.85, h: 0.38, fontSize: 14, bold: m === 'Ours-Offline', align: 'right', valign: 'middle', fontFace: 'Arial' });
  });
  text(s, lines, { x: x0 + 0.25, y: 4.75, w: w - 0.5, h: 1.95, fontSize: 14 });
}
if (D.typ) {
  const TF = D.typ_f;
  RISKS.forEach((risk) => {
    const am = D.alpha_main[risk], dss = Object.keys(D.typ[risk]);
    const isFnr = risk === 'fnr';
    const title = { selective_error: `放行错误率：每个数据集的典型对比（α = ${am}，B = ${TF} × c_ref）`, fpr: `FPR：每个数据集的典型对比（α = ${am}，B = ${TF} × c_ref）`, fnr: `FNR：每个数据集的典型对比（α = ${am}，B = ${TF} × c_ref）` }[risk];
    const s = slide(title, '结果', `每个数据集是 3 个 generator 的平均。${isFnr ? '纵轴是错误放行 e，越低越好。' : '纵轴是放行率，越高越好。'}${risk === 'fpr' ? '不测 MATH-500。' : ''}`);
    const n = dss.length, gap = 0.55, cw = (W - 2 * M - (n - 1) * gap) / n;
    const short = (m) => m === 'Self-consistency' ? 'SC' : m;
    const allv = dss.flatMap((ds) => Object.values(D.typ[risk][ds]).map((v) => 100 * v));
    const top = isFnr ? Math.ceil(Math.max(...allv) * 1.3 / 5) * 5 : 115;
    dss.forEach((ds, i) => {
      const x = M + i * (cw + gap), ms = D.methods.filter((m) => D.typ[risk][ds][m] != null);
      text(s, ds, { x, y: 1.3, w: cw, h: 0.35, fontSize: 16, bold: true, align: 'center' });
      text(s, isFnr ? '条形 = 错误放行 e（占全部题，越低越好）' : '条形 = 放行率（占全部题，越高越好）', { x, y: 1.62, w: cw, h: 0.28, fontSize: 11, color: C.text2, align: 'center' });
      CHARTS.push({ slide: SN, kind: 'hbar', x, y: 1.95, w: cw, h: 4.45, categories: ms.map(short), max: top, gap: 40,
        series: [{ name: isFnr ? '错误放行 e' : '放行率', values: ms.map((m) => +(100 * D.typ[risk][ds][m]).toFixed(1)), color: SERIES[0] }],
        pointColors: ms.map((m) => m === 'Ours-Offline' ? SERIES[0] : 'A9B4BA'), labelFmt: isFnr ? '0.0"%"' : '0"%"' });
    });
    text(s, (isFnr ? '蓝色是我们。FNR 下所有方法都要放行大部分题，差距天然较小。' : '蓝色是我们。') + ' SSV 只比放行错误率。', { x: M, y: 6.5, w: W - 2 * M, h: 0.35, fontSize: 12, color: C.text2 });
  });
  // actual cost vs budget
  {
    const s = slide('实际花费对预算：我们多数时候只用掉一部分预算', '结果',
      `α 取各风险的主要展示值，9 组数据集 · generator 的平均。纵轴是测试集上每题平均 verifier 花费占 c_ref 的比例，虚线是预算上限 f。`);
    const spend = ['Ours-Offline', 'Fixed-1', 'Fixed-3', 'Weaver', 'SSV'].filter((m) => D.methods.includes(m));
    legendRow(s, [...spend.map((m) => [m, MCOL(m)]), ['预算上限', '9AA5AB']], M, 1.3);
    const cw = (W - 2 * M - 2 * 0.3) / 3;
    RISKS.forEach((risk, i) => {
      const x = M + i * (cw + 0.3), cc = D.cost_curve[risk];
      const ms = spend.filter((m) => cc[m]);
      text(s, `${RN[risk]}（α = ${D.alpha_main[risk]}）`, { x, y: 1.8, w: cw, h: 0.35, fontSize: 15, bold: true, align: 'center' });
      CHARTS.push({ slide: SN, kind: 'line', x, y: 2.15, w: cw, h: 4.6, categories: D.budget_fractions.map(String), min: 0, max: 100, unit: 25, legend: false,
        series: [...ms.map((m) => ({ name: m, values: cc[m].map((v) => v == null ? null : +(100 * v).toFixed(1)), color: MCOL(m), bold: m === 'Ours-Offline' })),
          { name: '预算上限', values: D.budget_fractions.map((f) => 100 * f), color: '9AA5AB', dash: true }],
        catTitle: '预算 f', valTitle: '实际花费 / c_ref', valFmt: '0"%"' });
    });
  }
  // actual risk vs target
  {
    const s = slide('实际风险对目标 α：放行错误率贴着目标，FPR 偏高，FNR 偏低', '结果',
      `预算 B = ${D.typ_f} × c_ref，9 组数据集 · generator 的平均（FPR 6 组）。纵轴是测试集上的实际风险，虚线是目标 α。平均值会掩盖单个条件的超标，超标次数见后面的表。`);
    const ms0 = D.methods;
    legendGrid(s, [...ms0.map((m) => [m, MCOL(m)]), ['目标 α', '9AA5AB']], M, 1.25, 5, W - 2 * M);
    const cw = (W - 2 * M - 2 * 0.3) / 3;
    RISKS.forEach((risk, i) => {
      const x = M + i * (cw + 0.3), rc = D.risk_curve[risk], al = D.alpha[risk];
      const ms = ms0.filter((m) => rc[m]);
      const raw = Math.max(...ms.flatMap((m) => rc[m].filter((v) => v != null).map((v) => 100 * v)), ...al.map((a) => 100 * a)) * 1.1;
      const unit = raw <= 20 ? 5 : 10, top = Math.ceil(raw / unit) * unit;
      text(s, RN[risk], { x, y: 2.05, w: cw, h: 0.35, fontSize: 15, bold: true, align: 'center' });
      CHARTS.push({ slide: SN, kind: 'line', x, y: 2.4, w: cw, h: 4.4, categories: al.map(String), min: 0, max: top, unit, legend: false,
        series: [...ms.map((m) => ({ name: m, values: rc[m].map((v) => v == null ? null : +(100 * v).toFixed(1)), color: MCOL(m), bold: m === 'Ours-Offline' })),
          { name: '目标 α', values: al.map((a) => 100 * a), color: '9AA5AB', dash: true }],
        catTitle: '目标 α', valTitle: `实际${RN[risk]}`, valFmt: '0"%"' });
    });
  }
} else {
curveSlide('selective_error', '放行错误率：放行率领先，预算最多时 Fixed-3 追平', '图里是预算最多的 B = 1 × c_ref，这是 Fixed-3 最强的情况；预算少时 Fixed-3 问不起 3 个 verifier。',
  (s) => sidePanel(s, 'selective_error', `α = ${D.alpha_main.selective_error} 时的平均放行率`,
    `显著更好的条件数，对 Fixed-3：f ≤ 0.5 时 ${BB('selective_error', 'Fixed-3', ['0.0', '0.1', '0.25', '0.5'])}，f = 1 时 ${BB('selective_error', 'Fixed-3', ['1.0'])}。对 Cal-threshold 全部条件 ${S.selective_error.ours[2]}:${S.selective_error.base[2]}。`));
curveSlide('fpr', POOL ? 'FPR：领先阈值方法和 Fixed-1，预算多时与 Fixed-3 互有胜负' : 'FPR：领先不问 verifier 的方法，预算多时不如 Fixed', `不测 MATH-500：FPR 的分母是错误答案数，5 折测试合计 Qwen3 ${D.fpr_excl['Qwen3']} 道、gpt-oss ${D.fpr_excl['gpt-oss']} 道、Sonnet 5.5 ${D.fpr_excl['Sonnet 5.5']} 道（含没给答案的），一两道题就让 FPR 变化很多。`,
  (s) => sidePanel(s, 'fpr', `α = ${D.alpha_main.fpr} 时的平均放行率`,
    `对 Fixed-3：f ≤ 0.25 时 ${BB('fpr', 'Fixed-3', ['0.0', '0.1', '0.25'])}，f ≥ 0.5 时 ${BB('fpr', 'Fixed-3', ['0.5', '0.75', '1.0'])}。对 Cal-threshold ${S.fpr.ours[2]}:${S.fpr.base[2]}。不测 MATH-500：错误答案太少（${D.fpr_excl['Qwen3']} / ${D.fpr_excl['gpt-oss']} / ${D.fpr_excl['Sonnet 5.5']} 道）。`));
curveSlide('fnr', POOL ? 'FNR：错误放行显著更少的条件多于 Fixed' : 'FNR：比不问 verifier 的方法好，比 Fixed 差', '纵轴越低越好。三个不问 verifier 的方法几乎重合。',
  (s) => sidePanel(s, 'fnr', `α = ${D.alpha_main.fnr} 时的平均错误放行 e`,
    `错误放行显著更少的条件数：对 Cal-threshold ${S.fnr.ours[2]}:${S.fnr.base[2]}，对 Fixed-1 ${S.fnr.ours[3]}:${S.fnr.base[3]}，对 Fixed-3 ${S.fnr.ours[4]}:${S.fnr.base[4]}。`));

}

// ---------------------------------------------------------------- 13. why FNR loses to Fixed / pool share
if (POOL) {
  const s = slide('FPR 和 FNR 下，约一半条件挑中了固定策略', '结果',
    '条件 × 折，f > 0。固定 panel 候选只用 Ours 短名单里的 verifier，是 Fixed-1 / Fixed-3 候选的子集；挑中时 Ours 执行的就是一个 Fixed 也能选的策略。');
  const PS = D.pool_share;
  legendRow(s, [['单个 verifier', SERIES[0]], ['3 个一组', '8FB8EA']], M, 1.3);
  const cats = [];
  const one = [], three = [];
  RISKS.forEach((r) => ['lo', 'hi'].forEach((g) => {
    const [a, b, , t] = PS[r][g];
    cats.push(`${r === 'selective_error' ? '错误率' : RN[r]}·${g === 'lo' ? '低' : '高'}`);
    one.push(+(100 * a / t).toFixed(0)); three.push(+(100 * b / t).toFixed(0));
  }));
  text(s, '低 = 预算 f ≤ 0.25，高 = f ≥ 0.5；错误率 = 放行错误率。纵轴是挑中该类固定 panel 的比例。', { x: M, y: 6.45, w: 7.6, h: 0.4, fontSize: 12, color: C.text2 });
  CHARTS.push({ slide: SN, kind: 'col', x: M, y: 1.8, w: 7.6, h: 4.6, categories: cats, max: 70, gap: 60,
    series: [{ name: '单个 verifier', values: one, color: SERIES[0] }, { name: '3 个一组', values: three, color: '8FB8EA' }], labelFmt: '0"%"' });
  const x0 = 8.6, w = W - M - x0;
  const tot = (r) => { const a = PS[r].lo, b = PS[r].hi; return pct((a[0] + a[1] + b[0] + b[1]) / (a[3] + b[3])); };
  const facts = [
    [tot('fnr'), 'FNR 下挑中固定 panel 的比例；FPR 是 ' + tot('fpr') + '，放行错误率是 ' + tot('selective_error')],
    [`${S.fnr.ours[3]} : ${S.fnr.base[3]}`, `FNR 对 Fixed-1 错误放行显著更少 : 显著更多（原版本 25 : 75）`],
    [`${S.fnr.overrun['Ours-Offline']}`, `FNR 风险超标的条件数（原版本 70；Fixed-1 ${S.fnr.overrun['Fixed-1']}、Fixed-3 ${S.fnr.overrun['Fixed-3']}）`],
  ];
  facts.forEach(([big, d], i) => {
    const y = 1.5 + i * 1.75;
    card(s, x0, y, w, 1.55);
    text(s, big, { x: x0 + 0.25, y: y + 0.15, w: w - 0.5, h: 0.65, fontSize: 30, bold: true, color: C.accent1, fontFace: 'Arial' });
    text(s, d, { x: x0 + 0.25, y: y + 0.8, w: w - 0.5, h: 0.65, fontSize: 13, color: C.text2 });
  });
} else {
  const s = slide('FNR 为什么输给 Fixed：我们问得太少', '结果',
    'FNR 下大部分题本来就要放行，差距只能靠 verifier 把个别错答案的 p 拉到门槛以下。我们的方法在这里花钱太省。平均值取 f > 0 的条件。');
  const sp = D.spend.fnr, ms = ['Ours-Offline', 'Fixed-1', 'Fixed-3'];
  legendRow(s, [['用掉的预算比例', SERIES[0]], ['问了 verifier 的题', '8FB8EA']], M, 1.3);
  CHARTS.push({ slide: SN, kind: 'col', x: M, y: 1.8, w: 6.9, h: 4.9, categories: ms, max: 100, gap: 70,
    series: [{ name: '用掉的预算比例', values: ms.map((m) => +(100 * sp[m].budget_use).toFixed(0)), color: SERIES[0] },
             { name: '问了 verifier 的题', values: ms.map((m) => +(100 * sp[m].query_rate).toFixed(0)), color: '8FB8EA' }],
    labelFmt: '0"%"' });
  const fb = D.fnr_both['Fixed-1'];
  const x0 = 8.0, w = W - M - x0;
  const facts = [
    [pct(sp['Ours-Offline'].fallback), '的折 × 种子在验证部分没有可行候选，回退到放行所有有答案的题'],
    [`${S.fnr.ours[3]} : ${S.fnr.base[3]}`, `${S.fnr.n} 个条件里，对 Fixed-1 错误放行显著更少 : 显著更多`],
    [`+${(100 * fb.median).toFixed(2)} 个百分点`, `双方都达标的 ${fb.n} 个条件里，我们减 Fixed-1 的 e 中位数，差距很小`],
  ];
  facts.forEach(([big, d], i) => {
    const y = 1.5 + i * 1.75;
    card(s, x0, y, w, 1.55);
    text(s, big, { x: x0 + 0.25, y: y + 0.15, w: w - 0.5, h: 0.65, fontSize: 30, bold: true, color: C.accent1, fontFace: 'Arial' });
    text(s, d, { x: x0 + 0.25, y: y + 0.8, w: w - 0.5, h: 0.65, fontSize: 13, color: C.text2 });
  });
}

// ---------------------------------------------------------------- 13b. FNR by dataset (original version)
if (!POOL) {
  const s = slide('FNR 输在哪：问得少、风险额度没用完、还会回退', '结果',
    'f > 0 的条件平均。实际 FNR 是测试集上被拦下的正确答案比例，越接近 α 说明额度用得越满。');
  const SD = D.fnr_spend_ds, MS = ['Ours-Offline', 'Fixed-1', 'Fixed-3'];
  const trip = (ds, k, f) => MS.map((m) => f(SD[ds][m][k])).join(' / ');
  const rows = [[TH('数据集'), TH('问了 verifier 的题', { align: 'right' }), TH('用掉的预算', { align: 'right' }), TH('实际 FNR', { align: 'right' }), TH('Ours 回退', { align: 'right' })]];
  Object.keys(SD).forEach((ds) => rows.push([TD(ds, { bold: true }), TD(trip(ds, 'ask', (v) => pct(v)), { align: 'right', fontFace: 'Arial' }),
    TD(trip(ds, 'spend', (v) => pct(v)), { align: 'right', fontFace: 'Arial' }), TD(trip(ds, 'fnr', (v) => v.toFixed(3)), { align: 'right', fontFace: 'Arial' }),
    TD(pct(SD[ds]['Ours-Offline'].fallback), { align: 'right', fontFace: 'Arial' })]));
  table(s, rows, { y: 1.45, colW: [2.2, 2.9, 2.9, 2.6, 1.533], fontSize: 14, rowH: 0.5 });
  text(s, '每格依次是 Ours / Fixed-1 / Fixed-3。', { x: M, y: 3.55, w: W - 2 * M, h: 0.35, fontSize: 12, color: C.text2 });
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  [['问得少，钱没花完', 'GPQA 和 QuantumKatas 上 Ours 只问约四分之一的题，也只用掉约四分之一预算。FNR 的收益全靠 verifier 从「本来要放行」的题里挑出错答案。'],
   ['风险额度没用完', 'Ours 的实际 FNR 低于 Fixed，本可以多拦一些、少放一些错答案。花钱更多的候选常在验证部分 FNR 超标被淘汰，最后选中的多是按小预算规划的候选。'],
   ['回退到全放行', 'GPQA 和 QuantumKatas 有 15% 到 17% 的折 × 条件在验证部分找不到可行候选，只能放行所有有答案的题。']].forEach(([h, d], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 4.0, cw, 2.75);
    text(s, h, { x: x + 0.25, y: 4.15, w: cw - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
    text(s, d, { x: x + 0.25, y: 4.6, w: cw - 0.5, h: 2.05, fontSize: 13 });
  });
}

// ---------------------------------------------------------------- 14. overruns
{
  const s = slide('超标情况：单独列出，不影响主比较', '结果',
    '风险超标是测试集上实际风险 > α，任何超出都算，包括很小的超出。阈值方法不问 verifier，所以不会超预算。');
  const head = [TH('方法')];
  RISKS.forEach((r) => { head.push(TH(`${RN[r]}：风险超标`)); head.push(TH(`${RN[r]}：超预算`)); });
  const rows = [head];
  D.methods.forEach((m) => {
    const r = [TD(m, { bold: true, color: m === 'Ours-Offline' ? C.accent1 : C.text1 })];
    RISKS.forEach((risk) => {
      const none = !S[risk].n_cond[m];
      r.push(TD(none ? '—' : `${S[risk].overrun[m]} / ${S[risk].n_cond[m]}`, { align: 'right', fontFace: 'Arial' }));
      r.push(TD(none ? '—' : `${S[risk].budget_over[m]}`, { align: 'right', fontFace: 'Arial' }));
    });
    rows.push(r);
  });
  table(s, rows, { y: 1.45, colW: [2.3, 1.75, 1.55, 1.75, 1.55, 1.75, 1.483], fontSize: D.methods.length > 6 ? 13 : 14, rowH: D.methods.length > 6 ? 0.42 : 0.5 });
  text(s, [
    { text: POOL ? 'FPR 和 FNR 的超标比原版本多（原版本 58 和 70），FNR 已多于 Fixed；' : '我们的风险超标次数和 Fixed 相近，比三个阈值方法多；', options: { bold: true } },
    { text: POOL ? '候选池变大后，验证部分更容易挑到碰巧在验证部分达标的候选（推测）。超预算只发生在会问 verifier 的三个方法上。' : (D.methods.includes('Weaver') ? '验证部分选出的工作点放到测试集上常常略微超出 α。超预算只发生在会问 verifier 的方法上，Weaver 和 Fixed-3 最多；Self-consistency 很少超标。' : '验证部分选出的工作点放到测试集上常常略微超出 α。超预算只发生在会问 verifier 的三个方法上，Fixed-3 最多。') },
  ], { x: M, y: D.methods.length > 6 ? 6.0 : 5.35, w: W - 2 * M, h: 0.9, fontSize: D.methods.length > 6 ? 13 : 15 });
}

// ---------------------------------------------------------------- 15. raw vs calibrated
pres.addSection({ title: '对照与下一步' });
{
  const s = slide('对照：所有方法改读原始 p，结论方向不变', '对照与下一步',
    '同样的数据、划分、α 和预算，只是所有方法都读原始 p_correct：Ours 和 Fixed 不做 isotonic 校准，直接把原始 p 当先验（verifier 的 likelihood 不变）。');
  const head = [TH('对照方法')];
  RISKS.forEach((r) => { head.push(TH(`${RN[r]} 校准 p`)); head.push(TH(`${RN[r]} 原始 p`)); });
  const rows = [head];
  D.baselines.forEach((b, i) => {
    const name = b === 'Cal-threshold' ? '阈值方法' : b;
    const r = [TD(name, { bold: true })];
    RISKS.forEach((risk) => {
      const no = S[risk].nb && S[risk].nb[i] === 0;
      r.push(TD(no ? '—' : `${S[risk].ours[i]} : ${S[risk].base[i]}`, { align: 'right', fontFace: 'Arial' }));
      r.push(TD(no ? '—' : `${SR[risk].ours[i]} : ${SR[risk].base[i]}`, { align: 'right', fontFace: 'Arial', color: C.text2 }));
    });
    rows.push(r);
  });
  table(s, rows, { y: 1.45, colW: [2.2, 1.65, 1.65, 1.65, 1.65, 1.65, 1.683], fontSize: D.baselines.length > 5 ? 13 : 14, rowH: D.baselines.length > 5 ? 0.42 : 0.5 });
  const ic = D.input_compare.selective_error['Ours-Offline'];
  text(s, [
    { text: '数字是「我们显著更好 : 对照显著更好」的条件数。', options: { breakLine: true } },
    { text: `我们自己变化很小：放行错误率下平均放行率 ${pct(ic[0], 1)}（校准）对 ${pct(ic[1], 1)}（原始）。isotonic 校准不改变题目排序，所以阈值方法用校准 p 还是原始 p，分出的放行集合几乎一样。` },
  ], { x: M, y: D.baselines.length > 5 ? 5.45 : 4.85, w: W - 2 * M, h: 1.4, fontSize: D.baselines.length > 5 ? 13 : 15 });
}

// ---------------------------------------------------------------- 15b. variants tried (original version)
if (!POOL) {
  const s = slide('试过的改进：没有一种让 FNR 追上 Fixed，但都让超标更少', '对照与下一步',
    '都只用缓存，参数只在拟合部分定。收缩：isotonic 每块往总体正确率拉，(s + a·ȳ)/(n + a)。打折：第 2、3 个 verifier 的似然比取 γ 次方。加固定 panel：验证部分也可以挑「每题都问同一组 verifier」的策略。');
  const VR = D.variants, labs = Object.keys(VR);
  const pair = (x) => `${x[0]} : ${x[1]}`;
  const rows = [[TH('版本'), TH('放行错误率 对 Fixed-3（f = 1）', { align: 'right' }), TH('FPR 对 Fixed-1', { align: 'right' }), TH('FPR 对 Fixed-3', { align: 'right' }),
    TH('FNR 对 Fixed-1', { align: 'right' }), TH('FNR 对 Fixed-3', { align: 'right' }), TH('FNR 超标', { align: 'right' })]];
  labs.forEach((l) => {
    const v = VR[l];
    rows.push([TD(l, { bold: true, color: l === '原版本' ? C.accent1 : C.text1 }), TD(pair(v.sel_f1_vs_f3), { align: 'right', fontFace: 'Arial' }),
      TD(pair(v.fpr.f1), { align: 'right', fontFace: 'Arial' }), TD(pair(v.fpr.f3), { align: 'right', fontFace: 'Arial' }),
      TD(pair(v.fnr.f1), { align: 'right', fontFace: 'Arial' }), TD(pair(v.fnr.f3), { align: 'right', fontFace: 'Arial' }), TD(String(v.fnr.over), { align: 'right', fontFace: 'Arial' })]);
  });
  table(s, rows, { y: 1.45, colW: [2.0, 2.4, 1.6, 1.6, 1.6, 1.6, 1.333], fontSize: 14, rowH: 0.5 });
  text(s, [
    { text: '加固定 panel：', options: { bold: true } }, { text: 'FNR 变成赢，但约一半 FPR / FNR 条件实际执行的是固定策略，只作诊断。', options: { breakLine: true } },
    { text: '收缩：', options: { bold: true } }, { text: '更保守、超标更少，但 FNR 对 Fixed 更差。', options: { breakLine: true } },
    { text: '打折：', options: { bold: true } }, { text: 'γ 多在 0.05 到 0.9，verifier 之间确实相关；放行错误率高预算对 Fixed-3 从 8:11 变成 17:6。', options: { breakLine: true } },
    { text: '收缩 + 打折：', options: { bold: true } }, { text: 'FNR 超标从 70 降到 33（Fixed 是 74 和 72），FNR 对 Fixed 仍然输。' },
  ], { x: M, y: 4.65, w: W - 2 * M, h: 2.1, fontSize: 14, paraSpaceAfter: 4 });
}

// ---------------------------------------------------------------- 15c. overruns and directions (original version)
if (!POOL) {
  const s = slide('风险超标是所有方法的共同弱点；改进方向', '对照与下一步',
    '超标：测试集风险 > α，很小的超出也算。原因在选择方式：验证部分只有 50 到 100 题，点估计 ≤ α 就算达标，挑出的工作点压在 α 边上。');
  const cw = (W - 2 * M - 0.3) / 2;
  card(s, M, 1.45, cw, 5.3);
  text(s, '风险超标条件数', { x: M + 0.3, y: 1.6, w: cw - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  const ms = ['Ours-Offline', 'Fixed-1', 'Fixed-3', 'Cal-threshold'];
  const rows = [[TH('方法'), TH('放行错误率', { align: 'right' }), TH('FPR', { align: 'right' }), TH('FNR', { align: 'right' })]];
  ms.forEach((m) => rows.push([TD(m, { bold: m === 'Ours-Offline' }), ...RISKS.map((r) => TD(`${S[r].overrun[m]} / ${S[r].n_cond[m]}`, { align: 'right', fontFace: 'Arial' }))]));
  table(s, rows, { x: M + 0.3, y: 2.1, w: cw - 0.6, colW: [1.8, 1.3, 1.1, 1.12], fontSize: 13, rowH: 0.42 });
  text(s, '会问 verifier 的方法超标都多；FPR 和 FNR 下我们比 Fixed 少。修法：验证部分用风险的上置信界判断达标（如 Clopper–Pearson 上界 ≤ α，或 Learn-then-Test），对所有方法都生效，代价是放行变少。',
    { x: M + 0.3, y: 4.45, w: cw - 0.6, h: 2.2, fontSize: 13 });
  const x1 = M + cw + 0.3;
  card(s, x1, 1.45, cw, 5.3);
  text(s, '改进方向', { x: x1 + 0.3, y: 1.6, w: cw - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  text(s, [
    { text: 'FNR 多问：', options: { bold: true } }, { text: 'LP 里加「至少用掉 x% 预算」的下限，钱花在离门槛最近的题上。', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 6 } },
    { text: 'LP 用真实估计：', options: { bold: true } }, { text: '规则的放行率、风险、花费改用拟合部分交叉拟合的估计，不全靠模型。', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 6 } },
    { text: '收缩 + 打折：', options: { bold: true } }, { text: '已验证能明显减少超标，可作为默认校准。', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 6 } },
    { text: '上置信界选择：', options: { bold: true } }, { text: '让「满足约束」这句话站得住。' },
  ], { x: x1 + 0.3, y: 2.15, w: cw - 0.6, h: 4.4, fontSize: 14 });
}

// ---------------------------------------------------------------- 16. open items
{
  const s = slide('待定和没做的部分', '对照与下一步', '');
  const cw = (W - 2 * M - 0.3) / 2;
  card(s, M, 1.5, cw, 3.4);
  text(s, '待你决定', { x: M + 0.3, y: 1.7, w: cw - 0.6, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  text(s, [
    { text: '要不要加 Weaver、SSV、Self-consistency', options: { bold: true, breakLine: true } },
    { text: '它们来自之前的效用实验，不在这次手册的方法里。要加的话，各自做成「打分 + 每种风险在验证部分选阈值」，只用缓存。', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 8 } },
    { text: 'Self-consistency 在 QuantumKatas 上没有定义；SSV 需要先冻结一个版本。' },
  ], { x: M + 0.3, y: 2.2, w: cw - 0.6, h: 4.0, fontSize: 15 });
  const x1 = M + cw + 0.3;
  card(s, x1, 1.5, cw, 3.4);
  text(s, '手册里可选、这次没跑', { x: x1 + 0.3, y: 1.7, w: cw - 0.6, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  text(s, [
    { text: 'GPQA 固定美元预算的附录', options: { bullet: true, breakLine: true } },
    { text: 'Ours 的消融：原始先验、不问 verifier、单一规则', options: { bullet: true, breakLine: true } },
    { text: 'AT 的 γ_test = γ_fit 敏感性', options: { bullet: true, breakLine: true } },
    { text: '外部审计花费场景', options: { bullet: true } },
  ], { x: x1 + 0.3, y: 2.2, w: cw - 0.6, h: 4.0, fontSize: 15, paraSpaceAfter: 8 });
}

// ---------------------------------------------------------------- appendix
pres.addSection({ title: '附录' });
function mainTable(risk) {
  const am = D.alpha_main[risk];
  const goal = risk === 'fnr' ? '错误放行 e' : '放行率';
  const s = slide(`附录：${RN[risk]} α = ${am}，B = 1 × c_ref`, '附录',
    `每格是「${goal} / 测试集实际 ${RN[risk]}」；✕ 表示风险或预算未达标。`);
  const rows = [[TH('数据集 · generator'), ...D.methods.map((m) => TH(m, { align: 'right' }))]];
  D.main_f1[risk].forEach(([lab, cells]) => {
    rows.push([TD(lab, { bold: true }), ...cells.map((c) => c == null ? TD('—', { align: 'right' }) : (([o, r, ok]) => TD(`${pct(o, 0)} / ${r == null ? '—' : pct(r, 1)}${ok ? '' : ' ✕'}`,
      { align: 'right', fontFace: 'Arial', color: ok ? C.text1 : C.text2 }))(c))]);
  });
  const nm_ = D.methods.length, lw = 2.3;
  table(s, rows, { y: 1.45, colW: [lw, ...D.methods.map(() => (W - 2 * M - lw) / nm_)], fontSize: nm_ > 6 ? 10 : 13, rowH: 0.48 });
  text(s, `每格：${goal} / 测试集实际 ${RN[risk]}；✕ 表示风险或预算未达标（灰色）。`, { x: M, y: 6.45, w: W - 2 * M, h: 0.4, fontSize: 12, color: C.text2 });
}
RISKS.forEach(mainTable);
if (D.sc_appendix && Object.keys(D.sc_appendix).length) {
  const s = slide('附录：Self-consistency（不计入正文比较）', '附录',
    'Self-consistency 的分数是 generator 自己 10 个样本里和最终答案一致的比例，过阈值就放行，阈值在验证部分挑。它不问 verifier，靠的是多生成 10 次，这部分花费没有计入 verifier 预算。');
  const SA = D.sc_appendix;
  const rows = [[TH('数据集 · 风险'), TH('我们 : SC（显著更好的条件数）', { align: 'right' }), TH('SC 风险超标', { align: 'right' }), TH('SC 回退比例', { align: 'right' })]];
  Object.keys(SA).forEach((ds) => RISKS.forEach((r) => {
    const x = SA[ds][r];
    if (!x) return;
    rows.push([TD(`${ds} · ${RN[r]}`, { bold: true }), TD(`${x.ours} : ${x.base}（共 ${x.n}）`, { align: 'right', fontFace: 'Arial' }),
      TD(String(x.over), { align: 'right', fontFace: 'Arial' }), TD(pct(x.fallback), { align: 'right', fontFace: 'Arial' })]);
  }));
  table(s, rows, { y: 1.4, w: 7.6, colW: [2.6, 2.5, 1.2, 1.3], fontSize: 13, rowH: 0.42 });
  const x0 = 8.5, w = W - M - x0;
  card(s, x0, 1.4, w, 5.3);
  text(s, '10 个样本的花费（×10⁻⁴ 美元 / 题）', { x: x0 + 0.25, y: 1.55, w: w - 0.5, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  const lines = [];
  Object.keys(SA).forEach((ds) => {
    lines.push({ text: `${ds}（c_ref ${D.cref[ds][0].toFixed(0)}–${D.cref[ds][1].toFixed(0)}）`, options: { bold: true, breakLine: true } });
    Object.entries(SA[ds].usd).forEach(([g, v]) => lines.push({ text: `${g}：${v.toFixed(0)}`, options: { breakLine: true } }));
    lines.push({ text: ' ', options: { breakLine: true, fontSize: 6 } });
  });
  lines.push({ text: 'Sonnet 5.5 和 MATH-500 · Qwen3 的样本花费是整个 verifier 预算的几倍到几十倍。', options: {} });
  text(s, lines, { x: x0 + 0.25, y: 2.0, w: w - 0.5, h: 4.6, fontSize: 13 });
  text(s, 'GPQA 上 SC 很少能达标：10 个样本全一致的题，答对率也只有 71%（Qwen3）到 93%（Sonnet 5.5），放行错误率 α = 0.05 下 80% 的条件只能回退成全部拦下。MATH-500 上它比我们好，但靠的是没计入预算的生成花费。QuantumKatas 是代码题，没有定义。',
    { x: M, y: 5.0, w: 7.6, h: 1.7, fontSize: 12, color: C.text2 });
}
{
  const s = slide('附录：复现和核对', '附录', '');
  text(s, [
    { text: '只用缓存：', options: { bold: true } },
    { text: '所有命令都拦截网络连接，这次没有任何 generator、verifier 或评分调用。', options: { breakLine: true } },
    { text: '独立重算：', options: { bold: true } },
    { text: `从逐题日志重算 ${D.verify.cells} 个条件 × 方法（校准 p 和原始 p 两套），全部一致。`, options: { breakLine: true } },
    { text: '代码哈希：', options: { bold: true } },
    { text: `${D.code}，分支 claude/project-thread-m8td13。` },
  ], { x: M, y: 1.5, w: W - 2 * M, h: 1.6, fontSize: 16, paraSpaceAfter: 6 });
  card(s, M, 3.4, W - 2 * M, 2.3);
  text(s, [
    { text: `python -m vgx.offline_multirisk run --config ${D.cfg_path}`, options: { breakLine: true } },
    { text: `python -m vgx.offline_multirisk run --config ${D.cfg_path.replace('_calibrated', '_raw')}`, options: { breakLine: true } },
    { text: `python -m vgx.offline_multirisk verify --config ${D.cfg_path}`, options: { breakLine: true } },
    { text: `python -m vgx.offline_multirisk report --config ${D.cfg_path}`, options: { breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: `结果：${D.out_dir}/ 和 _raw/`, options: { breakLine: true } },
    { text: `网页报告：${D.report_dir}/results.html` },
  ], { x: M + 0.3, y: 3.6, w: W - 2 * M - 0.6, h: 1.9, fontSize: 13, fontFace: 'Courier New' });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  fs.writeFileSync(path.join(__dirname, 'charts.json'), JSON.stringify(CHARTS));
  console.log('wrote', OUT);
})();
