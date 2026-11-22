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
pres.title = '离线多风险实验：结果图';
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
    { text: { text: '离线多风险实验 · 结果图', options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
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
function legendGrid(s, items, x, y, perRow, w, fs) {
  const cw = w / perRow;
  items.forEach(([label, col], i) => {
    const cx = x + (i % perRow) * cw, cy = y + Math.floor(i / perRow) * 0.34;
    if (label.startsWith('虚线')) s.addShape(pres.shapes.LINE, { x: cx, y: cy + 0.16, w: 0.32, h: 0, line: { color: col, width: 1.5, dashType: 'dash' }, objectName: nm('legend') });
    else s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy + 0.08, w: 0.32, h: 0.16, rectRadius: 0.05, fill: { color: col }, line: { type: 'none' }, objectName: nm('legend') });
    text(s, label, { x: cx + 0.42, y: cy, w: cw - 0.5, h: 0.32, fontSize: fs || 13, color: C.text2, valign: 'middle' });
  });
}
function legendRow(s, items, x, y) {
  let cx = x;
  for (const [label, col] of items) {
    if (label.startsWith('虚线')) s.addShape(pres.shapes.LINE, { x: cx, y: y + 0.16, w: 0.32, h: 0, line: { color: col, width: 1.5, dashType: 'dash' }, objectName: nm('legend') });
    else s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: y + 0.08, w: 0.32, h: 0.16, rectRadius: 0.05, fill: { color: col }, line: { type: 'none' }, objectName: nm('legend') });
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


// ================================================================ graph-first deck
const G = JSON.parse(fs.readFileSync(process.env.CHARTS, 'utf8'));
const RISK2 = ['selective_error', 'fpr'];
const GM = G.methods;                       // main methods (Self-consistency only in the appendix)
const GCOL = (m) => SERIES[['Ours-Offline', 'AT-Frozen', 'AT+Cal-Frozen', 'Cal-threshold', 'Fixed-1', 'Fixed-3', 'Weaver', 'SSV'].indexOf(m)];
const legendMethods = (s, y, ms) => legendGrid(s, ms.map((m) => [m, GCOL(m)]), M, y, 5, W - 2 * M, 11);
function grid(s, panels, cols, x0, y0, w, h, make) {
  const rowsN = Math.ceil(panels.length / cols), gx = 0.18, gy = 0.12;
  const pw = (w - (cols - 1) * gx) / cols, ph = (h - (rowsN - 1) * gy) / rowsN;
  panels.forEach((p, i) => {
    const x = x0 + (i % cols) * (pw + gx), y = y0 + Math.floor(i / cols) * (ph + gy);
    text(s, p.label, { x, y, w: pw, h: 0.24, fontSize: 10.5, bold: true, align: 'center', color: C.text1 });
    CHARTS.push(Object.assign({ slide: SN, kind: 'line', x, y: y + 0.22, w: pw, h: ph - 0.22, small: true, legend: false }, make(p)));
  });
}
const lineSeries = (obj, ms, scale) => ms.filter((m) => obj[m]).map((m) => ({ name: m, values: obj[m].map((v) => v == null ? null : +(scale * v).toFixed(2)), color: GCOL(m), bold: m === 'Ours-Offline' }));

// ---------------------------------------------------------------- cover
pres.addSection({ title: '概览' });
{
  const s = pres.addSlide({ masterName: 'TITLE', sectionTitle: '概览' });
  SN += 1;
  s.addText('离线多风险实验：结果图', { placeholder: 'title' });
  s.addText([
    { text: '放行错误率和 FPR 约束下，按需验证和各对照的比较', options: { breakLine: true } },
    { text: 'GPQA · MATH-500 · QuantumKatas ｜ Qwen3 · gpt-oss · Sonnet 5.5 ｜ 只用缓存', options: { fontSize: 16, color: '9FB4C2' } },
  ], { placeholder: 'body' });
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


// ---------------------------------------------------------------- how to read
{
  pres.addSection({ title: '图例' });
  const s = slide('怎么读后面的图', '图例', '');
  const cw = (W - 2 * M - 0.3) / 2;
  card(s, M, 1.4, cw, 2.6);
  text(s, '两种约束（FNR 见单独一页）', { x: M + 0.25, y: 1.55, w: cw - 0.5, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  text(s, [
    { text: '放行错误率：', options: { bold: true } }, { text: '放行的答案里错的比例 ≤ α。', options: { breakLine: true } },
    { text: 'FPR：', options: { bold: true } }, { text: '所有错误答案里被放行的比例 ≤ α（不测 MATH-500，错误答案太少）。', options: { breakLine: true } },
    { text: '两者的目标都是放行率（放行的题占全部题），越高越好。' },
  ], { x: M + 0.25, y: 1.95, w: cw - 0.5, h: 1.9, fontSize: 13, paraSpaceAfter: 4 });
  const x1 = M + cw + 0.3;
  card(s, x1, 1.4, cw, 2.6);
  text(s, '预算和 α', { x: x1 + 0.25, y: 1.55, w: cw - 0.5, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  text(s, [
    { text: '预算 B = f × c_ref：c_ref 是一道题问满 5 个 verifier 的平均花费；f 取 0.1、0.5、1 三档画图。', options: { breakLine: true } },
    { text: `α：放行错误率 ${G.alpha.selective_error.join('、')}；FPR ${G.alpha.fpr.join('、')}。`, options: { breakLine: true } },
    { text: '每个小图是一组数据集 · generator，所有方法都读同一个校准后的 p_correct。' },
  ], { x: x1 + 0.25, y: 1.95, w: cw - 0.5, h: 1.9, fontSize: 13, paraSpaceAfter: 4 });
  card(s, M, 4.25, W - 2 * M, 2.45);
  text(s, '方法和颜色', { x: M + 0.25, y: 4.4, w: 4, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  legendGrid(s, GM.map((m) => [m, GCOL(m)]), M + 0.25, 4.85, 4, W - 2 * M - 0.5, 12);
  text(s, '会问 verifier 的：Ours（按需，最多 3 个）、Fixed-1 / Fixed-3（每题问满固定的 1 / 3 个）、Weaver（问满 5 个）、SSV（先问弱的，拿不准再问强的，只比放行错误率）。不问的：AT、AT+Cal、Cal-threshold。图里只画结果；是否超出风险和预算见后面「约束」几页。',
    { x: M + 0.25, y: 5.6, w: W - 2 * M - 0.5, h: 1.0, fontSize: 12, color: C.text2 });
}

// ---------------------------------------------------------------- per risk: alpha curves + mean bar, risk vs alpha, cost vs budget
RISK2.forEach((risk) => {
  pres.addSection({ title: RN[risk] });
  const P = G.panels[risk], al = G.alpha[risk].map(String), cols = 3;
  const ms = GM.filter((m) => P.some((p) => p.obj['1'][m]));
  G.shown.forEach((f) => {
    const s = slide(`${RN[risk]} · B = ${f} × c_ref：α 和放行率`, RN[risk], `每个小图横轴是目标 α，纵轴是测试集放行率（越高越好）。右边是同一预算下所有数据集 · generator 和 α 的平均放行率。`);
    legendMethods(s, 1.15, ms);
    grid(s, P, cols, M, 1.9, 8.55, 4.95, (p) => ({ categories: al, min: 0, max: 100, unit: 50, valFmt: '0"%"', series: lineSeries(p.obj[String(f)], ms, 100) }));
    const mb = G.meanbar[risk][String(f)], bm = ms.filter((m) => mb[m] != null);
    text(s, '平均放行率', { x: 9.45, y: 1.9, w: 3.3, h: 0.3, fontSize: 13, bold: true, align: 'center' });
    CHARTS.push({ slide: SN, kind: 'hbar', x: 9.35, y: 2.25, w: 3.4, h: 4.6, categories: bm, max: 100, gap: 40,
      series: [{ name: '平均放行率', values: bm.map((m) => +(100 * mb[m]).toFixed(1)), color: SERIES[0] }],
      pointColors: bm.map((m) => GCOL(m)), labelFmt: '0"%"', small: true });
  });
  G.shown.forEach((f) => {
    const s = slide(`${RN[risk]} · B = ${f} × c_ref：实际${RN[risk]}和目标 α`, RN[risk], '每个小图横轴是目标 α，纵轴是测试集上的实际风险，灰色虚线是目标 α（在它上方就是超标）。');
    legendGrid(s, [...ms.map((m) => [m, GCOL(m)]), ['虚线 = 目标 α', GRAY]], M, 1.15, 5, W - 2 * M, 11);
    const top = Math.round(200 * Math.max(...G.alpha[risk]));
    grid(s, P, cols, M, 1.9, W - 2 * M, 4.95, (p) => ({ categories: al, min: 0, max: top, unit: top / 4, valFmt: '0"%"',
      series: [...lineSeries(p.risk[String(f)], ms, 100), { name: '目标 α', values: G.alpha[risk].map((a) => 100 * a), color: GRAY, dash: true }] }));
  });
  {
    const am = G.alpha_main[risk];
    const s = slide(`${RN[risk]} · α = ${am}：实际花费和预算`, RN[risk], '横轴是预算比例 f，纵轴是测试集上每题平均 verifier 花费占 c_ref 的比例，灰色虚线是预算上限 f。只画会问 verifier 的方法。');
    const sm = ms.filter((m) => ['Ours-Offline', 'Fixed-1', 'Fixed-3', 'Weaver', 'SSV'].includes(m));
    legendGrid(s, [...sm.map((m) => [m, GCOL(m)]), ['虚线 = 预算上限 f', GRAY]], M, 1.15, 6, W - 2 * M, 11);
    grid(s, P, cols, M, 1.6, W - 2 * M, 5.25, (p) => ({ categories: G.fracs.map(String), min: 0, max: 125, unit: 25, valFmt: '0"%"',
      series: [...lineSeries(p.cost, sm, 100), { name: '预算上限', values: G.fracs.map((x) => 100 * x), color: GRAY, dash: true }] }));
  }
});

// ---------------------------------------------------------------- FPR: why Fixed-1 releases more
if (G.fpr_f1 && G.fpr_f1.n) {
  const F = G.fpr_f1;
  const s = slide(`FPR：Fixed-1 放行更多的 ${F.n} 个条件里，它的实际 FPR 全部超标`, 'FPR',
    'Fixed-1 显著放行更多的条件（配对 bootstrap），对比两者在测试集上的实际 FPR。左边两个小图是这些条件最集中的两组，预算 B = 1 × c_ref，灰色虚线是目标 α。');
  const pairs = Object.entries(F.by_pair).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([p]) => p);
  const ms = ['Ours-Offline', 'Fixed-1'];
  legendGrid(s, [...ms.map((m) => [m, GCOL(m)]), ['虚线 = 目标 α', GRAY]], M, 1.2, 3, 7.4, 12);
  const al = G.alpha.fpr.map(String);
  pairs.forEach((p, i) => {
    const P = G.panels.fpr.find((x) => x.label === p);
    const x = M + i * 3.75;
    text(s, p, { x, y: 1.75, w: 3.55, h: 0.3, fontSize: 13, bold: true, align: 'center' });
    CHARTS.push({ slide: SN, kind: 'line', x, y: 2.05, w: 3.55, h: 4.65, small: true, legend: false, categories: al, min: 0, max: 50, unit: 10, valFmt: '0"%"',
      catTitle: '目标 α', valTitle: '实际 FPR',
      series: [...lineSeries(P.risk['1'], ms, 100), { name: '目标 α', values: G.alpha.fpr.map((a) => 100 * a), color: GRAY, dash: true }] });
  });
  const x0 = 8.4, w = W - M - x0;
  const facts = [
    [`${F.f1_over} / ${F.n}`, `这些条件里 Fixed-1 的实际 FPR 超过 α；我们只有 ${F.ours_over} / ${F.n}`],
    [`${F.f1_ratio.toFixed(1)} × α`, `Fixed-1 的平均实际 FPR（我们是 ${F.ours_ratio.toFixed(1)} × α），平均超出 ${(100 * F.f1_ex).toFixed(1)} 个百分点`],
    [`${pairs.map((p) => F.by_pair[p]).reduce((a, b) => a + b, 0)} / ${F.n}`, `集中在 ${pairs.join('、')}：正确率高、错题少，验证部分估 FPR 很不稳`],
  ];
  facts.forEach(([big, d], i) => {
    const y = 1.2 + i * 1.85;
    card(s, x0, y, w, 1.65);
    text(s, big, { x: x0 + 0.25, y: y + 0.12, w: w - 0.5, h: 0.65, fontSize: 28, bold: true, color: C.accent1, fontFace: 'Arial' });
    text(s, d, { x: x0 + 0.25, y: y + 0.8, w: w - 0.5, h: 0.8, fontSize: 12, color: C.text2 });
  });
}

// ---------------------------------------------------------------- FNR note
pres.addSection({ title: 'FNR' });
{
  const s = slide('FNR 不作为比较指标：各方法都只能放行大部分题，区分度低', 'FNR', '');
  const rows = [[TH('数据集 · generator'), TH('测试集正确率', { align: 'right' }), TH('只看校准 p 时 FNR α = 0.05 的放行比例', { align: 'right' })]];
  const ex = Object.fromEntries((D.examples || []).map(([lab, cells]) => [lab, cells[2]]));
  G.accuracy.forEach(([lab, acc]) => rows.push([TD(lab, { bold: true }), TD(pct(acc), { align: 'right', fontFace: 'Arial' }), TD(ex[lab] || '—', { align: 'right' })]));
  table(s, rows, { y: 1.4, w: 7.6, colW: [2.8, 1.6, 3.2], fontSize: 13, rowH: 0.42 });
  const x0 = 8.6, w = W - M - x0;
  card(s, x0, 1.4, w, 5.3);
  text(s, [
    { text: '为什么不比 FNR', options: { bold: true, color: C.accent1, breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 6 } },
    { text: 'FNR 限制被拦下的正确答案 ≤ α × 正确答案数。正确答案占多数时，这个额度只够拦下 p 最低的一小段，所有方法都要放行 75% 到 94% 的题。', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 6 } },
    { text: '能比的只剩这一小段里挑出了几道错题，各方法的错误放行只差几个百分点，结论受少数题左右。', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 6 } },
    { text: 'MATH-500 和 Sonnet 5.5 的正确率在 80% 到 97%，这个问题最明显；正确率较低的 GPQA · Qwen3、QuantumKatas · Qwen3 也一样，因为 α 只有 0.01 到 0.1。' },
  ], { x: x0 + 0.25, y: 1.6, w: w - 0.5, h: 5.0, fontSize: 13 });
}

// ---------------------------------------------------------------- comparisons and constraints
pres.addSection({ title: '比较与约束' });
const BI2 = (b) => D.baselines.indexOf(b);
{
  const s = slide('和每个对照比：显著更好的条件数', '比较与约束',
    '一个条件 = 一组（数据集 · generator，α，预算 f）。在每个条件下比较测试集放行率，2000 次配对 bootstrap，95% 区间不含 0 才算一方显著更好。');
  legendRow(s, [['我们显著更好', SERIES[0]], ['对照显著更好', GRAY]], M, 1.3);
  const cw = (W - 2 * M - 0.4) / 2;
  RISK2.forEach((risk, i) => {
    const x = M + i * (cw + 0.4);
    const keep = D.baselines.map((b, j) => j).filter((j) => !S[risk].nb || S[risk].nb[j] > 0);
    text(s, `${RN[risk]}（${S[risk].n} 个条件）`, { x, y: 1.85, w: cw, h: 0.4, fontSize: 16, bold: true, align: 'center' });
    CHARTS.push({ slide: SN, kind: 'hbar', x, y: 2.3, w: cw, h: 4.5, categories: keep.map((j) => D.baselines[j]), max: 230, gap: 50,
      series: [{ name: '我们显著更好', values: keep.map((j) => S[risk].ours[j]), color: SERIES[0] }, { name: '对照显著更好', values: keep.map((j) => S[risk].base[j]), color: GRAY }], labelFmt: '0' });
  });
}
{
  const s = slide('按数据集看：显著更好的条件数（我们 : 对照）', '比较与约束', '括号里是低预算 f ≤ 0.25（含 0）/ 高预算 f ≥ 0.5。');
  const SG = D.sig_ds, cell = (x, split) => !x ? '—' : split ? `${x[0]}:${x[1]}（${x[3]}:${x[4]} / ${x[5]}:${x[6]}）` : `${x[0]}:${x[1]}`;
  const EX = ['Weaver', 'SSV'];
  const rows = [[TH('数据集 · 风险'), TH('条件数', { align: 'right' }), TH('对 AT', { align: 'right' }), TH('对 Cal-thr', { align: 'right' }),
    TH('对 Fixed-1（低 / 高）', { align: 'right' }), TH('对 Fixed-3（低 / 高）', { align: 'right' }), ...EX.map((m) => TH(`对 ${m}`, { align: 'right' }))]];
  Object.keys(SG).forEach((ds) => RISK2.forEach((r) => {
    const x = SG[ds][r];
    if (!x) return;
    rows.push([TD(`${ds} · ${RN[r]}`, { bold: true }), TD(String(x['AT-Frozen'][2]), { align: 'right', fontFace: 'Arial' }),
      TD(cell(x['AT-Frozen']), { align: 'right', fontFace: 'Arial' }), TD(cell(x['Cal-threshold']), { align: 'right', fontFace: 'Arial' }),
      TD(cell(x['Fixed-1'], true), { align: 'right', fontFace: 'Arial' }), TD(cell(x['Fixed-3'], true), { align: 'right', fontFace: 'Arial' }),
      ...EX.map((m) => TD(cell(x[m]), { align: 'right', fontFace: 'Arial' }))]);
  }));
  table(s, rows, { y: 1.4, colW: [2.8, 0.85, 1.0, 1.1, 2.2, 2.2, 1.0, 0.983], fontSize: 12, rowH: 0.42 });
}
{
  const s = slide('按预算看：和 Fixed-3 比，优势在低预算', '比较与约束', '每格是「我们显著更好 : Fixed-3 显著更好」的条件数。');
  const rows = [[TH('预算 f'), ...RISK2.map((r) => TH(`${RN[r]} 对 Fixed-3`, { align: 'right' }))]];
  ['0.0', '0.1', '0.25', '0.5', '0.75', '1.0'].forEach((f) => rows.push([TD(String(+f), { bold: true }),
    ...RISK2.map((r) => { const c = D.by_budget[r]['Fixed-3'][f]; return TD(`${c[1]} : ${c[2]}`, { align: 'right', fontFace: 'Arial' }); })]));
  table(s, rows, { y: 1.45, colW: [3.133, 4.5, 4.5], fontSize: 15, rowH: 0.5 });
  text(s, '预算少时 Fixed-3 只能随机抽一小部分题问满 3 个，我们把钱花在拿不准的题上；预算多时 Fixed-3 几乎每题都问，FPR 下高 p 题里的错答案被它挑出来，我们的花费停在约三分之一 c_ref。',
    { x: M, y: 5.2, w: W - 2 * M, h: 1.0, fontSize: 14 });
}
{
  const s = slide('约束：测试集上风险超标和预算超标的条件数', '比较与约束', '风险超标：测试集实际风险 > α，任何超出都算。预算超标：每题平均花费 > B。');
  const rows = [[TH('方法'), ...RISK2.flatMap((r) => [TH(`${RN[r]}：风险超标`, { align: 'right' }), TH(`${RN[r]}：预算超标`, { align: 'right' })])]];
  D.methods.filter((m) => m !== 'Self-consistency').forEach((m) => {
    rows.push([TD(m, { bold: true, color: m === 'Ours-Offline' ? C.accent1 : C.text1 }), ...RISK2.flatMap((r) => {
      const none = !S[r].n_cond[m];
      return [TD(none ? '—' : `${S[r].overrun[m]} / ${S[r].n_cond[m]}`, { align: 'right', fontFace: 'Arial' }), TD(none ? '—' : String(S[r].budget_over[m]), { align: 'right', fontFace: 'Arial' })];
    })]);
  });
  table(s, rows, { y: 1.45, colW: [2.533, 2.4, 2.4, 2.4, 2.4], fontSize: 14, rowH: 0.45 });
  text(s, '会问 verifier 的方法风险超标都不少：验证部分只有 50 到 100 题，挑出的工作点压在 α 边上，换到测试集常常略超。FPR 下我们比 Fixed 超标少。',
    { x: M, y: 6.05, w: W - 2 * M, h: 0.7, fontSize: 13 });
}

// ---------------------------------------------------------------- appendix: Self-consistency
pres.addSection({ title: '附录' });
if (D.sc_appendix && Object.keys(D.sc_appendix).length) {
  const s = slide('附录：Self-consistency（不计入正文比较）', '附录',
    'Self-consistency 的分数是 generator 10 个样本里和最终答案一致的比例，过阈值就放行，阈值在验证部分挑。它不问 verifier，靠多生成 10 次，花费没有计入预算。');
  const SA = D.sc_appendix;
  const rows = [[TH('数据集 · 风险'), TH('我们 : SC', { align: 'right' }), TH('SC 风险超标', { align: 'right' }), TH('SC 回退比例', { align: 'right' })]];
  Object.keys(SA).forEach((ds) => RISK2.forEach((r) => {
    const x = SA[ds][r];
    if (!x) return;
    rows.push([TD(`${ds} · ${RN[r]}`, { bold: true }), TD(`${x.ours} : ${x.base}（共 ${x.n}）`, { align: 'right', fontFace: 'Arial' }),
      TD(String(x.over), { align: 'right', fontFace: 'Arial' }), TD(pct(x.fallback), { align: 'right', fontFace: 'Arial' })]);
  }));
  table(s, rows, { y: 1.4, w: 7.6, colW: [2.6, 2.5, 1.2, 1.3], fontSize: 13, rowH: 0.42 });
  const x0 = 8.5, w = W - M - x0;
  card(s, x0, 1.4, w, 5.3);
  const lines = [{ text: '10 个样本的花费（×10⁻⁴ 美元 / 题）', options: { bold: true, color: C.text2, breakLine: true } }];
  Object.keys(SA).forEach((ds) => {
    lines.push({ text: `${ds}（c_ref ${D.cref[ds][0].toFixed(0)}–${D.cref[ds][1].toFixed(0)}）`, options: { bold: true, breakLine: true } });
    Object.entries(SA[ds].usd).forEach(([g, v]) => lines.push({ text: `${g}：${v.toFixed(0)}`, options: { breakLine: true } }));
  });
  text(s, lines, { x: x0 + 0.25, y: 1.6, w: w - 0.5, h: 5.0, fontSize: 13 });
  text(s, 'GPQA 上 10 个样本全一致的题答对率也只有 71% 到 93%，多数条件找不到达标的阈值而回退成全部拦下；MATH-500 上它更好，但靠的是没计入预算的生成花费。',
    { x: M, y: 4.2, w: 7.6, h: 1.2, fontSize: 12, color: C.text2 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  fs.writeFileSync(path.join(__dirname, 'charts.json'), JSON.stringify(CHARTS));
  console.log('wrote', OUT, 'charts', CHARTS.length);
})();
