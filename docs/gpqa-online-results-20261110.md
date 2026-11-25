# GPQA online 回放：结果（2026-11-10）

性质：在已有 GPQA 缓存输出上的追加分析，不是预注册实验。tony 在 2026-11-10 提出，用来检验 QuantumKatas
的结论是否只是数据集本身造成的。没有新的 API 调用。协议、学习算法、超参数（η0 = 0.01、γ = 0.1、α = 0.05、
B = 25%）与 QuantumKatas 回放完全相同，没有在 GPQA 上重新调。

配置：`configs/gpqa_online_20261110.json`（独立模型，A/B/C1/C2/D）、`configs/gpqa_online_dependence_20261110.json`
（依赖模型，A/B/C2/D）、`configs/gpqa_online_replicates_20261110.json`（样本 1–4，仅 A）。
导出：`scripts/export_gpqa_online_rows_20261110.py`。表格：`reports/gpqa_online_20261110/tables.md`。

## 0. 结论

1. **静态基线在 GPQA 上同样超标，问题不是 QuantumKatas 特有的。** A 组实际风险（模型估计都约 5%）：
   - qwen3：11.8% [6.4, 17.6]，发布率 14.7%；
   - gptoss：9.1% [5.9, 12.7]，发布率 56.7%；
   - sonnet55：8.9% [6.4, 11.6]，发布率 80.5%。
   换用其余 4 个生成样本结果一致：qwen3 11.2–12.6%，gptoss 7.6–8.5%，sonnet55 8.3–9.8%。
   依赖感知模型差别很小：qwen3 13.0%，gptoss 8.6%，sonnet55 8.7%。
2. **GPQA 的初始化数据更多，但没有解决问题。** 每折 89 题，D_init 中 qwen3 有 43–55 个错误样本（QuantumKatas 约 22–32），
   最强的 verifier（gemini_flash）区分度也更高（类均值差 0.6–0.78）。即使这样仍然超标。
3. **即使用数据流自身标签拟合（oracle，样本内），风险也降不到 5%。**

   | A 组实际风险（%） | φ0, θ0（实验） | φ*, θ*（oracle） |
   |---|---|---|
   | qwen3 独立 / 依赖 | 11.8 / 13.0 | 9.4 / 8.6（依赖模型此时只发布 2%） |
   | gptoss 独立 / 依赖 | 9.1 / 8.6 | 7.4 / 6.1 |
   | sonnet55 独立 / 依赖 | 8.9 / 8.7 | 6.5 / 6.2 |

   作为对照，此前的模拟里模型设定正确时（ci_known），同一套 DesignRouting 的实际风险是 5.2%（n = 194）和 4.5%（n = 1000）。
   所以剩下的偏差来自模型设定本身，而不只是参数估计。
4. **最可能的设定错误：给定 Y，generator 的置信度和 verifier 分数仍然相关。** 模型假设两者条件独立
   （这也是 Ẑ 无偏的前提，协议 §8 第 3 条）。实际在答错的题上，r 与 5 个 verifier 平均分的 Spearman 相关为：
   - QuantumKatas：qwen3 0.68，gptoss 0.35，sonnet55 0.30（答对的题上 0.59–0.70）；
   - GPQA：qwen3 0.00，gptoss 0.29，sonnet55 0.15（答对的题上 0.19–0.32）。
   直观上就是同一道题让 generator 和 verifier 一起犯错，先验和 verifier 把同一份证据算了两次，后验因此过度自信；
   发布规则又专挑后验最高的题，偏差被放大。QuantumKatas 的相关更强，超标也更严重，方向一致。
5. **学习组的结论与 QuantumKatas 基本一致**（q = 0.5，独立模型）：
   - B − A：Brier 改善 ≤ 0.0015，几乎不动；qwen3 上 B 的风险反而高 1.5pp [0.3, 2.6]；
   - C1、C2 仍然"多发布"：C1 − B 的发布率在 qwen3 上 +15.6pp、在 sonnet55 上 +4.5pp；C2 − B 在 qwen3 上 +8.7pp。
     风险在 qwen3 上没有显著变化（C1 −0.5pp，C2 −1.5pp），sonnet55 的 C1 +1.4pp [0.5, 2.3]；
   - C1 − C2：发布率更高，风险持平或更高（sonnet55 +1.4pp [0.7, 2.3]）；C1 的 θ 更新耗时约为 C2 的 11–18 倍，
     但重新设计占主要时间，所以总耗时差距不大；
   - D − B：风险降低 0.2–0.8pp，都不显著；GPQA 上标签带来的收益比 QuantumKatas 还小。
   q = 0.25、0.1 的方向相同。

## 1. 数据与设置

- 生成端：`reports/raw_generator_export_20261026/gpqa.jsonl.gz`，445 道题 × 3 个 generator × 10 个样本，
  有答案字母、自报 p_correct 和对照答案键的标签。主分析用样本 0。
- verifier：`reports/raw_verifier_export_20261027`，5 个 verifier（与 QuantumKatas 相同）对每题四个选项都打了分，
  所以每个样本的回答都有完整的面板。
- 费用：按 GPT-OSS GPQA 验证研究开发集上的每次调用平均费用估算，所有 generator 共用。
- 折：445 题按固定种子随机分成 5 折，每折 89 题，所有 generator 和样本共用。GPQA 没有题目组，所以 bootstrap 按题聚类。
- 可处理：返回了答案字母并且 p_correct ∈ [0, 1]（gptoss 有 18 题、sonnet55 有 6 题没有答案）。
- 网格：独立模型 975 个 run，依赖模型 750 个 run，样本 1–4 的 A 组 300 个 run。

## 2. 对五个问题的影响

- ① verifier 反馈改善校准：两个数据集上都几乎没有改善。
- ② 更新 likelihood 是否优于固定：无标签更新（C1、C2）只增加发布，不降低风险；有标签的 D 在 QuantumKatas 上有帮助，在 GPQA 上不显著。
- ③ C1 vs C2：C1 更激进，计算量大一个数量级，没有换来更低的风险。
- ④ 内部 likelihood 上升而真实指标变差：C1、C2 的 panel Brier 都比 B 差（例如 gptoss 依赖模型 C2 0.121 vs B 0.112）。
- ⑤ 学习收益是否值得审计成本：目前两个数据集上都看不出值得。

## 3. 这对方法意味着什么

现在的主要问题不在 online 学习算法，而在静态模型的两个独立性假设：
verifier 之间（依赖模型已部分处理），以及 generator 置信度与 verifier 之间（尚未处理）。
后者在两个数据集上都明显存在，而且会让发布风险系统性地高于名义值。
在这个问题解决之前，online 各组之间的比较都建立在一个本身不满足风险保证的基线上。

可能的下一步（都只用本地计算，不改变标签规则）：
1. 把 r 也放进联合模型：例如按 r 的分档建立 P(verifier 分档 | Y, r 档)，或让 g 依赖 r；
2. 风险约束加入经验校正：在 D_init 上用交叉拟合估计"名义风险到实际风险"的放大倍数，按倍数收紧 α；
3. 在设计阶段对后验做保守处理（例如对后验置信度做温度缩放，温度只在 D_init 上拟合）。
