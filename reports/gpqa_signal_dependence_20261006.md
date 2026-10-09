# Verifier 与原始 confidence 的依赖性检查

日期：2026-10-06。开发集描述性诊断；不校准 generator、不修改似然或停止规则、不读取 Diamond 答案、不调用 API。

## 为什么检查

现有单 verifier 更新使用按答案正确性分组的似然 P(v|Y)。这相当于在该更新模型中假设：固定正确性 Y 后，verifier 信号 v 不再依赖 generator 原始 confidence b。

如果在正确答案内部或错误答案内部，v 仍随 b 明显变化，那么总体估计的 P(v|Y) 未必适用于特定的 b 区域。多个 verifier 串联还需要检查它们彼此之间的条件依赖。来自不同模型家族不保证独立。

## Gemini 上的观察

有效初始候选共 247 个：211 个正确、36 个错误。以下为 verifier 概率报告与 generator 原始 confidence 的 Spearman 相关系数，分别在正确和错误候选内部计算；无效信号仅从对应相关计算中排除。

| 概率 verifier | 正确候选内部 | 错误候选内部 |
|---|---:|---:|
| Gemini Flash | 0.797（n=210） | 0.612（n=36） |
| Flash-Lite | 0.299（n=209） | 0.065（n=36） |
| Mistral Small | 0.273（n=211） | 0.401（n=36） |
| Mistral Medium | −0.021（n=211） | 0.193（n=36） |
| Claude Haiku | 0.516（n=211） | 0.404（n=36） |

这些是样本相关，不是总体相关的确定值，也不是独立性检验的最终结论。误答样本只有 36 个。

**Flash 与原始 confidence 存在明显的样本关联，即使把答案对错固定下来。** Verifier 提示词只包含问题、选项、固定候选和输出要求，并未传入 generator confidence；共同的题目难度、知识覆盖等仍可能造成关联，不能只归因于同一家族。

一个与实际似然分箱直接相关的例子：在错误候选中，Flash 报告落在最高概率箱 [2/3,1] 的比例为：

- 原始 confidence <0.95：11/17，64.7%；
- 原始 confidence ≥0.95：17/19，89.5%。

因此，高 confidence 区域里错误答案也经常得到 Flash 的高概率认可。全体误答上的平均似然可能掩盖这种差异。我们仍需更多数据和折外比较才能决定是否要显式加入条件 b。

## 多 verifier 的关联

不仅同家族存在关联。例如在正确候选内部，Mistral Small 与 Mistral Medium 的二元判断相关约 0.542；在错误候选内部，Flash 与 Mistral Small 的概率报告相关约 0.594。

这些例子说明，在选定三个 verifier 后，不能简单地把所有似然比视作独立证据相乘。接近零的相关也不能证明独立，非线性依赖和稀疏取值仍可能存在。

## 当前处理与边界

- 本次仅诊断，不把按 b 分组得到的频率替换进已冻结的算法，不事后调参。
- 同时保存二元／概率信号、两个开发集范围、给定正确性后的 verifier 两两相关和按 0.95 划分的信号箱计数。
- Qwen 缓存未补齐前不报告其选择性子集上的相关；补齐后用同样方法分析。
- 后续若研究 P(v|Y,b) 或依赖感知更新，需要单独固定模型、训练折拟合、验证折比较。它仍可以保留原始 b 为初始概率，但不保证未经校准的 b 就是经验上的真实概率。
- 249 与 199 两种范围重叠；分析不能当成两份独立证据。

## 复现

```sh
.venv/bin/python -m vgx.gpqa.signal_dependence \
  --config configs/gpqa_raw_prior_analysis.json \
  --output results/gpqa_signal_dependence_20261006
```

机器可读结果位于该目录的 report.json；源缓存、候选、配置和实现哈希记录在 protocol.json。
