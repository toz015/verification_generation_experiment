# 离线多风险实验 Agent 执行手册

版本日期：2026-11-16。适用仓库：`toz015/verification_generation_experiment`。

本手册用于实现并运行三个数据集、三个 generator、三种风险约束下的离线比较实验。先修复已审计的问题，再使用现有缓存重跑。完成标准是计算正确、比较公平、结果可复现，不要求 Ours 获胜。

## 1 给执行 Agent 的启动指令

从头阅读本手册和已有 offline 协议，检查当前代码与审计基准的差异，然后按第 14 节的顺序实施。不要停留在提出方案；完成已明确授权的代码修改、测试、缓存实验和报告。先报告影响数学定义或比较公平性的实质歧义，其余常规实现选择自行处理并记录。

- 仓库：<https://github.com/toz015/verification_generation_experiment>
- 工作分支：`claude/project-thread-m8td13`
- 审计基准：`4709be4b5db96c493936a1a36c1402106f49bd19`
- 现有协议：`docs/offline-at-frozen-experiment-protocol-20261110.md`
- 本轮手册：`docs/offline-multirisk-agent-handbook-20261116.md`

如果当前提交晚于审计基准，保留已有进展，先核对哪些修复已经完成。不要 reset、覆盖未提交文件或重复实现已存在的正确修复。本手册没有要求自动 commit、push 或合并分支；版本发布按用户后续指示执行。

**禁止新增任何 generator、verifier 或付费评分 API 调用。** 缺失缓存、失败请求和无法解析的记录都不能触发自动补采或重试。测试与实验执行期间启用网络请求阻断，并记录缓存完整性检查结果。

## 2 任务边界与核心问题

每个问题已有一个固定候选答案和 generator 自报的 `p_correct`。策略可以调用缓存中的弱 verifier，最终放行原答案或弃权，不生成新答案、不修复答案。Test 上策略冻结，官方真实标签只供独立 evaluator 评价。

本轮回答：

1. 在不同风险容忍度和平均预算下，各方法的性能如何变化？
2. Ours 是否能适应放行错误率、FPR、FNR 三种不同约束？
3. 差异是否在不同数据集和 generator 上一致？
4. 当双方的风险和预算同时达标时，性能差异还剩多少？

沿用官方 checker 和缓存评分。原版在线 AT、无标签 EM、全历史 EM 与递推 online EM 不属于本轮主表。TriviaQA、Mini-Sudoku、HH-RLHF 留作后续扩展，不在本轮新增采集。

## 3 实验矩阵

| 维度 | 本轮设置 |
| --- | --- |
| 数据集 | GPQA、MATH-500、QuantumKatas |
| Generator | Qwen3、gpt-oss、Sonnet 5.5 |
| 风险目标 | `selective_error`、`fpr`、`fnr` |
| 主方法 | Ours-Offline、AT-Frozen、AT+Cal-Frozen、Raw-threshold、Fixed-1-Mix、Fixed-3-Mix |
| 边界参照 | Release-eligible、Abstain-all |
| 主预算比例 | 0、0.1、0.25、0.5、0.75、1 |

所有数据集、generator 和主方法都参加三种风险实验。不能将 FNR 只安排给 MATH-500。

| 风险目标 | `alpha_target` | 正文主要展示点 |
| --- | --- | --- |
| 放行错误率 | 0.01、0.02、0.05、0.10、0.20 | 0.05 |
| FPR | 0.05、0.10、0.20、0.30 | 0.10 |
| FNR | 0.01、0.02、0.05、0.10 | 0.05 |

三个 generator、三个数据集、十三个风险与 alpha 组合、六个预算，共 **702 个主实验条件**；六个主方法共 4,212 个方法结果行，尚不包括边界参照、附表、seed 和 fold 明细。实际缺失或不可评价的条件保留状态行，不直接删除。

MATH-500 额外突出放行错误率 1% 和 2% 的结果，但不删除其共同网格。主要展示点只用于组织正文，不用于筛选好看的结果。预算和 alpha 相邻条件可能选择同一个策略，不是新的独立证据。

## 4 风险定义与目标函数

令 `Y_i = 1` 表示固定候选按官方评分正确；`a_i` 是策略在题目 i 上的放行概率；`c_i` 是相应的期望额外推理成本。确定性策略的 `a_i` 为 0 或 1，混合策略可为分数值。

定义：

$$
p=\frac{1}{n}\sum_i a_i,\qquad
e=\frac{1}{n}\sum_i a_i(1-Y_i),\qquad
m=\frac{1}{n}\sum_i(1-a_i)Y_i,
$$

$$
q=\frac{1}{n}\sum_iY_i,\qquad
\bar c=\frac{1}{n}\sum_i c_i.
$$

三组优化问题必须分别实现：

| 分支 | 优化目标 | 风险约束 | 成本约束 |
| --- | --- | --- | --- |
| `selective_error` | 最大化 p | e <= alpha * p | mean_cost <= B |
| `fpr` | 最大化 p | e <= alpha * (1 - q) | mean_cost <= B |
| `fnr` | 最小化 e | m <= alpha * q | mean_cost <= B |

对应评价指标：

$$
R_{\mathrm{sel}}=e/p,\qquad
R_{\mathrm{FP}}=e/(1-q),\qquad
R_{\mathrm{FN}}=m/q.
$$

FPR 的正类约定是“候选正确”，所以 FPR 是所有错误候选中的放行比例。它对应 AT 原文“错误中有多少未经审计”的分母。FNR 是所有正确候选中的拒绝比例。

实现时利用恒等式 `m = q - p + e` 交叉检查各统计量。不得把每条规则的风险比率先取平均；先合并错误数、放行数等分子分母，再求比率。

**FNR 必须同时改变优化目标。** 如果继续最大化放行率，全部放行就会获得 FNR 为零，不能回答筛选错误的实验问题。

不同风险下同一个 alpha 不代表同样严格程度。同一策略可以附带报告三种指标，但每个主分支必须针对其风险重新规划和选择工作点，不能仅更换评价字段。

## 5 数据划分与信息边界

### 5.1 保留已有划分

- GPQA：249 个 calibration questions，按既有固定划分分为 199 Fit、50 Validation；196 Diamond questions 为 Test。
- MATH-500：五折。外层第 k 折为 Test，下一折为 Validation，其余三折为 Fit。
- QuantumKatas：同样五折，使用既有 merged kata groups；内层划分若存在，也必须按组。

所有方法共享相同题目 ID、fold、Fit、Validation 和 Test。不得为了增加错误样本而查看当前 Test 标签后重排划分。

### 5.2 各阶段允许使用的信息

| 阶段 | 允许操作 |
| --- | --- |
| Fit | 拟合校准器、verifier likelihood、AT 状态和成本参考 |
| 模型规划 | 使用 Fit 模型和 Validation 输入特征计算候选策略的模型期望值 |
| Validation 选择 | 用 Validation 标签、实际回放成本评价候选并选择工作点 |
| Test | 冻结执行；真实标签只供 evaluator 关联和评分 |

完成 Validation 选择后，不合并 Fit 与 Validation 重拟合。Test confidence、verifier 分数和调用价格都不能用于定义预算、选择 panel、调整参数或选择工作点。

模型可以为评价加载缓存数组，但每个问题的决策只能读取其已查询的 verifier 前缀。未来 verifier 分数、未调用 verifier 的价格或测试标签不能影响当前动作。

### 5.3 失败候选与标签缺失

分开保存 `input_eligible`、`label_available` 和真实 `Y`。是否可放行由输入条件决定，不能依赖是否拥有 Test 标签。

- 无答案、无法执行等失败按预先冻结的官方基准规则处理，并标记来源。
- 有答案但 confidence 缺失，不等于答案错误；若因此必须拒绝，正确答案仍应计入 FNR 的正确候选分母。
- 缺失官方标签不得自动填零。若影响完整主评价，列出受影响条件并明确无法完成的部分，不能补采 API 或静默缩小样本集合。
- 端到端表保留全部计划任务。另给有效候选子集分析，检查自动拒绝生成失败是否显著降低了 FPR。

所有生成失败、输入不可用和标签缺失计数按 dataset、generator、fold 输出。

## 6 预算与成本

每个 dataset、generator、fold 定义：

$$
C_{\mathrm{ref}}=\text{Fit 有效候选上五个 verifier 缓存调用成本之和的均值},
\qquad B=fC_{\mathrm{ref}}.
$$

记录 `C_ref` 的分母、样本数、各 verifier 成本和价格表版本。同一条件的所有方法共用预算；只允许 Fit 数据参与参考成本计算。

实际评价使用查询轨迹中的缓存逐调用价格。混合策略的成本是规则成本的权重平均。离线回放所有规则是评价计算，不能把所有回放规则的成本相加充当一次部署成本。

原始 generator 调用成本单列。已包含在 generator 输出中的 confidence 不重复收费；任何额外 confidence 获取成本必须统一计费。失败请求按已有计费证据处理，未知费用明确标记。

跨 fold 用稳定的 `budget_id` 合并；每个 fold 的 B 不同则保留 `budget_usd_by_fold`，pooled B 和成本按 Test 题数加权。附带报告各 fold 的超标状态，不能仅用 pooled 结果掩盖局部超标。

GPQA 历史固定美元预算可单独保留为附表，不计入 702 个主条件。

本轮 Test 的 defer 不获取 oracle 标签，也不等于发生了人工审计。若保留接管成本敏感性分析，统一假设每个 defer 都交给外部服务，并计算：

$$
C_{\mathrm{total}}=C_{\mathrm{verifier}}+c_{\mathrm{audit}}(1-A).
$$

该场景单独呈现，不改变主筛选预算。Fit 和 Validation 标签池对各方法一致可用；另报各方法实际读取的唯一标签数和训练计算时间。多 seed、多候选重读同一标签，不算新购买的标签。

## 7 Ours 的模型与规划

### 7.1 拟合与参数复用

每个 Fit split 拟合一次 isotonic PAV 正确率校准器，以及每个 verifier 的五 bin likelihood，Laplace smoothing 为 1。不同风险、alpha、B 复用同一组拟合模型。

`alpha_target` 是最终评价约束；`alpha_plan` 是生成候选策略的内部参数。alpha 改变时重规划策略，不人为改变概率校准的定义。

规划网格：

```json
{
  "selective_error": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
  "fpr": [0.025, 0.05, 0.10, 0.20, 0.30, 0.40],
  "fnr": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
}
```

规划预算使用共同六点网格。保留当前最多调用三个 verifier 的结构：从五个 verifier 中考虑长度三的 60 个顺序，按模型期望目标对候选顺序排序，保留既有 shortlist 大小 3。FNR 排序使用其目标值，不能仍按最大 coverage 排序。所有排序、平均范围和 tie-break 记录到配置；不读取 Test。

规划只使用 Fit 模型和 Validation 输入分布。Validation 标签用于第 9 节的共同外层选择，不直接替代 Ours 的模型规划值。若实现经验标签驱动的规划，必须作为另一个明确命名的变体，不能替换主方法。

### 7.2 LP 与 Bellman 一致性

每条规则 j 存储模型期望 `p_j, e_j, m_j, c_j`。权重满足 `w_j >= 0`、`sum(w_j) = 1`，加权后使用第 4 节的目标和约束。

令 b 为正确性的当前后验概率，lambda、mu 为非负风险和成本乘子。以弃权价值 0 为基准，立即放行价值为：

$$
g(b)=
\begin{cases}
1-\lambda[(1-b)-\alpha_{\mathrm{plan}}],&\text{selective error},\\
1-\lambda(1-b),&\text{FPR},\\
\lambda b-(1-b),&\text{FNR}.
\end{cases}
$$

查询下一个 verifier v 的价值为：

$$
-\mu\widehat c_v+\sum_zP(z\mid b)V_{\mathrm{next}}(b_z).
$$

Bellman 取立即拒绝、立即放行和继续查询的最大值。FNR 表达式来自最小化 `e + lambda*m + mu*c`，去掉与策略无关的常数。可以使用等价的乘子归一化，但必须给出推导并通过小型枚举测试。

FNR 的立即放行边界为 `b >= 1/(1 + lambda)`；不得未经证明复用旧的 `1 - 1/omega` 及原 multiplier 网格。规划中的类比例用模型估计，Validation 和 Test 的类比例由 evaluator 计算；两者分别存储，禁止用 Test 类比例求 LP。

LP 和列生成保留 release、abstain 边界规则；处理不可行状态，不把 solver 失败解释成合法全零权重。不要为了强行得到两到三条规则而截断权重。保存 solver 状态、权重、约束残差及候选生成配置。

### 7.3 混合语义

部署时每题按固定权重抽一条规则执行。缓存评价时可以回放所有规则再加权，得到该冻结策略在固定测试集上的精确期望：

$$
a_i=\sum_r w_r a_{ir},\qquad
c_i=\sum_r w_r c_{ir}.
$$

至少调用一次的概率为：

$$
u_i=\sum_r w_r\mathbf1\{N_{ir}>0\},
$$

不同于 `1{sum_r w_r N_ir > 0}`。保留 20 个预设 mixture draw seeds，报告采样执行与期望评价的差异。期望满足约束不表示每次随机实现都会满足约束。

## 8 Baseline 定义

### 8.1 AT-Frozen 与 AT+Cal-Frozen

保留 Fit 上的原更新，以下 `e_t = 1 - Y_t` 是单题错误指示，不是第 4 节的汇总错误放行概率：

$$
\tau_{t+1}=\tau_t+\eta(1-\pi_t-\alpha_{\mathrm{AT}})e_tH_t/\pi_t.
$$

低于阈值时 `pi=1`，否则 `pi=gamma_fit`。保持阈值不裁剪；无审计反馈时不更新。

```json
{
  "alpha_at": [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90],
  "eta": [0.001, 0.003, 0.01, 0.03, 0.10],
  "beta": [0.001, 0.01, 0.10, 1.00],
  "gamma_fit": 0.10,
  "gamma_test": 0.0,
  "tau_initial": 0.5,
  "fit_epochs": 1,
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
}
```

AT+Cal 保留 21 个有限阈值 `0, 0.05, ..., 1` 加 infinity、22 个初始均匀权重，以及原文的逆概率加权梯度和指数权重更新。不能替换为 Ours 的 PAV。

每种风险分支用第 9 节的规则重新选择候选。内部 `alpha_at` 不必等于 `alpha_target`。选择性风险和 FNR 标注为 offline adaptation；即使使用 FPR，冻结版也不声称继承原版在线风险保证。

`gamma_test=gamma_fit` 的历史敏感性结果可以保留在附表，不混入主表。

### 8.2 Raw-threshold

使用 raw confidence 阈值。候选阈值由 Fit 或 Validation 分数产生，含放行全部 eligible 候选和全部弃权两个边界。提前固定相等分数的处理，Test 上不能拆分 ties 重新调阈值。

保持主版本为直接阈值选择。若增加阈值混合版本，单独命名作为敏感性分析，不静默改变 Raw-threshold。

### 8.3 Fixed-1-Mix 与 Fixed-3-Mix

使用和 Ours 相同的 Fit 校准器、likelihood 和成本模型。

- Fixed-1 的 panel 候选为五个单 verifier。
- Fixed-3 的 panel 候选为五选三的十个组合。
- 每个候选混合只使用一个固定 panel；不要在同一混合内跨 panel 选 verifier。
- 一个完整 panel 规则被抽中后，对 eligible 输入调用全部 k 个 verifier，再按冻结的后验阈值决策，不进行基于 confidence 的自适应提前停止。
- 输入无效和 verifier 服务失败路径提前规定并计费。不能在结果中隐去已产生的失败调用成本。
- 阈值候选使用冻结规划 belief grid，补充明确的全放行、全弃权边界。
- 每个 panel 用同一风险 LP 将完整 panel 规则与零查询边界规则混合；权重由模型规划产生，再由共同 Validation 选择 panel 和候选混合。

零查询边界为 Release-eligible 和 Abstain-all。混合规则在读取付费 verifier 输出之前选择。计算的是平均预算，所以随机选择一部分题调用完整 panel 是允许的；这不属于按 verifier 反馈提前停止。

另外可做使用 Ours 同一 panel 的控制分析，命名为 same-panel，不能与独立选择 panel 的主 baseline 混称。不得将 same-panel 的胜负解释为对所有最佳固定 panel 的胜负。

### 8.4 边界与消融

Release-eligible 不调用 verifier；Abstain-all 不放行。二者始终作为参照显示。

主实验完成后，可报告 Ours raw-prior、Ours no-verifier 和 single-rule 版本。它们使用同一划分、风险定义和预算，不引入 API。将新增消融与主六方法分开统计。

## 9 Validation 选择与不可评价状态

对每个风险、目标 alpha、B、方法和预设 seed，先评价候选的 Validation 风险、目标值和实际回放成本，再进行选择。

| 分支 | 可行候选中主目标 | 平局顺序 |
| --- | --- | --- |
| 放行错误率或 FPR | coverage 最大 | 成本更低、规则更简单、固定配置 ID 更小 |
| FNR | 错误放行概率 e 最小 | 成本更低、FNR 更低、规则更简单、配置 ID 更小 |

复杂度定义在运行前固定，例如非零混合规则数，再以配置 ID 决定最终平局。候选集、复杂度和筛选规则相同的版本只允许一个确定性选择结果，不按 Test 改选。

状态至少包含：

- `selected`：存在正常选中的策略。
- `no_feasible_candidate`：没有满足 Validation 约束的候选。
- `risk_not_estimable`：FPR 没有错误分母或 FNR 没有正确分母。
- `zero_release`：评价时无放行，选择性风险为 NA。
- `missing_required_cache`：必需输入缺失。

全 abstain 可以是选择性风险或 FPR 的保守动作，但不能记为非退化的成功；它通常违反 FNR 约束。无可行候选时，先记录选择失败。工程上若需要 fallback，提前定义，独立记录其策略身份、理由和行为，不将其误写为正常选中的候选。尤其不能用 FNR 的全 abstain fallback 宣称约束达标。

Validation 选择成功、Test 经验达标、是否采用 fallback 三者分别记录。全体结果展示 fallback 比例；报告双方达标比较时说明 fallback 的纳入方式，并给出剔除 fallback 的敏感性结果。不得临时改变纳入规则以增加胜场。

数值比较使用写入配置的固定浮点容差；在相同单位上执行。容差只处理浮点误差，不额外宽限风险或预算。风险分母为零时不使用 epsilon 替代。

MATH-500 的部分 generator/fold 错误样本很少，尤其要检查 FPR 分母。混合策略能平滑数值，但不会增加错误样本数量。不得通过 Test 标签重分 fold 来解决这个问题。

## 10 评价口径与决策日志

### 10.1 每条件指标

至少保存以下字段，并保存原始计数以支持独立重建：

```text
n_tasks, n_eligible, n_labeled, n_correct, n_wrong
n_released, n_wrong_released, n_correct_deferred
coverage, release_error_rate, fpr, fnr
joint_wrong_release_rate, correct_retention_rate
mean_screen_cost, budget_usd, cost_excess_usd, cost_excess_relative
weak_verifier_calls_per_task, verifier_query_rate
risk_excess, risk_estimable, empirical_feasible
selection_status, fallback_used, zero_release
```

预算为零时相对超支为 NA，仍报告绝对超支。`empirical_feasible` 使用当前风险分支的实际风险和平均成本，不使用规划风险替代。零放行、不存在相关类别或缺失评价不能记为有意义的胜出。

多 seed 主表先对每题放行概率、成本取均值，再汇总分子分母；另报 per-seed 分布和超标比例。不要把 seed、mixture draws 或重复预算条件当成新增测试题。

### 10.2 决策与调用轨迹

逐题日志包含：

```text
dataset, generator, task_id, group_id, fold
risk_metric, alpha_target, budget_id, budget_usd
method, seed, selected_config_id, selection_status, fallback_used
input_eligible, raw_confidence, calibrated_confidence
release_probability, expected_cost, expected_verifier_calls
verifier_query_probability, rule_ids, mixture_weights
```

逐规则轨迹另存实际调用顺序、已读分数、每次费用、停止原因、最终后验和动作。保存 selected config 的完整规则，而不只存 weights 或一个不可重建的 hash。

真实标签由 evaluator 在独立的评价层关联。若输出带标签的评价文件，明确它不是 policy 输入。所有 Test 调用轨迹与模型、候选、配置 hash 对齐。

### 10.3 Brier 附表

Raw 和 Ours prior 在共同、完整的有效且有标签的 Test 集合上计算平方误差总和除以样本数。无可行路由不影响 prior 的 Brier 可评价性。相同 prior 不因 alpha、预算重复而获得额外样本权重。

AT+Cal 无选中候选时记为 `no_selected_calibrator`，不得使用候选 0 代替。比较它与其他分数时，另给共同可评价 task×seed 集合上的配对 Brier，报告题目数、task×seed 数、fold/seed 覆盖率。不能把删掉缺失项后的结果称为完整测试集 Brier。

## 11 统计比较与图表

主区间采用 2,000 次配对 bootstrap，沿用固定 seed 20261111。GPQA、MATH 以题目为单位，QuantumKatas 以 merged kata group 为单位；同一比较的各方法使用同一重采样索引。保留 fold 信息并说明分层策略。

点估计是实际测试差值，不是 bootstrap 均值。风险比率在每个 replicate 中重新由分子分母计算；零分母记 NA，报告有效 replicate 数和零分母比例。少量错误或零错误造成的退化区间不得写成严格风险认证。此处区间描述冻结预测下的评价不确定性，不包含完整训练过程的重新抽样。

报告分为三层：

1. **全部条件表**：保留超标、零放行、不可估计、无可行候选和缺失状态。
2. **经验可行状态表**：双方达标、仅 Ours 达标、仅 baseline 达标、双方未达标、不可评价；报告明确分母与 fallback 口径。
3. **双方均达标的性能表**：选择性风险/FPR 比较 coverage；FNR 比较错误放行概率 e。记录 paired interval、纳入条件数、数据集和 generator。

FNR 差值建议同时保存 `ours_e_minus_baseline_e`，负值表示 Ours 更好；如另设统一“正值有利于 Ours”字段，明确符号定义。其余两种风险的主差值为 `ours_coverage_minus_baseline_coverage`。

“双方均达标”是测试集上的事后描述子集，不得据此重新选择部署策略。保留完整分母和失败条件，不能宣称对总体已获得严格风险保证。点式区间没有自动完成多重比较校正，逐条件显著胜出计数只作描述，不作为独立胜场统计。

至少生成：

- 固定预算下 alpha 与性能曲线。
- 固定 alpha 下预算与性能曲线。
- 实际风险与目标的偏离，以及实际成本与预算的偏离。
- 双方均达标的配对表。
- 类别分母、有效候选比例、重复工作点、seed 稳定性诊断。

超标点保留并明确标记，不能从图中删除。正文主要展示 alpha 见第 3 节，预算展示点固定为 0.1、0.5、1，完整六点保留在附件。所有文字计数从数据生成，禁止硬编码结论。

## 12 审计修复清单

下面的数字是审计基准提交的核对点，不是新结果必须达到的目标。候选、划分或协议修复后，结果变化应如实报告。

| 问题 | 修改要求 | 验证方式 |
| --- | --- | --- |
| AT 边界参数漏项 | 修复 `gamma_fit <= 1-alpha_at` 的浮点边界，保留 alpha_at=0.9 | 当前配置 AT 40 个、AT+Cal 160 个候选；原为 35、140 |
| mixture query rate 错误 | 加权“至少调用一次”的指示，不对期望调用数二值化 | 半数权重零调用、半数权重一次调用，query rate 为 0.5 |
| Brier 人群与权重错误 | 不跳过 Ours 无可行策略的 fold；样本加权；AT+Cal 无选择不拿候选 0 | 原 MATH gpt-oss Ours prior 全有效样本 Brier 约 0.0394701，而报告约 0.0446423 |
| 风险超标口径混用 | pooled risk 超标和多数 seeds 超标分别命名 | 原 AT 63 是多数 seeds 口径，按展示的 pooled risk 为 81 |
| 预算超标文字错误 | 由数据产生计数和位置 | 原 Ours 23 个为 22 GPQA、1 QK；QK 为 gpt-oss、alpha=0.1、frac=0.1 |
| 旧 QK 内层划分跨组 | 使用 merged kata groups | Fit、Validation、Test 和所有内层划分均检查组交集 |
| 旧预算参考使用全数据 | 改为 Fit/相应训练部分，保留稳定预算 ID | 扰动 Test 价格不改变 C_ref 和预算 |
| 空分母 bootstrap 伪造零风险 | NA 与有效重复数 | 全拒绝、无错误类别、无正确类别测试 |
| 差值点估计使用 bootstrap 均值 | 保存真实测试差值 | 直接逐题重算核对 |
| 有效性与标签耦合 | 分离输入条件和标签状态 | 扰动评价标签不改变 eligible 或决策 |

PAV 最小二乘拟合、逐规则实际费用累计、混合比率汇总及现有 release-error LP 的核心逻辑已通过审计。只在新分支需要时作有依据的重构，避免连带改变已正确的部分。

旧结果中的覆盖率胜出并不表示风险与预算同时达标。新报告不沿用旧 headline counts 作为结论，必须重算。

## 13 代码位置与配置模板

现有主要文件：

| 用途 | 路径 |
| --- | --- |
| 数据加载与既有 baseline | `src/vgx/baselines.py` |
| PAV | `src/vgx/gpqa/reporting_trial.py` |
| LP、规则、DP 原型 | `src/vgx/gpqa/fpr_routing.py` |
| 旧风险 baseline 与比较入口 | `src/vgx/gpqa/fpr_baselines.py`、`fpr_compare.py` |
| offline 主实现与报告 | `src/vgx/offline_frozen.py`、`offline_frozen_report.py` |
| AT 原始逻辑与说明 | `src/vgx/selective_auditing.py`、`docs/selective-auditing-baselines.md` |
| 当前配置 | `configs/offline_at_frozen_20261111.json` |
| 当前测试 | `tests/test_offline_frozen.py`、`tests/test_selective_auditing.py` |

新增结果使用：

```text
configs/offline_multirisk_20261116.json
results/offline_multirisk_20261116/
reports/offline_multirisk_20261116/
```

下列为待实现的新配置骨架，补齐 seed、AT、bootstrap、数值容差、日志及缺失处理字段。不要把它当成现有程序已经支持的 schema。

```json
{
  "schema": 2,
  "kind": "offline_multirisk_cached",
  "datasets": ["gpqa", "math500", "qkatas"],
  "generators": ["qwen3", "gptoss", "sonnet55"],
  "risk_metrics": ["selective_error", "fpr", "fnr"],
  "alpha_target": {
    "selective_error": [0.01, 0.02, 0.05, 0.10, 0.20],
    "fpr": [0.05, 0.10, 0.20, 0.30],
    "fnr": [0.01, 0.02, 0.05, 0.10]
  },
  "alpha_plan": {
    "selective_error": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
    "fpr": [0.025, 0.05, 0.10, 0.20, 0.30, 0.40],
    "fnr": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
  },
  "budget_fractions": [0, 0.1, 0.25, 0.5, 0.75, 1],
  "budget_reference": "fit_mean_all_five_verifiers_on_eligible_candidates",
  "calibration": {"method": "isotonic_pav", "fit_split_only": true},
  "likelihood": {"bins": 5, "laplace": 1.0},
  "ours": {"max_verifiers": 3, "orders_shortlist": 3, "mixture_draws": 20},
  "bootstrap": {"replicates": 2000, "seed": 20261111, "qkatas_unit": "merged_kata_group"},
  "execution": {"cache_only": true, "network_allowed": false},
  "output_dir": "results/offline_multirisk_20261116",
  "report_dir": "reports/offline_multirisk_20261116"
}
```

避免依赖 `VGX_CONSTRAINT` 等未记录的全局状态决定风险。风险参数应显式传递到规划、选择、评价和报告，确保同一进程依次跑三种风险不会相互污染。

## 14 实施顺序与运行接口

### 阶段 A 盘点与差异说明

读取 `git status`、HEAD、已有协议和缓存清单。记录已完成修复、待修复项、需要新增的接口；保留无关文件。核对缓存是否足以完成全部 702 个条件，不把 loader 连接到远程服务作为兜底。

### 阶段 B 审计修复与回归测试

现有测试命令：

```bash
python -m pytest tests/test_offline_frozen.py tests/test_selective_auditing.py
```

审计时这组共有 26 个通过的测试；新增测试后总数可变化。修复旧 pipeline 时保留兼容入口，并允许显式输出到新的目录，不能覆盖历史结果来验证修复。

### 阶段 C 三类风险统一接口

实现显式风险对象或等价的纯函数接口，统一 LP、Bellman、候选评价、Validation 选择、Test 评价与报告。可新增 `vgx.offline_multirisk` 作为薄入口，复用正确的旧模块。

将拟合模型、规则回放和统计汇总分离。相同 Fit 模型及同一规则的逐题回放可缓存复用，避免为每个 alpha/B 重新读取和计算全部内容；缓存键包含数据、划分、配置和代码版本。

### 阶段 D 数学测试与小规模回放

至少新增：

- 三类风险分母、优化方向和 LP 约束的解析例子。
- FNR 中 Release-eligible、Abstain-all 和无正确样本的边界。
- 小型可枚举 verifier 问题的 Bellman、LP 与列生成一致性。
- mixture 的费用、query rate、调用次数和风险分子分母。
- AT 0.9 参数边界及候选数量。
- 不等大小 fold、无可行策略和缺失 calibrator 的 Brier。
- 所有数据集 ID 隔离与 QK 组隔离。
- 扰动 Test 标签不影响拟合、预算、候选和动作。
- 扰动 Test 调用价格不影响预算或动作，只影响实际费用评价。
- 扰动未查询 verifier 输出不影响此前动作。
- Fixed-panel 无自适应提前停止。
- 缺失缓存无法触发网络调用。
- 同一进程切换风险指标的顺序不影响各自结果。

小规模回放用于查实现错误，不用来按 Test 表现挑选有利参数。发现 bug 后修复、记录版本并统一重跑受影响条件。

### 阶段 E 冻结配置与完整回放

以下是需要实现的入口约定，不是声称当前代码已有这些命令。若选择扩展旧入口，提供参数等价的准确命令并更新本节的实现说明。

```bash
python -m vgx.offline_multirisk preflight --config configs/offline_multirisk_20261116.json
python -m vgx.offline_multirisk verify --config configs/offline_multirisk_20261116.json
python -m vgx.offline_multirisk run --config configs/offline_multirisk_20261116.json
python -m vgx.offline_multirisk report --config configs/offline_multirisk_20261116.json
```

入口需要支持按 dataset、generator、risk 分片运行和安全恢复。恢复只能使用哈希完全匹配的中间产物，不能混用不同配置或代码版本。每次运行保存网络阻断证据、开始结束状态、错误日志和资源耗时。

### 阶段 F 独立重建与复现检查

从逐题和逐规则日志独立重算汇总值，核对模型/标签路径、混合费用、所有风险比率、可行状态与 headline counts。再次从同一缓存、配置与代码版本生成结果，比较规范化 JSON、CSV、解压后的日志和报告。运行时长、生成时间、gzip 时间戳等非确定性字段单独列明，不要求原始二进制文件必然相同。

## 15 交付清单与验收

交付内容：

1. 实现差异说明和审计修复对照表。
2. 代码、必要测试和完整配置。
3. 可直接运行的精确命令及依赖版本。
4. 数据、缓存、split 与代码版本清单和哈希。
5. 完整选参记录、逐题决策、逐规则调用轨迹和 evaluator 汇总。
6. 全部条件表、经验可行状态表、双方均达标比较表、HTML 和图表。
7. 原版本与新版本差异，区分实现修复、统计口径变化和新增实验目标。
8. 未完成或不可评价条件及原因；不得隐藏。

验收清单：

- [ ] 三个数据集、三个 generator 都覆盖三类风险。
- [ ] 主网格完整，缺失条件有状态行，额外附表未混入主分母。
- [ ] FNR 最小化错误放行概率，LP、DP 和选择准则一致。
- [ ] 所有 calibration、likelihood、预算参考和策略选择不使用 Test 标签。
- [ ] QK 的所有划分及统计重采样按既有 merged groups。
- [ ] AT 网格、gamma、校准权重和冻结语义正确。
- [ ] Fixed-panel 成员独立选择，混合和费用语义明确。
- [ ] 所有方法使用同一任务集合、风险分母、预算及选择规则。
- [ ] query rate、Brier、空分母、超标计数等审计问题已修复。
- [ ] 无新 API 调用；缺失缓存没有触发重试。
- [ ] 未覆盖历史结果或改动无关文件。
- [ ] 从决策日志可独立重建全部指标，重复运行可复现。
- [ ] 同时展示全部结果、双方达标结果及失败状态。
- [ ] 没有根据 Test 改选 alpha、seed、策略或删掉失败条件。
- [ ] 没有把经验达标、pointwise 区间或在线原版保证写成 offline 严格风险保证。

报告应直接回答哪些条件下 Ours 有优势、哪些没有，以及风险或预算超标的大小。若结果不支持原结论，修改结论，不修改评价标准。
