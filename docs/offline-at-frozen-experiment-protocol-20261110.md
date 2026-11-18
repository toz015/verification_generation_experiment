# Offline 实验实施协议

日期：2026-11-10

本指令供执行 agent 讨论、实现和运行实验。核心设置是：离线拟合、测试时冻结；统一控制放行错误率；分别统计“拦截／送审选择”和“实际执行的审计”；按真实使用的资源计算成本。本文件是实验实施协议，不是实验结果。

## 1 本次任务与范围

请在现有仓库中实现并比较以下三种方法：

1. **AT-Frozen**
2. **AT＋Calibration-Frozen**
3. **Ours-Offline：我们的离线生成—验证策略**

本轮回答的问题是：

> 在相同候选答案、数据划分和成本预算下，各方法能放行多少答案，同时将放行答案中的错误比例控制在目标 $\alpha$ 附近？各方法拦截错误和误拦正确答案的能力如何？

本轮采用固定候选答案，不重新生成答案，不实现审计后的纠正流程，不在测试阶段进行 EM、在线校准、阈值更新或 likelihood 更新。

AT 原论文研究在线选择性审计，审计者可以接受或纠正答案。本次将其适配为离线筛选基线，必须明确标为 **offline adaptation**，不能将原论文的在线风险保证直接用于这些冻结版本。参考 [Reliable AI-Assisted Decisions via Selective Auditing，arXiv:2610.31662v1](https://arxiv.org/abs/2610.31662v1)。

先检查现有实现与下述定义的差异，形成简短的实现说明，然后完成代码、验证和缓存数据实验。保持无关文件和历史结果不变。已有 [selective-auditing baselines](selective-auditing-baselines.md) 记录的是在线审计实验；本轮使用独立的配置与输出，不覆盖其结果。

## 2 统一决策语义

对每个固定候选答案，定义：

- $Y_i=1$：候选答案正确；$e_i=1-Y_i$。
- $a_i=1$：系统自动放行原始候选答案。
- $d_i=1-a_i$：系统拦截原始答案，包括弃答或转交外部处理。
- $Q_i$：系统实际查询的弱 verifier 集合。
- $o_i=1$：系统实际获取了一次外部 oracle／人工审计结果。

**不要默认 $d_i=o_i$。**决定弃答或送审，不等于已经获取真实标签。

本轮主表采用统一的“原始答案筛选”任务：

- AT 的送审决定映射为“不自动放行原始答案”。
- Ours 的弃答同样映射为“不自动放行原始答案”。
- 两者都不执行外部纠错。
- 测试真值仅由独立 evaluator 使用，不提供给策略。
- 因此，主表中策略实际获得的测试 oracle 标签数为零。

可以用 evaluator 的真值计算“所选送审集合中有多少错误”，但应称为**送审选择质量／错误拦截能力**，不能写成策略已经免费获得了这些审计标签。

弱 verifier 查询后仍可能放行，因此“查询过 verifier”也不等于“拦截”。

## 3 统一风险目标和工作点选择

主风险指标固定为：

$$
\widehat R_{\mathrm{release}}
=
\frac{\sum_i a_i e_i}{\sum_i a_i}.
$$

放行率和平均决策成本分别为：

$$
\widehat{\mathrm{Coverage}}
=
\frac1n\sum_i a_i,
\qquad
\widehat C
=
\frac1n\sum_i C_i.
$$

所有方法统一按以下目标选择工作点：

$$
\max \widehat{\mathrm{Coverage}}
\quad\text{subject to}\quad
\sum_i a_i(e_i-\alpha)\le 0,\qquad
\widehat C\le B.
$$

这里的工作点选择只使用开发／验证数据。

默认：

- 主目标：$\alpha=0.05$。
- 补充目标：$\alpha\in\{0.01,0.10\}$。
- 预算是**平均每题的决策成本上限**，不是每题必须花掉的金额。

必须区分三个参数：

- `alpha_target`：统一比较的放行错误率目标。
- `alpha_at`：原版 AT 更新公式内部的风险参数。
- `alpha_plan`：Ours 内部模型规划使用的风险参数，如需要调节。

**不得直接令 `alpha_at = alpha_target` 并宣称风险目标已经统一。**原版 AT 控制的指标分母是全部原始错误数，与放行错误率不同。

若没有满足开发集风险和预算约束的非零放行策略，记录：

`no_nontrivial_feasible_policy`

可以部署全弃答回退策略，但报告放行数为零、经验放行错误率为 `NA`；不要把它包装成“零错误的有效放行方案”。

开发集满足约束不等于测试集获得有限样本风险保证。测试阶段应如实报告风险和预算超限。

## 4 数据划分和冻结要求

复用现有候选答案、checker、verifier 缓存和外层划分：

- GPQA：保留现有 calibration／Diamond test 边界。
- QuantumKatas：保留现有分组五折及已排除的开发题目，继续使用官方 checker 定义。
- MATH-500：保留现有候选答案、评分规则和外层划分。
- 同一 dataset／generator 下，三种方法使用完全相同的候选答案和评估任务集合。
- 不同 generator 分别拟合参数、分别报告结果。

在每个外层训练池内，再划分：

| 数据部分 | 用途 |
|---|---|
| Fit | 拟合 AT 状态、校准函数、verifier likelihood |
| Design／Validation | 构建路由候选、选择超参数和满足共同约束的工作点 |
| Test | 冻结策略后，最终评价 |

建议默认：

- GPQA 的 calibration 池按固定种子划分约 80% Fit、20% Validation，存在关联任务时按组划分。
- QuantumKatas／MATH 的外层五折中，一折 Test；其余四折中固定一折 Validation、三折 Fit。所有方法使用同一安排。
- 选定工作点后，本轮不再合并 Fit 和 Validation 重训，以避免实际测试的策略与经过验证的策略不同。
- 如果采用更复杂的内层交叉验证与重训流程，必须明确记录，三种方法保持一致。

重新运行 Ours，不能拿“使用更多训练数据的历史 Ours 结果”与新的基线直接比较。

缺失和失败处理：

- 不根据测试效果挑选完整数据子集。
- 区分无效候选、verifier 返回失败、尚未采集信号、缺失评估标签。
- 缺失真值不能默认记为错误。
- 策略输入的有效性判断不得依赖隐藏真值或 checker 标签是否存在。
- 若需要公共可评估子集，应在运行方法前固定，并同时报告排除原因和全任务覆盖情况。

## 5 AT-Frozen 的实现

在 Fit 集上按固定随机顺序运行原版 AT。设当前输入 confidence 为 $C_t$，使用原始 generator confidence：

$$
\pi_t=
\begin{cases}
1,&C_t<\tau_t,\\
\gamma_{\mathrm{fit}},&C_t\ge\tau_t,
\end{cases}
\qquad
H_t\sim\mathrm{Bernoulli}(\pi_t).
$$

只有被抽中时，训练更新接口才读取该样本标签：

$$
\tau_{t+1}
=
\tau_t+
\eta(1-\pi_t-\alpha_{\mathrm{at}})
e_t\frac{H_t}{\pi_t}.
$$

实现要求：

- 默认 $\tau_1=0.5$。
- 不把 $\tau_t$ 强行裁剪到 $[0,1]$。
- 每个配置默认单次遍历 Fit。
- 完成后冻结最终阈值。
- 不在 Validation 或 Test 上继续执行更新公式。

本轮主表采用：

$$
\gamma_{\mathrm{test}}=0,
\qquad
a_i=\mathbf 1\{C_i\ge\tau_{\mathrm{frozen}}\}.
$$

这是明确的离线适配：训练阶段保留探索，测试阶段不再为未来学习进行随机抽查。

另外保留一个敏感性版本：

$$
\gamma_{\mathrm{test}}=\gamma_{\mathrm{fit}},
$$

即冻结参数后仍保留论文的随机送审规则。报告中明确区分，不能悄悄切换。主结论不能依赖给基线保留无学习用途的随机拦截。

## 6 AT＋Calibration-Frozen 的实现

按论文 Algorithm 1，在 Fit 上联合学习阈值与单调校准函数。

设置固定网格 $\theta_j$，包含末端 $\theta_{k+1}=\infty$，初始化：

$$
w_{1,j}=\frac1{k+1},
\qquad
\tau_1=0.5.
$$

每步计算：

$$
x_{t,j}=\mathbf 1\{C_t\ge\theta_j\},
\qquad
S_t=\sum_jw_{t,j}x_{t,j}.
$$

用 $S_t$ 决定审计概率，随后更新：

$$
\widehat g_{t,j}
=
\frac{H_t}{\pi_t}(S_t-Y_t)x_{t,j},
$$

$$
w_{t+1,j}
\propto
w_{t,j}\exp(-\beta\widehat g_{t,j}).
$$

阈值更新与 AT 相同，使用更新前的 $\pi_t$ 和当轮状态。指数权重更新采用数值稳定的实现。

训练结束后同时冻结：

- 网格；
- 权重 $w$；
- 阈值 $\tau$。

主表测试规则为：

$$
a_i=\mathbf 1\{S_{\mathrm{frozen}}(C_i)\ge\tau_{\mathrm{frozen}}\},
$$

同样采用 $\gamma_{\mathrm{test}}=0$，并提供保留随机送审的敏感性版本。

**不要用 isotonic、Platt 或其他离线校准器替换论文的指数权重算法后，仍称为 AT＋Calibration-Frozen。**这些可以另列为补充基线。

特别注意：冻结后的单调校准不会改变原 confidence 的排序，可能合并一些分数。因此，它的离线收益主要来自阈值学习、概率刻度和有限样本效应，不应预设其一定改善完整的 risk–coverage 前沿。

上述 AT 更新依据[原论文第 3 节与 Algorithm 1](https://arxiv.org/html/2610.31662v1)。

## 7 Ours-Offline 的实现与约束检查

Ours 在 Fit 上完成：

- confidence reporting／calibration；
- verifier 条件分布拟合；
- 所需的固定模型参数估计。

随后在 Design 上构建固定路由策略。测试时只允许根据当前题目已经查询到的 verifier 输出更新当前题目的 belief，不更新跨题共享参数。

如果本轮采用草稿中的 Brier 最小化校准，应明确实现该目标，例如采用单调约束下的 Brier 拟合。现有 Laplace 平滑单调分组方法如保留，应单独说明，不能直接声称它与纯 Brier 最小化完全相同。

路由优化应对应：

$$
\max_w\sum_j w_jp_j
$$

约束：

$$
\sum_jw_j(e_j-\alpha_{\mathrm{plan}}p_j)\le0,
\qquad
\sum_jw_jc_j\le B_{\mathrm{plan}},
\qquad
w_j\ge0,\quad \sum_jw_j=1.
$$

对应的立即放行价值是：

$$
A_\omega(b)=1-\omega(1-b-\alpha_{\mathrm{plan}}).
$$

必须检查 LP、Bellman 递推和列生成使用同一个风险定义。

仓库实现重点：

- [统一数据与基线入口](../src/vgx/baselines.py)
- [风险与预算路由原型](../src/vgx/gpqa/fpr_routing.py)
- [QuantumKatas 离线流程](../src/vgx/quantum_katas/crossfit.py)

其中路由原型默认是 FPR，已有 `fdr` 分支及乘子缩放逻辑。请核对并复用正确的等价实现，不要仅修改输出指标名称。新实验配置应显式写明 `risk_metric: selective_error`，避免依赖未记录的环境变量。

项目草稿的 8 页版本中，第 2–4 页正文与第 8 页附录 Algorithm 2 的风险约束存在不一致：本轮以正文的放行错误率约束为准，不沿用附录中的旧分母。本节给出了本轮实施所需的风险目标、LP 约束与放行价值。

还需区分：

- Ours 内部按拟合模型计算的规划风险；
- Validation 上用真实标签计算的经验风险。

本轮三种方法统一经过第二层的 Validation 工作点选择。若将 Ours 内部优化也改为直接使用 Validation 标签，应明确标为实现变体，不能与草稿的模型规划版本混称。

## 8 统一选参规则与建议配置

以下网格是本项目建议设置，不是声称来自论文的默认参数：

~~~text
alpha_target: [0.01, 0.05, 0.10]
alpha_at:     [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
gamma_fit:    0.10
gamma_test:   0.00
eta:          [0.001, 0.003, 0.01, 0.03, 0.10]
beta:         [0.001, 0.01, 0.10, 1.00]
calibration_grid: [0, 0.05, ..., 1] plus infinity
fit_epochs:   1
~~~

保证 $\gamma_{\mathrm{fit}}\le1-\alpha_{\mathrm{at}}$。

对每个 dataset、generator、外层划分和 $(\alpha_{\mathrm{target}},B)$：

1. 在 Fit 上生成候选冻结模型。
2. 在 Validation 上测量放行率、放行错误率和成本。
3. 从满足共同风险、成本约束的候选中，选择放行率最高者。
4. 平局时依次选择成本更低、规则更简单、配置序号更小者。
5. 冻结后评价 Test。

Ours 可以预先设置 `alpha_plan`、规划预算等候选值，再经过同样的验证规则。

不得：

- 用测试标签选择阈值、规则、混合权重或超参数。
- 因测试风险超标而回头选择另一个工作点，并仍称为独立测试。
- 从多个随机种子中只保留最好结果。
- 将历史的固定 utility 参数，例如 $L=19$，直接视为已经实现了 5% 放行错误率控制。

随机训练顺序与抽查使用预先固定的种子；建议完整流程运行 20 个种子，报告全部结果或汇总分布，不挑种子。

## 9 成本核算

主表控制的是**筛选决策成本**：

$$
C_i^{\mathrm{screen}}
=
C_i^{\mathrm{extra\ confidence}}
+
\sum_{j\in Q_i}\kappa_{ij}.
$$

其中：

- generator 已经输出的原始 confidence，不重复收费；
- 额外调用模型获取 confidence，必须收费；
- Ours 只计策略实际读取的 verifier 调用；
- 返回格式错误但实际发生的付费调用，也计费；
- 复用同一次调用的结果，不重复计费；
- 本地读取缓存不意味着反事实部署成本为零。

主表中不执行外部审计，因此不要给任何一方加上虚构的 oracle 调用，也不要宣称它获得了免费 oracle 反馈。

在这个定义下，仅使用已有 generator confidence 的 AT-Frozen，新增 API 决策成本可能为零。其结果可能在多个 $B$ 下保持相同，这是正常结果；不能强迫它花满预算。

另报告完整成本分项：

| 成本 | 处理方式 |
|---|---|
| 固定候选答案生成成本 | 各方法共同基础成本，单列 |
| 额外 confidence 与弱 verifier 成本 | 纳入主表预算 $B$ |
| 实际外部审计成本 | 仅在确实执行该服务的设置中计入 |
| Fit／Validation 数据与标签采集成本 | 单列，并可报告部署规模下的摊销成本 |
| 测试评估标签成本 | 公共评估成本，不算策略获得的反馈 |
| 优化、拟合与回放计算时间 | 单列运行时间 |

训练池的标签和缓存是共同的数据资源。AT 拟合中的多次模拟审计读取同一已有标签，不应被重复计成多次真实标签采集；同时也不能把整个训练数据采集成本隐去。

价格采用固定、有版本记录的计价表。区分：

- 决策时可用的训练估计调用成本；
- 测试回放后测得的实际 token 计价成本。

策略不能提前读取尚未发生调用的真实 token 用量来决定是否查询。

预算建议：

- GPQA 优先复用现有美元预算网格，并增加 $B=0$。
- 跨数据集另给归一化预算：以 Fit 上固定 verifier 全面板的平均成本 $c_{\mathrm{ref}}$ 为单位，使用 $B/c_{\mathrm{ref}}\in\{0,0.1,0.25,0.5,0.75,1\}$。
- 预算只根据 Fit 成本确定，不根据 Test 表现重设。

### 外部审计成本敏感性分析

额外分析一个明确不同的服务情景：

> 所有被拦截的答案，都转交同一种外部审计服务。

此时三种方法统一采用：

$$
o_i=d_i,
\qquad
C_i^{\mathrm{service}}
=
C_i^{\mathrm{screen}}+d_i c_A.
$$

这只是统一下游服务的补充情景，不能反过来规定主表中的弃答必须购买真实标签。

若没有可信的审计单价，报告审计次数，并对预设 $c_A/c_{\mathrm{ref}}$ 做敏感性分析，明确标为情景估算。需要比较总成本预算下的最佳策略时，仍在 Validation 上重新选择工作点，再冻结评价 Test。

## 10 审计与拦截结果的比较

对三种方法统一报告：

| 指标 | 定义 |
|---|---|
| 放行错误率 | $\sum a_ie_i/\sum a_i$ |
| 放行率 | $\sum a_i/n$ |
| 拦截率 | $\sum d_i/n$ |
| 错误拦截率 | $\sum d_ie_i/\sum e_i$ |
| 拦截集合中的错误比例 | $\sum d_ie_i/\sum d_i$ |
| 正确答案误拦率 | $\sum d_i(1-e_i)/\sum(1-e_i)$ |
| verifier 查询率与平均调用数 | 从实际查询日志计算 |
| 外部标签获取数 | $\sum o_i$，主表为零 |

各指标分母为零时记 `NA`。

补充比较可以包含：

- 相同目标放行错误率下，谁的放行率更高；
- 相同筛选成本预算下，谁拦截了更多错误；
- 相近送审比例下，谁的送审集合包含更多错误、误拦更少正确答案。

注意：如果严格固定拦截率，放行率已经由 $1-\mathrm{拦截率}$ 决定，不能再用放行率宣称优势。

匹配送审比例的工作点也应在 Validation 上选取，Test 上报告实际比例。测试集上的事后曲线可以作为描述性分析，但不能据此选出主表的“最佳部署点”。

## 11 统计报告与必要补充对照

主表按 dataset、generator、$\alpha$、$B$ 分组，至少包含：

~~~text
method
n_tasks
n_released
n_wrong_released
release_error_rate
coverage
mean_screen_cost
weak_verifier_calls_per_task
risk_target_exceeded
budget_exceeded
~~~

审计选择指标放第二张表；外部审计服务成本情景另表报告。

同时输出：

- 放行错误率—放行率图；
- 在不同预算下的放行率图，并标出风险超标点；
- 错误拦截率—拦截率图；
- AT 校准前后的 Brier 与可靠性诊断。

三种方法使用配对评估。QuantumKatas 按既有任务组做 bootstrap，而不是把同组题目当成完全独立。建议 2,000 次配对重采样。

需要明确：

- 小样本下，零观测错误不等于风险已被证明很低。
- 全零错误时，普通 percentile bootstrap 可能给出退化区间，不能将其当作可靠的零风险证明。
- 固定策略上的 bootstrap 不包含全部训练不确定性；训练种子变化单独报告。
- 使用随机策略混合时，分别报告期望计数结果和抽样运行结果。不要混淆“期望计数之比”与“每次运行比例的期望”。

必要补充对照：

1. **离线原始 confidence 阈值基线**：直接在 Validation 上选阈值，满足同一风险和成本规则。用于判断 AT-Frozen 是否仅因有限训练未找到好的阈值。
2. **AT 两种测试抽查设置**：主表关闭抽查；补充保留 $\gamma_{\mathrm{fit}}$。
3. **Ours 原始／校准 prior 对照**：量化我们的 confidence 校准贡献。

这些对照可以放附表。不要把 Ours 相对一个训练不足的 AT 阈值的优势，直接解释为相对所有 confidence 筛选方法的优势。

## 12 实现验收日志和交付

建议提供统一接口：

~~~python
fit(fit_data, config)
select_policy(validation_data, alpha_target, budget)
freeze()
decide(candidate_features, verifier_accessor)
evaluate(decisions, private_labels)
~~~

策略接口不得接收测试标签。verifier accessor 只返回已请求的信号。

每个决策至少记录：

~~~text
dataset / generator / task_id / split_id / seed
method / frozen_state_hash
raw_confidence / calibrated_confidence
release_or_defer
queried_verifiers / observed_scores
screen_cost / external_audit_count
selected_config_id
~~~

真值由 evaluator 在决策完成后关联，不写入策略输入。

完成以下有实际意义的验证：

- 修改隐藏测试标签，不改变任何策略决策。
- 修改未查询的 verifier 输出，不改变已有决策。
- 测试前后阈值、校准器、likelihood、路由及混合权重的哈希不变。
- AT 与 AT＋Calibration 的训练更新符合手工可检查的小例子。
- LP 确实约束 $e-\alpha p$，没有使用旧 FPR 分母。
- 成本日志与实际读取的调用集合一致。
- 混合策略只执行选中的规则，不能把所有候选规则的调用成本相加。
- 拦截、弱 verifier 查询和真实标签获取三类计数彼此独立。

执行顺序：

1. 检查现有数据、代码和协议差异，固定配置与划分。
2. 用合成例子和少量缓存任务验证公式、接口、成本和冻结要求。
3. 在现有缓存上运行完整实验。
4. 输出主表、审计选择表、成本敏感性表及图。
5. 汇总风险超标、预算超标、不可行工作点和数据缺失情况。
6. 提供复现命令、配置、逐题日志、冻结模型状态、测试结果和简短结论。

本轮优先使用现有缓存完成，不新增生成或 verifier API 调用。结果必须允许 AT、AT＋Calibration 与 Ours 中任何一种胜出；若结果相近、风险控制失败或样本量不足，也应直接报告。
