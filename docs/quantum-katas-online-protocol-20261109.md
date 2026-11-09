# QuantumKatas online learning 回放：协议（2026-11-09）

性质：在已有缓存输出上的追加分析，不是此前预注册的主实验。不新增任何 LLM API 调用。
配置：`configs/quantum_katas_online_20261109.json`；代码：`src/vgx/online/`；测试：`tests/test_online_replay.py`。

## 1. 数据与信息隔离

- 数据：`results/quantum_katas_full_{qwen3,sonnet55,gptoss}_20261019`，通过 `crossfit.load_rows` 导出为
  `reports/quantum_katas_online_20261109/rows/`：
  - `learner_<gen>.jsonl.gz`：任务 id、折、组、raw r、是否有代码、五个 verifier 的缓存分数/失败/估算费用。无标签，也无"是否有标签"。
  - `labels_<gen>.jsonl.gz`：官方 checker 标签，只给 evaluator、初始化（仅 D_init）和 D 组（仅审计轮）。
- 可处理性只用公开信息：有可提取代码且 r ∈ [0,1]。不可处理的任务弃答、不审计、计入发布率分母。
  （原 crossfit 的 `valid` 要求有官方标签；三个 generator 中有代码的候选都有标签，所以实际无差别。）
- 访问控制（`vgx.online.views`）：
  - `ScoreSource` 只通过当前轮的 `RoundAccess` 给出分数并记录每次读取；
  - `InitLabels` 只放行 D_init 标签；
  - `LabelGate` 只在本轮已抽中审计后放行当前题的标签（仅 D 组持有）；
  - `LabelStore` 只给 evaluator。
- 28 道 pilot 题不参与。

## 2. 角色与随机数

- 轮换 k = 0..4：D_init = 第 k 折，D_design = 第 (k+1) mod 5 折，其余三折为数据流（约 194 题）。
- 每种轮换有 5 个题序 o = 0..4。`SeedSequence([20261109, k, o])` 依次给出数据流排列、规则抽样
  u_rule[t] 和审计抽样 u_audit[t]。所有组、所有 generator、所有 q 共用这些随机数；
  审计规则为 I_t = 1[u_audit[t] < q]，所以不同 q 下的审计集合是嵌套的。
- 重复题序只用于评估顺序敏感性。区间估计按 35 个题目组做聚类 bootstrap：每个被抽中的组带上它的全部
  (轮换, 题序) 记录。

## 3. 初始化（各组共用，只用 D_init 标签与 D_design 公开字段）

- φ0：直接最小化 Brier 损失。校准器为 p̂ = σ(a + b·logit(clip(r, 1e-3, 1−1e-3)))，a ∈ [−10, 10]，
  b ∈ [0, 10]；用 L-BFGS-B，固定 4 个起点。
- 成本：D_init 可处理题上每个 verifier 的平均估算费用。B = 0.25 × 五个 verifier 平均费用之和（USD，冻结）。
- 审计 verifier：在无约束的标注拟合下，从 gap ≥ γ 的 verifier 中选 E[Var(A|Y)]/gap² 最小的；
  若都不满足，选 gap 最大的。
- θ0：在 D_init 的完整 panel 上做带标签的 MAP 拟合（见 §4）。若违反均值差约束，按约束 M 步投影。
- hierarchy：在 60 个有序三元组中，以 D_init 的报告为设计人群、在 (φ0, θ0) 下运行 DesignRouting，
  选拟合发布率最高的；平局按 verifier 配置顺序的字典序。

## 4. Likelihood 模型（C1/C2/D 共用）

- 每个 verifier 用三档等宽分箱；g[j, y, b] = P(第 b 档 | Y = y)。
- 可行域 Θ：每行严格正；审计 verifier 的编码均值差 μ1 − μ0 ≥ γ = 0.1，编码为 A ∈ {0, ½, 1}。
- 平滑：每个格子加 1 个伪计数，即 Dirichlet(2) 先验下的 MAP，不是无正则 MLE。
- 带约束 M 步：解 KKT 条件（对 τ、λ 一维求根），与 SLSQP 交叉核对过。
- 缺失 panel：θ 只用完整 panel 更新。φ 更新只需要审计 verifier 的分数。

## 5. 每轮顺序（`vgx.online.runner.Run.step`）

1. 读取公开字段；
2. 计算 p̂_t = r_φt(r)；
3. 用 u_rule[t] 从 R_t 中按逆 CDF 抽一条规则；
4. NestedVerify：只在决定查询时读取下一层分数；分数缺失则弃答；
5. 固定并记录决策、信念和决策成本；
6. 若 u_audit[t] < q 则审计；
7. 补齐其余 verifier（已读取的复用，只对新读取的计费）；
8. 用 θ_t、φ_t 更新参数；
9. 在 D_design 上重新设计 R_{t+1}。

- Ẑ = (A − μ0,t)/(μ1,t − μ0,t)，A 为审计 verifier 的档位编码，μ 由 θ_t 计算，不截断。
- φ 更新：φ_next = Π_box[φ_t + (2η_n/q)(Ẑ − p̂_t)∇p̂_t]，其中 η_n = η0/√(1 + n)，n 为之前的更新次数。
  步长中出现非有限值时跳过该步并记录。
- C1（全历史 EM）：
  - 历史 = D_init 完整 panel（当作无标签）+ 已审计的完整 panel；
  - π_s = r_φt(U_s) 全部重算，EM 期间固定；从 θ_t 起步；
  - 目标为 L_t(θ) + 对数先验；最多 500 轮，相对容差 1e-9；
  - 记录目标值、迭代次数、停止原因、单调性违例、处理记录数和耗时。
- C2（递推 online EM）：
  - S_0 = D_init 完整 panel 在 (φ0, θ0) 下的平均软计数；
  - S_n = (1 − γ_n)S_{n−1} + γ_n s_n，γ_n = 1/(N0 + n)；
  - 在 (N0 + n)S_n + 伪计数上做 M 步；
  - 过去记录的软权重不再重算。
- D：用真实 Y 代替 Ẑ；θ 由"初始化标签计数 + 审计轮标签计数"做 MAP 拟合。
- C1F/C2F：φ 冻结在 φ0，θ 的更新分别同 C1/C2。

## 6. DesignRouting（按正文，修正附录 Algorithm 2）

- 规则：对给定乘子 (ω, ρ) 的 Bellman 策略。
  - 立即发布的价值 A(b) = 1 − ω(1 − b − α)，G(b) = max(0, A)；
  - 继续查询当且仅当 Q_k > G + 1e-12（平局时停止）；停止后，A ≥ 0 则发布。
- 拟合分布：Y ~ Bern(r_φ(U_i))，i ∈ D_design；各层分档独立地服从 g[·, Y]。
  p、e、c 在 3³ 条分档路径上精确计算，不用网格；不可处理题贡献 0。
- 混合 LP：max Σw p，约束 Σw(e − αp) ≤ 0、Σw c ≤ B、Σw = 1（HiGHS）。
- 列生成：
  - 定价值为 p − ω(e − αp) − ρc；
  - 新规则只有超过当前候选的最大惩罚值 1e-9 以上才加入；
  - 从 {全部发布, 全部弃答} 两条规则开始，最多 200 轮。
- 与 3600 个 (ω, ρ) 网格点的 LP 对照：列生成的发布率不低于网格结果（见测试）。

## 7. 实验组与网格

- 组：A、B、C1、C2、D，以及诊断组 C1F、C2F。
- 网格：q ∈ {0.5, 0.25, 0.1}（A 无审计，只跑一次），3 个 generator × 5 种轮换 × 5 个题序。
- B 约束的是决策阶段的期望成本（拟合值），不含审计成本。
- 实际新增 API 支出为 0。"审计成本"是按缓存估算费用计的反事实成本。

## 8. 与论文的差异

1. 附录 Algorithm 2 的 LP 约束和定价式漏了 α·p 项，按正文修正。
2. 论文 (5) 的历史只含审计记录；这里 C1 的历史还包含 D_init 的 panel（当作无标签）。
   C2 用同一批 panel 构造 S_0。
3. 论文中的 μ_y(x) 依赖上下文；这里用与上下文无关的 μ_y(θ)。Ẑ 无偏要求 E[A | Y, r] = μ_Y，
   即给定 Y 时审计分数与 generator 置信度条件独立。
4. 论文中 Θ 的单调似然比（MLR）没有强制；只强制审计 verifier 的均值差约束。
5. 平滑使目标变为 MAP。
6. 论文中的分数是连续的，这里用三档离散化，审计分数按档位编码。

## 9. 超参数来源

- η0：只根据模拟实验选（`reports/quantum_katas_online_20261109/sim/eta_sweep.json`），在完整网格运行前冻结。
- γ = 0.1、伪计数 1、EM 容差：事先固定，未根据真实数据调整。

## 10. 追加：依赖感知 likelihood（2026-11-09，在独立模型结果之后，经 tony 选择）

性质：在独立模型结果出来之后追加的分析，不是预先设定的主分析。配置
`configs/quantum_katas_online_dependence_20261109.json`，代码 `vgx.online.joint`，测试 `tests/test_online_joint.py`。
只跑 A、B、C2、D，q ∈ {0.5, 0.25, 0.1}，B = 25%；其余设置、随机数、题序与 §1–§9 完全相同。

- 模型：对每个类别 y，hierarchy 三个 verifier 的 27 个分档格子用联合表
  T[y] = (C[y] + α·∏_k g[h_k, y]) / (n_y + α)，与 `vgx.gpqa.dependence_triples` 相同。
  C 为格子计数（D_init 为硬计数），g 为原来的边际（含审计 verifier 的均值差约束）。α = ∞ 即原独立模型。
- hierarchy 之外的两个 verifier 仍用边际 g，并假设给定 Y 时与 hierarchy 独立；它们只进入完整 panel 后验
  （C2 的 E 步和评估），不进入路由。
- Ẑ 仍用审计 verifier 的约束边际 g[a] 计算，所以 B 组的 φ 轨迹在两种模型下相同，差异只来自路由。
- α：每个轮换、每个 verifier 三元组，在 D_init 完整 panel 上做 5 折内层交叉验证（留出格子的类条件对数似然），
  候选 {0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, ∞}，平局取较大 α。之后在整个 run 内冻结；
  C2 与 D 更新计数，不重选 α。
- hierarchy：60 个有序三元组在 (φ0, g0, T0) 下运行 DesignRouting，取拟合发布率最高者。
- 路由：DesignRouting 与 NestedVerify 的 Bayes 更新改为 P(第 k 层分档 | 前面各层分档, Y)，仍在 3³ 树上精确计算。
- C2：在 E 步用联合模型的后验权重，同时递推边际软计数和格子软计数，γ_n 与 §5 相同。
- D：初始化标签与审计轮标签的格子硬计数加边际硬计数。
- C1、C1F、C2F 未在依赖模型下实现（会报错）。
- 评估：brier_panel 和 label_ll 用 T_t（hierarchy）乘其余两个 verifier 的边际。

评估端 oracle 诊断（`scripts/online_oracle_decomposition_20261109.py`）：用数据流自身的官方标签在样本内拟合 φ*、
θ*/T*（联合模型保留 D_init 选出的 α），在相同 hierarchy、预算、设计集和随机数下重放 A。它只用于拆解偏差来源，
不是实验组，也不反馈给任何学习器。
