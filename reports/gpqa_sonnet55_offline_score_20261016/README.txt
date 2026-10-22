Sonnet / Gemini / Qwen：冻结答案离线评分
2026-10-16, America/Los_Angeles
==============================================================

一、范围与口径
本次按用户授权读取答案键，对相同的249道开发题、196道Diamond题分别评分。
候选答案、原始confidence、解析器及历史实验全部保持不变；无API请求，无校准、重拟合或修复。
开发集在历史文件中名为calibration；这个名称不表示本次对confidence做了校准。
正确率 = 正确答案数 / 全部指定题数；无法解析的答案计为未答对。
平均confidence、答错时confidence与Brier只使用答案和confidence均有效的题目。
Brier = mean((原始confidence - 是否答对)^2)，越小越好。不是四选项概率向量的多分类Brier。
答错时confidence的分母是有有效confidence的错误答案数量，不含未解析答案。

二、全部指定题目
--------------------------------------------------------------

开发集
Model | Correct / N | Accuracy | Mean confidence | Confidence when wrong (n) | Raw Brier (n)
Gemini 3.8 Flash | 211/249 | 84.74% | 93.44% | 88.56% (36) | 0.125491 (247)
Qwen3-235B (two-step) | 157/249 | 63.05% | 91.99% | 89.17% (92) | 0.301451 (249)
Sonnet 5.5 | 217/249 | 87.15% | 79.77% | 66.39% (31) | 0.102596 (248)

Diamond
Model | Correct / N | Accuracy | Mean confidence | Confidence when wrong (n) | Raw Brier (n)
Gemini 3.8 Flash | 180/196 | 91.84% | 94.79% | 84.33% (15) | 0.061073 (195)
Qwen3-235B (two-step) | 125/196 | 63.78% | 92.31% | 90.21% (71) | 0.299911 (196)
Sonnet 5.5 | 180/196 | 91.84% | 83.73% | 66.12% (16) | 0.070458 (196)

三、共同有效题目（补充比较）
--------------------------------------------------------------
在读取标签前确定三模型均有有效答案与confidence的交集：开发246题、Diamond195题。

开发集
Model | Correct / N | Accuracy | Mean confidence | Confidence when wrong (n) | Raw Brier (n)
Gemini 3.8 Flash | 210/246 | 85.37% | 93.58% | 88.56% (36) | 0.125350 (246)
Qwen3-235B (two-step) | 157/246 | 63.82% | 92.00% | 89.09% (89) | 0.294852 (246)
Sonnet 5.5 | 215/246 | 87.40% | 79.77% | 66.39% (31) | 0.103105 (246)

Diamond
Model | Correct / N | Accuracy | Mean confidence | Confidence when wrong (n) | Raw Brier (n)
Gemini 3.8 Flash | 180/195 | 92.31% | 94.79% | 84.33% (15) | 0.061073 (195)
Qwen3-235B (two-step) | 124/195 | 63.59% | 92.35% | 90.21% (71) | 0.301333 (195)
Sonnet 5.5 | 180/195 | 92.31% | 83.76% | 65.40% (15) | 0.067778 (195)

四、完整性与解释
--------------------------------------------------------------
开发集缺失/无法解析答案：Gemini 2题、Qwen 0题、Sonnet 1题；Diamond：1、0、0题。
Gemini开发集另有1条无有效答案但存在数值confidence；不把它纳入概率指标。
主表概率指标样本数：开发Gemini247、Qwen249、Sonnet248；Diamond195、196、196。
若只在有效答案内计正确率：开发Gemini85.43%、Qwen63.05%、Sonnet87.50%；Diamond92.31%、63.78%、91.84%。
共同有效题目中，Brier的排序与主表一致：开发Sonnet更低，Diamond Gemini更低，Qwen均最高。
Sonnet答错时的confidence更低，但答对时也较保守。Diamond全有效题中，Sonnet答对时平均confidence为85.29%，Gemini为95.66%。
因此，较低的答错confidence并不保证整体Brier更低；Brier同时包含正确和错误回答的概率误差。
本次没有显著性检验，也不声称模型总体优劣或概率已得到校准保证。
Qwen使用两步生成，Sonnet使用adaptive thinking/high effort；设置与服务日期不同，比较对象是模型加生成协议。
开发集已用于先前探索；原50题参与过格式协议检查。Diamond上的新评分不用于本次调参。

五、验证与文件
--------------------------------------------------------------
已确认三模型题目及选项顺序相同，开发/Diamond无交集，候选与标签文件校验和有效。
原50题已发布的三模型正确数、平均confidence、答错confidence和Brier全部复现。
评分前后输入文件、预算账本和历史结果未改变。未运行新增单元测试。
此前采集报告重复生成的tuple/list问题未改动；本次直接读取已冻结候选并成功完成独立评分。
目录包含result.json、run_stdout.txt、README.txt和sha256.txt，仅含汇总指标与来源哈希。
不含题干、答案键、逐题结果或凭证。未提交、推送或合并。

复现命令：
.venv/bin/python -m vgx.gpqa.sonnet_offline_score --output results/gpqa_sonnet55_offline_score_20261016

Result ID: 6e8567e921247f0b0c91b1dc827e2218501e8bb906e6cdeee1d670c43a778150
