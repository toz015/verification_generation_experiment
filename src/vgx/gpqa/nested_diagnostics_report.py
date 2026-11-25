"""Produce a short diagnosis and aggregate CSV from frozen diagnostic output."""
import argparse
import csv
import json
from pathlib import Path

from vgx.common.storage import atomic_json, file_digest


def render(root, output):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'report.json').read_text())
    primary={g:report['results'][g+'/all_calibration/seed_20261005']['19.0'] for g in ('gemini','qwen')}
    rows=[]
    for run,values in report['results'].items():
        for loss,s in values.items():
            rows.append({'run':run,'loss':loss,**{k:v for k,v in s.items() if not isinstance(v,(dict,list))},
                **{'counterfactual_'+k:v for k,v in s['counterfactual'].items() if not isinstance(v,dict)}})
    with (output/'diagnostic_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# 三层验证：提前输出与价值估计诊断','',
        '**结论：主分析中的错误输出，大多不能靠继续询问剩余 verifier 避免。后验概率低估实际错误风险，是目前直接观察到的问题；现有证据还不能把原因唯一归于 prior、likelihood 或模型依赖。**','',
        '范围：沿用冻结策略；主表为 249 道开发题、第一种三折划分、R=1/L=19/每美元10效用。原始 confidence、候选答案、likelihood、顺序和成本设定均未改变。新增 API 支出 $0。','',
        '## 1. 停得太早能解释多少错误？','',
        '| Generator | 输出中的全部错误 | 提前停止输出的错误 | 继续询问全部剩余 verifier 能拦下 |',
        '|---|---:|---:|---:|']
    for g,s in primary.items():
        c=s['counterfactual'];text.append(f"| {g} | {s['wrong_released']} | {c['early_wrong_released']} | {c['early_wrong_rejected_if_force_all']} |")
    text += ['', '这是用已有缓存做的事后对照：原始顺序策略仍然只能看到已查询信号。额外信号只用于诊断，没有倒灌给原决策。两个策略都只决定输出／abstain，不修改候选答案。', '',
        '## 2. 后验说“接近99%”，实际却没有达到','',
        '| Generator | 输出数 | 平均后验 | 实际输出正确率 | 按后验预计错误数 | 实际错误数 |',
        '|---|---:|---:|---:|---:|---:|']
    for g,s in primary.items():
        text.append(f"| {g} | {s['released']} | {s['mean_released_posterior']:.2%} | {s['released_accuracy']:.2%} | {s['posterior_predicted_wrong']:.2f} | {s['wrong_released']} |")
    text += ['', '“预计错误数”是所有输出的 (1−posterior) 之和。这是验证概率含义的诊断，没有拟合或替换 generator confidence。','',
        '一个 Qwen 错误案例：原始 confidence=0.99，Flash 返回1.0，Bayes 后验升到0.99717。立即输出效用 A=0.94331，继续验证 Q=0.92617，于是停止。缓存中后两位分别返回1.0、0.95，全部询问后仍会输出该错误答案。这个案例说明，继续增加同样的支持信号无法纠正错误。','',
        '另一个 Qwen 案例确实因提前停止而漏检：Flash=0.85、Flash-Lite=1.0 后，后验0.98851，A=0.77021>Q=0.74551，因此输出；如果继续，Haiku=0.35 会把后验降到0.94882，从而 abstain。这是主划分中那1个可被拦下的错误，不能由此推出应当一律继续询问。','',
        '## 3. 修正之前对“价值高估”的概括','',
        '| Generator | 模型预测总效用 | 实际总效用 | 模型预测验证增益 | 实际验证增益 |',
        '|---|---:|---:|---:|---:|']
    for g,s in primary.items():
        text.append(f"| {g} | {s['predicted_mean_utility']:+.3f} | {s['actual_mean_utility']:+.3f} | {s['predicted_verification_gain']:+.3f} | {s['actual_verification_gain']:+.3f} |")
    text += ['', '总效用从首次查询前计算，包含 verifier 成本；验证增益是三层策略减去仅原始 confidence 策略，按相同题目平均。预测值用原始 prior 和训练折 likelihood；实际值用留出题对错评分。','',
        '**总效用过于乐观，但验证相对“不验证”的增量改善并未在这两个主结果中被高估。** 仅原始 confidence 的决策已经明显低估错误风险。这个代数分解说明误差起点，不能单独证明原始 prior 是唯一原因。','',
        '另一个值得跟进的现象：在错误候选中，三位 verifier 都给出支持信号的数量，Gemini 是16（独立模型预计8.04），Qwen 是15（预计7.93）。这里“支持”定义为所在信号区间的 likelihood ratio>1，并非 score>0.5。样本只包含三信号都有效的题目。边际 likelihood 误差和信号依赖都可能造成差异；按对错分组的统计，不能直接否定论文按 (Y,w) 条件独立的假设。','',
        '## 4. 数值与一致性核对','',
        '- 120 组已保存价值表均与冻结参数重新构建的表一致。',
        '- 枚举三层所有可能信号、去掉网格插值后：249题、L=19、两种 generator、五种折分的决策路径均不变。主结论不能用网格误差解释。',
        '- 全部敏感性分析中有41个“题目×折分×惩罚”记录的路径变化，其中6个最终输出／abstain 决策变化；这些不是41道独立题。边界差异已保留，未修改旧策略。',
        '- 逐题效用、原始策略对照和预测误差分解均核对一致；来源分析文件哈希未变。',
        '- 未读取 Diamond 答案键。本报告是开发集事后诊断，重复折分和199题子集不构成独立验证。','',
        '## 建议的下一步','',
        '保持论文算法和原始 confidence 不变，先针对“多个 verifier 同时支持错误答案”检查信号和 likelihood：区分边际估计不稳、分箱过粗、以及联合信号与乘积模型的偏差。先明确是哪一环缺乏区分力，再决定改提示词、换 verifier 或增加稳健性对照；当前结果尚不支持直接换一个三人组合就能解决问题。','',
        f'完整诊断：{root}/report.json；逐题错误案例：{root}/wrong_release_cases.json；各状态 A/Q 与实际后续效用：{root}/state_value_diagnostics.csv。',
        '本目录 diagnostic_summary.csv 汇总所有原有折分、题集和惩罚设置。','']
    (output/'SUMMARY_ZH.md').write_text('\n'.join(text))
    atomic_json(output/'manifest.json',{'diagnostic_id':report['diagnostic_id'],'source_report_sha256':file_digest(root/'report.json'),
        'builder_sha256':file_digest(Path(__file__)),
        'files':{p.name:file_digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/gpqa_nested_diagnostics_20261008_v2')
    p.add_argument('--output',required=True)
    args=p.parse_args();render(args.input,args.output)
