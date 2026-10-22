"""Compact Chinese report and shareable aggregate tables; no question text."""
import argparse
import csv
import json
from pathlib import Path

from vgx.common.storage import file_digest, atomic_json


NAMES = {'gemini_flash':'Flash', 'gemini_lite':'Flash-Lite', 'mistral_small':'Mistral Small',
         'mistral_medium':'Mistral Medium', 'claude_haiku':'Haiku'}


def order_text(order):
    return ' → '.join(NAMES[t.split(':')[0]]+'（'+('二元' if t.endswith(':binary') else '概率')+'）' for t in order)


def interval(value):
    return f"{value['estimate']:+.3f} [{value['ci95']['low']:+.3f}, {value['ci95']['high']:+.3f}]"


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'report.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    source=Path(protocol['config']['source_results'])
    rows=[]; selected_rows=[]; likelihood_rows=[]
    for key, result in report['results'].items():
        g, cohort, seed=key.split('/')
        for loss, policies in result['policies'].items():
            for name,s in policies.items():
                rows.append({'generator':g,'cohort':cohort,'seed':seed,'loss':loss,'policy':name,
                    **{k:s[k] for k in ('n','released','wrong_released','coverage','released_accuracy','queries',
                        'mean_utility','expected_cost_usd','usage_estimated_cost_usd','signal_failures_queried')},
                    'utility_difference_vs_selected_single':s['paired_utility_vs_selected_single']['estimate'],
                    'difference_ci_low':s['paired_utility_vs_selected_single']['ci95']['low'],
                    'difference_ci_high':s['paired_utility_vs_selected_single']['ci95']['high'],
                    **{'queried_'+str(n):s['query_count_histogram'].get(str(n),0) for n in range(4)}})
        detailed=json.loads((root/key/'results.json').read_text())
        for loss, selections in detailed['selections'].items():
            for fold in selections:
                for rank,entry in enumerate(fold['ranked_candidates'],1):
                    selected_rows.append({'generator':g,'cohort':cohort,'seed':seed,'loss':loss,'fold':fold['fold'],
                        'training_rank':rank,'combination':entry['combination'],'order':' > '.join(entry['order']),
                        'training_model_value':entry['training_model_value']})
        fitted=json.loads((source/(key+'.json')).read_text())
        for fold_id,fold in enumerate(fitted['folds']):
            for tag,arm in fold['arms'].items():
                likelihood=arm['likelihood']
                for bin_id,(p1,p0) in enumerate(zip(likelihood['p_bin_if_correct'],likelihood['p_bin_if_incorrect'])):
                    likelihood_rows.append({'generator':g,'cohort':cohort,'seed':seed,'fold':fold_id,'verifier_signal':tag,
                        'bin':bin_id,'lower':likelihood['edges'][bin_id],'upper':likelihood['edges'][bin_id+1],
                        'p_signal_given_correct':p1,'p_signal_given_wrong':p0,'likelihood_ratio':p1/p0,
                        'expected_usd_per_call':arm['expected_cost_usd'],
                        'cost_utility_at_alpha10':10*arm['expected_cost_usd'],
                        'fit_correct':arm['fit_correct'],'fit_wrong':arm['fit_wrong']})
    write_csv(output/'policy_results.csv',rows)
    write_csv(output/'training_rankings.csv',selected_rows)
    write_csv(output/'fold_likelihood_cost.csv',likelihood_rows)
    text=['# 固定三层 verifier：组合、成本与停止实验', '',
        '**结论：已经把论文的三层价值表和顺序停止逻辑应用于 Gemini、Qwen 的冻结答案。三层比单 verifier 的额外收益尚不稳定，不应据此宣布最佳组合。**', '',
        '## 1. 如何选择“便宜又好用”', '',
        '- 五种模型中选三个：10 种组合 × 6 种顺序 × 8 种二元／概率格式搭配 = 480 个候选方案。',
        '- 每个训练折用原有 likelihood、该折平均 token 费用和原始 generator confidence，构建每个方案的 J/Q 表，以训练题原始 prior 上的平均 J₀ 最大者选择方案；完全相同时按标签字典序。',
        '- 每个组合内部也独立选择格式和顺序，作为十组比较。留出题的正确答案、信号、费用、prior 都不参与该折方案选择。',
        '- 固定后按顺序执行；根据当前 belief 决定停止或询问下一位，不在执行中更换 verifier。各折可能选择不同方案，所以主结果评价的是这个选择方法，不是一个全局固定三人组合。',
        '- 单 verifier 基线也从十个模型／格式选项中用同样的训练 J₀ 标准选择。不是依据留出集表现挑选。', '',
        '**这不是按 likelihood ratio 降序或按 token 单价升序排队。** 单个 ratio 不代表整个信号分布的信息价值；每次成本还依赖输入、输出和推理 token 用量。J₀ 同时考虑信息价值、提前停止和成本，但依赖概率模型正确，因此预测值不等于实测效用。', '',
        '## 2. 停止与成本规则', '',
        '主设置 R=1，L=19；A(b)=20b−19，直接输出门槛为 0.95。J₃=max(0,A)；Qₖ=−dₖ₊₁+E[Jₖ₊₁(更新后的 b)]；Jₖ=max(0,A,Qₖ)。', '',
        '若 Qₖ > max(0,A)，只询问下一层；否则停止，A≥0 则输出原答案，A<0 则 abstain。平手停止。查询后按该 verifier 的似然进行 Bayes 更新；格式无效则 abstain。候选答案从不修改。', '',
        '成本换算 d=10×训练折估算美元费用。这个系数是沿用的实验效用设定，需要以后做敏感性分析；不是自动由 API 价格确定。费用包含已返回的解析失败。Q 表尚未显式建模解析失败概率。', '',
        '## 3. 主结果：249 道开发题，预先固定的第一种三折划分', '',
        'Gemini 有 247 个有效 generator 回答；Qwen 有 249 个。覆盖率和平均效用均以 249 题为分母，包含无效回答的 abstain；已输出答案正确率则以输出数为分母。', '',
        '| Generator | 策略 | 输出数 | 输出中的错误 | verifier 查询数 | 平均效用 | 策略估算费用（美元） |',
        '|---|---|---:|---:|---:|---:|---:|']
    names={'raw':'仅原始 confidence','selected_single':'训练选择单 verifier','selected_three':'训练选择三层顺序验证',
           'always_selected_three':'同组三位全部询问'}
    for g in ('gemini','qwen'):
        main=report['results'][g+'/all_calibration/seed_20261005']['policies']['19.0']
        for name,label in names.items():
            s=main[name]
            text.append(f"| {g} | {label} | {s['released']} | {s['wrong_released']} | {s['queries']} | {s['mean_utility']:.3f} | {s['expected_cost_usd']:.4f} |")
    text += ['', '费用是按训练折平均请求成本计算的反事实策略费用，**不是本次新支出，也不是确认账单**；CSV 同时列出按实际缓存请求 token 用量估算的费用。所有方案的 generator 成本相同，未计入 verifier 增量比较。', '',
        '**所有主表策略的平均效用仍为负；全部 abstain 的效用为 0。改善相对于较差基线，不代表已达到值得部署的收益。**', '',
        '### 三层相比单 verifier 的增量', '']
    for g in ('gemini','qwen'):
        s=report['results'][g+'/all_calibration/seed_20261005']['policies']['19.0']['selected_three']
        text.append(f"- {g}：每题效用差 {interval(s['paired_utility_vs_selected_single'])}；95% 区间包含零。")
    text += ['', '区间为固定折外预测的配对 bootstrap；没有包含重新拟合、方案选择的全部不确定性，也未做多重比较校正。', '',
        '### 实际停止位置', '', '| Generator | 0 次查询 | 1 次 | 2 次 | 3 次 | 比全部询问少查询 |',
        '|---|---:|---:|---:|---:|---:|']
    for g in ('gemini','qwen'):
        p=report['results'][g+'/all_calibration/seed_20261005']['policies']['19.0']
        s=p['selected_three']; h=s['query_count_histogram']; savings=1-s['queries']/p['always_selected_three']['queries']
        text.append(f"| {g} | "+' | '.join(str(h.get(str(i),0)) for i in range(4))+f" | {savings:.1%} |")
    text += ['', 'Gemini 的 0 次查询包含两个无效 generator 回答。少查询不保证实测效用更高：Qwen 全部询问的主划分效用为 −0.901，顺序停止为 −0.951。这是需要关注的模型预测与实际表现差距。', '',
        '## 4. 训练折选择的组合与顺序', '', '| Generator | 折 | 固定顺序 |', '|---|---:|---|']
    for g in ('gemini','qwen'):
        r=json.loads((root/g/'all_calibration/seed_20261005/results.json').read_text())
        for f in r['selections']['19.0']:
            text.append(f"| {g} | {f['fold']+1} | {order_text(f['ranked_candidates'][0]['order'])} |")
    text += ['', '以下十组采用各自训练折选出的格式和顺序。是开发集探索性比较，不将表中最高者直接宣布为获胜者。', '',
        '| 三模型组合 | Gemini 平均效用 | Qwen 平均效用 |', '|---|---:|---:|']
    gem=report['results']['gemini/all_calibration/seed_20261005']['policies']['19.0']
    qwen=report['results']['qwen/all_calibration/seed_20261005']['policies']['19.0']
    for name in sorted(gem):
        if name.startswith('combo:'):
            label=' + '.join(NAMES[x] for x in name[6:].split('+'))
            text.append(f"| {label} | {gem[name]['mean_utility']:.3f} | {qwen[name]['mean_utility']:.3f} |")
    text += ['', '## 5. 稳定性与边界', '', '| Generator / 题集 | 五种折分的三层效用范围 | 相对单 verifier 的效用差范围 |',
        '|---|---:|---:|']
    for g in ('gemini','qwen'):
        for cohort in ('all_calibration','remaining_calibration'):
            vals=[v['policies']['19.0']['selected_three'] for k,v in report['results'].items() if k.startswith(g+'/'+cohort+'/')]
            u=[v['mean_utility'] for v in vals]; d=[v['paired_utility_vs_selected_single']['estimate'] for v in vals]
            text.append(f"| {g} / {'249' if cohort=='all_calibration' else '199'} | [{min(u):.3f}, {max(u):.3f}] | [{min(d):+.3f}, {max(d):+.3f}] |")
    text += ['', '- 199 题是去掉最初 50 题 pilot 的敏感性分析，与 249 题重叠，不是另一份独立测试集。',
        '- L=99 的主划分三层平均效用：Gemini −1.325，Qwen −2.062。提高惩罚并未解决概率与实际风险不匹配的问题。',
        '- 原始 confidence 完全保留；但 verifier likelihood 仍由训练折已知对错估计。这与校准 generator confidence 是不同步骤。',
        '- 沿用论文条件独立模型。实际模型相关性、未经校准的先验、少量错误样本和探索后选择都可能影响可靠性；这里不声称已确定哪一种是主要原因。',
        '- 此次实现的是固定单 verifier 层的价值表、查询、Bayes 更新和停止。LLM 内部信念反馈、终局审计、奖励训练未在本次验证。没有安装新的在线部署策略。',
        '- Qwen 使用两阶段生成，Gemini 设置不同；两个 generator 的结果差异不能全部归因于模型。', '',
        '## 6. 验证和文件', '',
        f"- 完成 {sum(v['replay_agreements'] for v in report['results'].values()):,} 次执行／回放一致性核对，并核对已查询前缀的逐步 Bayes 与乘积似然更新；这不是同等数量的独立题目或 API 实验。",
        '- 执行器仅接收当前 prior、固定 planner 与惰性查询接口；正确答案在外部评分。每次读取核对查询顺序，不读取未选中的未来信号。',
        '- 原候选、已缓存响应及原三折拟合文件 SHA-256 均未改变；未运行新 API，新增 API 支出 $0。',
        '- 本次未读取 Diamond 答案键。历史上 Diamond 曾用于其他实验，不能称其为从未使用的独立测试集。',
        '- `policy_results.csv`：所有 generator、题集、折分、惩罚和策略结果。',
        '- `training_rankings.csv`：所有 480 个方案在各训练折的模型预测排名。',
        '- `fold_likelihood_cost.csv`：每折、每个信号区间的 P(v|正确)、P(v|错误)、likelihood ratio 和成本。',
        '- 原结果目录包含完整逐题 trace，以及被选中三层方案的压缩 J/Q 表和重建参数。', '',
        '## 下一步建议', '',
        '先检查三层何时提前输出了错误答案、何时多花成本却未改变决策，并比较预测信息价值与折外实际收益。保持原始 confidence 不变，不用增加 verifier 数量来替代这项诊断。待解释清楚这些差距后，再固定一个 generator 对应的具体组合与顺序，讨论后续评价。', '']
    (output/'SUMMARY_ZH.md').write_text('\n'.join(text))
    atomic_json(output/'manifest.json',{'source_analysis_id':report['analysis_id'],'source_report_sha256':file_digest(root/'report.json'),
        'report_builder_sha256':file_digest(Path(__file__)), 'files':{p.name:file_digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); render(args.input,args.output)
