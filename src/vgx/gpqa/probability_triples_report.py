"""Readable summaries of exhaustive fixed-order probability-verifier experiments."""
import argparse
import csv
import json
from pathlib import Path

from vgx.common.storage import atomic_json, file_digest
from vgx.gpqa.nested_combinations_report import NAMES


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def short(order):
    return ' → '.join(NAMES[t.split(':')[0]] for t in order)


def render(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root/'protocol.json').read_text())
    report = json.loads((root/'report.json').read_text())
    status = json.loads((root/'status.json').read_text())
    if (status['status'] != 'complete' or status['analysis_id'] != protocol['analysis_id']
            or not report['source_integrity_unchanged'] or report['analysis_id'] != protocol['analysis_id']):
        raise ValueError('completed matching analysis required')
    metrics, ranks, selections, likelihoods = [], [], [], []
    for key, run in report['results'].items():
        g, cohort, seed = key.split('/')
        detailed = json.loads((root/key/'results.json').read_text())
        for loss, policies in run['policies'].items():
            context = {'generator': g, 'cohort': cohort, 'seed': seed, 'loss': float(loss)}
            for policy, s in policies.items():
                metrics.append({**context, 'policy': policy,
                    **{k: s[k] for k in ('n','released','correct_released','wrong_released','coverage','conditional_error',
                        'queries','expected_cost_usd','usage_estimated_cost_usd','mean_utility','mean_utility_at_observed_usage',
                        'utility_difference_vs_raw','utility_difference_vs_flash','utility_difference_vs_matched_raw',
                        'matched_raw_expected_wrong','signal_failures_queried','mean_reported_posterior_on_released')},
                    **{f'queries_{i}': s['query_count_histogram'].get(str(i),0) for i in range(4)},
                    **{k: v for k, v in s['transitions_vs_raw'].items()},
                    **{f'vs_flash_{k}': v for k, v in s['transitions_vs_flash'].items()}})
            for selection in detailed['selections'][loss]:
                fold = selection['fold']
                for r in selection['ranked_orders']:
                    ranks.append({**context, 'fold': fold, 'order': short(r['order']), 'order_id': r['order_id'],
                                  'combination': r['combination'], 'length': r['length'],
                                  'training_model_value': r['training_model_value'],
                                  'sum_expected_cost_usd': sum(r['expected_cost_usd']),
                                  'table_sha256': r['table_sha256']})
                selections.append({**context, 'fold': fold, 'selected_order': short(selection['selected_triple']['order']),
                    'selected_single': short(selection['selected_single']['order']),
                    'cheapest_three': short(selection['cheapest_order'][:3]), 'selection_id': selection['selection_id']})
                for tag, arm in selection['fitted_arms'].items():
                    model = next(m for m in protocol['frozen_sources']['models'] if m['id'] == tag.split(':')[0])
                    price = protocol['frozen_sources']['pricing']['usd_per_million_tokens'][model['model']]
                    table = arm['likelihood']
                    for i, (p1, p0) in enumerate(zip(table['p_bin_if_correct'],table['p_bin_if_incorrect'])):
                        likelihoods.append({**context, 'fold': fold, 'verifier': tag, 'model': model['model'],
                            'bin': i, 'lower': table['edges'][i], 'upper': table['edges'][i+1],
                            'p_given_correct': p1, 'p_given_wrong': p0, 'likelihood_ratio': p1/p0,
                            'expected_call_usd': arm['expected_cost_usd'], 'input_usd_per_million': price['input'],
                            'output_usd_per_million': price['output'], 'cached_usd_per_million': price.get('cached_input'),
                            'correct_fit_n': arm['fit_correct'], 'wrong_fit_n': arm['fit_wrong']})
    write_csv(output/'all_policy_results.csv', metrics)
    write_csv(output/'training_order_rankings.csv', ranks)
    write_csv(output/'selected_orders.csv', selections)
    write_csv(output/'likelihood_and_cost.csv', likelihoods)
    # All six fixed orders are retained. Ranges are descriptive, never a selection rule.
    combo_rows = []
    for g in protocol['config']['generators']:
        for cohort in protocol['source_config']['cohorts']:
            for loss in protocol['config']['losses']:
                combos = sorted({'+'.join(sorted(r['policy'][7:].split('>'))) for r in metrics
                                 if r['policy'].startswith('triple:')})
                for combo in combos:
                    sub = [r for r in metrics if r['generator']==g and r['cohort']==cohort and r['loss']==loss
                           and r['policy'].startswith('triple:') and '+'.join(sorted(r['policy'][7:].split('>')))==combo]
                    primary = [r for r in sub if r['seed'] == 'seed_'+str(protocol['source_config']['seeds'][0])]
                    combo_rows.append({'generator':g,'cohort':cohort,'loss':loss,'combination':combo,
                        'fixed_orders':len(primary),'order_seed_rows':len(sub),
                        **{f'{k}_{side}': fn(r[k] for r in primary if r[k] is not None)
                           for k in ('released','wrong_released','queries','mean_utility','usage_estimated_cost_usd')
                           for side,fn in [('min',min),('max',max)]},
                        'utility_all_seeds_min':min(r['mean_utility'] for r in sub),
                        'utility_all_seeds_max':max(r['mean_utility'] for r in sub)})
    write_csv(output/'ten_combinations_six_orders.csv', combo_rows)
    lines = ['三 verifier 概率信号：60 种固定顺序实验', '',
        '主结果为 249 道开发题、主折分；保留其他四个折分及排除 pilot 的 199 题结果。',
        '全部使用概率回复；五种模型选三个，10 个组合、每组 6 种顺序。',
        '固定顺序策略：每个次序在所有留出题上保持一致；训练选择策略：各训练折可能选出不同顺序。',
        '所有方案保留原始答案与 confidence。只有输出/拒答和查询次数变化。',
        'Qwen 有249条有效候选；Gemini 有247条，另外2条保持拒答，指标分母仍为249题。',
        '5% 不是成功/失败门槛；报告错误率、接受率、效用与查询费用。',
        'R=1；主 L=19、敏感性 L=99；美元成本乘 10 转成效用成本，沿用历史设定。',
        '因此无后续验证时的输出阈值仍为0.95/0.99；这是效用参数的结果，不是实际错误率保证。',
        '美元到效用的倍率是实验偏好参数；不是由token价目表自动推出的事实。',
        '成本包含已返回的无效信号；预期成本来自训练折，实际用量估算只用于事后报告。',
        '预计信息价值/训练 J0 用于选序；实际留出效果不参与选择。便宜优先排序单列作基线。', '']
    primary_seed = protocol['source_config']['seeds'][0]
    loss_key = str(protocol['config']['primary_loss'])
    names = [('raw','原始 confidence'),('single:gemini_flash','单 Flash'),('selected_single','训练选择单 verifier'),
             ('selected_triple','训练选择三 verifier'),('cheapest_three','最便宜三位、费用升序'),
             ('always_selected_triple','训练所选三位全部询问'),('abstain_all','全部拒答')]
    for g in protocol['config']['generators']:
        key = f'{g}/all_calibration/seed_{primary_seed}'
        policies = report['results'][key]['policies'][loss_key]
        lines += [g.upper(), '策略 | 输出数 | 错误数 | 接受率 | 输出错误率 | 查询数 | 平均效用 | 用量估算USD']
        for policy, name in names:
            s = policies[policy]; risk = '-' if s['conditional_error'] is None else f"{s['conditional_error']:.1%}"
            lines.append(f"{name} | {s['released']} | {s['wrong_released']} | {s['coverage']:.1%} | {risk} | "
                         f"{s['queries']} | {s['mean_utility']:+.4f} | {s['usage_estimated_cost_usd']:.4f}")
        s = policies['selected_triple']
        t = s['transitions_vs_flash']
        lines.append('相对单 Flash 的逐题变化：新增输出正确/错误 '
                     f"{t.get('added_correct',0)}/{t.get('added_wrong',0)}；取消输出正确/错误 "
                     f"{t.get('removed_correct',0)}/{t.get('removed_wrong',0)}。总数相同不代表逐题决策相同。")
        for base in ('raw','flash','matched_raw'):
            d = s['paired_utility_vs_'+base]
            lines.append(f"三层相对 {base} 的每题效用差：{d['estimate']:+.4f}，描述性95%区间 "
                         f"[{d['ci95']['low']:+.4f}, {d['ci95']['high']:+.4f}]")
        lines.append('停止位置（查询次数:题数）：'+json.dumps(s['query_count_histogram'],ensure_ascii=False))
        for r in selections:
            if r['generator']==g and r['cohort']=='all_calibration' and r['seed']==f'seed_{primary_seed}' and r['loss']==float(loss_key):
                lines.append(f"训练折 {r['fold']+1} 选择：{r['selected_order']}")
        for cohort in protocol['source_config']['cohorts']:
            values=[run['policies'][loss_key]['selected_triple']['utility_difference_vs_flash']
                    for k,run in report['results'].items() if k.startswith(g+'/'+cohort+'/')]
            lines.append(f"{cohort}：五折分中三层相对单 Flash 的平均效用差范围 "
                         f"[{min(values):+.4f}, {max(values):+.4f}]")
        lines.append('')
    lines += ['主结果的解释',
        'Qwen：三层与单 Flash 的输出数、错误数相同，逐题决策有变化；额外查询增加了成本。',
        'Gemini：三层比单 Flash 净减少1个错误输出，同时净减少7个正确输出，接受率下降。',
        '两个 generator 的三层增量效用区间都跨过0，其他折分也有方向变化；目前没有稳定优于单 Flash 的证据。',
        '主设定下各主要输出策略的平均效用仍为负，优于原始confidence基线不等于达到正效用。',
        '这不能证明任意三 verifier 组合都无用；下表保留全部组合和顺序供诊断，但不能事后挑最优格子当作确认性结果。', '']
    lines += ['十组组合：主折分中六种固定顺序的范围（结果不用于宣布最优组合）',
              'generator | 组合 | 输出数 | 错误数 | 查询数 | 平均效用']
    for r in combo_rows:
        if r['cohort']=='all_calibration' and r['loss']==float(loss_key):
            name=' + '.join(NAMES[t] for t in r['combination'].split('+'))
            spans=[f"{r[k+'_min']}–{r[k+'_max']}" for k in ('released','wrong_released','queries')]
            lines.append(f"{r['generator']} | {name} | "+' | '.join(spans)+
                         f" | {r['mean_utility_min']:+.4f}–{r['mean_utility_max']:+.4f}")
    checks=sum(r['replay_agreements'] for r in report['results'].values())
    refits=sum(r['likelihood_refit_checks'] for r in report['results'].values())
    lines += ['', '论文算法的实现边界',
        'Algorithm 1：固定每层一位 verifier，构造三层 J/Q 表；A(b)=(R+L)b-L；平手停止。',
        'Algorithm 2：h 取恒等函数；按固定顺序仅查询被选中的下一层；Bayes 更新后重新比较继续/停止价值。',
        '每个概率信号使用训练折拟合的三个等宽箱；箱内LR=P(信号箱|正确)/P(信号箱|错误)，Laplace平滑为1。',
        '排序同时使用全部信号箱的likelihood和预期请求费，比较训练先验上的平均J0；不按某一个“赞成”箱的LR直接排序。',
        '只在停止时选择输出原候选或拒答；无效信号会付费后拒答。',
        '第18步反馈记录为可传送的 queried-score 事件；本轮没有实际再次调用 generator，未测量其私有信念变化。',
        '终局独立审计、激励相容和奖励训练不由这次离线控制器实验验证。', '', '结果解释边界',
        '沿用给定正确性条件下的 verifier 独立乘积模型；信号相关性仍可能造成过度自信和错误价值估计。',
        '价值表仅建模有效信号箱，未显式建模格式失败概率；实际执行遇到失败会拒答。',
        '原始 confidence 未校准；效用为负时，全部拒答的零效用基线必须保留。',
        '开发集已经参与前期探索，199题是249题的子集；重复折分不增加独立样本量。',
        '配对区间条件于固定折外结果，不包含重拟合与筛选不确定性；不证明某个组合获胜。',
        '相同接受数的 confidence 基线是事后描述性比较，同分按随机选择的期望计算。',
        'Qwen 两阶段生成与 Gemini 的设置不同；不能把结果差异完全归因于模型家族。',
        '没有读取 Diamond 答案键；历史 Diamond 已被使用，不宣称是全新的测试集。', '', '执行记录',
        f'完成 {len(report["results"])} 个 generator×cohort×seed 运行；每个运行含两种惩罚。',
        f'训练 likelihood/成本复核 {refits} 个；懒加载执行/回放核对 {checks} 次。',
        '这些是软件核对次数，不是独立题目或新 API 请求数。',
        '所有旧候选、缓存和源拟合哈希保持不变；本次新增 API 请求 0、费用 $0。',
        '策略费用是基于冻结 provider token 用量和历史价目表的反事实估算，不是确认账单。',
        '计费项目（历史缓存）：'+protocol['frozen_sources']['project'], '', '文件',
        'all_policy_results.csv：所有固定顺序、选择策略、基线在两种惩罚/两个题集/五折分上的指标。',
        'ten_combinations_six_orders.csv：十组各六个顺序的结果范围，含跨折分效用范围。',
        'training_order_rankings.csv：训练 J0、预期费用、全部固定顺序及价值表哈希。',
        'likelihood_and_cost.csv：P(v|正确)、P(v|错误)、LR、输入/输出单价和训练折预期请求费。',
        'selected_orders.csv：训练选序和便宜优先基线。',
        '原结果目录包含每题已查询前缀、A/Q、停止位置、反馈事件以及封存的三层 planner。']
    (output/'SUMMARY_ZH.txt').write_text('\n'.join(lines)+'\n')
    atomic_json(output/'manifest.json', {'analysis_id':report['analysis_id'],'source_report_sha256':file_digest(root/'report.json'),
        'report_implementation_sha256':file_digest(Path(__file__)),
        'files':{p.name:file_digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'}})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();render(a.root,a.output)
