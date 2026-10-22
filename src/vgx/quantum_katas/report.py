"""Offline pilot report: reads every artifact, including private official results.

Every metric states its effective n. AUROC is N/A when only one label class is
present; small-sample ECE is a development description, not a calibration claim.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics

from vgx.common.billing import summarize_vertex_usage
from vgx.common.storage import atomic_json, read_jsonl
from vgx.gpqa.score import binary_forecast_metrics
from vgx.quantum_katas import oracle
from vgx.quantum_katas.run_pilot import Run, SignalStore


def _describe(values):
    if not values:
        return {'n': 0}
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values),
            'histogram_0.1': [sum(1 for v in values if lo/10 <= v < (lo+1)/10 or (lo == 9 and v == 1)) for lo in range(10)]}


def _metrics(pairs):
    if not pairs:
        return {'n': 0, 'note': 'no labeled observations'}
    y, p = [a for a, _ in pairs], [b for _, b in pairs]
    m = binary_forecast_metrics(y, p).to_dict()
    m['positives'] = sum(y)
    if m['auroc'] is None:
        m['auroc'] = 'N/A (one label class)'
    m['ece_note'] = 'development description only (small n); not evidence of calibration'
    return m


def _latencies(cache) -> dict:
    by_role = {}
    for log in cache.logs().values():
        for row in log.records():
            if row.get('error') is None:
                by_role.setdefault((row.get('meta') or {}).get('role', 'unknown'), []).append(row['latency_s'])
    return {role: _describe(values) for role, values in sorted(by_role.items())}


def build_report(root) -> dict:
    run = Run(root)
    tasks = run.tasks
    planned = len(tasks)
    candidates = run.candidates.by_task()
    store = oracle.OfficialStore(run.root/'private'/'official_results.jsonl')
    states = store.states() if store.path.exists() else {}
    official = {s['terminal']['candidate_id']: s['terminal'] for s in states.values() if s['status'] == 'official'}
    unresolved = Counter(s['status'] if s['terminal'] is None else f"service_error:{s['terminal']['reason']}"
                         for s in states.values() if s['status'] != 'official')
    signals = SignalStore(run.root/'verifier_signals.jsonl').index()

    with_code = [c for c in candidates.values() if c.has_code]
    with_r = [c for c in with_code if c.has_valid_confidence]
    labeled = [c for c in with_code if c.candidate_id in official]
    labels = {cid: row['label'] for cid, row in official.items()}
    official_rows = [official[c.candidate_id] for c in labeled]
    generator = {
        'planned_tasks': planned, 'generated': len(candidates), 'not_generated': planned - len(candidates),
        'format_failures': dict(Counter(c.generator_failure for c in candidates.values() if not c.has_code)),
        'service_failures': [r['task_id'] for r in read_jsonl(run.root/'collection'/'abandoned_requests.jsonl')
                             if r['role'] == 'generator'],
        'truncated': sum(c.finish_status == 'truncated' for c in candidates.values()),
        'finish_status': dict(Counter(c.finish_status for c in candidates.values())),
        'code_extractable': len(with_code), 'code_extractable_rate_of_planned': len(with_code)/planned,
        'confidence_valid': len(with_r), 'confidence_valid_rate_of_planned': len(with_r)/planned,
        'confidence_status_with_code': dict(Counter(c.confidence_status for c in with_code)),
    }
    passed = sum(r['label'] for r in official_rows)
    times = [r['evaluate_elapsed_s'] for r in official_rows if r.get('evaluate_elapsed_s') is not None]
    walls = [r['supervisor_elapsed_s'] for r in official_rows if r.get('supervisor_elapsed_s') is not None]
    checker = {
        'candidates_with_code': len(with_code), 'official_returns': len(official_rows), 'passed': passed,
        'failed': len(official_rows) - passed, 'unresolved': dict(unresolved),
        'not_scored': len(with_code) - len(official_rows) - sum(unresolved.values()),
        'error_types': dict(Counter(r['official']['error_type'] or 'none' for r in official_rows)),
        'syntax_invalid': sum(not r['official']['syntax_valid'] for r in official_rows),
        'end_to_end_pass_rate': {'passed': passed, 'denominator_all_planned_tasks': planned, 'rate': passed/planned},
        'pass_rate_with_official_return': ({'passed': passed, 'n': len(official_rows), 'rate': passed/len(official_rows)}
                                           if official_rows else {'n': 0}),
        'evaluate_elapsed_s': _describe(times), 'supervisor_wall_elapsed_s': _describe(walls),
        'backend': run.plan['checker_limits']['backend'], 'limits': run.plan['checker_limits'],
        'note': 'Local checker compute time; not comparable to API dollars without a common cost model.',
    }
    forecasts = {'generator_r': _metrics([(labels[c.candidate_id], c.raw_p_correct) for c in with_r
                                          if c.candidate_id in labels]),
                 'generator_r_distribution': _describe([c.raw_p_correct for c in with_r])}
    verifier_counts = {}
    for spec in run.verifiers:
        rows = [signals.get((c.candidate_id, spec.id)) for c in with_code]
        valid = [(c, r) for c, r in zip(with_code, rows) if r and r['p_correct'] is not None]
        forecasts[spec.id] = _metrics([(labels[c.candidate_id], r['p_correct']) for c, r in valid
                                       if c.candidate_id in labels])
        forecasts[spec.id + '_distribution'] = _describe([r['p_correct'] for _, r in valid])
        verifier_counts[spec.id] = {'recorded': sum(r is not None for r in rows), 'valid': len(valid),
                                    'failures': dict(Counter(r['failure'] for r in rows if r and r['failure'])),
                                    'not_collected': sum(r is None for r in rows)}
    samples = {'planned': planned, 'with_code': len(with_code), 'with_valid_r': len(with_r),
               'with_official_label': len(labeled), 'with_label_and_r': sum(c.candidate_id in labels for c in with_r),
               'verifiers': verifier_counts}

    budget = run.config.get('budget') or {}
    cache = run.cache()
    usage = summarize_vertex_usage(cache.logs(), budget.get('pricing'))
    calls = {'by_role_model': usage['by_role_model_partition'], 'estimated_usd_for_priced_calls': usage['estimated_usd_for_priced_calls'],
             'successful_executions': usage['successful_calls'], 'rejected_attempt_records': usage['rejected_attempt_records'],
             'unresolved_attempts': len(usage['unresolved_attempts']), 'confirmed_billing_usd': None,
             'pricing_basis': budget.get('pricing_source', 'none'),
             'latency_s': _latencies(cache),
             'collection_status': {n: json.loads((run.root/'collection'/f'{n}.json').read_text())
                                   for n in ('generators', 'verifiers') if (run.root/'collection'/f'{n}.json').exists()}}
    for status in calls['collection_status'].values():
        status.pop('rows', None)
    sequential = None
    path = run.root/'sequential'/'decisions.json'
    if path.exists():
        value = json.loads(path.read_text())
        results = value['results']
        sequential = {'policy_kind': value['policy_kind'], 'policy_id': value['policy_id'],
                      'actions': dict(Counter(r['action'] for r in results)),
                      'failures': dict(Counter(r['failure'] for r in results if r.get('failure'))),
                      'logical_verifier_queries': value['logical_verifier_queries'],
                      'new_provider_executions': value['new_provider_executions'],
                      'replay_agreement': value.get('replay_agreement'),
                      'accepted_with_label': {'n': sum(r['action'] == 'assert' and r['candidate_id'] in labels for r in results),
                                              'passed': sum(r['action'] == 'assert' and labels.get(r['candidate_id']) == 1 for r in results)},
                      'note': value['note']}
    report = {'schema': 1, 'plan_id': run.plan_id, 'mode': run.plan['mode'], 'partition': 'development',
              'samples': samples, 'generator': generator, 'official_checker': checker, 'forecasts': forecasts,
              'calls_and_costs': calls, 'sequential': sequential,
              'caveats': [
                  'Development pilot: 28 fixed tasks, one generation each; not a representative performance sample.',
                  'Raw generator r is reported as-is; no prior or verifier likelihood is fitted or deployed here.',
                  'No 95%/99% acceptance guarantee or advantage over published full-benchmark results is claimed.',
                  'Same official checker is necessary but not sufficient for cross-paper comparability '
                  '(task subset and added confidence output differ).',
                  'Logical sequential queries over collected signals are not saved API spend.',
              ] + (['DRY RUN: mock provider responses and reviewed reference/control fixtures; '
                    'no model was called and no number here describes a model.'] if run.plan['mode'] == 'dry_run' else [])}
    atomic_json(run.root/'report.json', report)
    (run.root/'report.txt').write_text(render_text(report))
    return report


def render_text(r: dict) -> str:
    g, c, s = r['generator'], r['official_checker'], r['samples']
    lines = [f"QuantumKatas 官方 checker pilot 报告（{r['mode']}，plan {r['plan_id'][:12]}）", '',
             f"计划 {g['planned_tasks']} 题；已生成 {g['generated']}；可提取代码 {g['code_extractable']}；"
             f"置信度有效 {g['confidence_valid']}；截断 {g['truncated']}。",
             f"生成服务失败（结果未知、未重发）：{g['service_failures'] or '无'}；格式失败：{g['format_failures'] or '无'}；有代码但 r 无效：{ {k: v for k, v in g['confidence_status_with_code'].items() if k != 'valid'} or '无'}",
             f"官方返回 {c['official_returns']}：通过 {c['passed']}，失败 {c['failed']}；未解决 {c['unresolved'] or '无'}；未评分 {c['not_scored']}。",
             f"错误类型：{c['error_types']}",
             f"端到端通过比例 {c['end_to_end_pass_rate']['passed']}/{c['end_to_end_pass_rate']['denominator_all_planned_tasks']}；"
             f"有官方返回的通过比例 {c['pass_rate_with_official_return']}",
             f"样本数：有代码 {s['with_code']}，有 r {s['with_valid_r']}，有标签 {s['with_official_label']}，"
             f"有标签且有 r {s['with_label_and_r']}。", '', '预测指标（各自有效 n）：']
    for name, value in r['forecasts'].items():
        if name.endswith('_distribution'):
            continue
        lines.append(f"  {name}: n={value['n']} brier={value.get('brier')} log_loss={value.get('log_loss')} auroc={value.get('auroc')}")
    calls = r['calls_and_costs']
    lines += ['', f"调用：成功执行 {calls['successful_executions']}，估算费用 ${calls['estimated_usd_for_priced_calls']:.4f}"
                  f"（{calls['pricing_basis']}；非账单），已确认账单：未知。"]
    if r['sequential']:
        q = r['sequential']
        lines.append(f"顺序执行（{q['policy_kind']}）：动作 {q['actions']}；逻辑查询 {q['logical_verifier_queries']}；"
                     f"新增实际调用 {q['new_provider_executions']}；replay 一致 {q['replay_agreement']}")
    lines += ['', '说明：'] + [f'  - {x}' for x in r['caveats']]
    return '\n'.join(lines) + '\n'
