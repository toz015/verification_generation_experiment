"""Aggregate-only offline audit/export for the authorized GPT-OSS resume."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from vgx.common.api import RequestCache
from vgx.common.storage import digest, file_digest, read_jsonl
from vgx.gpqa import gptoss_resume as resume
from vgx.gpqa import gptoss_continuation as base
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.sequential import seal, unseal
from vgx.gpqa.sonnet_offline_score import summarize


def run(root=resume.ROOT, out=Path('reports/gpqa_gptoss_resume_20261017')):
    state=resume.audit(root)
    if state['stage']!='complete': raise ValueError('collection must finish before scoring audit/export')
    amendment, parent, items, runner, cache=resume.load(root)
    result=json.loads((root/'comparison.json').read_text()); unseal(result,'result_id')
    if result['continuation_id']!=parent['continuation_id']: raise ValueError('comparison identity mismatch')
    frozen=json.loads((root/'frozen_candidates.json').read_text()); unseal(frozen,'bundle_id')
    if result['bundle_id']!=frozen['bundle_id']: raise ValueError('candidate bundle mismatch')
    before={str(p):file_digest(p) for p in root.rglob('*') if p.is_file()}
    ledger=json.loads((root/'cache/budget.json').read_text())['requests']
    assert set(ledger)==set(parent['request_keys'])-set(parent['original_response_hashes'])
    totals=Counter(); hashes={}; versions=Counter(); rejected=Counter(); parsed_failures=Counter()
    for item,key in zip(items,parent['request_keys']):
        req=generator_request(item,parent['parent_study_id'])
        call,_=RequestCache.get(cache,runner,req)
        estimate=base.checked(call,runner,req,parent['config'])
        hashes[key]=digest(asdict(call))
        records,attempts,starts,successes=resume.history(cache,key)
        assert len(successes)==1
        rejected.update(str(r['meta']['http_status']) for r in records if r.get('error'))
        parsed=base.original.parse_call(call)
        if not parsed.ok: parsed_failures[parsed.failure]+=1
        totals['input_tokens']+=estimate['usage']['input_tokens']
        totals['output_tokens_including_reasoning']+=estimate['usage']['output_tokens']
        totals['estimated_usd']+=estimate['estimated_usd']
        versions[str((call.meta['provider_response'].get('metadata') or {}).get('weight_version'))]+=1
        if key in parent['original_response_hashes']:
            assert hashes[key]==parent['original_response_hashes'][key]
        else:
            entry=ledger[key]
            assert entry['status']=='settled' and not entry['issues']
            assert abs(entry['estimated_usd']-estimate['estimated_usd'])<1e-12
            assert entry['execution_id']==call.meta['execution_id']
            completed=[a for a in attempts if a['event']=='completed']
            reject_ids={a['execution_id'] for a in attempts if a['event']=='rejected'}
            assert len(completed)==1 and completed[0]['execution_id']==call.meta['execution_id']
            assert {a['execution_id'] for a in starts}==reject_ids|{call.meta['execution_id']}
            assert len(starts)<=3
    assert dict(parsed_failures)==state['invalid']
    for field in ('input_tokens','output_tokens_including_reasoning'): assert totals[field]==result[field]
    assert abs(totals['estimated_usd']-result['total_gptoss_estimated_usd'])<1e-10
    # Determine matched probability denominators from frozen predictions alone.
    groups=frozen['groups']
    expected_ids=[i.item_id for i in items]
    common=set.intersection(*[{c['item_id'] for c in rows if c['answer'] in ('A','B','C','D') and c['p_correct'] is not None}
                              for rows in groups.values()])
    for rows in groups.values():
        assert len(rows)==50 and [c['item_id'] for c in rows]==expected_ids
        assert all(c['partition']=='calibration' for c in rows)
    label_path=Path(parent['config']['baseline_bundle'])/'calibration_records.jsonl'
    assert file_digest(label_path)==result['development_label_sha256']
    label_rows=read_jsonl(label_path)
    assert all(r['partition']=='calibration' for r in label_rows)
    labels={r['item_id']:r['correct_index'] for r in label_rows}
    for name,rows in groups.items():
        pairs=[(c['p_correct'],int(c['answer']=='ABCD'[labels[c['item_id']]])) for c in rows
               if c['answer'] in ('A','B','C','D') and c['p_correct'] is not None]
        correct=sum(c['answer']=='ABCD'[labels[c['item_id']]] for c in rows)
        metric=result['generators'][name]
        assert correct==metric['correct'] and len(pairs)==metric['brier_n']
        for measured,expected in ((correct/len(rows),metric['accuracy_all']),
            (sum(p for p,y in pairs)/len(pairs),metric['mean_confidence']),
            (sum((p-y)**2 for p,y in pairs)/len(pairs),metric['raw_brier'])):
            assert abs(measured-expected)<1e-12
        wrong=[p for p,y in pairs if not y]
        assert len(wrong)==metric['wrong_with_confidence_n']
        if wrong: assert abs(sum(wrong)/len(wrong)-metric['mean_confidence_if_wrong'])<1e-12
    matched=seal({'source_result_id':result['result_id'],'n':len(common),
        'membership_digest':digest(sorted(common)), 'membership_basis':'intersection of usable frozen answer/confidence pairs, without labels',
        'confidence_calibrated':False,'evaluation_labels_read':False,
        'generators':{name:summarize([c for c in rows if c['item_id'] in common],labels) for name,rows in groups.items()}},'comparison_id')
    audit=seal({**state,'audit':'passed','result_id':result['result_id'],
        'independent_metric_arithmetic_matches':True, 'candidate_bundle_sha256':file_digest(root/'frozen_candidates.json'),
        'response_fingerprints_digest':digest(hashes),'duplicate_successes':0,
        'provider_weight_version_counts':dict(versions),'all_rejections':dict(rejected),
        'development_keys_read_for_scoring_only':True,'evaluation_keys_read':False,
        'new_api_requests_in_audit':0,'unit_tests_run':False,
        'export_implementation_sha256':file_digest(__file__)},'audit_id')
    names={'gemini_baseline':'Gemini','qwen_baseline':'Qwen（两步）','sonnet_5_5':'Sonnet 5.5','gpt_oss_120b':'gpt-oss-120b'}
    lines=['GPT-OSS 50题开发集：30秒续跑与四模型离线比较','2026-10-17','',
        '本次复用18条已有回复，只新增剩余32条。原有格式失败保持原样。confidence均未校准。',
        '原始50题为主分析；无效答案在全题正确率中计为未答对；Brier及confidence均值只使用有效答案/概率配对。',
        '', '模型 | 正确/50 | 完整答案及confidence | 平均confidence | 答错confidence（题数） | raw Brier（题数）']
    for name in names:
        m=result['generators'][name]; wrong=m['mean_confidence_if_wrong']
        lines.append(f"{names[name]} | {m['correct']}/50 ({m['accuracy_all']:.2%}) | {m['confidence_n']}/50 | {m['mean_confidence']:.2%} | {('未定义' if wrong is None else f'{wrong:.2%}')} ({m['wrong_with_confidence_n']}) | {m['raw_brier']:.6f} ({m['brier_n']})")
    lines.extend(['',f"另附common_valid_comparison.json：四模型均有有效答案与confidence的同一{len(common)}题；成员在读取标签前确定。",
        '', '协议：openai/gpt-oss-120b-maas，Vertex us-central1，medium reasoning，temperature=0，top_p=1，max_tokens=32768。',
        '使用已冻结原prompt和严格final-content解析器；reasoning_content不用于提取或修复答案。',
        '续跑相邻请求在前一次完成后至少等待30秒；明确429后60/120秒退避，单题总计最多3次尝试。',
        '历史429计入总次数；已等待超过退避时间的旧429无需重复等待。所有错误记录保留，未知请求结果不重发。',
        f"续跑新增token费用估算：USD {state['resume_increment_estimated_usd']:.8f}。",
        f"从原5题以后全部45题费用估算：USD {result['budget']['estimated_usd']:.8f}，沿用同一USD1预算。",
        f"原5题费用估算：USD {result['original_gptoss_estimated_usd']:.8f}。",
        f"GPT-OSS全部50条费用估算：USD {result['total_gptoss_estimated_usd']:.8f}。",
        f"输入token：{result['input_tokens']}；输出token（含reasoning）：{result['output_tokens_including_reasoning']}。",
        '计费项目llm-applications-490420。美元数来自provider token用量和冻结价格，不是确认账单；未扩大预算。',
        f"finish_reason统计：{result['finish_reasons']}；包含reasoning_content的回复：{result['reasoning_content_present_n']}/50。",
        f"解析失败：{state['invalid']}；累计明确HTTP拒绝：{dict(rejected)}。",
        '', '审计：原18条缓存逐字节不变，旧报告/源文件/请求身份不变；唯一成功回复、退避、预算和token核对通过。',
        '先冻结全部候选，再读取开发标签；独立重算正确数、均值、答错均值和Brier与主报告一致。',
        '无Diamond评分，无confidence校准，无verifier调用。此次未新增或运行单元测试。',
        '50题是反复用于探索的开发题；不同模型的生成协议和服务日期不同，不能将差异全部归因于模型。',
        '本目录仅包含汇总结果、计划、状态、审计、协议和哈希；无题干、答案键、逐题对错、原始回复或凭证。',
        '本轮未提交或推送。'])
    out.mkdir(parents=True,exist_ok=False)
    for src,dest in [(root/'comparison.json','result.json'),(root/'plan.json','original_plan.json'),
        (root/resume.AMENDMENT/'plan.json','plan.json'),(root/resume.AMENDMENT/'status.json','status.json'),
        (root/resume.AMENDMENT/'billing_check.json','billing_check.json'),
        (root/resume.AMENDMENT/'previous_status.json','previous_status.json'),
        (root/resume.AMENDMENT/'collection_stdout.json','collection_stdout.json'),
        (root/resume.AMENDMENT/'dry_run_stdout.json','dry_run_stdout.json')]: shutil.copyfile(src,out/dest)
    shutil.copyfile('docs/gpqa-gptoss-resume-20261017.txt',out/'protocol.txt')
    base.immutable(out/'audit.json',audit)
    base.immutable(out/'common_valid_comparison.json',matched)
    (out/'README.txt').write_text('\n'.join(lines)+'\n')
    banned={'question','choices','answer','correct_index','correct_answer','response','provider_response','prompt','system','access_token','refresh_token','private_key','client_secret'}
    def safe(value):
        if isinstance(value,dict):
            assert not banned.intersection(value), 'unsafe export field'
            for v in value.values(): safe(v)
        elif isinstance(value,list):
            for v in value: safe(v)
    for p in out.glob('*.json'): safe(json.loads(p.read_text()))
    for p,h in before.items(): assert file_digest(p)==h,'run artifact changed during export'
    (out/'sha256.txt').write_text(''.join(f'{file_digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()))
    print(json.dumps({'report':str(out),'audit_id':audit['audit_id'],'generators':result['generators'],
        'resume_increment_estimated_usd':state['resume_increment_estimated_usd'],
        'total_gptoss_estimated_usd':result['total_gptoss_estimated_usd']},indent=2))


if __name__=='__main__': run()
