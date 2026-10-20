"""Export the blocked 291-response checkpoint; no API calls or label reads.

Run from the original checkout with its private results/ artifacts present.
This is an immutable checkpoint export, not a collector or correctness scorer.
"""
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import shutil

from vgx.common.api import RequestCache
from vgx.common.storage import digest, file_digest, read_jsonl
from vgx.gpqa import gptoss_retry179_ledgerfix as run
from vgx.gpqa.fresh import generator_request
from vgx.gpqa.sequential import unseal

OUTPUT = Path('reports/gpqa_gptoss_handoff_20261019')


def main():
    with (run.ROOT/'collection.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, parent, items, runner, budget = run.load(run.ROOT)
        state = budget._load(); run.check_ledger(plan, budget, state)
        source = run.ROOT/run.DIRECTORY
        status = json.loads((source/'status.json').read_text())
        assert (status['stage'], status['responses'], status['current_position'], status['error_type']) == (
            'blocked', 291, 292, 'UncertainRequestError')
        analysis_path = run.ROOT/'offline_score_445_retry179_ledgerfix_20261018/plan.json'
        analysis = json.loads(analysis_path.read_text()); unseal(analysis, 'analysis_id')
        for path, sha in analysis['source_sha256'].items():
            assert file_digest(path) == sha, path
        guarded = {str(p): file_digest(p) for p in (source/'status.json', budget.path, analysis_path)}
        rows = []; costs = []; positions = {}; timings = []
        for position, item in enumerate(items, 1):
            req = generator_request(item, parent['parent_study_id'])
            key = runner.cache_key(req); positions[key] = position
            cache = run.cache_for(run.ROOT, position)
            records = list(cache.log(key).records())
            events = read_jsonl(str(cache.log(key).path)+'.attempts.jsonl')
            if position <= 291:
                successes = [r for r in records if r.get('error') is None and r.get('key') == key]
                assert len(successes) == 1, position
                call, _ = cache.get(runner, req)  # Offline default; never allow_api.
                estimate = run.base.checked(call, runner, req, parent['config'])
                costs.append(estimate)
                if position <= 178:
                    assert digest(asdict(call)) == plan['existing_response_hashes'][key]
                    assert digest(run.candidate(item, req, call)) == plan['existing_candidate_hashes'][key]
                else:
                    entry = state['requests'][run.entry_key(plan, position)]
                    assert entry['status'] == 'settled'
                    assert entry['execution_id'] == call.meta['execution_id']
                    assert math.isclose(entry['estimated_usd'], estimate['estimated_usd'], abs_tol=1e-12)
                    assert [event['event'] for event in events] == ['started', 'completed']
                    timings.append((successes[0]['created'], successes[0]['created']+successes[0]['latency_s']))
                parsed = run.base.original.parse_call(call)
                rows.append({'partition': item.partition, 'valid': parsed.ok,
                             'failure': parsed.failure, 'confidence': parsed.p_correct})
            elif position == 292:
                assert not records and [e['event'] for e in events] == ['started', 'uncertain']
                assert state['requests'][key]['status'] == 'unresolved'
                blocked_seconds = events[-1]['created']-events[0]['created']
            else:
                assert not records and not events and key not in state['requests'], position
        for part, expected in status['partitions'].items():
            selected = [r for r in rows if r['partition'] == part]
            probs = [r['confidence'] for r in selected if r['confidence'] is not None]
            assert len(selected) == expected['responses']
            assert sum(r['valid'] for r in selected) == expected['valid']
            assert dict(Counter(r['failure'] for r in selected if r['failure'])) == expected['invalid']
            assert len(probs) == expected['confidence_n']
            assert math.isclose(sum(probs)/len(probs), expected['mean_raw_confidence'], abs_tol=1e-12)
        total = sum(c['estimated_usd'] for c in costs)
        summary = budget.summary()
        assert math.isclose(total, status['all_received_gptoss_estimated_usd'], abs_tol=1e-12)
        assert math.isclose(total-parent['original_gptoss_estimated_usd'], summary['estimated_usd'], abs_tol=1e-12)
        gaps = [b[0]-a[1] for a, b in zip(timings, timings[1:])]
        assert min(gaps) >= 29.99
        unknown = sorted([{'position': positions[k], 'status': e['status'], 'reserved_usd': e['reserved_usd']}
                          for k, e in state['requests'].items() if e['status'] not in ('settled','rejected')],
                         key=lambda e: e['position'])
        assert [e['position'] for e in unknown] == [22,49,65,96,176,179,292]
        assert math.isclose(sum(e['reserved_usd'] for e in unknown), summary['outstanding_reserved_usd'], abs_tol=1e-12)
        audit = {'checkpoint': '291 received; unknown292;153 never attempted',
            'expansion_id': plan['expansion_id'], 'analysis_id': analysis['analysis_id'],
            'passed': True, 'source_seals_and_protected_files_match': True,
            'original178_reply_and_candidate_hashes_match': True,
            'received_call_identity_and_priced_usage_checked_n': len(costs),
            'received_responses_single_success_record_each': True,
            'partition_counts_and_confidence_summary_recomputed': True,
            'usage_reconciles_with_shared_ledger_and_imported_first5': True,
            'minimum_new_request_gap_after_completion_seconds': min(gaps),
            'blocked292_elapsed_seconds': blocked_seconds,
            'new_api_calls': 0, 'answer_keys_read': False, 'correctness_scored': False,
            'confidence_calibrated': False, 'unit_tests_run': False,
            'full_completion_audit_passed': False,
            'note': 'Partial checkpoint audit only. Full collection audit correctly blocks on unknown292.'}
        budget_export = {'project': summary['project'], 'limit_usd': summary['limit_usd'],
            'received_reply_estimated_usd': total, 'shared_ledger_estimated_usd': summary['estimated_usd'],
            'imported_first5_estimated_usd': parent['original_gptoss_estimated_usd'],
            'latest_continuation_increment_estimated_usd': status['additional_estimated_usd'],
            'unknown_reserved_usd': summary['outstanding_reserved_usd'], 'unknown_requests': unknown,
            'confirmed_billing_usd': None, 'actual_total_cost_unknown': True}
        local_files = set(plan['preserved_files_sha256']) | set(analysis['source_sha256'])
        for root in (run.ROOT, run.base.PARENT):
            local_files.update(str(p) for p in root.rglob('*') if p.is_file()
                               and not p.name.endswith('.lock') and 'labels' not in p.name)
        local_manifest = {'note': 'Paths and checksums only; raw artifacts are intentionally absent from Git.',
                          'files': {p: file_digest(p) for p in sorted(local_files) if 'labels' not in Path(p).name}}
        source_paths = sorted(Path('src/vgx/gpqa').glob('gptoss*.py'))
        source_paths += [Path('configs/gpqa_replacement_generators_20261014.json'), Path(__file__)]
        code_manifest = {str(p.relative_to(Path.cwd()) if p.is_absolute() else p): file_digest(p) for p in source_paths}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for filename, value in (('audit.json',audit), ('budget_summary.json',budget_export),
                                ('local_artifacts_manifest.json',local_manifest), ('code_sha256.json',code_manifest)):
            run.base.immutable(OUTPUT/filename,value)
        for src, name in [(source/'status.json','status.json'), (source/'plan.json','collection_plan.json'),
                          (source/'billing_check.json','billing_check.json'),
                          (source/'budget_inventory.json','budget_inventory_at_launch.json'),
                          (analysis_path,'analysis_plan.json')]:
            dst=OUTPUT/name
            if dst.exists(): assert file_digest(dst)==file_digest(src), name
            else: shutil.copyfile(src,dst)
        assert all(file_digest(p)==sha for p,sha in guarded.items())
        run.load(run.ROOT)
        print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
