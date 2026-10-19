"""QuantumKatas official-checker pilot CLI.

  prepare             freeze tasks/prompts/models/checker environment into a new run dir
  dry-run             full offline pipeline with mock providers and reviewed fixtures
  collect-generators  one frozen candidate per task (real calls need --allow-api)
  collect-verifiers   development collection: all verifiers for every candidate with code
  score-official      private: one official evaluate_solution per frozen candidate
  synthetic-policy    write an explicitly synthetic engineering policy (not fitted)
  run-sequential      policy-driven queries; nothing past the stopping point is requested
  report              offline report (reads private results)

API access is off by default. prepare, dry-run and report need no credentials or network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from vgx.common.storage import atomic_json, digest, file_lock
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.score import VerifierLikelihood
from vgx.gpqa.sequential import seal, unseal
from vgx.quantum_katas import dryrun, load, run_pilot, sequential
from vgx.quantum_katas.report import build_report
from vgx.quantum_katas.run_pilot import Run, SignalStore


def synthetic_policy(root) -> dict:
    run = Run(root)
    model = VerifierLikelihood((0., 1/3, 2/3, 1.), (.15, .25, .6), (.6, .25, .15))
    planner = NestedPlanner([model] * len(run.verifiers), [.02 + .01*i for i in range(len(run.verifiers))], 1., 4., grid_size=1001)
    return sequential.make_policy(plan_id=run.plan_id, verifiers=run.verifiers, planner=planner,
                                  kind='synthetic_engineering_dry_run',
                                  note='Hand-set likelihoods and costs for interface testing only; not fitted to '
                                       'QuantumKatas data and not GPQA parameters.')


def run_sequential(root, policy: dict, *, allow_api=False) -> dict:
    run = Run(root)
    specs, _ = sequential.policy_parts(policy)
    if policy['plan_id'] != run.plan_id:
        raise ValueError('policy was prepared for a different run')
    known = {v.identity for v in run.verifiers}
    if any(s.identity not in known for s in specs):
        raise ValueError('policy verifier is not part of this run')
    run.check_api(allow_api, specs)
    cache = run.cache()
    output = run.root/'sequential'
    binding = {'schema': 1, 'plan_id': run.plan_id, 'policy_id': policy['policy_id'], 'task_ids': run.plan['task_ids']}
    with file_lock(output/'execution.lock'):
        path = output/'execution_manifest.json'
        if path.exists():
            execution = json.loads(path.read_text())
            unseal(execution, 'execution_id')
            if execution['binding'] != binding:
                raise ValueError('sequential directory belongs to a different policy')
        else:
            execution = seal({'binding': binding, 'operation_id': str(uuid.uuid4())}, 'execution_id')
            atomic_json(path, execution)
        atomic_json(output/'policy.json', policy)
        query = run_pilot.verifier_query(run, cache, allow_api=allow_api, operation_id=execution['operation_id'])
        candidates = run.candidates.by_task()
        results, missing = [], []
        for task in run.tasks:
            candidate = candidates.get(task.task_id)
            if candidate is None:
                missing.append(task.task_id)
                continue
            results.append(sequential.execute_candidate(candidate, policy, query,
                                                        output/'items'/f'{digest(candidate.candidate_id)}.json'))
        # Replay over development signals happens after live decisions and reads only the
        # layers the planner selects; it demonstrates logic, not saved spend.
        agreement = None
        signal_path = run.root/'verifier_signals.jsonl'
        if signal_path.exists():
            index = SignalStore(signal_path).index()
            agree = []
            for result in results:
                candidate = next(c for c in candidates.values() if c.candidate_id == result['candidate_id'])
                scores = [(index.get((candidate.candidate_id, s.id)) or {}).get('p_correct') for s in specs]
                replayed = sequential.replay(candidate, policy, scores)
                agree.append(all(replayed[k] == result[k] for k in ('action', 'verifiers_used'))
                             and (replayed['failure'] is None) == (result['failure'] is None)
                             and (result['posterior'] is None or replayed['posterior'] == result['posterior']))
            agreement = {'n': len(agree), 'agree': sum(agree)}
        value = seal({**execution, 'policy_id': policy['policy_id'], 'policy_kind': policy['kind'],
                      'results': results, 'tasks_without_candidate': missing,
                      'logical_verifier_queries': sum(r['verifiers_used'] for r in results),
                      'new_provider_executions': sum(r['new_provider_executions'] for r in results),
                      'replay_agreement': agreement,
                      'note': ('Engineering dry run with a synthetic policy.' if policy['kind'] != 'fitted_development'
                               else 'Development policy.') +
                              ' Logical queries include cached responses; paid executions are counted separately.'},
                     'decisions_id')
        atomic_json(output/'decisions.json', value)
    return {k: value[k] for k in ('logical_verifier_queries', 'new_provider_executions', 'replay_agreement')} | {
        'actions': {a: sum(r['action'] == a for r in results) for a in ('assert', 'abstain')},
        'tasks_without_candidate': len(missing)}


def dry_run(root, *, checker='reviewed_local', interpreter=None) -> dict:
    """Prepare, collect (mock), freeze, score, resume, sequential and report — all offline."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    cfg = dryrun.config(checker, **({'interpreter': interpreter} if interpreter else {}))
    public, private = load.load_tasks(cfg['source'], cfg['pilot'])
    responses, fixtures, expected = dryrun.build(public, private)
    (root/'inputs').mkdir()
    atomic_json(root/'inputs'/'dry_run_config.json', cfg)
    mock_env = None
    mock_checker = None
    if checker == 'mock':
        mock_env, mock_checker = mock_checker_parts(private, fixtures)
    run_dir = root/'run'
    steps = {'prepare': run_pilot.prepare(root/'inputs'/'dry_run_config.json', run_dir, mock_responses=responses,
                                          reviewed_fixtures=fixtures, mock_environment=mock_env)}
    steps['collect_generators_smoke'] = _brief(run_pilot.collect_generators(run_dir, allow_api=True, limit=3))
    steps['collect_generators'] = _brief(run_pilot.collect_generators(run_dir, allow_api=True))
    steps['collect_generators_resume_offline'] = _brief(run_pilot.collect_generators(run_dir, allow_api=False))
    steps['collect_verifiers'] = _brief(run_pilot.collect_verifiers(run_dir, allow_api=True))
    steps['collect_verifiers_resume_offline'] = _brief(run_pilot.collect_verifiers(run_dir, allow_api=False))
    steps['score_official_smoke'] = run_pilot.score_official(run_dir, limit=3, mock=mock_checker)
    steps['score_official'] = run_pilot.score_official(run_dir, mock=mock_checker)
    steps['score_official_resume'] = run_pilot.score_official(run_dir, mock=mock_checker)
    policy = synthetic_policy(run_dir)
    atomic_json(root/'inputs'/'synthetic_policy.json', policy)
    steps['run_sequential'] = run_sequential(run_dir, policy, allow_api=False)
    steps['run_sequential_resume'] = run_sequential(run_dir, policy, allow_api=False)
    report = build_report(run_dir)
    steps['report'] = {'samples': report['samples'], 'official_checker': {
        k: report['official_checker'][k] for k in ('official_returns', 'passed', 'failed', 'unresolved', 'error_types')}}
    atomic_json(root/'dry_run_summary.json', {'expected_fixtures': expected, 'steps': steps})
    return steps


def _brief(status):
    return {k: v for k, v in status.items() if k not in ('rows', 'budget', 'updated_utc', 'operation_id')}


def mock_checker_parts(private, fixtures):
    """Main-env test double: passes reference/probe fixtures and fails the rest, without executing code."""
    reviewed = {f['code_sha256']: f['kind'] for f in fixtures}
    env = {'packages': {'qiskit': '2.3.0', 'qiskit-aer': '0.17.2', 'numpy': '2.4.1', 'scipy': '1.17.0',
                        'rustworkx': '0.17.1', 'dill': '0.4.0', 'stevedore': '5.6.0', 'psutil': '7.2.1',
                        'python-dateutil': '2.9.0.post0', 'six': '1.17.0', 'typing_extensions': '4.15.0'},
           'python': '3.12.13 (mock)', 'executable_realpath': 'mock', 'prefix': 'mock', 'base_prefix': 'mock'}

    def evaluate(code, task):
        kind = reviewed.get(load.sha256_text(code))
        ok = kind in ('reference', 'reference_json_fence', 'reference_in_markdown', 'probe')
        syntax = kind != 'syntax_error'
        return {'official': {'passed': ok, 'syntax_valid': syntax, 'test_passed': ok,
                             'error_message': None if ok else 'mock failure',
                             'error_type': None if ok else ('SyntaxError' if not syntax else 'AssertionError'),
                             'stdout': '' if syntax else None},
                'evaluate_elapsed_s': 0.0, 'supervisor_elapsed_s': 0.0, 'executed_code_sha256': None}
    return env, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('dry-run'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checker', choices=('reviewed_local', 'macos_sandbox', 'mock'), default='reviewed_local')
    p.add_argument('--interpreter')
    for name in ('collect-generators', 'collect-verifiers'):
        p = sub.add_parser(name); p.add_argument('--root', type=Path, required=True)
        p.add_argument('--allow-api', action='store_true'); p.add_argument('--limit', type=int)
    p = sub.add_parser('score-official'); p.add_argument('--root', type=Path, required=True); p.add_argument('--limit', type=int)
    p = sub.add_parser('synthetic-policy'); p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run-sequential'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True); p.add_argument('--allow-api', action='store_true')
    p = sub.add_parser('report'); p.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = run_pilot.prepare(args.config, args.output)
    elif args.command == 'dry-run':
        result = dry_run(args.root, checker=args.checker, interpreter=args.interpreter)
    elif args.command == 'collect-generators':
        result = _brief(run_pilot.collect_generators(args.root, allow_api=args.allow_api, limit=args.limit))
    elif args.command == 'collect-verifiers':
        result = _brief(run_pilot.collect_verifiers(args.root, allow_api=args.allow_api, limit=args.limit))
    elif args.command == 'score-official':
        result = run_pilot.score_official(args.root, limit=args.limit)
    elif args.command == 'synthetic-policy':
        if args.output.exists():
            raise SystemExit('policy output exists; refusing to overwrite')
        result = synthetic_policy(args.root)
        atomic_json(args.output, result)
        result = {'policy_id': result['policy_id'], 'kind': result['kind'], 'output': str(args.output)}
    elif args.command == 'run-sequential':
        result = run_sequential(args.root, json.loads(args.policy.read_text()), allow_api=args.allow_api)
    else:
        report = build_report(args.root)
        result = {'report': str(Path(args.root)/'report.txt'), 'samples': report['samples']}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
