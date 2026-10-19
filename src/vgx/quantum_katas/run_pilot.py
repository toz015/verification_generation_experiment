"""Run directory: sealed plan, paid/mock collection, and private official scoring.

Boundaries:
  prepare            no network or credentials; probes the local quantum interpreter.
  collect-*          read plan.json, public_tasks.jsonl, frozen candidates and cache only;
                     never private/. Provider calls need allow_api (and a budget for
                     real providers); otherwise a cache miss is reported, not sent.
  score-official     reads private/oracle_tasks.jsonl; one official evaluation per candidate.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, RequestCache, UncertainRequestError
from vgx.common.budget import BudgetLimitError, BudgetedRequestCache
from vgx.common.llm import Request
from vgx.common.storage import append_jsonl, atomic_json, digest, file_digest, file_lock, read_jsonl
from vgx.gpqa.sequential import seal, unseal
from vgx.quantum_katas import artifacts, load, oracle, prompt, runners
from vgx.quantum_katas.artifacts import CandidateStore, FrozenCodeCandidate, candidate_identity, verifier_view
from vgx.quantum_katas.load import PublicTask, sha256_text

PACKAGE = Path(__file__).resolve().parent
COMMON = PACKAGE.parent/'common'
# Files that determine requests, parsing, candidate identity or official labels.
BOUND_FILES = {
    **{f'quantum_katas/{n}': PACKAGE/n for n in ('load.py', 'prompt.py', 'artifacts.py', 'runners.py',
                                                 'oracle.py', 'oracle_worker.py', 'run_pilot.py')},
    **{f'common/{n}': COMMON/n for n in ('api.py', 'llm.py', 'storage.py', 'vertex.py', 'vertex_partner.py',
                                         'budget.py', 'billing.py')},
}


def implementation() -> dict:
    return {name: file_digest(path) for name, path in BOUND_FILES.items()}


class Run:
    """A prepared run directory whose plan and inputs are re-verified on every load."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        plan = json.loads((self.root/'plan.json').read_text())
        unseal(plan, 'plan_id')
        if plan['implementation'] != implementation():
            raise ValueError('bound implementation files changed after prepare; prepare a new run')
        if file_digest(self.root/'public_tasks.jsonl') != plan['public_tasks_sha256']:
            raise ValueError('public task file changed')
        self.plan = plan
        self.config = plan['config']
        self.tasks = load.read_public_tasks(self.root/'public_tasks.jsonl')
        if [t.task_id for t in self.tasks] != plan['task_ids']:
            raise ValueError('public task order changed')
        self.generator, self.verifiers = runners.load_specs(self.config['models'])
        self.mock_responses = None
        if plan['mock_provider_sha256'] is not None:
            path = self.root/'mock_provider'/'responses.json'
            if file_digest(path) != plan['mock_provider_sha256']:
                raise ValueError('mock provider fixture changed')
            self.mock_responses = json.loads(path.read_text())
        self.candidates = CandidateStore(self.root/'frozen_candidates.jsonl')

    @property
    def plan_id(self):
        return self.plan['plan_id']

    def runner(self, spec):
        return spec.runner(self.mock_responses)

    def cache(self):
        budget = self.config.get('budget')
        if budget is None:
            return RequestCache(self.root/'cache')
        return BudgetedRequestCache(self.root/'cache', project=budget['project'], pricing=budget['pricing'],
                                    limit_usd=budget['limit_usd'],
                                    gemini_output_reserve=budget.get('gemini_output_reserve', 131072))

    def task_map(self) -> dict[str, PublicTask]:
        return {t.task_id: t for t in self.tasks}

    def check_api(self, allow_api: bool, specs) -> None:
        if not allow_api:
            return
        if all(s.provider == 'mock' for s in specs):
            return
        if self.config.get('budget') is None or not self.config.get('paid_collection_authorized'):
            raise ValueError('paid collection needs an explicit budget and paid_collection_authorized=true in the run config')


def generator_request(generation_id: str, task: PublicTask, operation_id=None) -> Request:
    # Provider identity is runner + prompt + system; generation_id is metadata only, so a
    # second independent sample would need a new prompt or runner identity, not new meta.
    return Request(f'generator|{task.task_id}|{generation_id}', prompt.build_generator_prompt(task),
                   prompt.GENERATOR_SYSTEM, {'role': 'generator', 'task_id': task.task_id, 'partition': task.partition,
                                             'generation_id': generation_id, 'operation_id': operation_id})


def verifier_prompt_for(task: PublicTask, code: str) -> str:
    return prompt.build_verifier_prompt({'task_id': task.task_id, 'prompt': task.prompt, 'entry_point': task.entry_point,
                                         'required_functions': list(task.required_functions), 'code': code})


def verifier_request(spec, candidate: FrozenCodeCandidate, task: PublicTask, operation_id=None) -> Request:
    view = verifier_view(candidate, task)
    return Request(f'{spec.id}|{candidate.candidate_id}', prompt.build_verifier_prompt(view), prompt.VERIFIER_SYSTEM,
                   {'role': 'verifier', 'task_id': task.task_id, 'partition': task.partition,
                    'candidate_id': candidate.candidate_id, 'verifier_id': spec.id, 'operation_id': operation_id})


def _reservations(run_config, specs_requests):
    budget = run_config.get('budget')
    if budget is None:
        return None
    cache = BudgetedRequestCache('/nonexistent-reservation-only', project=budget['project'], pricing=budget['pricing'],
                                 limit_usd=budget['limit_usd'],
                                 gemini_output_reserve=budget.get('gemini_output_reserve', 131072))
    return sum(cache.reservation(runner, request) for runner, request in specs_requests)


def prepare(config_path: str | Path, output: str | Path, *, mock_responses: dict | None = None,
            reviewed_fixtures: list[dict] | None = None, mock_environment: dict | None = None) -> dict:
    """Freeze tasks, prompts, model specs, checker environment and identities. Never overwrites."""
    config = json.loads(Path(config_path).read_text())
    if config.get('schema') != 1 or config.get('kind') != 'quantum_katas_official_pilot':
        raise ValueError('unsupported QuantumKatas run config')
    if config.get('generation_id') != 'g0':
        raise ValueError('this pilot freezes exactly one generation per task (generation_id g0)')
    if config.get('prompt_version') != prompt.PROMPT_VERSION:
        raise ValueError('run config prompt_version differs from the implemented parser')
    generator, verifiers = runners.load_specs(config['models'])
    if not 3 <= len(verifiers) <= 5:
        raise ValueError('the pilot uses three to five configured verifiers')
    is_mock = generator.provider == 'mock' and all(v.provider == 'mock' for v in verifiers)
    if is_mock != (mock_responses is not None):
        raise ValueError('mock providers and mock responses must be used together (dry run only)')
    if not is_mock and any(s.provider == 'mock' for s in (generator, *verifiers)):
        raise ValueError('mixing mock and real providers is not allowed')
    source = Path(config['source'])
    sources = load.verify_sources(source)
    protocol = json.loads(Path(config['protocol']).read_text())
    if (protocol['official_evaluator']['source_sha256'] != load.SOURCE_SHA256['benchmark/evaluator.py']
            or protocol['official_evaluator']['dataset_sha256'] != load.SOURCE_SHA256['dataset/qiskit_quantumkatas.jsonl']
            or protocol['revision'] != load.REVISION or protocol['excluded_task_ids'] != []
            or protocol['official_evaluator']['timeout_s'] != oracle.OFFICIAL_TIMEOUT_S
            or protocol['official_evaluator']['evaluations_per_frozen_candidate'] != 1):
        raise ValueError('run config does not match the official protocol')
    public, private = load.load_tasks(source, config['pilot'])
    checker = oracle.CheckerConfig(**config['checker'])
    if (checker.backend == 'mock') != (mock_environment is not None):
        raise ValueError('a synthetic environment is allowed only with the mock checker backend')
    env = mock_environment or oracle.probe_environment(checker.interpreter)
    pinned = oracle.check_requirements(env, config['requirements'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'private').mkdir()
    artifacts.write_jsonl_once(output/'public_tasks.jsonl', [t.to_row() for t in public])
    artifacts.write_jsonl_once(output/'private'/'oracle_tasks.jsonl', [asdict(t) for t in private])
    if reviewed_fixtures is not None:
        artifacts.write_jsonl_once(output/'private'/'reviewed_fixtures.jsonl', reviewed_fixtures)
    mock_sha = None
    if mock_responses is not None:
        (output/'mock_provider').mkdir()
        atomic_json(output/'mock_provider'/'responses.json', mock_responses)
        (output/'mock_provider'/'README.txt').write_text(
            'Offline dry-run provider double. Responses were authored locally from reviewed\n'
            'reference/control fixtures; they are not model outputs and contain reference code.\n')
        mock_sha = file_digest(output/'mock_provider'/'responses.json')
    shutil.copy(source/'LICENSE', output/'LICENSE.upstream.txt')
    (output/'ATTRIBUTION.txt').write_text(
        f'Source: {load.DATASET} (https://github.com/qiskit-community/Qiskit-QuantumKatas)\n'
        f'Pinned revision: {load.REVISION}\nLicense: CC-BY-NC-SA-4.0 (LICENSE.upstream.txt).\n'
        'public_tasks.jsonl projects unmodified upstream prompts and identifiers; private/oracle_tasks.jsonl\n'
        'holds the unmodified tests and reference solutions for official scoring only. "private" is an\n'
        'experimental boundary, not a claim that the upstream data is secret or absent from pretraining.\n')
    gen_runner = generator.runner(mock_responses)
    gen_requests = [generator_request(config['generation_id'], t) for t in public]
    plan = seal({
        'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'dry_run' if is_mock else 'pilot',
        'config': config, 'config_sha256': file_digest(config_path),
        'sources': sources, 'pilot_config_sha256': file_digest(config['pilot']),
        'protocol_sha256': file_digest(config['protocol']), 'requirements_sha256': file_digest(config['requirements']),
        'pinned_requirements': pinned, 'implementation': implementation(),
        'task_ids': [t.task_id for t in public], 'public_tasks_sha256': file_digest(output/'public_tasks.jsonl'),
        'oracle_tasks_sha256': file_digest(output/'private'/'oracle_tasks.jsonl'),
        'reviewed_fixtures_sha256': file_digest(output/'private'/'reviewed_fixtures.jsonl') if reviewed_fixtures is not None else None,
        'mock_provider_sha256': mock_sha, 'prompt_version': prompt.PROMPT_VERSION,
        'generator_runner_identity': gen_runner.identity,
        'verifier_runner_identities': {v.id: v.runner(mock_responses).identity for v in verifiers},
        'generator_request_keys': {r.meta['task_id']: gen_runner.cache_key(r) for r in gen_requests},
        'environment': env, 'environment_id': oracle.environment_identity(env, checker),
        'checker_limits': checker.limits(),
        'request_ceilings': {'generator': len(public), 'verifier': len(public) * len(verifiers),
                             'official_checker': len(public), 'smoke_tasks': config.get('smoke_tasks', 3)},
        'generator_reservation_usd': _reservations(config, [(gen_runner, r) for r in gen_requests]),
        'paid_api_calls_at_prepare': 0,
    }, 'plan_id')
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'manifest.json', {
        'plan_id': plan['plan_id'], 'mode': plan['mode'], 'tasks': len(public),
        'files': {'plan': 'plan.json', 'public_tasks': 'public_tasks.jsonl',
                  'frozen_candidates': 'frozen_candidates.jsonl', 'cache': 'cache/',
                  'official_results': 'private/official_results.jsonl', 'verifier_signals': 'verifier_signals.jsonl',
                  'sequential': 'sequential/', 'report': ['report.json', 'report.txt']},
        'note': 'Development partition only. Private files are read by score-official and report only.'})
    return {'stage': 'prepared', 'plan_id': plan['plan_id'], 'root': str(output), 'tasks': len(public),
            'mode': plan['mode'], 'generator_reservation_usd': plan['generator_reservation_usd']}


def _status(run: Run, name: str, value: dict) -> dict:
    value = {'plan_id': run.plan_id, 'updated_utc': datetime.now(timezone.utc).isoformat(), **value}
    atomic_json(run.root/'collection'/f'{name}.json', value)
    append_jsonl(run.root/'collection'/'history.jsonl', {'name': name, **value})
    return value


def _blocking(error):
    return isinstance(error, (UncertainRequestError, BudgetLimitError, ProviderRejectedError))


def collect_generators(root, *, allow_api=False, limit=None) -> dict:
    run = Run(root)
    run.check_api(allow_api, [run.generator])
    cache, runner = run.cache(), run.runner(run.generator)
    operation_id = str(uuid.uuid4())
    tasks = run.tasks[:limit] if limit else run.tasks
    rows, errors = [], []
    with file_lock(run.root/'collection'/'generators.lock'):
        for task in tasks:
            request = generator_request(run.config['generation_id'], task, operation_id)
            if runner.cache_key(request) != run.plan['generator_request_keys'][task.task_id]:
                raise ValueError('generator request identity changed since prepare')
            try:
                call, reused = cache.get(runner, request, allow_api=allow_api)
            except OfflineCacheMiss:
                rows.append({'task_id': task.task_id, 'status': 'missing_cache'})
                continue
            except Exception as error:  # stop: never resend an uncertain/rejected/over-budget request
                if not _blocking(error):
                    raise
                errors.append({'task_id': task.task_id, 'type': type(error).__name__,
                               'http_status': getattr(error, 'status', None), 'message': str(error)})
                break
            finish = runners.finish_status(call)
            parsed = prompt.parse_generator_response(call.response, truncated=finish == 'truncated')
            code_sha = sha256_text(parsed.code) if parsed.code is not None else None
            response_sha = sha256_text(call.response)
            meta = call.meta or {}
            candidate = FrozenCodeCandidate(
                task_id=task.task_id, generation_id=run.config['generation_id'],
                candidate_id=candidate_identity(task.task_id, run.config['generation_id'], call.key, response_sha, code_sha),
                partition=task.partition, raw_code=parsed.code, code_sha256=code_sha, raw_p_correct=parsed.p_correct,
                generator_response_status=parsed.response_status, generator_failure=parsed.failure,
                confidence_status=parsed.confidence_status, finish_status=finish,
                model_identity={'spec_id': run.generator.id, 'model': run.generator.model,
                                'runner_identity': runner.identity,
                                'provider_response_model': meta.get('provider_response_model'),
                                'model_version': meta.get('model_version')},
                request_key=call.key, provider_execution_id=meta.get('execution_id'),
                prompt_sha256=sha256_text(request.prompt), system_sha256=sha256_text(request.system),
                raw_response_sha256=response_sha, prompt_version=prompt.PROMPT_VERSION,
                dataset_revision=task.revision, task_prompt_sha256=task.prompt_sha256)
            new = run.candidates.freeze(candidate)
            rows.append({'task_id': task.task_id, 'status': 'frozen' if new else 'already_frozen',
                         'used_cached_response': reused, 'has_code': candidate.has_code,
                         'confidence_status': candidate.confidence_status, 'failure': candidate.generator_failure})
    summary = {'requested_tasks': len(tasks), 'processed': len(rows), 'errors': errors,
               'new_provider_executions': sum(r.get('used_cached_response') is False for r in rows),
               'cache_hits': sum(r.get('used_cached_response') is True for r in rows),
               'missing_cache': sum(r['status'] == 'missing_cache' for r in rows),
               'frozen_new': sum(r['status'] == 'frozen' for r in rows), 'rows': rows,
               'budget': cache.summary() if isinstance(cache, BudgetedRequestCache) else None,
               'stage': 'blocked' if errors else 'complete', 'allow_api': allow_api, 'operation_id': operation_id}
    return _status(run, 'generators', summary)


class SignalStore:
    """verifier_signals.jsonl: one immutable row per (candidate_id, verifier_id)."""

    def __init__(self, path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        rows = read_jsonl(self.path)
        keys = [(r['candidate_id'], r['verifier_id']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate verifier signal rows')
        return rows

    def index(self) -> dict:
        return {(r['candidate_id'], r['verifier_id']): r for r in self.rows()}

    def add(self, row: dict) -> bool:
        with file_lock(str(self.path) + '.lock'):
            existing = self.index().get((row['candidate_id'], row['verifier_id']))
            if existing is not None:
                if existing != row:
                    raise ValueError('verifier signal cannot be revised')
                return False
            append_jsonl(self.path, row)
            return True


def verifier_query(run: Run, cache, *, allow_api: bool, operation_id: str):
    """Return query(spec, candidate) -> observation; exactly one request for exactly one layer."""
    tasks = run.task_map()

    def query(spec, candidate):
        task = tasks[candidate.task_id]
        runner = run.runner(spec)
        request = verifier_request(spec, candidate, task, operation_id)
        call, reused = cache.get(runner, request, allow_api=allow_api)  # OfflineCacheMiss propagates
        finish = runners.finish_status(call)
        parsed = prompt.parse_verifier_response(call.response, truncated=finish == 'truncated')
        return {'score': parsed.p_correct, 'failure': parsed.failure, 'request_key': call.key,
                'execution_id': (call.meta or {}).get('execution_id'), 'used_cached_response': reused,
                'finish_status': finish, 'prompt_sha256': sha256_text(request.prompt),
                'raw_response_sha256': sha256_text(call.response), 'latency_s': call.latency_s}
    return query


def collect_verifiers(root, *, allow_api=False, limit=None) -> dict:
    """Development collection: every verifier for every candidate with code (all calls are spent)."""
    run = Run(root)
    run.check_api(allow_api, run.verifiers)
    cache = run.cache()
    operation_id = str(uuid.uuid4())
    store = SignalStore(run.root/'verifier_signals.jsonl')
    query = verifier_query(run, cache, allow_api=allow_api, operation_id=operation_id)
    candidates = run.candidates.by_task()
    task_ids = [t.task_id for t in (run.tasks[:limit] if limit else run.tasks)]
    rows, errors = [], []
    with file_lock(run.root/'collection'/'verifiers.lock'):
        for tid in task_ids:
            candidate = candidates.get(tid)
            if candidate is None or not candidate.has_code:
                continue
            for spec in run.verifiers:
                try:
                    observation = query(spec, candidate)
                except OfflineCacheMiss:
                    rows.append({'task_id': tid, 'verifier_id': spec.id, 'status': 'missing_cache'})
                    continue
                except Exception as error:
                    if not _blocking(error):
                        raise
                    errors.append({'task_id': tid, 'verifier_id': spec.id, 'type': type(error).__name__,
                                   'http_status': getattr(error, 'status', None), 'message': str(error)})
                    break
                row = {'candidate_id': candidate.candidate_id, 'task_id': tid, 'verifier_id': spec.id,
                       'verifier_identity': spec.identity, 'p_correct': observation['score'],
                       'failure': observation['failure'], 'finish_status': observation['finish_status'],
                       'request_key': observation['request_key'], 'execution_id': observation['execution_id'],
                       'prompt_sha256': observation['prompt_sha256'],
                       'raw_response_sha256': observation['raw_response_sha256'], 'latency_s': observation['latency_s']}
                new = store.add(row)
                rows.append({'task_id': tid, 'verifier_id': spec.id, 'status': 'recorded' if new else 'already_recorded',
                             'used_cached_response': observation['used_cached_response'], 'failure': row['failure']})
            if errors:
                break
    summary = {'mode': 'collect-development', 'processed': len(rows), 'errors': errors,
               'new_provider_executions': sum(r.get('used_cached_response') is False for r in rows),
               'cache_hits': sum(r.get('used_cached_response') is True for r in rows),
               'missing_cache': sum(r['status'] == 'missing_cache' for r in rows),
               'invalid_signals': sum(bool(r.get('failure')) for r in rows if r['status'] != 'missing_cache'),
               'budget': cache.summary() if isinstance(cache, BudgetedRequestCache) else None,
               'stage': 'blocked' if errors else 'complete', 'allow_api': allow_api, 'operation_id': operation_id,
               'note': 'Development collection queries every verifier; none of these calls is saved by a policy.'}
    return _status(run, 'verifiers', summary)


def official_checker(run: Run, *, mock=None):
    """Private side only: allowlist = reference solutions plus any reviewed fixtures."""
    private = run.root/'private'
    if file_digest(private/'oracle_tasks.jsonl') != run.plan['oracle_tasks_sha256']:
        raise ValueError('private oracle tasks changed')
    oracle_tasks = load.read_oracle_tasks(private/'oracle_tasks.jsonl')
    allow = {t.canonical_solution_sha256 for t in oracle_tasks.values()}
    if run.plan['reviewed_fixtures_sha256'] is not None:
        if file_digest(private/'reviewed_fixtures.jsonl') != run.plan['reviewed_fixtures_sha256']:
            raise ValueError('reviewed fixtures changed')
        allow |= {r['code_sha256'] for r in read_jsonl(private/'reviewed_fixtures.jsonl')}
    cfg = oracle.CheckerConfig(**run.config['checker'])
    env = run.plan['environment'] if cfg.backend == 'mock' else oracle.probe_environment(cfg.interpreter)
    checker = oracle.OfficialChecker(cfg, Path(run.config['source'])/'benchmark/evaluator.py', env,
                                     allowlist=allow, repo_root=Path.cwd(), mock=mock)
    if checker.environment_id != run.plan['environment_id']:
        raise ValueError('quantum execution environment differs from the prepared plan')
    return checker, oracle_tasks


def score_official(root, *, limit=None, mock=None) -> dict:
    run = Run(root)
    checker, oracle_tasks = official_checker(run, mock=mock)
    candidates = run.candidates.by_task()
    task_ids = [t.task_id for t in (run.tasks[:limit] if limit else run.tasks)]
    chosen = [candidates[t] for t in task_ids if t in candidates]
    store = oracle.OfficialStore(run.root/'private'/'official_results.jsonl')
    started = time.perf_counter()
    summary = oracle.score_candidates(chosen, oracle_tasks, checker, store, operation_id=str(uuid.uuid4()))
    summary.update(candidates_considered=len(chosen), tasks_without_candidate=len(task_ids) - len(chosen),
                   wall_elapsed_s=time.perf_counter() - started, backend=checker.cfg.backend)
    atomic_json(run.root/'private'/'score_status.json', {'plan_id': run.plan_id, **summary})
    return summary
