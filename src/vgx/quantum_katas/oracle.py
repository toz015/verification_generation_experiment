"""Official-checker adapter: one pinned evaluate_solution call per frozen candidate.

Primary label y = int(official.passed). The complete official return value is
persisted and never overridden. Every run records its intent first; a key with
an intent but no official return value is unresolved and is never re-run, so a
stochastic test is never re-drawn. A supervisor timeout or worker failure is a
service error (unresolved), not an official Timeout.

Execution backends:
  reviewed_local  fresh clean-env subprocess; refuses any code that is not in
                  the reviewed-fixture allowlist (reference/control fixtures).
  macos_sandbox   the same worker under sandbox-exec: network denied, reads of
                  the home directory and repository denied (except the pinned
                  interpreter), writes confined to the job directory, cleared
                  environment, CPU-time and file-size rlimits. macOS does not
                  enforce memory or process-count rlimits; unreviewed model code
                  additionally requires checker.model_code_execution_approved.
  mock            test double; never executes code.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import tempfile
import time

from vgx.common.storage import append_jsonl, digest, file_digest, file_lock, read_jsonl
from vgx.quantum_katas.load import SOURCE_SHA256, OracleTask

WORKER = Path(__file__).with_name('oracle_worker.py')
OFFICIAL_TIMEOUT_S = 30.0
OFFICIAL_FIELDS = ('passed', 'syntax_valid', 'test_passed', 'error_message', 'error_type', 'stdout')
BACKENDS = ('reviewed_local', 'macos_sandbox', 'mock')
SANDBOX_PROFILE = """\
(version 1)
(allow default)
(deny network*)
(deny mach-lookup (global-name "com.apple.SecurityServer") (global-name "com.apple.securityd"))
(deny file-read-data (subpath "{home}") (subpath "{repo}"))
(allow file-read-data (subpath "{prefix}") (subpath "{base_prefix}") (subpath "{job}"))
(deny file-write*)
(allow file-write* (subpath "{job}") (literal "/dev/null"))
"""


class ServiceError(RuntimeError):
    def __init__(self, reason, detail=None):
        self.reason, self.detail = reason, detail
        super().__init__(reason)


class UnreviewedCodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckerConfig:
    interpreter: str
    backend: str = 'reviewed_local'
    supervisor_timeout_s: float = 120.0
    cpu_seconds: int = 120
    file_size_bytes: int = 64 * 2**20
    workers: int = 1
    model_code_execution_approved: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f'unknown checker backend: {self.backend}')
        if self.supervisor_timeout_s <= OFFICIAL_TIMEOUT_S + 15:
            raise ValueError('supervisor timeout must leave startup/teardown margin over the official 30 s')
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or not 1 <= self.workers <= 4:
            raise ValueError('checker workers must be an integer in [1, 4]')
        if self.model_code_execution_approved and self.backend != 'macos_sandbox':
            raise ValueError('model code may only be approved for an isolated backend')

    def limits(self) -> dict:
        enforced = {'cpu_seconds_rlimit': self.cpu_seconds, 'file_size_bytes_rlimit': self.file_size_bytes,
                    'core_dumps': 0, 'supervisor_wall_timeout_s': self.supervisor_timeout_s,
                    'environment': 'cleared (PATH, HOME/TMPDIR=job dir, single-thread BLAS/OMP)'}
        if self.backend == 'macos_sandbox':
            enforced.update(network='denied by sandbox-exec', home_and_repo_reads='denied except pinned interpreter',
                            writes='job directory only', keychain_services='denied')
        return {'backend': self.backend, 'enforced': enforced,
                'not_enforced': ['memory limit (macOS ignores RLIMIT_AS/RLIMIT_DATA)',
                                 'process-count limit (RLIMIT_NPROC is per user on macOS)'] +
                                ([] if self.backend == 'macos_sandbox' else ['network', 'filesystem isolation'])}


def clean_env(job_dir: str) -> dict:
    return {'PATH': '/usr/bin:/bin', 'HOME': job_dir, 'TMPDIR': job_dir, 'LANG': 'en_US.UTF-8',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'VECLIB_MAXIMUM_THREADS': '1', 'RAYON_NUM_THREADS': '1', 'QISKIT_PARALLEL': 'FALSE',
            'PYTHONDONTWRITEBYTECODE': '1'}


def probe_environment(interpreter: str) -> dict:
    with tempfile.TemporaryDirectory(prefix='vgx-qkatas-probe-') as directory:
        directory = os.path.realpath(directory)
        shutil.copy(WORKER, Path(directory)/'oracle_worker.py')
        process = subprocess.run([os.path.abspath(interpreter), '-I', 'oracle_worker.py', '--probe'], cwd=directory,
                                 env=clean_env(directory), capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise RuntimeError('quantum interpreter probe failed: ' + process.stderr[-2000:])
    return json.loads(process.stdout)


def check_requirements(env: dict, requirements: str | Path) -> dict:
    def norm(name):
        return name.lower().replace('_', '-')
    pinned = {}
    for line in Path(requirements).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            name, version = line.split('==')
            pinned[norm(name)] = version
    found = {norm(k): v for k, v in env['packages'].items()}
    mismatched = {k: {'pinned': v, 'found': found.get(k)} for k, v in pinned.items() if found.get(k) != v}
    if mismatched or not env['python'].startswith('3.12.'):
        raise ValueError(f'quantum environment differs from pinned requirements: {mismatched}')
    return pinned


def environment_identity(env: dict, checker: CheckerConfig) -> str:
    return digest({'schema': 1, 'probe': env, 'limits': checker.limits(),
                   'worker_sha256': file_digest(WORKER),
                   'sandbox_profile_sha256': digest(SANDBOX_PROFILE) if checker.backend == 'macos_sandbox' else None})


def official_key(task: OracleTask, candidate, environment_id: str) -> str:
    return digest({'schema': 1, 'task_id': task.task_id, 'candidate_id': candidate.candidate_id,
                   'code_sha256': candidate.code_sha256, 'test_sha256': task.test_sha256,
                   'entry_point': task.entry_point, 'timeout_s': OFFICIAL_TIMEOUT_S,
                   'evaluator_sha256': SOURCE_SHA256['benchmark/evaluator.py'],
                   'environment_id': environment_id})


class OfficialStore:
    """private/official_results.jsonl: intent, then exactly one terminal event per key."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def events(self) -> list[dict]:
        return read_jsonl(self.path, tolerate_tail=True)

    def states(self) -> dict[str, dict]:
        states = {}
        for event in self.events():
            key = event['official_key']
            state = states.setdefault(key, {'intents': 0, 'terminal': None})
            if event['event'] == 'intent':
                state['intents'] += 1
            elif event['event'] in ('result', 'service_error'):
                if state['terminal'] is not None:
                    raise ValueError(f'multiple terminal events for official key {key}')
                state['terminal'] = event
            else:
                raise ValueError('unknown official store event')
        for key, state in states.items():
            if state['intents'] != 1:
                raise ValueError(f'official key {key} must have exactly one recorded intent')
            terminal = state['terminal']
            state['status'] = ('unresolved_lost_result' if terminal is None else
                               'official' if terminal['event'] == 'result' else 'unresolved_service_error')
        return states

    def results(self) -> dict[str, dict]:
        """candidate_id -> official result rows (only events with an official return value)."""
        out = {}
        for state in self.states().values():
            if state['status'] == 'official':
                out[state['terminal']['candidate_id']] = state['terminal']
        return out


class OfficialChecker:
    def __init__(self, checker: CheckerConfig, evaluator_path: str | Path, env: dict, *,
                 allowlist: set[str] | None = None, repo_root: str | Path | None = None, mock=None):
        self.cfg, self.env = checker, env
        self.evaluator_path = Path(evaluator_path)
        if file_digest(self.evaluator_path) != SOURCE_SHA256['benchmark/evaluator.py']:
            raise ValueError('pinned evaluator hash mismatch')
        self.allowlist = set(allowlist or ())
        self.repo_root = os.path.realpath(repo_root or Path.cwd())
        self.mock = mock
        if checker.backend == 'mock' and mock is None:
            raise ValueError('mock backend requires a mock evaluator')
        self.environment_id = environment_identity(env, checker)

    def authorize(self, code_sha256: str) -> str:
        if code_sha256 in self.allowlist:
            return 'reviewed_fixture'
        if self.cfg.backend == 'mock':
            return 'mock_not_executed'
        if self.cfg.backend == 'macos_sandbox' and self.cfg.model_code_execution_approved:
            return 'model_code_isolated'
        raise UnreviewedCodeError('unreviewed code requires an approved isolated backend; nothing was executed')

    def run(self, code: str, task: OracleTask) -> dict:
        """Return the worker payload or raise ServiceError. No retry is attempted."""
        if self.cfg.backend == 'mock':
            return self.mock(code, task)
        nonce = 'VGXQK-' + secrets.token_hex(16) + ':'
        job = {'code': code, 'test': task.test, 'entry_point': task.entry_point,
               'timeout': OFFICIAL_TIMEOUT_S, 'nonce': nonce,
               'rlimits': {'cpu_seconds': self.cfg.cpu_seconds, 'file_size_bytes': self.cfg.file_size_bytes},
               'evaluator_sha256': SOURCE_SHA256['benchmark/evaluator.py']}
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='vgx-qkatas-job-') as directory:
            directory = os.path.realpath(directory)
            shutil.copy(WORKER, Path(directory)/'oracle_worker.py')
            shutil.copy(self.evaluator_path, Path(directory)/'evaluator.py')
            command = [os.path.abspath(self.cfg.interpreter), '-I', 'oracle_worker.py']
            if self.cfg.backend == 'macos_sandbox':
                profile = Path(directory)/'sandbox.sb'
                profile.write_text(SANDBOX_PROFILE.format(
                    home=os.path.realpath(Path.home()), repo=self.repo_root, prefix=os.path.realpath(self.env['prefix']),
                    base_prefix=os.path.realpath(self.env['base_prefix']), job=directory))
                command = ['/usr/bin/sandbox-exec', '-f', str(profile), *command]
            process = subprocess.Popen(command, cwd=directory, env=clean_env(directory), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       start_new_session=True)
            try:
                stdout, stderr = process.communicate(json.dumps(job), timeout=self.cfg.supervisor_timeout_s)
            except subprocess.TimeoutExpired:
                _kill_group(process.pid)
                process.communicate()
                raise ServiceError('supervisor_timeout', {'supervisor_timeout_s': self.cfg.supervisor_timeout_s}) from None
            finally:
                _kill_group(process.pid)  # reap stray grandchildren of a finished worker
        elapsed = time.perf_counter() - started
        if process.returncode:
            raise ServiceError('worker_exit_nonzero', {'returncode': process.returncode, 'stderr_tail': stderr[-4000:]})
        lines = [line[len(nonce):] for line in stdout.splitlines() if line.startswith(nonce)]
        if len(lines) != 1:
            raise ServiceError('missing_or_ambiguous_worker_result', {'stderr_tail': stderr[-4000:]})
        try:
            payload = json.loads(lines[0])
        except ValueError:
            raise ServiceError('invalid_worker_result') from None
        official = payload.get('official')
        if not isinstance(official, dict) or set(official) != set(OFFICIAL_FIELDS) or not isinstance(official['passed'], bool):
            raise ServiceError('invalid_official_fields')
        if payload.get('evaluator_sha256') != SOURCE_SHA256['benchmark/evaluator.py']:
            raise ServiceError('evaluator_hash_mismatch')
        if {k: payload['environment'].get(k) for k in ('packages', 'python', 'executable_realpath')} != \
                {k: self.env.get(k) for k in ('packages', 'python', 'executable_realpath')}:
            raise ServiceError('environment_changed')
        return {**payload, 'supervisor_elapsed_s': elapsed, 'worker_stderr_tail': stderr[-4000:]}


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def score_candidates(candidates, oracle_tasks: dict[str, OracleTask], checker: OfficialChecker,
                     store: OfficialStore, *, operation_id: str) -> dict:
    """Score each candidate with code at most once; reuse or report everything already recorded."""
    summary = {'already_official': 0, 'already_unresolved': 0, 'new_official': 0,
               'new_service_errors': 0, 'skipped_no_code': 0, 'refused_unreviewed': 0, 'errors': []}
    with file_lock(str(store.path) + '.score.lock'):
        states = store.states()
        pending = []
        for candidate in candidates:
            if not candidate.has_code:
                summary['skipped_no_code'] += 1
                continue
            task = oracle_tasks[candidate.task_id]
            key = official_key(task, candidate, checker.environment_id)
            state = states.get(key)
            if state is not None:
                summary['already_official' if state['status'] == 'official' else 'already_unresolved'] += 1
                continue
            try:
                authorization = checker.authorize(candidate.code_sha256)
            except UnreviewedCodeError as error:
                summary['refused_unreviewed'] += 1
                summary['errors'].append({'task_id': candidate.task_id, 'error': str(error)})
                continue
            pending.append((candidate, task, key, authorization))

        def execute(entry):
            candidate, task, key, authorization = entry
            base = {'official_key': key, 'task_id': task.task_id, 'candidate_id': candidate.candidate_id,
                    'code_sha256': candidate.code_sha256, 'test_sha256': task.test_sha256,
                    'evaluator_sha256': SOURCE_SHA256['benchmark/evaluator.py'],
                    'environment_id': checker.environment_id, 'backend': checker.cfg.backend,
                    'authorization': authorization, 'operation_id': operation_id}
            append_jsonl(store.path, {**base, 'event': 'intent', 'created': time.time()})
            try:
                payload = checker.run(candidate.raw_code, task)
            except ServiceError as error:
                append_jsonl(store.path, {**base, 'event': 'service_error', 'reason': error.reason,
                                          'detail': error.detail, 'created': time.time()})
                return 'new_service_errors'
            official = payload['official']
            append_jsonl(store.path, {**base, 'event': 'result', 'official': official,
                                      'label': int(official['passed']),
                                      'executed_code_sha256': payload.get('executed_code_sha256'),
                                      'evaluate_elapsed_s': payload.get('evaluate_elapsed_s'),
                                      'supervisor_elapsed_s': payload.get('supervisor_elapsed_s'),
                                      'worker_stderr_tail': payload.get('worker_stderr_tail'),
                                      'created': time.time()})
            return 'new_official'

        with ThreadPoolExecutor(max_workers=checker.cfg.workers) as pool:
            for outcome in pool.map(execute, pending):
                summary[outcome] += 1
    return summary
