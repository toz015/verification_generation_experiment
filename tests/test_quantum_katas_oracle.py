"""Official-checker adapter: one evaluation per candidate, unresolved handling, Qiskit integration.

Integration tests run the pinned evaluator under the separate quantum interpreter;
the main environment never imports Qiskit.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vgx.quantum_katas import load, oracle
from vgx.quantum_katas.artifacts import FrozenCodeCandidate, candidate_identity

INTERPRETER = Path('tmp/quantum_katas_pilot_20261018/venv/bin/python')
EVALUATOR = load.DEFAULT_SOURCE/'benchmark/evaluator.py'
needs_source = pytest.mark.skipif(not EVALUATOR.exists(), reason='pinned evaluator not downloaded')
needs_qiskit = pytest.mark.skipif(not INTERPRETER.exists() or not EVALUATOR.exists(),
                                  reason='separate quantum interpreter not installed')
MOCK_ENV = {'packages': {}, 'python': '3.12.13 (mock)', 'executable_realpath': 'mock', 'prefix': 'mock', 'base_prefix': 'mock'}


def candidate(task_id, code, tag='r'):
    sha = load.sha256_text(code)
    return FrozenCodeCandidate(
        task_id=task_id, generation_id='g0', candidate_id=candidate_identity(task_id, 'g0', 'k', tag, sha),
        partition='development', raw_code=code, code_sha256=sha, raw_p_correct=None,
        generator_response_status='code_extracted', generator_failure=None, confidence_status='missing',
        finish_status='complete', model_identity={}, request_key='k', provider_execution_id=None, prompt_sha256='p',
        system_sha256='s', raw_response_sha256=tag, prompt_version='v', dataset_revision=load.REVISION,
        task_prompt_sha256='t')


@pytest.fixture(scope='module')
def tasks():
    if not EVALUATOR.exists():
        pytest.skip('pinned source missing')
    _, private = load.load_tasks(load.DEFAULT_SOURCE, load.DEFAULT_PILOT)
    return {t.task_id: t for t in private}


def mock_checker(results, allow=()):
    calls = []
    def evaluate(code, task):
        calls.append(task.task_id)
        outcome = results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {'official': outcome, 'evaluate_elapsed_s': 0.1, 'supervisor_elapsed_s': 0.2}
    cfg = oracle.CheckerConfig('unused', backend='mock')
    return oracle.OfficialChecker(cfg, EVALUATOR, MOCK_ENV, allowlist=set(allow), mock=evaluate), calls


OFFICIAL_FAIL = {'passed': False, 'syntax_valid': True, 'test_passed': False, 'error_message': 'Timeout after 30.0s',
                 'error_type': 'Timeout', 'stdout': None}


@needs_source
def test_intent_then_result_and_completed_result_is_reused(tmp_path, tasks):
    checker, calls = mock_checker([dict(OFFICIAL_FAIL)])
    store = oracle.OfficialStore(tmp_path/'official.jsonl')
    c = candidate('TruthTables/5', 'def tt_or(a, b):\n    while True: pass\n')
    first = oracle.score_candidates([c], tasks, checker, store, operation_id='op')
    assert first['new_official'] == 1
    events = store.events()
    assert [e['event'] for e in events] == ['intent', 'result']
    assert events[1]['official'] == OFFICIAL_FAIL and events[1]['label'] == 0  # official Timeout keeps passed=False
    again = oracle.score_candidates([c], tasks, checker, store, operation_id='op2')
    assert again['already_official'] == 1 and calls == ['TruthTables/5']


@needs_source
def test_service_error_and_lost_result_are_unresolved_and_never_rerun(tmp_path, tasks):
    checker, calls = mock_checker([oracle.ServiceError('supervisor_timeout')])
    store = oracle.OfficialStore(tmp_path/'official.jsonl')
    c1 = candidate('TruthTables/5', 'x = 1\n', 'a')
    assert oracle.score_candidates([c1], tasks, checker, store, operation_id='op')['new_service_errors'] == 1
    state = next(iter(store.states().values()))
    assert state['status'] == 'unresolved_service_error' and 'label' not in state['terminal']
    # A crash after the intent was written leaves no terminal event: unresolved, not re-drawn.
    c2 = candidate('CHSHGame/2.4', 'y = 2\n', 'b')
    key = oracle.official_key(tasks['CHSHGame/2.4'], c2, checker.environment_id)
    from vgx.common.storage import append_jsonl
    append_jsonl(store.path, {'official_key': key, 'event': 'intent', 'candidate_id': c2.candidate_id})
    summary = oracle.score_candidates([c1, c2], tasks, checker, store, operation_id='op2')
    assert summary['already_unresolved'] == 2 and summary['new_official'] == 0 and calls == ['TruthTables/5']
    assert {s['status'] for s in store.states().values()} == {'unresolved_service_error', 'unresolved_lost_result'}
    assert store.results() == {}


@needs_source
def test_unreviewed_code_is_refused_before_any_intent(tmp_path, tasks):
    cfg = oracle.CheckerConfig(str(INTERPRETER), backend='reviewed_local')
    checker = oracle.OfficialChecker(cfg, EVALUATOR, MOCK_ENV, allowlist=set())
    store = oracle.OfficialStore(tmp_path/'official.jsonl')
    summary = oracle.score_candidates([candidate('TruthTables/5', 'import os\n')], tasks, checker, store, operation_id='op')
    assert summary['refused_unreviewed'] == 1 and not store.path.exists()
    with pytest.raises(ValueError):
        oracle.CheckerConfig('x', backend='reviewed_local', model_code_execution_approved=True)
    with pytest.raises(ValueError):
        oracle.CheckerConfig('x', supervisor_timeout_s=40)


@needs_source
def test_environment_identity_binds_the_official_key(tasks):
    c = candidate('TruthTables/5', 'x = 1\n')
    t = tasks['TruthTables/5']
    assert oracle.official_key(t, c, 'env-a') != oracle.official_key(t, c, 'env-b')
    assert oracle.official_key(t, c, 'env-a') != oracle.official_key(t, candidate('TruthTables/5', 'x = 2\n'), 'env-a')


def direct_official(code, task):
    """Call the pinned evaluate_solution directly, outside the adapter."""
    script = ('import json, sys, dataclasses\nsys.path.insert(0, sys.argv[1])\nimport evaluator\n'
              'job = json.load(sys.stdin)\n'
              'r = evaluator.evaluate_solution(job["code"], job["test"], job["entry_point"], timeout=30.0)\n'
              'print("RESULT" + json.dumps(dataclasses.asdict(r)))\n')
    process = subprocess.run([str(INTERPRETER.absolute()), '-c', script, str(EVALUATOR.parent.absolute())],
                             input=json.dumps({'code': code, 'test': task.test, 'entry_point': task.entry_point}),
                             capture_output=True, text=True, timeout=120)
    line = [x for x in process.stdout.splitlines() if x.startswith('RESULT')]
    assert line, process.stderr
    return json.loads(line[-1][6:])


@pytest.fixture(scope='module')
def real_checker():
    if not INTERPRETER.exists():
        pytest.skip('quantum interpreter missing')
    env = oracle.probe_environment(str(INTERPRETER))
    oracle.check_requirements(env, load.DEFAULT_REQUIREMENTS)
    return env


DETERMINISTIC = [
    ('TruthTables/5', None),  # reference
    ('TruthTables/5', 'class TruthTable:\n    pass\n\ndef tt_or(tt1, tt2):\n    return None\n'),
    ('TruthTables/5', 'def tt_or(tt1, tt2:\n    return tt1\n'),
    ('TruthTables/5', 'def something_else():\n    return 1\n'),
    ('SolveSATWithGrover/3.1', 'def test_grovers_with_sat():\n    return True\n'),
    ('BasicGates/2.5', None),
    ('Measurements/1.13', 'MARKDOWN'),
]


@needs_qiskit
@pytest.mark.parametrize('backend', ['reviewed_local'] + (['macos_sandbox'] if sys.platform == 'darwin' else []))
def test_adapter_matches_direct_official_evaluate_solution(tasks, real_checker, backend):
    codes = []
    for tid, code in DETERMINISTIC:
        reference = tasks[tid].canonical_solution
        codes.append((tid, reference if code is None else f'```python\n{reference}\n```' if code == 'MARKDOWN' else code))
    checker = oracle.OfficialChecker(oracle.CheckerConfig(str(INTERPRETER), backend=backend), EVALUATOR, real_checker,
                                     allowlist={load.sha256_text(c) for _, c in codes})
    for tid, code in codes:
        adapted = checker.run(code, tasks[tid])['official']
        direct = direct_official(code, tasks[tid])
        assert {k: adapted[k] for k in ('passed', 'syntax_valid', 'test_passed', 'error_type')} == \
               {k: direct[k] for k in ('passed', 'syntax_valid', 'test_passed', 'error_type')}, tid
    sat = checker.run('def test_grovers_with_sat():\n    return True\n', tasks['SolveSATWithGrover/3.1'])['official']
    assert sat['passed'] is True  # the official checker accepts it; it stays a pass


@needs_qiskit
def test_supervisor_timeout_is_a_service_error_not_official_timeout(tasks, real_checker):
    cfg = oracle.CheckerConfig(str(INTERPRETER), backend='reviewed_local')
    object.__setattr__(cfg, 'supervisor_timeout_s', 3.0)  # test-only: shorter than the worker's own startup
    checker = oracle.OfficialChecker(cfg, EVALUATOR, real_checker, allowlist=set())
    slow = 'import time\ntime.sleep(20)\ndef tt_or(a, b):\n    return a\n'
    with pytest.raises(oracle.ServiceError) as error:
        checker.run(slow, tasks['TruthTables/5'])
    assert error.value.reason == 'supervisor_timeout'


@needs_qiskit
@pytest.mark.skipif(sys.platform != 'darwin', reason='sandbox-exec is macOS only')
def test_sandbox_denies_network_home_and_repository_reads(tasks, real_checker):
    checker = oracle.OfficialChecker(oracle.CheckerConfig(str(INTERPRETER), backend='macos_sandbox'), EVALUATOR,
                                     real_checker, allowlist=set())
    probes = {
        'network': 'import socket\nsocket.create_connection(("1.1.1.1", 443), timeout=3)\n',
        'home': f'open({str(Path.home()/".zshrc")!r}).read()\n',
        'repo': f'open({str(Path("pyproject.toml").resolve())!r}).read()\n',
        'write': 'open("/tmp/vgx-qkatas-sandbox-probe", "w").write("x")\n',
    }
    for name, body in probes.items():
        code = body + 'def tt_or(a, b):\n    return a\n'
        checker.allowlist.add(load.sha256_text(code))
        official = checker.run(code, tasks['TruthTables/5'])['official']
        assert official['passed'] is False and official['error_type'] in ('PermissionError', 'FileNotFoundError'), (name, official)
