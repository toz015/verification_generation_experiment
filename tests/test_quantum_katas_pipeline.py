"""Mock end-to-end flow, identity/information boundaries, resume, and sequential access."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import pytest

from vgx.common.api import RequestCache
from vgx.common.storage import read_jsonl
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.score import VerifierLikelihood
from vgx.quantum_katas import dryrun, load, run_pilot, sequential, workflow
from vgx.quantum_katas.artifacts import CandidateStore, FrozenCodeCandidate, candidate_identity, verifier_view
from vgx.quantum_katas.oracle import OfficialStore
from vgx.quantum_katas.runners import MockRunner, ModelSpec

pytestmark = pytest.mark.skipif(not (load.DEFAULT_SOURCE/'dataset/qiskit_quantumkatas.jsonl').exists()
                                or not dryrun.AUDIT_CONTROLS.exists(), reason='pinned source or audit fixtures missing')


@pytest.fixture(scope='module')
def dry(tmp_path_factory):
    root = tmp_path_factory.mktemp('qk')/'dry'
    steps = workflow.dry_run(root, checker='mock')
    return root/'run', steps


def test_end_to_end_mock_flow_and_denominators(dry):
    run_dir, steps = dry
    report = json.loads((run_dir/'report.json').read_text())
    g, c = report['generator'], report['official_checker']
    assert g['planned_tasks'] == 28 and g['generated'] == 28
    assert g['format_failures'] == {'invalid_json': 1, 'truncated_output': 1}
    assert g['code_extractable'] == 26 and g['confidence_valid'] == 24
    assert c['official_returns'] == 26 and c['end_to_end_pass_rate']['denominator_all_planned_tasks'] == 28
    assert report['samples']['verifiers']['verifier_c']['failures'] == {'confidence_non_numeric': 1, 'duplicate_json_key': 1}
    # Resume: no new provider executions and no second official evaluation.
    assert steps['collect_generators_resume_offline']['new_provider_executions'] == 0
    assert steps['collect_verifiers_resume_offline']['new_provider_executions'] == 0
    assert steps['score_official_resume']['new_official'] == 0 and steps['score_official_resume']['already_official'] == 26
    assert steps['run_sequential']['new_provider_executions'] == 0
    assert steps['run_sequential']['replay_agreement'] == {'n': 28, 'agree': 28}
    assert sum(e['event'] == 'intent' for e in OfficialStore(run_dir/'private/official_results.jsonl').events()) == 26


def test_invalid_r_and_no_code_abstain_without_queries(dry):
    run_dir, _ = dry
    results = {r['task_id']: r for r in json.loads((run_dir/'sequential/decisions.json').read_text())['results']}
    for tid in ('RippleCarryAdder/3.3', 'Superposition/1.5'):
        assert results[tid]['failure'] == 'invalid_generator_confidence'
        assert results[tid]['verifiers_used'] == 0 and results[tid]['prior'] is None
    for tid in ('QFT/1.3', 'GHZGame/2.2'):
        assert results[tid]['failure'] == 'no_extractable_code' and results[tid]['verifiers_used'] == 0
    # Invalid-r code still has its official label.
    labeled = OfficialStore(run_dir/'private/official_results.jsonl').results()
    candidates = CandidateStore(run_dir/'frozen_candidates.jsonl').by_task()
    assert candidates['RippleCarryAdder/3.3'].candidate_id in labeled


def test_verifier_prompt_ignores_r_model_and_labels_but_tracks_code(dry):
    run_dir, _ = dry
    run = run_pilot.Run(run_dir)
    candidate = run.candidates.by_task()['BoundedKnapsack/1.4']
    task = run.task_map()[candidate.task_id]
    spec = run.verifiers[0]
    base = run_pilot.verifier_request(spec, candidate, task)
    other = replace(candidate, raw_p_correct=0.01, model_identity={'model': 'other'}, provider_execution_id='x',
                    candidate_id=candidate.candidate_id)
    assert run_pilot.verifier_request(spec, other, task).prompt == base.prompt
    assert set(verifier_view(candidate, task)) == {'task_id', 'prompt', 'entry_point', 'required_functions', 'code'}
    code = candidate.raw_code + '\n# changed\n'
    changed_id = candidate_identity(candidate.task_id, 'g0', candidate.request_key, candidate.raw_response_sha256,
                                    load.sha256_text(code))
    changed = replace(candidate, raw_code=code, code_sha256=load.sha256_text(code), candidate_id=changed_id)
    assert changed_id != candidate.candidate_id
    runner = run.runner(spec)
    assert runner.cache_key(run_pilot.verifier_request(spec, changed, task)) != runner.cache_key(base)
    # Request identity ignores metadata such as operation IDs or labels.
    assert runner.cache_key(replace(base, meta={**base.meta, 'label': 1})) == runner.cache_key(base)
    with pytest.raises(ValueError):
        replace(candidate, raw_p_correct=0.5, confidence_status='valid', candidate_id='forged')


def test_frozen_candidate_cannot_be_revised(dry):
    run_dir, _ = dry
    store = CandidateStore(run_dir/'frozen_candidates.jsonl')
    candidate = store.by_task()['BoundedKnapsack/1.4']
    assert store.freeze(candidate) is False
    with pytest.raises(ValueError, match='cannot be revised'):
        store.freeze(replace(candidate, raw_p_correct=0.1))


def test_collection_and_sequential_never_read_private(dry, monkeypatch):
    run_dir, _ = dry
    private = (run_dir/'private').resolve()
    import builtins, io
    real_open = io.open
    def guarded(path, *a, **k):
        if isinstance(path, (str, os.PathLike)):
            assert private not in Path(path).resolve().parents, f'private file opened: {path}'
        return real_open(path, *a, **k)
    monkeypatch.setattr(io, 'open', guarded)
    monkeypatch.setattr(builtins, 'open', guarded)
    original_read = real_open
    assert run_pilot.collect_generators(run_dir)['new_provider_executions'] == 0
    assert run_pilot.collect_verifiers(run_dir)['new_provider_executions'] == 0
    policy = json.loads(original_read(run_dir/'sequential/policy.json').read())
    assert workflow.run_sequential(run_dir, policy)['new_provider_executions'] == 0
    with pytest.raises(AssertionError, match='private file opened'):  # the guard itself works
        run_pilot.score_official(run_dir, mock=lambda *a: None)


def test_offline_cache_miss_is_reported_and_never_sent(dry, tmp_path, monkeypatch):
    run_dir, _ = dry
    copy = tmp_path/'copy'
    import shutil
    shutil.copytree(run_dir, copy)
    shutil.rmtree(copy/'cache')
    (copy/'frozen_candidates.jsonl').unlink()
    monkeypatch.setattr(MockRunner, 'run', lambda *a, **k: pytest.fail('offline collection sent a request'))
    status = run_pilot.collect_generators(copy)
    assert status['missing_cache'] == 28 and status['frozen_new'] == 0 and status['stage'] == 'complete'


def test_real_provider_needs_budget_and_authorization(tmp_path):
    cfg = dryrun.config('mock')
    cfg['models'] = {'generator': {'id': 'generator', 'provider': 'vertex_openai', 'model': 'google/example',
                                   'location': 'global', 'project': 'p', 'generation': {'max_tokens': 64}},
                     'verifiers': [{'id': f'v{i}', 'provider': 'vertex_openai', 'model': f'google/example-{i}',
                                    'location': 'global', 'project': 'p', 'generation': {'max_tokens': 64}}
                                   for i in range(3)]}
    cfg.pop('budget')
    path = tmp_path/'cfg.json'
    path.write_text(json.dumps(cfg))
    public, private = load.load_tasks(cfg['source'], cfg['pilot'])
    env, _ = workflow.mock_checker_parts(private, [])
    run_pilot.prepare(path, tmp_path/'run', mock_environment=env)
    with pytest.raises(ValueError, match='paid collection'):
        run_pilot.collect_generators(tmp_path/'run', allow_api=True, limit=1)
    status = run_pilot.collect_generators(tmp_path/'run', limit=1)  # offline: no auth, no network
    assert status['missing_cache'] == 1


def test_bound_implementation_or_task_changes_block_resume(dry, tmp_path):
    run_dir, _ = dry
    import shutil
    copy = tmp_path/'tampered'
    shutil.copytree(run_dir, copy)
    rows = (copy/'public_tasks.jsonl').read_text().replace('Task', 'Tusk', 1)
    (copy/'public_tasks.jsonl').write_text(rows)
    with pytest.raises(ValueError, match='public task file changed'):
        run_pilot.Run(copy)


def _candidate(p=0.7, code='def f():\n    return 1\n'):
    sha = load.sha256_text(code) if code else None
    kwargs = dict(task_id='T/1', generation_id='g0', partition='development', raw_code=code, code_sha256=sha,
                  raw_p_correct=p, generator_response_status='code_extracted' if code else 'format_failure',
                  generator_failure=None if code else 'invalid_json',
                  confidence_status='valid' if p is not None else ('missing' if code else 'not_applicable'),
                  finish_status='complete', model_identity={}, request_key='k', provider_execution_id=None,
                  prompt_sha256='p', system_sha256='s', raw_response_sha256='r', prompt_version='v',
                  dataset_revision=load.REVISION, task_prompt_sha256='t')
    return FrozenCodeCandidate(candidate_id=candidate_identity('T/1', 'g0', 'k', 'r', sha), **kwargs)


def _policy(costs=(.01, .01, .01)):
    specs = [ModelSpec(f'v{i}', 'mock', f'mock/v{i}', generation={'max_tokens': 8}) for i in range(3)]
    informative = VerifierLikelihood((0., .5, 1.), (.1, .9), (.9, .1))
    planner = NestedPlanner([informative]*3, list(costs), 1., 4., grid_size=201)
    return sequential.make_policy(plan_id='x', verifiers=specs, planner=planner, kind='synthetic_engineering_dry_run')


@pytest.mark.parametrize('prior,scores', [(.7, [.9, .9, .9]), (.7, [.1, .9, .2]), (.95, [.1, .1, .1]),
                                          (.3, [.9, .9, .9]), (.7, [None, .9, .9])])
def test_live_matches_replay_and_never_reads_future_signals(tmp_path, prior, scores):
    policy = _policy()
    candidate = _candidate(prior)
    asked = []
    def query(spec, c):
        assert c is candidate
        asked.append(spec.id)
        score = scores[len(asked)-1]
        return {'score': score, 'failure': None if score is not None else 'duplicate_json_key',
                'request_key': spec.id, 'used_cached_response': True}
    live = sequential.execute_candidate(candidate, policy, query, tmp_path/'state.json')
    replayed = sequential.replay(candidate, policy, scores)
    assert live['action'] == replayed['action'] and live['verifiers_used'] == replayed['verifiers_used']
    assert replayed['accessed_layers'] == list(range(live['verifiers_used'])) == list(range(len(asked)))
    if scores[0] is None:
        assert live['action'] == 'abstain' and live['verifiers_used'] == 1
    again = sequential.execute_candidate(candidate, policy, lambda *a: pytest.fail('resumed decision queried again'),
                                         tmp_path/'state.json')
    assert again == live


def test_interrupted_query_resumes_from_checkpoint_without_repeating(tmp_path):
    policy = _policy()
    candidate = _candidate(.7)
    calls = []
    def flaky(spec, c):
        calls.append(spec.id)
        if len(calls) == 2:
            raise RuntimeError('transport interrupted')
        return {'score': .9, 'failure': None, 'request_key': spec.id, 'used_cached_response': False}
    with pytest.raises(RuntimeError):
        sequential.execute_candidate(candidate, policy, flaky, tmp_path/'s.json')
    resumed = []
    def query(spec, c):
        resumed.append(spec.id)
        return {'score': .9, 'failure': None, 'request_key': spec.id, 'used_cached_response': True}
    result = sequential.execute_candidate(candidate, policy, query, tmp_path/'s.json')
    assert resumed[0] == 'v1'  # v0 observation came from the checkpoint
    assert result['verifiers_used'] == 1 + len(resumed)


def test_no_prior_means_abstain_without_queries(tmp_path):
    policy = _policy()
    for candidate, failure in ((_candidate(None), 'invalid_generator_confidence'),
                               (_candidate(None, code=None), 'no_extractable_code')):
        result = sequential.execute_candidate(candidate, policy, lambda *a: pytest.fail('queried'), tmp_path/f'{failure}.json')
        assert result['action'] == 'abstain' and result['failure'] == failure and result['prior'] is None


def test_http_200_throttle_body_is_recorded_as_429_not_a_cached_output(tmp_path, monkeypatch):
    import requests
    from vgx.common.api import ProviderRejectedError
    from vgx.common.llm import CallLog, Request
    runner = ModelSpec('q', 'vertex_openai', 'qwen/qwen3-235b-a22b-instruct-2507-maas', 'us-south1', 'p',
                       {'max_tokens': 64, 'response_format': {'type': 'json_object'}}).runner()

    class Response:
        status_code = 200

        def json(self):
            return {'error': 'Prediction failed: request is throttled due to too many requests.', 'id': 'x'}

    class Session:
        def post(self, *a, **k):
            return Response()

        def close(self):
            pass
    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr('vgx.common.vertex._get_adc_token', lambda project: 'token')
    log = CallLog(tmp_path/'calls.jsonl')
    with pytest.raises(ProviderRejectedError) as error:
        runner.run([Request('k', 'prompt', None)], log)
    assert error.value.status == 429 and log.responses() == {}
