from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from vgx.common.api import RequestCache
from vgx.common.storage import atomic_json, digest
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.score import VerifierLikelihood, simulate_sequential_decision
from vgx.gpqa.sequential import execute_bundle, execute_candidate, policy_parts, seal
from vgx.gpqa.workflow import prepare_policy, score_execution


def make_policy(run, likelihoods=None, costs=None, reward=1., loss=1.):
    models = likelihoods or [VerifierLikelihood((0.,.5,1.),(0.,1.),(1.,0.))]*2
    planner = NestedPlanner(models, [.01, .01] if costs is None else costs, reward, loss, grid_size=101)
    return seal({'schema':1,'bundle_id':run.manifest['bundle_id'],
                 'verifiers':[asdict(s) for s in run.specs],
                 'verifier_identities':[s.identity for s in run.specs],
                 'planner':planner.to_dict(),'cost_mapping':{'kind':'normalized_sensitivity'}}, 'policy_id')


@pytest.mark.parametrize('prior,score',[(.1,0.),(.1,1.),(.5,0.),(.5,1.),(.9,0.),(.9,1.),(0.,1.),(1.,0.)])
def test_live_replay_consistency_and_no_future_signal_access(frozen_run, prior, score):
    run = frozen_run
    _, candidates = load_candidates(run.bundle)
    candidate = replace(candidates[0],p_correct=prior)
    policy = make_policy(run)
    requested = []
    def query(spec, c):
        assert not hasattr(c,'correct_index') and not hasattr(c,'verifiers')
        requested.append(spec.id)
        assert spec.id == run.specs[0].id  # perfect first observation makes second unnecessary
        return {'score':score,'failure':None,'request_key':'synthetic'}
    live = execute_candidate(candidate,policy,query,run.root/'state.json')
    class Scores:
        def __len__(self): return 2
        def __getitem__(self,index):
            assert index == 0, 'future signal inspected'
            return score
    _, planner = policy_parts(policy)
    replay = planner.replay(prior,Scores())
    for key in ('action','posterior','verifiers_used','failure','trace'):
        assert live[key] == replay[key]
    assert len(requested) == live['verifiers_used']
    resumed = execute_candidate(candidate,policy,lambda *a:pytest.fail('completed decision queried again'),run.root/'state.json')
    assert resumed == live


def test_stop_tie_terminal_threshold_and_no_need_for_verifiers():
    model = VerifierLikelihood((0.,.5,1.),(0.,1.),(1.,0.))
    planner = NestedPlanner([model],[.5],1,1,grid_size=101)
    assert planner.decide(0,.5)['action'] == 'assert'  # stop == continue == 0
    assert planner.decide(1,.49)['action'] == 'abstain'
    assert planner.decide(1,.5)['action'] == 'assert'
    assert planner.decide(0,1.)['action'] == 'assert'
    assert planner.decide(0,0.)['action'] == 'abstain'
    assert NestedPlanner([],[],1,4).decide(0,.79)['action'] == 'abstain'


def test_impossible_observation_abstains_and_charges_one_call(frozen_run):
    run = frozen_run
    model = VerifierLikelihood((0.,1/3,2/3,1.),(0.,0.,1.),(1.,0.,0.))
    policy = make_policy(run,[model,model])
    _, candidates = load_candidates(run.bundle)
    live = execute_candidate(replace(candidates[0],p_correct=.5),policy,
                             lambda *a:{'score':.5},run.root/'state.json')
    assert live['action'] == 'abstain' and live['verifiers_used'] == 1
    assert live['failure'] == 'impossible_verifier_observation'
    assert live['expected_verification_cost_utility'] == .01
    replay = simulate_sequential_decision(.5,[.5,None],[model,model],[.01,.01],1,1)
    assert replay.action == 'abstain' and replay.failure == live['failure']


def test_missing_score_abstains_and_does_not_query_later(frozen_run):
    run = frozen_run
    _, candidates = load_candidates(run.bundle)
    called = []
    def query(spec,candidate):
        called.append(spec.id)
        return {'score':None,'failure':'missing_verifier_score'}
    result = execute_candidate(candidates[0],make_policy(run),query,run.root/'state.json')
    assert result['action'] == 'abstain' and result['verifiers_used'] == len(called) == 1


def test_resume_after_response_before_checkpoint_does_not_repeat_api(frozen_run, monkeypatch):
    run = frozen_run
    _, candidates = load_candidates(run.bundle)
    candidate, policy = candidates[0], make_policy(run)
    # Existing original response is already durable in the shared cache.
    monkeypatch.setattr('vgx.common.vertex.VertexBatchRunner.run',lambda *a,**k:pytest.fail('must reuse response'))
    def crash_after_response(spec,c):
        run.cache.get(spec.runner(),spec.request(c))
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute_candidate(candidate,policy,crash_after_response,run.root/'state.json')
    assert json.loads((run.root/'state.json').read_text())['pending']['stage'] == 0
    def query(spec,c):
        call,cached = run.cache.get(spec.runner(),spec.request(c))
        assert cached
        score,failure = spec.signal(call.response,c)
        return {'score':score,'failure':failure,'request_key':call.key}
    resumed = execute_candidate(candidate,policy,query,run.root/'state.json')
    assert resumed['verifiers_used'] == 1


def test_resume_rejects_changed_candidate_policy_and_tampered_state(frozen_run):
    run = frozen_run
    _, candidates = load_candidates(run.bundle)
    candidate, policy = candidates[0], make_policy(run)
    path = run.root/'state.json'
    execute_candidate(candidate,policy,lambda *a:{'score':1.},path)
    with pytest.raises(ValueError,match='differs'):
        execute_candidate(replace(candidate,p_correct=.7),policy,lambda *a:None,path)
    with pytest.raises(ValueError,match='differs'):
        execute_candidate(candidate,make_policy(run,costs=[.02,.02]),lambda *a:None,path)
    value = json.loads(path.read_text())
    value['observations'][0]['score'] = 0.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='checksum'):
        execute_candidate(candidate,policy,lambda *a:None,path)


def test_invalid_generator_answer_or_confidence_abstains_without_query(frozen_run):
    run = frozen_run
    _, candidates = load_candidates(run.bundle)
    result = execute_candidate(replace(candidates[0],p_correct=None),make_policy(run),
                               lambda *a:pytest.fail('no query'),run.root/'state.json')
    assert result['action'] == 'abstain' and result['verifiers_used'] == 0


def test_executor_opens_no_label_files_and_reuses_completed_run(frozen_run, monkeypatch):
    run = frozen_run
    policy = make_policy(run)
    original_read = Path.read_text
    def guarded_read(path,*args,**kwargs):
        assert path.name not in ('calibration_records.jsonl','evaluation_labels.jsonl','pilot_records.jsonl')
        return original_read(path,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path,'read_text',guarded_read)
        m.setattr('vgx.common.vertex.VertexBatchRunner.run',lambda *a,**k:pytest.fail('no paid calls'))
        first = execute_bundle(run.bundle,policy,run.cache,run.root/'execution')
        resumed = execute_bundle(run.bundle,policy,run.cache,run.root/'execution')
        assert first == resumed
    scored = score_execution(run.bundle,policy,first)
    assert scored['n'] == len(run.split.evaluation)
    assert scored['mean_queries'] == 1.
    assert scored['accuracy_among_released'] == 1.


def test_tables_roundtrip_and_invalid_parameters():
    model = VerifierLikelihood((0.,.5,1.),(.2,.8),(.8,.2))
    planner = NestedPlanner([model],[.03],1,4,grid_size=101)
    assert NestedPlanner.from_dict(planner.to_dict()).decide(0,.75) == planner.decide(0,.75)
    for cost in (float('nan'),float('inf'),-1,True):
        with pytest.raises(ValueError): NestedPlanner([model],[cost],1,4)
    bad = VerifierLikelihood((0.,.5,1.),(.2,.2),(.8,.2))
    with pytest.raises(ValueError): NestedPlanner([bad],[.01],1,4)
    tables = planner.to_dict()
    tables['Q'][0][0] += .1
    with pytest.raises(ValueError,match='tables'): NestedPlanner.from_dict(tables)


def test_grid_refinement_close_to_exhaustive_one_step():
    model = VerifierLikelihood((0.,.5,1.),(.2,.8),(.8,.2))
    planner = NestedPlanner([model],[.03],1,4,grid_size=10001)
    for prior in (.137,.533,.777,.913):
        expected = -.03
        for p1,p0 in zip(model.p_bin_if_correct,model.p_bin_if_incorrect):
            mass = prior*p1+(1-prior)*p0
            expected += mass*max(0.,5*prior*p1/mass-4)
        assert planner.decide(0,prior)['continuation_value'] == pytest.approx(expected,abs=.001)
