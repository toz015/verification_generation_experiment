from dataclasses import replace
import json
from types import SimpleNamespace
import sys

import pytest

from vgx.common.api import RequestCache
from vgx.common.jev import JevRunner
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.verifiers import VerifierSpec
from vgx.gpqa.workflow import prepare_comparison


def spec(mode='choice'):
    return VerifierSpec('jev_'+mode,'typesafe','jev-1.13.0',mode=mode,prompt_version='gpqa_jev_v1')


def test_choice_extracts_candidate_probability_not_argmax_or_confidence(frozen_run):
    _, candidates = load_candidates(frozen_run.bundle)
    candidate = replace(candidates[0],answer='A')
    response = {'model':'jev-1.13.0','answers':{'verification':{'type':'choice','choice':'B','confidence':.8,
                'probabilities':{'A':.1,'B':.85,'C':.03,'D':.02}}}}
    assert spec().signal(json.dumps(response),candidate) == (.1,None)
    response['answers']['verification']['probabilities']['A'] = .9
    assert spec().signal(json.dumps(response),candidate)[0] is None


def test_noul_uses_yes_probability_and_checks_revision(frozen_run):
    _, candidates = load_candidates(frozen_run.bundle)
    response = {'model':'jev-1.13.0','answers':{'verification':{'type':'noul','noul':.25}}}
    assert spec('noul').signal(json.dumps(response),candidates[0]) == (.25,None)
    response['model'] = 'jev-latest'
    assert spec('noul').signal(json.dumps(response),candidates[0]) == (None,'model_revision_mismatch')
    with pytest.raises(ValueError,match='pin'): JevRunner('jev-latest')


def test_requests_hide_generator_confidence_and_gold_and_preserve_order(frozen_run):
    _, candidates = load_candidates(frozen_run.bundle)
    candidate = candidates[0]
    for mode in ('choice','noul'):
        request = spec(mode).request(candidate)
        body = json.loads(request.prompt)
        assert body['state']['options'] == dict(zip('ABCD',candidate.choices))
        assert 'p_correct' not in request.prompt and 'correct_index' not in request.prompt
        assert len(body['questions']) == 1 and request.system is None
        if mode == 'noul': assert body['state']['fixed_candidate']['option'] == candidate.answer
        else: assert 'fixed_candidate' not in body['state']


def test_mock_transport_and_shared_cache(frozen_run, monkeypatch):
    _, candidates = load_candidates(frozen_run.bundle)
    candidate = candidates[0]
    runner = spec('noul').runner()
    request = spec('noul').request(candidate)
    monkeypatch.setenv('TYPESAFE_API_KEY','not-logged-test-secret')
    seen = []
    class Session:
        def post(self,url,**kwargs):
            seen.append((url,kwargs))
            return SimpleNamespace(status_code=200,json=lambda:{'model':'jev-1.13.0',
                'answers':{'verification':{'type':'noul','noul':.8}},'usage':{'input_tokens':20,'output_tokens':5}})
    monkeypatch.setitem(sys.modules,'requests',SimpleNamespace(Session=Session))
    first,cached = frozen_run.cache.get(runner,request,allow_api=True)
    second,cached = frozen_run.cache.get(runner,request)
    assert cached and first == second and len(seen) == 1
    assert seen[0][1]['json']['model'] == 'jev-1.13.0'
    for p in (frozen_run.cache.root/'calls').glob('*.jsonl'):
        assert 'not-logged-test-secret' not in p.read_text()


def test_comparison_does_not_chain_jev_modes_as_independent(frozen_run):
    from dataclasses import asdict
    config = {'schema':1,'verifiers':[asdict(spec()),asdict(spec('noul'))],
              'arms':[{'name':'bad','order':['jev_choice','jev_noul']}]}
    with pytest.raises(ValueError,match='separate arms'):
        prepare_comparison(frozen_run.bundle,config,frozen_run.cache)
