"""Synthetic checks only: no dataset access or paid model requests."""
from copy import deepcopy
import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from vgx.common.api import execute_request, RequestCache
from vgx.common.budget import BudgetedRequestCache, BudgetLimitError
from vgx.common.llm import Request
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import fresh


@pytest.fixture
def fresh_data(tmp_path):
    source=tmp_path/'source'
    source.mkdir()
    fields=['Question','Correct Answer','Incorrect Answer 1','Incorrect Answer 2','Incorrect Answer 3','High-level domain']
    for subset,n in [('extended',24),('main',20),('diamond',8)]:
        with (source/f'gpqa_{subset}.csv').open('w') as stream:
            writer=csv.writer(stream)
            writer.writerow(fields)
            for i in range(n):
                writer.writerow([f'Synthetic question {i}',f'right {i}',f'wrong a {i}',f'wrong b {i}',f'wrong c {i}','Physics' if i%2 else 'Biology'])
    config={'seed':123,'models':{'generator':'google/gemini-3.8-flash','verifiers':['m1','m2']},
            'generation':{'max_tokens':32,'temperature':0.,'top_p':1.,'reasoning_effort':'low'},
            'inference':{'project_id':'project','model_locations':{'google/gemini-3.8-flash':'global','m1':'us','m2':'us'}},
            'api_pricing':{'usd_per_million_tokens':{m:{'input':1.,'output':1.} for m in ['google/gemini-3.8-flash','m1','m2']}},
            'budget_usd':200.,'sample_size':20,'calibration_size':12,
            'policy_scenarios':[{'name':'primary','correct_reward':1.,'incorrect_loss':4.,'utility_per_usd':10.}],
            'protocol':{'primary_scenario':'primary','primary_order':['verifier_1','verifier_2'],'assumptions':[]}}
    root=tmp_path/'run'
    return source,config,root


@pytest.fixture
def fake_vertex(monkeypatch):
    calls=[]
    def run(self, requests, log):
        for request in requests:
            calls.append((self.model,request))
            text='{"answer":"B","p_correct":0.7}' if request.meta['role']=='generator' else '{"p_correct":0.8}'
            result={'model':self.model,'choices':[{'message':{'content':text}}],
                    'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}
            execute_request(self,request,log,lambda:(200,result))
    monkeypatch.setattr(VertexBatchRunner,'run',run)
    return calls


def test_subset_difference_public_keys_and_reproducibility(fresh_data):
    source,config,root=fresh_data
    first=fresh.prepare(source,'a'*40,config,root)
    assert fresh.prepare(source,'a'*40,config,root)==first
    assert first['counts']=={'calibration':12,'evaluation':8}
    _,items=fresh.inputs(root)
    assert not any('correct' in key for item in items for key in vars(item))
    assert {i.item_id for i in items if i.partition=='calibration'}.isdisjoint(i.item_id for i in items if i.partition=='evaluation')
    with pytest.raises(ValueError,match='refusing to change'):
        fresh.prepare(source,'a'*40,{**config,'seed':999},root)


def test_normalized_duplicate_questions_rejected(fresh_data):
    source,config,root=fresh_data
    path=source/'gpqa_main.csv'
    rows=list(csv.DictReader(path.open()))
    rows[1]['Question']='  Synthetic   question 0  '
    with path.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='duplicate normalized'):
        fresh.prepare(source,'a'*40,config,root)


def test_conflicting_overlapping_answer_keys_rejected(fresh_data):
    source,config,root=fresh_data
    path=source/'gpqa_diamond.csv'
    rows=list(csv.DictReader(path.open()))
    rows[0]['Correct Answer'],rows[0]['Incorrect Answer 1']=rows[0]['Incorrect Answer 1'],rows[0]['Correct Answer']
    with path.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='different options or answer keys'):
        fresh.prepare(source,'a'*40,config,root)


def test_new_verifiers_and_prices_do_not_repeat_generator(fresh_data,fake_vertex):
    source,config,root=fresh_data
    fresh.prepare(source,'a'*40,config,root)
    cache=RequestCache(root/'cache')
    original=fresh.collect_generators(root,config,cache,allow_api=True)
    assert len(fake_vertex)==20
    changed=deepcopy(config)
    changed['models']['verifiers'].append('another-verifier')
    changed['api_pricing']['arbitrary_metadata']='changed'
    assert fresh.collect_generators(root,changed,cache)==original
    assert len(fake_vertex)==20
    fresh.freeze(root,config,cache)
    from vgx.gpqa.artifacts import load_candidates
    manifest,candidates=load_candidates(root/'frozen')
    assert len(candidates)==20 and manifest['origin']=='fresh_huggingface'


def test_generator_collection_never_opens_answer_keys(fresh_data,fake_vertex,monkeypatch):
    source,config,root=fresh_data
    fresh.prepare(source,'a'*40,config,root)
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.name!='labels.jsonl'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    fresh.collect_generators(root,config,RequestCache(root/'cache'),allow_api=True)


def test_budget_refuses_wrong_project_and_reserves_before_call(tmp_path,fake_vertex):
    pricing={'usd_per_million_tokens':{'m':{'input':1.,'output':1.}}}
    cache=BudgetedRequestCache(tmp_path,project='project',pricing=pricing,limit_usd=.00001)
    request=Request('r','prompt',meta={'role':'generator'})
    with pytest.raises(ValueError,match='authorized project'):
        cache.get(VertexBatchRunner('m','global','wrong'),request,allow_api=True)
    with pytest.raises(BudgetLimitError,match='insufficient budget'):
        cache.get(VertexBatchRunner('m','global','project'),request,allow_api=True)
    assert not fake_vertex


def test_budget_resume_counts_execution_once(tmp_path,fake_vertex):
    pricing={'usd_per_million_tokens':{'m':{'input':1.,'output':1.}}}
    cache=BudgetedRequestCache(tmp_path,project='project',pricing=pricing,limit_usd=1.)
    request=Request('r','prompt',meta={'role':'generator'})
    runner=VertexBatchRunner('m','global','project')
    cache.get(runner,request,allow_api=True)
    cache.get(runner,replace(request,key='new-label'),allow_api=True)
    assert cache.summary()['successful_priced_calls']==1
    assert cache.summary()['estimated_usd']==pytest.approx(.00011)
    assert len(fake_vertex)==1


def test_fresh_complete_pipeline_and_resume(fresh_data,fake_vertex,monkeypatch):
    source,config,root=fresh_data
    fresh.prepare(source,'a'*40,config,root)
    from vgx.gpqa.experiment import run
    from vgx.gpqa import report
    original=report.build_report
    monkeypatch.setattr(report,'build_report',lambda *a,**k:original(*a,**k,bootstrap_repeats=3))
    result=run(root,config,allow_api=True,workers=2)
    assert result['fresh_protocol']['live_replay_matched_n']==8
    assert len(fake_vertex)==60
    assert json.loads((root/'progress.json').read_text())['phase']=='complete'
    result2=run(root,config,allow_api=False,workers=2)
    assert result2['fresh_protocol']['primary_live']==result['fresh_protocol']['primary_live']
    assert len(fake_vertex)==60
    live=json.loads((root/'live'/'primary'/'execution_manifest.json').read_text())
    plan=root/'analysis_plan.json'
    assert plan.is_file() and live['binding']['partition']=='evaluation'


def test_vertex_request_billing_project_is_explicit(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace
    from vgx.common.llm import CallLog
    sent=[]
    class Session:
        def post(self,url,headers,json,timeout):
            sent.append((url,headers))
            return SimpleNamespace(status_code=200,json=lambda:{'choices':[{'message':{'content':'ok'}}]})
        def close(self):
            pass
    monkeypatch.setitem(sys.modules,'requests',SimpleNamespace(Session=Session))
    monkeypatch.setattr('vgx.common.vertex._get_adc_token',lambda project:'local-test-token')
    runner=VertexBatchRunner('m','global','authorized-project')
    runner.run([Request('r','prompt')],CallLog(tmp_path/'call.jsonl'))
    assert '/projects/authorized-project/' in sent[0][0]
    assert sent[0][1]['x-goog-user-project']=='authorized-project'
    row=list(CallLog(tmp_path/'call.jsonl').records())[0]
    assert row['meta']['resource_project']==row['meta']['quota_project']=='authorized-project'


def test_budget_retries_only_explicit_429_rejections(monkeypatch,tmp_path):
    pricing={'usd_per_million_tokens':{'m':{'input':1.,'output':1.}}}
    cache=BudgetedRequestCache(tmp_path,project='project',pricing=pricing,limit_usd=1.)
    runner=VertexBatchRunner('m','global','project')
    calls=[]
    sleeps=[]
    def fake(requests,log):
        calls.append(1)
        execute_request(runner,requests[0],log,lambda:(429,{'error':{'status':'RESOURCE_EXHAUSTED'}}) if len(calls)==1
                        else (200,{'choices':[{'message':{'content':'ok'}}],
                                   'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}))
    monkeypatch.setattr(runner,'run',fake)
    monkeypatch.setattr('vgx.common.budget.time.sleep',sleeps.append)
    request=Request('r','prompt',meta={'role':'generator'})
    cache.get(runner,request,allow_api=True)
    assert calls==[1,1] and sleeps==[5]
    assert cache.summary()['estimated_usd']==pytest.approx(.00011)
    assert len(list(cache.log(runner.cache_key(request)).records()))==2


def test_budget_preserves_unknown_outcome_without_retry(monkeypatch,tmp_path):
    from vgx.common.api import UncertainRequestError
    pricing={'usd_per_million_tokens':{'m':{'input':1.,'output':1.}}}
    cache=BudgetedRequestCache(tmp_path,project='project',pricing=pricing,limit_usd=1.)
    runner=VertexBatchRunner('m','global','project')
    calls=[]
    def fake(requests,log):
        calls.append(1)
        def fail():
            raise TimeoutError()
        execute_request(runner,requests[0],log,fail)
    monkeypatch.setattr(runner,'run',fake)
    request=Request('r','prompt',meta={'role':'generator'})
    with pytest.raises(UncertainRequestError):
        cache.get(runner,request,allow_api=True)
    with pytest.raises(BudgetLimitError,match='unresolved'):
        cache.get(runner,replace(request,prompt='another prompt'),allow_api=True)
    assert calls==[1] and cache.summary()['outstanding_reserved_usd']>0


def test_bounded_submission_stops_after_first_failure():
    from vgx.common.concurrency import parallel_map
    calls=[]
    def fail(item):
        calls.append(item)
        raise ValueError('stop')
    with pytest.raises(ValueError,match='stop'):
        parallel_map(fail,list(range(100)),workers=2)
    assert len(calls)<=2
