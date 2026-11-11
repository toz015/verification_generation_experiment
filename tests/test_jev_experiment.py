from dataclasses import replace
import json
from pathlib import Path

import pytest

from vgx.common.api import ProviderRejectedError, UncertainRequestError, execute_request
from vgx.common.budget import BudgetLimitError
from vgx.common.jev import JevRunner
from vgx.common.jev_budget import JevBudgetCache, load_jev_key
from vgx.common.llm import Request
from vgx.common.storage import atomic_json, digest
from vgx.gpqa.jev_experiment import ComparisonCache, run


PRICING = {'usd_per_million_tokens':{'jev-1.13.0':{'input':.042,'output':0.}}}


class FakeJev(JevRunner):
    calls = 0
    status = 200
    lost = False

    def run(self, requests, log):
        result = {}
        for request in requests:
            self.calls += 1
            def send():
                if self.lost:
                    raise TimeoutError('synthetic timeout')
                return self.status, {'model':self.model,'usage':{'input_tokens':100,'output_tokens':20},'answers':{}}
            result[request.key] = execute_request(self,request,log,send).response
        return result


def cache(tmp_path, limit=10.):
    return JevBudgetCache(tmp_path,pricing=PRICING,limit_usd=limit,model='jev-1.13.0',account_scope='test')


def test_budget_prices_input_only_and_resume_never_calls(tmp_path):
    runner = FakeJev('jev-1.13.0','test')
    request = Request('one','{}',None,{'role':'verifier'})
    c = cache(tmp_path)
    call, reused = c.get(runner,request,allow_api=True)
    assert not reused and runner.calls==1
    assert c.summary()['estimated_usd']==pytest.approx(.0000042)
    assert cache(tmp_path).get(runner,request)[1]
    assert runner.calls==1
    # Metadata is not part of the paid inference identity.
    assert c.get(runner,replace(request,meta={'role':'verifier','price':'changed'}))[1]
    with pytest.raises(ValueError,match='binding'):
        cache(tmp_path,9.).get(runner,request)


def test_budget_blocks_before_spending_and_wrong_account(tmp_path):
    runner = FakeJev('jev-1.13.0','test')
    request = Request('one','{}',None,{})
    with pytest.raises(BudgetLimitError,match='insufficient'):
        cache(tmp_path,1e-6).get(runner,request,allow_api=True)
    assert runner.calls==0
    with pytest.raises(ValueError,match='authorized'):
        cache(tmp_path).get(FakeJev('jev-1.13.0','another'),request,allow_api=True)


@pytest.mark.parametrize('lost',[True,False])
def test_jev_failures_retain_reservation_and_block_retries(tmp_path,lost):
    runner = FakeJev('jev-1.13.0','test')
    runner.status,runner.lost = 429,lost
    c = cache(tmp_path)
    req = Request('one','{}',None,{})
    with pytest.raises(UncertainRequestError if lost else ProviderRejectedError):
        c.get(runner,req,allow_api=True)
    assert c.summary()['outstanding_reserved_usd']>0
    with pytest.raises(BudgetLimitError,match='unresolved'):
        c.get(runner,req,allow_api=True)
    assert runner.calls==1


def test_credential_loader_does_not_execute_or_print(tmp_path,monkeypatch,capsys):
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    file = tmp_path/'.env'
    file.write_text('UNRELATED=$(do-not-execute)\nexport TYPESAFE_API_KEY="synthetic-secret"\n')
    load_jev_key(file)
    import os
    assert os.environ['TYPESAFE_API_KEY']=='synthetic-secret'
    assert capsys.readouterr().out==''


def test_comparison_never_enables_vertex_calls(tmp_path):
    class Vertex:
        def get(self,runner,request,*,allow_api):
            assert allow_api is False
            return 'cached',True
    class Runner:
        identity={'provider':'vertex_ai'}
    assert ComparisonCache(cache(tmp_path),Vertex()).get(Runner(),None,allow_api=True)==('cached',True)


def test_complete_jev_experiment_and_zero_call_resume(frozen_run,monkeypatch):
    from vgx.gpqa.artifacts import load_candidates
    manifest,_ = load_candidates(frozen_run.bundle)
    rates = {model:{'input':.75,'output':3.75} for model in frozen_run.config['models']['verifiers']}
    manifest['collection_config']['api_pricing']={'usd_per_million_tokens':rates}
    manifest.pop('bundle_id')
    manifest['bundle_id']=digest(manifest)
    atomic_json(frozen_run.bundle/'bundle_manifest.json',manifest)
    atomic_json(frozen_run.root/'report.json',{'synthetic_baseline':True})
    config=json.loads(Path('configs/gpqa_jev_comparison.json').read_text())
    config.update(source_root=str(frozen_run.root),bundle_id=manifest['bundle_id'],pilot_calibration_size=2,bootstrap_repeats=8)
    calls=[]
    def fake_run(self,requests,log):
        outputs={}
        for request in requests:
            calls.append(request.key)
            mode=json.loads(request.prompt)['questions']['verification']['type']
            answer={'type':mode,'noul':.8} if mode=='noul' else {'type':mode,'choice':'A','confidence':.3,'probabilities':dict(zip('ABCD',[.4,.3,.2,.1]))}
            payload={'model':self.model,'answers':{'verification':answer},'usage':{'input_tokens':100,'output_tokens':25}}
            outputs[request.key]=execute_request(self,request,log,lambda:(200,payload)).response
        return outputs
    monkeypatch.setenv('TYPESAFE_API_KEY','synthetic-secret')
    monkeypatch.setattr(JevRunner,'run',fake_run)
    root=frozen_run.root/'jev'
    first=run(root,config,allow_api=True,phase='complete')
    count=len(calls)
    assert count==20 and first['generator_calls']==first['new_vertex_calls']==0
    second=run(root,config,allow_api=False,phase='complete')
    assert first==second and len(calls)==count
    assert all(a['live_replay_matched_n']==4 for a in first['arms'].values())


def test_rounding_sensitivity_is_bounded_and_keeps_frozen_candidate():
    from vgx.gpqa.jev_rounding import rounded_choice
    payload={'model':'jev-1.13.0','answers':{'verification':{'type':'choice','choice':'D','confidence':.9,
             'probabilities':{'A':.28,'B':.09,'C':.06,'D':.56}}}}
    assert rounded_choice(payload,'A','jev-1.13.0')==(.28,True)
    assert rounded_choice(payload,'A','jev-1.13.0',normalize=True)[0]==pytest.approx(.28/.99)
    assert rounded_choice(payload,'A','jev-9.9.9')==(None,False)
    payload['answers']['verification']['probabilities']['A']=.10
    assert rounded_choice(payload,'A','jev-1.13.0')==(None,False)
    payload['answers']['verification']['probabilities']['A']=.283
    assert rounded_choice(payload,'A','jev-1.13.0')==(None,False)
