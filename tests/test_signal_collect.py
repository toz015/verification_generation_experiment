import json

import pytest

from vgx.common.api import execute_request
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa.signal_collect import Pacer,collect
from vgx.gpqa.signal_study import prepare
from test_signal_study import study_config


def test_parallel_collection_preserves_failures_and_reuses_exact_calls(frozen_run,monkeypatch):
    cfg=study_config(frozen_run)
    cfg['collection']={'workers':4,'minimum_request_interval_seconds':{'gemini_flash':0}}
    out=frozen_run.root/'parallel';prepare(cfg,out)
    calls=[]
    def fake(self,requests,log):
        req=requests[0];calls.append(req)
        assert req.meta['role']=='verifier' and req.meta['partition']=='calibration'
        text='{"correct":true}' if req.meta['signal']=='binary' else ''
        execute_request(self,req,log,lambda:(200,{'choices':[{'message':{'content':text}}],
                              'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}))
    monkeypatch.setattr(VertexBatchRunner,'run',fake)
    monkeypatch.setattr('vgx.gpqa.signal_collect.check_billing',lambda project:{'project':project})
    first=collect(out,limit=3,allow_api=True)
    assert first['completed']==6 and len(calls)==6
    assert first['parse_failures']=={'gemini_flash:probability:invalid_json':3}
    second=collect(out,limit=3)
    assert second['reused']==6 and len(calls)==6
    assert second['budget']['successful_priced_calls']==6


@pytest.mark.parametrize('interval',[-1,float('nan'),float('inf'),True])
def test_pacer_rejects_invalid_intervals(interval):
    with pytest.raises(ValueError):Pacer(interval)


def test_output_budget_change_is_a_new_provider_request(frozen_run):
    from vgx.gpqa.signal_study import candidates_for,request_for,runner_for
    cfg=study_config(frozen_run);_,candidates=candidates_for(cfg)
    model=cfg['models'][0];request=request_for(candidates[0],model,'binary')
    old=runner_for(model,cfg['project']).cache_key(request)
    updated={**model,'generation':{**model['generation'],'max_tokens':4096}}
    assert runner_for(updated,cfg['project']).cache_key(request)!=old
