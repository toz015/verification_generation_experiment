from dataclasses import replace
import json
from pathlib import Path

import pytest

from vgx.common.api import execute_request
from vgx.common.billing import reconcile_usage
from vgx.common.llm import Request
from vgx.common.vertex import VertexBatchRunner
from vgx.common.vertex_partner import VertexPartnerRunner
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.signal_study import (collect, parse_signal, prepare, prompt, report, request_for, runner_for)


@pytest.mark.parametrize('text,signal,expected',[
    ('{"correct":true}', 'binary', (1.,None)),
    ('{"correct":false}', 'binary', (0.,None)),
    ('{"correct":1}', 'binary', (None,'expected_boolean_correct')),
    ('{"p_correct":1}', 'binary', (None,'expected_boolean_correct')),
    ('{"p_correct":0.9}', 'probability', (.9,None)),
    ('{"p_correct":0.1}', 'probability', (.1,None)),
    ('{"p_correct":0}', 'probability', (0.,None)),
    ('{"p_correct":true}', 'probability', (None,'expected_numeric_probability')),
    ('{"p_correct":1.1}', 'probability', (None,'invalid_probability')),
    ('{"p_correct":NaN}', 'probability', (None,'invalid_probability')),
    ('{"p_correct":0.2,"answer":"A"}', 'probability', (None,'expected_numeric_probability')),
    ('not json', 'probability', (None,'invalid_json')),
])
def test_signal_contract(text, signal, expected):
    assert parse_signal(text,signal)==expected


def test_native_partner_bodies_and_usage():
    request=Request('id','question','system')
    claude=VertexPartnerRunner('anthropic/claude-haiku-4-5','global','test-project')
    body=claude.request_body(request)
    assert body['system']=='system' and body['anthropic_version']=='vertex-2023-10-16'
    assert 'model' not in body and 'thinking' not in body and 'top_p' not in body
    assert '/publishers/anthropic/models/claude-haiku-4-5:rawPredict' in claude._endpoint()
    assert claude.response_text({'content':[{'type':'thinking','thinking':'hidden'}, {'type':'text','text':'json'}]})=='json'
    usage=reconcile_usage({'input_tokens':100,'output_tokens':10})
    assert usage.complete and usage.input_tokens==100 and usage.output_tokens==10
    mistral=VertexPartnerRunner('mistralai/mistral-small-2503','us-central1','test-project')
    assert mistral.request_body(request)['model']=='mistral-small-2503'
    assert mistral.request_body(request)['messages'][0]['role']=='system'
    assert mistral.response_text({'choices':[{'message':{'content':'text'}}]})=='text'
    assert claude.cache_key(request)!=mistral.cache_key(request)


def study_config(run):
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/gpqa_signal_study.json').read_text())
    cfg['bundle']=str(run.bundle)
    cfg['models']=cfg['models'][:1]
    return cfg


def test_prompts_hide_prior_and_signal_version_changes_cache(frozen_run):
    _,candidates=load_candidates(frozen_run.bundle)
    c=candidates[0];cfg=study_config(frozen_run);model=cfg['models'][0]
    runner=runner_for(model,cfg['project'])
    assert prompt(c,'probability')==prompt(replace(c,p_correct=.12345),'probability')
    assert runner.cache_key(request_for(c,model,'binary'))!=runner.cache_key(request_for(c,model,'probability'))
    assert runner.cache_key(request_for(c,model,'binary','a'))==runner.cache_key(request_for(c,model,'binary','b'))
    with pytest.raises(ValueError): prompt(replace(c,answer=None),'binary')


def test_prepare_is_calibration_only_and_model_addition_keeps_keys(frozen_run,monkeypatch):
    cfg=study_config(frozen_run)
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.name not in ('calibration_records.jsonl','evaluation_labels.jsonl')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    first=prepare(cfg,frozen_run.root/'study')
    assert first['generator_calls_planned']==first['evaluation_calls_planned']==0
    assert set(first['candidate_ids'])<=set(frozen_run.manifest['calibration_ids'])
    assert prepare(cfg,frozen_run.root/'study')==first
    expanded=json.loads(json.dumps(cfg));expanded['models'].append({**cfg['models'][0],'id':'other','model':'google/gemini-3.5-flash-lite'})
    second=prepare(expanded,frozen_run.root/'expanded')
    assert first['arms'][0]['request_keys']==second['arms'][0]['request_keys']
    repriced=json.loads(json.dumps(cfg));repriced['pricing']['usd_per_million_tokens'][cfg['models'][0]['model']]['input']=1
    third=prepare(repriced,frozen_run.root/'repriced')
    assert first['arms'][0]['request_keys']==third['arms'][0]['request_keys']
    with pytest.raises(ValueError,match='study changed'): prepare(expanded,frozen_run.root/'study')


def test_collection_resume_and_report_never_read_evaluation_labels(frozen_run,monkeypatch):
    cfg=study_config(frozen_run);output=frozen_run.root/'study';prepare(cfg,output)
    calls=[]
    def fake(self,requests,log):
        request=requests[0];calls.append(request)
        assert request.meta['role']=='verifier' and request.meta['partition']=='calibration'
        response='{"correct":true}' if request.meta['signal']=='binary' else '{"p_correct":0.9}'
        execute_request(self,request,log,lambda:(200,{'choices':[{'message':{'content':response}}],
                                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}))
    monkeypatch.setattr(VertexBatchRunner,'run',fake)
    monkeypatch.setattr('vgx.gpqa.signal_study.check_billing',lambda project:{'project':project,'billing_enabled':True})
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.name!='evaluation_labels.jsonl'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    first=collect(output,limit=2,allow_api=True,workers=1)
    assert len(calls)==4 and first['models'][0]['completed']==4
    second=collect(output,limit=2,allow_api=False,workers=1)
    assert len(calls)==4 and second['models'][0]['reused']==4
    result=report(output)
    assert result['arms'][0]['returned']==result['arms'][1]['returned']==2
    assert result['arms'][0]['score_counts']=={'1.0':2}
    assert result['arms'][1]['score_counts']=={'0.9':2}


def test_existing_vertex_body_is_unchanged():
    runner=VertexBatchRunner('google/gemini-3.7-flash','global','project')
    assert runner.request_body(Request('key','prompt','sys'))=={
        'model':runner.model,'messages':[{'role':'system','content':'sys'},{'role':'user','content':'prompt'}],
        'stream':False,'max_tokens':256,'reasoning_effort':'low'}
