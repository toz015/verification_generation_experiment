import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_completion_study import completion_prepared
from test_reasoning_study import reasoning_prepared
from test_generator_study import prepared
from test_fresh_gpqa import fresh_data, fake_vertex
from vgx.common.api import execute_request
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa import completion_study as controls, staged_generator as study, llama_resume as resume


def test_only_missing_llama_calls_and_preserves_all_existing(completion_prepared,monkeypatch):
    cfg,control_root=completion_prepared;calls=[]
    def fake(self,requests,log):
        req=requests[0];calls.append((self.model,req.meta['role']))
        text='{"answer":"B","p_correct":0.7}'
        if req.meta['role']!='generator_final':text='<final>'+text+'</final>'
        execute_request(self,req,log,lambda:(200,{'choices':[{'message':{'content':text},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130}}))
    monkeypatch.setattr(VertexBatchRunner,'run',fake)
    monkeypatch.setattr(controls,'check_billing',lambda p:{'project':p})
    monkeypatch.setattr(study,'check_billing',lambda p:{'project':p})
    controls.prepare(cfg,control_root);controls.collect(control_root,allow_api=True)
    config={k:cfg[k] for k in ('project','pricing','previous_results')}
    config.update(scope='pilot',stage_budget_usd=2.,reuse_caches=[str(control_root/'cache')])
    pilot=control_root.parent/'staged';study.prepare(config,pilot);study.collect(pilot,allow_api=True)
    full=dict(config,scope='all_questions',pilot_root=str(pilot),reuse_caches=[])
    root=control_root.parent/'full';plan=study.prepare(full,root);_,items=study.load(root)
    cache=study.direct.budget(full,root)
    qwen=study.runner_for(plan['models']['qwen'],full['project'])
    final=study.runner_for(plan['models']['qwen_final'],full['project'])
    llama=study.runner_for(plan['models']['llama'],full['project'])
    for index,item in enumerate(items):
        draft,_=cache.get(qwen,study.draft_request(item),allow_api=True)
        cache.get(final,study.final_request(item,draft),allow_api=True)
        if index<len(items)-2:cache.get(llama,study.previous.generator_request(item),allow_api=True)
    previous={'error':{'http_status':429},'completed':2*len(items)-2}
    (root/'collection.json').write_text(json.dumps(previous))
    monkeypatch.setattr(study,'Pacer',lambda seconds:SimpleNamespace(acquire=lambda:None))
    before=len(calls);output=root/'resume'
    result=resume.run(root,output,allow_api=True)
    assert result['stage']=='complete'
    assert calls[before:]==[(llama.model,'generator')]*2
    assert result['new_calls_this_run']==2
    assert json.loads((output/'previous_collection.json').read_text())==previous
    assert result['evaluation_labels_read'] is False
    before=len(calls)
    assert resume.run(root,output)['stage']=='complete'
    assert len(calls)==before


def test_interval_cannot_silently_increase_request_rate(tmp_path):
    with pytest.raises(ValueError,match='at least 10'):
        resume.run(tmp_path,tmp_path/'resume',interval=1)
