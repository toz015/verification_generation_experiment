from copy import deepcopy
import pytest

from vgx.gpqa import generator_calibration as study
from vgx.gpqa.artifacts import FrozenCandidate


def test_prior_fits_only_training_folds_and_rejects_evaluation():
    rows=[{'item_id':str(i),'partition':'calibration','generator_answer':'A',
           'generator_p_correct':.6 if i%3==0 else .9,'correct_index':1 if i%3==0 else 0,
           'outcome':int(i%3!=0),'verifiers':{}} for i in range(36)]
    result=study.fit_prior(rows);fit=result['primary_oof']
    assert fit['n']==36
    validations=[]
    for fold in fit['folds']:
        assert set(fold['train']).isdisjoint(fold['validation'])
        validations+=fold['validation']
    assert sorted(validations)==sorted(r['item_id'] for r in rows)
    assert len(result['fold_seed_sensitivity'])==5
    assert 'base_rate' in fit['metrics']
    bad=deepcopy(rows);bad[0]['partition']='evaluation'
    with pytest.raises(ValueError,match='calibration rows'):study.fit_prior(bad)


def test_cache_audit_counts_zero_signals_missing_and_invalid_separately():
    candidate=FrozenCandidate('x','calibration','Physics','q',('a','b','c','d'),'A',.9,True,None,
        'generator','old-key','prompt-hash','response-hash','system','verifier prompt')
    model={'id':'test','model':'google/gemini-3.7-flash','transport':'chat_completions','location':'global',
           'generation':{'max_tokens':4096,'reasoning_effort':'low'}}
    project='llm-applications-490420';runner=study.runner_for(model,project)
    req=study.request_for(candidate,model,'binary');key=runner.cache_key(req)
    call={'key':key,'prompt':req.prompt,'params':runner.params,'response':'{"correct":false}',
          'meta':{'system':req.system,'runner_identity':runner.identity,'resource_project':project,'quota_project':project}}
    result=study.audit_coverage({'qwen':[candidate]},{'x':1},[model],project,{key:call})
    assert result['coverage']['qwen']['test:binary']['cached_valid_wrong']==1
    assert result['coverage']['qwen']['test:probability']['missing']==1
    assert result['unique_missing_requests']==1
    malformed=deepcopy(call);malformed['response']='invalid'
    result=study.audit_coverage({'qwen':[candidate]},{'x':1},[model],project,{key:malformed})
    assert result['coverage']['qwen']['test:binary']['cached_invalid']==1
    poisoned=deepcopy(call);poisoned['prompt']='different candidate'
    with pytest.raises(ValueError,match='provenance mismatch'):
        study.audit_coverage({'qwen':[candidate]},{'x':1},[model],project,{key:poisoned})
