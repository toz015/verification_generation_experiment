from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_fresh_gpqa import fresh_data, fake_vertex
from vgx.common.api import RequestCache, OfflineCacheMiss
from vgx.gpqa import fresh, generator_study as study


@pytest.fixture
def prepared(fresh_data, fake_vertex):
    source, cfg, root = fresh_data
    fresh.prepare(source, 'a'*40, cfg, root)
    cache = RequestCache(root/'cache')
    fresh.collect_generators(root, cfg, cache, allow_api=True)
    fresh.freeze(root, cfg, cache)
    config = json.loads((Path(__file__).parents[1]/'configs/gpqa_generator_comparison.json').read_text())
    config.update(dataset_root=str(root), baseline_bundle=str(root/'frozen'), sample_size=12, reuse_caches=[])
    for m in config['generators'] + config['verifiers']:
        m['interval_seconds'] = 0
    # Use the synthetic chat adapter for both verifier families.
    for m in config['verifiers']:
        m['transport'] = 'chat_completions'
    out = root/'comparison'
    study.prepare(config, out)
    fake_vertex.clear()
    return config, out


def test_frozen_selection_resume_and_shared_verifier_requests(prepared, fake_vertex, monkeypatch):
    cfg, out = prepared
    monkeypatch.setattr(study, 'check_billing', lambda project: {'project':project})
    original = Path.open
    def guard(path, *args, **kwargs):
        assert path.name not in ('labels.jsonl','calibration_records.jsonl','evaluation_labels.jsonl')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, 'open', guard)
        result = study.collect(out, 'generators', allow_api=True)
        assert result['completed'] == len(fake_vertex) == 24
        assert all(r.meta['partition']=='calibration' for _,r in fake_vertex)
        assert not any(m.startswith('google/') for m,_ in fake_vertex)
        resumed = study.collect(out, 'generators')
        assert resumed['cache_reused']==24 and len(fake_vertex)==24
        study.collect(out, 'verifiers', allow_api=True)
        # All three generators chose B: only 12 * 2 distinct verifier calls.
        assert len(fake_vertex)==48
        assert study.collect(out, 'verifiers')['cache_reused']==72
    result = study.report(out)
    assert set(result['generators'])=={'gemini_baseline','qwen','llama'}
    assert result['new_collection_budget']['successful_priced_calls']==48
    assert all(s['n']==12 for s in result['generators'].values())
    from vgx.gpqa.generator_analysis import analyze
    matched=analyze(out)
    assert matched['common_confidence_n']==12
    assert all(p['raw_brier_difference']==0 for p in matched['paired_raw_forecasts'].values())
    audit=json.loads((out/'audit.json').read_text())
    assert audit['unique_executions']==48 and audit['evaluation_calls']==0


def test_verifiers_cannot_fall_back_to_paid_generation(prepared, fake_vertex):
    _, out = prepared
    with pytest.raises(OfflineCacheMiss):
        study.collect(out, 'verifiers', allow_api=True)
    assert fake_vertex==[]


def test_prices_and_verifier_pool_do_not_change_generator_keys(prepared):
    cfg, out = prepared
    old = study.load(out)[0]
    changed = deepcopy(cfg)
    changed['pricing']['checked_on']='future'
    changed['verifiers']=changed['verifiers'][:1]
    new = study.prepare(changed, out.parent/'other')
    assert old['generator_arms']==new['generator_arms']
    with pytest.raises(ValueError, match='changed study'):
        study.prepare(changed, out)


def test_repriced_study_imports_success_without_new_execution(prepared, fake_vertex, monkeypatch):
    cfg, out = prepared
    monkeypatch.setattr(study, 'check_billing', lambda project: {'project':project})
    study.collect(out, 'generators', limit=1, allow_api=True)
    assert len(fake_vertex)==2
    changed=deepcopy(cfg)
    for model in changed['generators']:
        rate=changed['pricing']['usd_per_million_tokens'][model['model']]
        rate['cached_input']=rate['input']
    changed['reuse_caches']=[str(out/'cache')]
    other=out.parent/'repriced'
    study.prepare(changed,other)
    result=study.collect(other,'generators',limit=1,allow_api=True)
    assert len(fake_vertex)==2 and result['cache_reused']==2
    # Imported executions are historical costs, not new spend in this ledger.
    assert result['budget']['successful_priced_calls']==0


def test_invalid_confidence_keeps_valid_answer_accuracy(prepared, fake_vertex, monkeypatch):
    from vgx.common.api import execute_request
    from vgx.common.vertex import VertexBatchRunner
    _, out = prepared
    monkeypatch.setattr(study, 'check_billing', lambda project: {'project':project})
    def fake(self, requests, log):
        req = requests[0]
        text = '{"answer":"B","p_correct":"bad"}' if req.meta['role']=='generator' else '{"p_correct":0.8}'
        execute_request(self,req,log,lambda:(200,{'choices':[{'message':{'content':text}}],
            'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}))
    monkeypatch.setattr(VertexBatchRunner,'run',fake)
    study.collect(out,'generators',allow_api=True)
    # A valid answer still receives verification even if confidence parsing failed.
    assert study.collect(out,'verifiers',allow_api=True)['completed']==72
    result=study.report(out)
    for name in ('qwen','llama'):
        assert result['generators'][name]['valid_answer_n']==12
        assert result['generators'][name]['confidence_n']==0
        assert result['generators'][name]['correct']==result['generators']['gemini_baseline']['correct']
