from dataclasses import replace
import json

import pytest

from vgx.common.api import execute_request, OfflineCacheMiss, RequestCache, UncertainRequestError
from vgx.common.llm import Call, CallLog, Request
from vgx.common.storage import append_jsonl
from vgx.common.vertex import VertexBatchRunner


def runner():
    return VertexBatchRunner('test-model','global','test-project')


def response():
    return 200, {'id':'response-one','model':'test-model', 'choices':[{'message':{'content':'{"p_correct":0.8}'}}],
                 'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}


def test_append_after_truncated_tail_recovers_next_success(tmp_path):
    path = tmp_path/'calls.jsonl'
    path.write_text('{"key":')
    log = CallLog(path)
    log.append(Call('new','m','p','r',{},'api',0.,0.))
    assert CallLog(path).has('new')
    assert list(CallLog(path).records())[0]['response'] == 'r'
    assert path.read_text().startswith('{"key":\n')


def test_cache_identity_excludes_role_labels_and_analysis_metadata():
    r = Request('original','prompt','system',{'role':'generator','partition':'calibration'})
    assert runner().cache_key(r) == runner().cache_key(replace(r,key='another',meta={'price':99,'role':'other'}))
    assert runner().cache_key(r) != runner().cache_key(replace(r,prompt='changed'))
    assert runner().cache_key(r) != runner().cache_key(replace(r,system='changed'))


def test_durable_response_reused_when_completion_event_is_missing(tmp_path):
    r, req, log = runner(), Request('request','prompt'), CallLog(tmp_path/'calls.jsonl')
    calls = []
    execute_request(r,req,log,lambda: (calls.append(1), response())[1])
    path = str(log.path)+'.attempts.jsonl'
    lines = open(path).readlines()
    open(path,'w').write(lines[0])  # crash between durable response and completion
    again = execute_request(r,req,CallLog(log.path),lambda: pytest.fail('must not resend'))
    assert again.response and calls == [1]


def test_unknown_transport_failure_blocks_automatic_repeat(tmp_path):
    r, req, log = runner(), Request('request','prompt'), CallLog(tmp_path/'calls.jsonl')
    def fail():
        raise TimeoutError('private-header-must-not-appear')
    with pytest.raises(UncertainRequestError):
        execute_request(r,req,log,fail)
    with pytest.raises(UncertainRequestError):
        execute_request(r,req,log,lambda: pytest.fail('must not retry'))
    assert 'private-header' not in (tmp_path/'calls.jsonl.attempts.jsonl').read_text()


def test_rejected_request_is_recorded_and_can_be_retried_explicitly(tmp_path):
    r, req, log = runner(), Request('request','prompt'), CallLog(tmp_path/'calls.jsonl')
    with pytest.raises(RuntimeError,match='HTTP 429'):
        execute_request(r,req,log,lambda:(429,{}))
    execute_request(r,req,log,response)
    rows = list(log.records())
    assert len(rows) == 2 and rows[0]['error'] == 'http_429' and rows[1]['error'] is None
    assert rows[0]['meta']['execution_id'] != rows[1]['meta']['execution_id']


def test_cache_is_offline_by_default_and_reuses_shared_exact_request(tmp_path, monkeypatch):
    cache, r, req = RequestCache(tmp_path/'cache'), runner(), Request('a','prompt')
    with pytest.raises(OfflineCacheMiss):
        cache.get(r,req)
    count = []
    def fake(requests, log):
        count.append(1)
        execute_request(r, requests[0], log, response)
    monkeypatch.setattr(r,'run',fake)
    first, reused = cache.get(r,req,allow_api=True)
    assert not reused
    second, reused = RequestCache(tmp_path/'cache').get(r,replace(req,key='b',meta={'pricing_version':2}))
    assert reused and first.response == second.response and count == [1]


def test_missing_nontext_provider_answer_is_cached_not_retried(tmp_path):
    r, req, log = runner(), Request('request','prompt'), CallLog(tmp_path/'calls.jsonl')
    result = execute_request(r,req,log,lambda:(200,{'usage':{'prompt_tokens':3,'completion_tokens':0},'choices':[]}))
    assert result.response == '' and result.error is None
    assert execute_request(r,req,log,lambda:pytest.fail('no retry')).response == ''
