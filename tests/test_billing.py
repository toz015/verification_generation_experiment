from dataclasses import replace
import json
import math

import pytest

from vgx.common.billing import estimate_call, reconcile_usage, summarize_vertex_usage
from vgx.common.llm import Call, CallLog
from vgx.common.storage import append_jsonl


@pytest.mark.parametrize('usage,output', [
    ({'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'completion_tokens_details':{'reasoning_tokens':15}},20),
    ({'prompt_tokens':100,'completion_tokens':5,'total_tokens':120,'completion_tokens_details':{'reasoning_tokens':15}},20),
    ({'promptTokenCount':100,'candidatesTokenCount':5,'thoughtsTokenCount':15,'totalTokenCount':120},20),
    ({'input_tokens':100,'output_tokens':20},20),
    ({'prompt_tokens':100,'completion_tokens':20,'total_tokens':120},20),
])
def test_provider_usage_reconciles_without_double_count(usage, output):
    result = reconcile_usage(usage)
    assert result.complete and result.output_tokens == output and result.input_tokens == 100


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens':-1,'completion_tokens':2},
    {'prompt_tokens':True,'completion_tokens':2}, {'prompt_tokens':1.5,'completion_tokens':2},
    {'prompt_tokens':1,'input_tokens':2,'completion_tokens':2},
    {'prompt_tokens':1,'completion_tokens':float('nan')},
    {'prompt_tokens':100,'completion_tokens':5,'completion_tokens_details':{'reasoning_tokens':15}},
    {'prompt_tokens':100,'completion_tokens':5,'total_tokens':999,'completion_tokens_details':{'reasoning_tokens':15}},
    {'prompt_tokens':10,'completion_tokens':2,'prompt_tokens_details':{'cached_tokens':11}},
])
def test_ambiguous_invalid_or_missing_usage_is_unpriced(usage):
    assert not reconcile_usage(usage).complete


def test_explicit_semantics_needs_consistent_evidence():
    usage = {'prompt_tokens':100,'completion_tokens':5,'completion_tokens_details':{'reasoning_tokens':15}}
    assert reconcile_usage(usage, semantics='completion_excludes_reasoning').output_tokens == 20
    assert not reconcile_usage(usage, semantics='completion_includes_reasoning').complete
    usage['total_tokens'] = 120
    assert not reconcile_usage(usage, semantics='completion_includes_reasoning').complete
    assert not reconcile_usage({'prompt_tokens':100,'completion_tokens':5}, model='google/gemini-3.8-flash').complete


def test_cached_token_rates_and_missing_rates():
    row = {'model':'m','meta':{'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,
                                     'prompt_tokens_details':{'cached_tokens':30}}}}
    pricing = {'usd_per_million_tokens':{'m':{'input':2.,'output':4.}}}
    assert estimate_call(row, pricing)['estimated_usd'] is None
    pricing['usd_per_million_tokens']['m']['cached_input'] = .5
    assert estimate_call(row, pricing)['estimated_usd'] == pytest.approx((70*2+30*.5+20*4)/1e6)
    pricing['usd_per_million_tokens']['m']['output'] = -1
    assert estimate_call(row, pricing)['estimated_usd'] is None


def call(key, execution, *, response_id=None, error=None, operation='run'):
    return Call(key,'m','prompt','response',{},'managed_api',.1,1.,error=error,
                meta={'execution_id':execution,'response_id':response_id,'role':'verifier',
                      'partition':'evaluation','operation_id':operation,
                      'usage':{'input_tokens':100,'output_tokens':20}})


def test_account_executions_not_logical_request_keys(tmp_path):
    log = CallLog(tmp_path/'calls.jsonl')
    log.append(call('same-request','one'))
    log.append(call('same-request','two'))  # two real executions are two estimates
    copy = CallLog(tmp_path/'copy.jsonl')
    copy.append(call('same-request','one'))
    result = summarize_vertex_usage({'log':log,'copy':copy}, {'usd_per_million_tokens':{'m':{'input':1.,'output':1.}}})
    assert result['successful_calls'] == result['priced_calls'] == 2
    assert result['duplicate_execution_rows_ignored'] == 1
    assert result['estimated_usd_for_priced_calls'] == pytest.approx(.00024)
    assert result['confirmed_billing']['usd'] is None


def test_account_provider_ids_and_conflicting_duplicates(tmp_path):
    log = CallLog(tmp_path/'calls.jsonl')
    log.append(call('one',None,response_id='provider-one'))
    log.append(call('two',None,response_id='provider-two'))
    result = summarize_vertex_usage({'log':log})
    assert result['successful_calls'] == 2
    broken = call('copy',None,response_id='provider-one')
    log.append(replace(broken,response='different'))
    with pytest.raises(ValueError, match='conflicting'):
        summarize_vertex_usage({'log':log})


def test_pending_attempt_and_rejection_are_not_hidden(tmp_path):
    log = CallLog(tmp_path/'calls.jsonl')
    log.append(call('success','ok'))
    log.append(call('rejected','rejected',error='http_429'))
    path = str(log.path)+'.attempts.jsonl'
    append_jsonl(path, {'key':'lost','execution_id':'lost','event':'started','operation_id':'run'})
    result = summarize_vertex_usage({'log':log}, {'usd_per_million_tokens':{'m':{'input':1.,'output':1.}}})
    assert result['unresolved_attempts'] == ['lost']
    assert result['rejected_attempt_records'] == 1
    assert not result['all_calls_priced']
    assert result['estimate_complete_for_observed_successes']


def test_incremental_operation_scope_and_billing_evidence(tmp_path):
    log = CallLog(tmp_path/'calls.jsonl')
    log.append(call('past','past',operation='historical'))
    log.append(call('now','now',operation='new'))
    evidence = {'currency':'USD','usd':.5,'source':'synthetic invoice', 'reference':'line-1','scope':'new run only'}
    result = summarize_vertex_usage({'log':log}, operation_id='new', confirmed_billing=evidence)
    assert result['successful_calls'] == 1
    assert result['confirmed_billing']['usd'] == .5
    assert result['estimated_usd_for_priced_calls'] == 0  # no supplied rates
    assert not result['all_calls_priced']
    with pytest.raises(ValueError):
        summarize_vertex_usage({'log':log}, confirmed_billing={'usd':1})


def test_rejected_attempt_is_not_claimed_as_fully_priced(tmp_path):
    log = CallLog(tmp_path/'calls.jsonl')
    log.append(call('rejected','failed',error='http_500'))
    result = summarize_vertex_usage({'log':log})
    assert result['successful_calls'] == 0 and not result['all_calls_priced']
    assert result['confirmed_billing']['status'] == 'not_provided'


def test_ledger_is_identical_when_cache_inventory_order_changes(tmp_path):
    logs={}
    for i in range(5):
        log=CallLog(tmp_path/f'{i}.jsonl')
        row=call(str(i),f'execution-{i}')
        row.meta['usage']={'input_tokens':31+i*101,'output_tokens':27+i}
        log.append(row)
        logs[str(i)]=log
    pricing={'usd_per_million_tokens':{'m':{'input':.042,'output':0.}}}
    forward=summarize_vertex_usage(logs,pricing)
    backward=summarize_vertex_usage(dict(reversed(list(logs.items()))),pricing)
    assert forward==backward
