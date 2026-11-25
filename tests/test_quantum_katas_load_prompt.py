"""Loader integrity, public/private split, and strict generator/verifier parsing."""
import json
from pathlib import Path
import shutil

import pytest

from vgx.quantum_katas import load
from vgx.quantum_katas.load import PublicTask
from vgx.quantum_katas.prompt import parse_generator_response, parse_verifier_response

needs_source = pytest.mark.skipif(not (load.DEFAULT_SOURCE/'dataset/qiskit_quantumkatas.jsonl').exists(),
                                  reason='pinned QuantumKatas source not downloaded')


@needs_source
def test_loader_pins_sources_and_keeps_all_28_tasks_in_order():
    public, oracle = load.load_tasks(load.DEFAULT_SOURCE, load.DEFAULT_PILOT)
    ids = json.loads(load.DEFAULT_PILOT.read_text())['task_ids']
    assert [t.task_id for t in public] == ids == [o.task_id for o in oracle]
    assert len(public) == 28
    for required in ('SolveSATWithGrover/3.1', 'QFT/1.3', 'QEC_BitFlipCode/6', 'CHSHGame/2.4'):
        assert required in ids
    by_id = {t.task_id: t for t in public}
    assert by_id['MagicSquareGame/2.6'].required_functions == ('alice_quantum', 'bob_quantum')
    assert all(t.entry_point in t.required_functions for t in public)
    # entry_point repeats across tasks (e.g. alice_quantum), so task_id is the key.
    assert sum(t.entry_point == 'alice_quantum' for t in public) == 2


@needs_source
def test_public_projection_has_no_reference_or_test_fields():
    public, oracle = load.load_tasks(load.DEFAULT_SOURCE, load.DEFAULT_PILOT)
    for task, secret in zip(public, oracle):
        row = task.to_row()
        assert 'canonical_solution' not in row and 'test' not in row
    with pytest.raises(ValueError):
        PublicTask.from_row({**public[0].to_row(), 'test': oracle[0].test})


@needs_source
def test_loader_rejects_modified_source(tmp_path):
    copy = tmp_path/'src'
    shutil.copytree(load.DEFAULT_SOURCE, copy, ignore=shutil.ignore_patterns('__pycache__'))
    path = copy/'benchmark/evaluator.py'
    path.write_text(path.read_text() + '\n# edited\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load.load_tasks(copy, load.DEFAULT_PILOT)


def test_pilot_config_requires_28_unique_ids(tmp_path):
    config = json.loads(load.DEFAULT_PILOT.read_text())
    bad = tmp_path/'pilot.json'
    bad.write_text(json.dumps({**config, 'task_ids': config['task_ids'][:27] + config['task_ids'][:1]}))
    with pytest.raises(ValueError, match='28 unique'):
        load.load_pilot_ids(bad)


CODE = 'from qiskit import QuantumCircuit\n\nTABLE = {"a": {"b": [1, 2]}}\n\ndef f(qc):\n    s = "quote \\" and } brace"\n    return qc\n'


def test_generator_json_with_nested_dicts_quotes_and_multiline_code():
    text = json.dumps({'code': CODE, 'p_correct': 0.62})
    parsed = parse_generator_response(text)
    assert parsed.code == CODE and parsed.p_correct == 0.62
    assert parsed.response_status == 'code_extracted' and parsed.confidence_status == 'valid'
    fenced = parse_generator_response('```json\n' + text + '\n```')
    assert fenced.code == CODE and fenced.p_correct == 0.62


@pytest.mark.parametrize('text,failure', [
    ('Here is my answer: ' + json.dumps({'code': CODE, 'p_correct': .5}), 'invalid_json'),
    ('```json\n{"code": "x", "p_correct": 0.5}\n```\n```json\n{"code": "y", "p_correct": 0.5}\n```', 'invalid_json'),
    ('{"code": "def f():\\n    return 1", "p_correct": 0.5', 'invalid_json'),
    ('{"code": "a", "code": "b", "p_correct": 0.5}', 'duplicate_json_key'),
    ('[1, 2]', 'json_not_object'),
    ('', 'empty_response'),
    ('{"code": "x", "p_correct": 0.5, "explanation": "y"}', 'unexpected_fields'),
    ('{"p_correct": 0.5}', 'missing_code'),
    ('{"code": 3, "p_correct": 0.5}', 'invalid_code_type'),
    ('{"code": "   ", "p_correct": 0.5}', 'empty_code'),
])
def test_generator_format_failures_have_no_code(text, failure):
    parsed = parse_generator_response(text)
    assert parsed.code is None and parsed.response_status == 'format_failure' and parsed.failure == failure


@pytest.mark.parametrize('literal,status', [
    ('NaN', 'non_finite'), ('Infinity', 'non_finite'), ('-Infinity', 'non_finite'), ('true', 'boolean'),
    ('1.0001', 'out_of_range'), ('-0.1', 'out_of_range'), ('"0.8"', 'non_numeric'), ('null', 'non_numeric'),
])
def test_invalid_confidence_keeps_code_but_no_probability(literal, status):
    parsed = parse_generator_response('{"code": ' + json.dumps(CODE) + ', "p_correct": ' + literal + '}')
    assert parsed.code == CODE and parsed.p_correct is None and parsed.confidence_status == status


def test_missing_confidence_and_truncation():
    parsed = parse_generator_response(json.dumps({'code': CODE}))
    assert parsed.code == CODE and parsed.confidence_status == 'missing'
    truncated = parse_generator_response(json.dumps({'code': CODE, 'p_correct': .5}), truncated=True)
    assert truncated.code is None and truncated.failure == 'truncated_output'


@pytest.mark.parametrize('text,p,failure', [
    ('{"p_correct": 0.25}', .25, None), ('```json\n{"p_correct": 1}\n```', 1., None),
    ('{"p_correct": NaN}', None, 'confidence_non_finite'), ('{"p_correct": false}', None, 'confidence_boolean'),
    ('{"p_correct": 2}', None, 'confidence_out_of_range'), ('{"p_correct": 0.2, "p_correct": 0.3}', None, 'duplicate_json_key'),
    ('{"p_correct": 0.3, "code": "x"}', None, 'unexpected_or_missing_fields'), ('p_correct = 0.3', None, 'invalid_json'),
])
def test_verifier_parse(text, p, failure):
    parsed = parse_verifier_response(text)
    assert (parsed.p_correct, parsed.failure) == (p, failure)
    assert parse_verifier_response('{"p_correct": 0.4}', truncated=True).failure == 'truncated_output'


@needs_source
def test_prompt_v2_only_appends_json_only_instruction_and_v1_is_unchanged():
    from vgx.quantum_katas import prompt
    public, _ = load.load_tasks(load.DEFAULT_SOURCE, load.DEFAULT_PILOT)
    task = public[0]
    v1, v2 = prompt.build_generator_prompt(task), prompt.build_generator_prompt(task, prompt.PROMPT_VERSION_V2)
    assert v2 == v1 + prompt.V2_REPLY and prompt.generator_system() == prompt.GENERATOR_SYSTEM
    assert prompt.verifier_system(prompt.PROMPT_VERSION_V2).startswith(prompt.VERIFIER_SYSTEM)
    with pytest.raises(ValueError):
        prompt.build_generator_prompt(task, 'unknown')
