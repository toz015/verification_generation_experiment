"""Offline dry-run inputs: mock provider responses built from reviewed fixtures.

Generator "responses" are the upstream reference solutions or the audit's
reviewed control programs (noop, SAT return True, H-only QFT, hard-coded QEC),
plus deliberate format failures. Verifier "responses" are fixed numbers. None of
this is a model output; it exercises parsing, freezing, isolation boundaries,
official scoring, resume and the sequential interface.
"""
from __future__ import annotations

import json
from pathlib import Path

from vgx.common.storage import digest, file_digest
from vgx.quantum_katas import load, prompt
from vgx.quantum_katas.load import sha256_text

AUDIT_CONTROLS = Path('reports/quantum_katas_pilot_20261018/private/control_codes.jsonl')
MOCK_GENERATOR = 'mock/generator-dry-run'
MOCK_VERIFIERS = ('mock/verifier-a', 'mock/verifier-b', 'mock/verifier-c')
SYNTAX_ERROR_CODE = 'def tt_or(tt1, tt2:\n    return tt1\n'

# task_id -> (fixture kind, raw p_correct literal as JSON text)
SPECIAL = {
    'SolveSATWithGrover/3.1': ('probe', '0.9'),            # constant True; official checker accepts
    'PhaseEstimation/2': ('probe', '0.7'),                 # H-only; official checker accepts
    'QEC_BitFlipCode/6': ('probe', '0.75'),                # hard-coded indices; official checker accepts
    'BasicGates/2.5': ('noop', '0.6'),
    'GroversAlgorithm/1.3': ('noop', '0.4'),
    'Teleportation/1.4': ('noop', '0.3'),
    'TruthTables/5': ('syntax_error', '0.5'),
    'QFT/1.3': ('invalid_json', None),
    'GHZGame/2.2': ('truncated', '0.8'),
    'RippleCarryAdder/3.3': ('reference', '1.7'),           # code kept; r out of range -> policy abstains
    'Superposition/1.5': ('reference', 'true'),             # boolean r
    'DeutschJozsa/1.6': ('reference_json_fence', '0.85'),   # allowed single ```json fence
    'Measurements/1.13': ('reference_in_markdown', '0.8'),  # official extractor strips the inner fence
}
VERIFIER_SCORES = {  # kind -> scores for verifier a, b, c
    'reference': ('0.85', '0.7', '0.6'), 'probe': ('0.55', '0.8', '0.65'),
    'noop': ('0.1', '0.3', '0.45'), 'syntax_error': ('0.05', '0.2', '0.3'),
}
VERIFIER_C_INVALID = {'MagicSquareGame/2.6': '{"p_correct": "high"}',
                      'SimonsAlgorithm/1.3': '{"p_correct": 0.4, "p_correct": 0.6}'}


def mock_key(model: str, prompt_text: str, system: str) -> str:
    return digest({'model': model, 'prompt': prompt_text, 'system': system})


def _controls():
    rows = [json.loads(line) for line in AUDIT_CONTROLS.read_text().splitlines() if line.strip()]
    noop = {r['task_id']: r['code'] for r in rows if r['kind'] == 'noop'}
    probes = {r['task_id']: r['code'] for r in rows if r['kind'] not in ('noop', 'reference')}
    return noop, probes


def build(public, oracle_rows):
    """Return (mock_responses, reviewed_fixtures, expected) for the 28 fixed tasks."""
    from vgx.quantum_katas.run_pilot import verifier_prompt_for
    oracle_by_id = {o.task_id: o for o in oracle_rows}
    noop, probes = _controls()
    responses, fixtures, expected = {}, [], {}

    def fixture(task_id, kind, code, source):
        fixtures.append({'task_id': task_id, 'kind': kind, 'code_sha256': sha256_text(code), 'source': source})

    for index, task in enumerate(public):
        tid = task.task_id
        kind, p = SPECIAL.get(tid, ('reference', ('0.95', '0.8', '0.65', '0.9')[index % 4]))
        reference = oracle_by_id[tid].canonical_solution
        code = {'reference': reference, 'reference_json_fence': reference, 'truncated': reference,
                'reference_in_markdown': f'```python\n{reference}\n```', 'noop': noop.get(tid),
                'probe': probes.get(tid), 'syntax_error': SYNTAX_ERROR_CODE, 'invalid_json': reference}[kind]
        if code is None:
            raise ValueError(f'no reviewed control fixture for {tid}')
        source = {'noop': 'audit control_codes.jsonl (noop)', 'probe': 'audit control_codes.jsonl (targeted probe)',
                  'syntax_error': 'local syntax-error control (never executed by the official checker)'}.get(
                      kind, 'upstream canonical_solution')
        fixture(tid, kind, code, source)
        if kind == 'invalid_json':
            text = '{"code": ' + json.dumps(code)[:-40]  # cut mid-string: no extractable code
        else:
            text = '{"code": ' + json.dumps(code) + ', "p_correct": ' + p + '}'
            if kind == 'reference_json_fence':
                text = '```json\n' + text + '\n```'
        finish = 'length' if kind == 'truncated' else 'stop'
        responses[mock_key(MOCK_GENERATOR, prompt.build_generator_prompt(task), prompt.generator_system())] = {
            'text': text, 'finish_reason': finish}
        has_code = kind not in ('invalid_json', 'truncated')
        expected[tid] = {'kind': kind, 'has_code': has_code}
        if not has_code:
            continue
        verifier_prompt = verifier_prompt_for(task, code)
        scores = VERIFIER_SCORES['reference' if kind.startswith('reference') else kind]
        for model, score in zip(MOCK_VERIFIERS, scores):
            body = '{"p_correct": ' + score + '}'
            if model == MOCK_VERIFIERS[2] and tid in VERIFIER_C_INVALID:
                body = VERIFIER_C_INVALID[tid]
            responses[mock_key(model, verifier_prompt, prompt.verifier_system())] = {'text': body, 'finish_reason': 'stop'}
    return responses, fixtures, expected


def config(checker_backend='reviewed_local', interpreter='tmp/quantum_katas_pilot_20261018/venv/bin/python',
           workers=2) -> dict:
    def spec(spec_id, model):
        return {'id': spec_id, 'provider': 'mock', 'model': model, 'generation': {'max_tokens': 8192}}
    return {
        'schema': 1, 'kind': 'quantum_katas_official_pilot', 'status': 'offline_dry_run',
        'source': str(load.DEFAULT_SOURCE), 'pilot': str(load.DEFAULT_PILOT),
        'protocol': str(load.DEFAULT_PROTOCOL), 'requirements': str(load.DEFAULT_REQUIREMENTS),
        'generation_id': 'g0', 'prompt_version': prompt.PROMPT_VERSION, 'smoke_tasks': 3,
        'models': {'generator': spec('generator', MOCK_GENERATOR),
                   'verifiers': [spec(f'verifier_{n}', m) for n, m in zip('abc', MOCK_VERIFIERS)]},
        'budget': {'project': 'offline-mock', 'limit_usd': 1.0, 'pricing_source': 'synthetic zero-cost dry-run pricing',
                   'pricing': {'usd_per_million_tokens': {m: {'input': 0.0, 'output': 0.0}
                                                          for m in (MOCK_GENERATOR, *MOCK_VERIFIERS)}}},
        'paid_collection_authorized': False,
        'checker': {'interpreter': interpreter, 'backend': checker_backend, 'workers': workers},
        'audit_controls_sha256': file_digest(AUDIT_CONTROLS),
    }
