"""Generator/verifier prompts and strict response parsing, fixed before any model call.

Accepted final text (PROMPT_VERSION): the whole text is one JSON object, or the
whole text is exactly one fenced block (```json or ```) containing one JSON
object. Nothing else is repaired, searched for, or re-requested. Thinking or
reasoning blocks are never parsed: runners pass the provider's final text only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

from vgx.quantum_katas.load import PublicTask

PROMPT_VERSION = 'qkatas_official_v1_json_or_single_fence'

EXECUTION_CONDITIONS = """\
Execution conditions (public, fixed for every task):
- The code is run once as a single Python 3.12 module with qiskit==2.3.0,
  qiskit-aer==0.17.2, numpy==2.4.1 and scipy==1.17.0; no other packages, no
  network access and no files are available.
- The module must contain all imports it needs, every function the task asks
  for ({functions}) and any helpers they use, not only a function body.
- The benchmark's fixed official checker then runs a hidden, unmodified test
  that calls `{entry_point}` (and anything it depends on) in the same
  namespace, with a 30 second limit. Syntax errors, exceptions, assertion
  failures and timeouts all count as not passing.
- Do not add example usage, prints or tests that run at import time."""

GENERATOR_SYSTEM = ('You write complete Python modules that solve Qiskit QuantumKatas tasks '
                    'and report a calibrated probability that they pass the official checker.')
VERIFIER_SYSTEM = ('You review one fixed candidate Python module for a Qiskit QuantumKatas task '
                   'and report a calibrated probability that it passes the official checker.')

GENERATOR_TEMPLATE = """\
Solve the following task from the Qiskit QuantumKatas benchmark.

Task (verbatim from the benchmark):
<<<TASK
{prompt}
TASK>>>

{conditions}

Reply with exactly one JSON object and nothing else:
{{"code": "<the complete Python module, as a JSON string>", "p_correct": <number from 0 to 1>}}
p_correct is your probability that this exact code passes the benchmark's fixed
official checker."""

VERIFIER_TEMPLATE = """\
Review a fixed candidate solution to the following task from the Qiskit
QuantumKatas benchmark. The candidate is frozen and will be submitted exactly
as shown. Do not write, repair or suggest a different solution.

Task (verbatim from the benchmark):
<<<TASK
{prompt}
TASK>>>

{conditions}

Candidate module (frozen):
<<<CODE
{code}
CODE>>>

Reply with exactly one JSON object and nothing else:
{{"p_correct": <number from 0 to 1>}}
p_correct is your probability that this exact code passes the benchmark's fixed
official checker."""


def _conditions(functions, entry_point: str) -> str:
    return EXECUTION_CONDITIONS.format(functions=', '.join(f'`{f}`' for f in functions),
                                       entry_point=entry_point)


def build_generator_prompt(task: PublicTask) -> str:
    return GENERATOR_TEMPLATE.format(prompt=task.prompt,
                                     conditions=_conditions(task.required_functions, task.entry_point))


def build_verifier_prompt(view: dict) -> str:
    """view is the whitelisted verifier projection from artifacts.verifier_view."""
    if set(view) != {'task_id', 'prompt', 'entry_point', 'required_functions', 'code'}:
        raise ValueError('verifier prompt accepts only the whitelisted public view')
    return VERIFIER_TEMPLATE.format(prompt=view['prompt'], code=view['code'],
                                    conditions=_conditions(view['required_functions'], view['entry_point']))


class _NonFinite:
    """Marker for NaN/Infinity so a bad probability does not erase extractable code."""


_FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n(.*)\r?\n[ \t]*```', re.DOTALL)


class StrictJSONError(ValueError):
    def __init__(self, failure):
        self.failure = failure
        super().__init__(failure)


def strict_json_object(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise StrictJSONError('empty_response')
    body = text.strip()
    fenced = _FENCE.fullmatch(body)
    if fenced:
        body = fenced.group(1)

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise StrictJSONError('duplicate_json_key')
        return dict(items)

    try:
        value = json.loads(body, object_pairs_hook=pairs, parse_constant=lambda _: _NonFinite())
    except StrictJSONError:
        raise
    except (ValueError, RecursionError):
        raise StrictJSONError('invalid_json') from None
    if not isinstance(value, dict):
        raise StrictJSONError('json_not_object')
    return value


def _probability_status(value) -> tuple[float | None, str]:
    if isinstance(value, _NonFinite):
        return None, 'non_finite'
    if isinstance(value, bool):
        return None, 'boolean'
    if not isinstance(value, (int, float)):
        return None, 'non_numeric'
    if not math.isfinite(value):
        return None, 'non_finite'
    if not 0 <= value <= 1:
        return None, 'out_of_range'
    return float(value), 'valid'


@dataclass(frozen=True)
class GeneratorParse:
    code: str | None
    p_correct: float | None
    response_status: str          # 'code_extracted' | 'format_failure'
    failure: str | None           # generator format failure reason
    confidence_status: str        # 'valid' | 'missing' | 'non_numeric' | ... | 'not_applicable'


def parse_generator_response(text: str, *, truncated: bool = False) -> GeneratorParse:
    def fail(reason):
        return GeneratorParse(None, None, 'format_failure', reason, 'not_applicable')
    if truncated:
        return fail('truncated_output')
    try:
        payload = strict_json_object(text)
    except StrictJSONError as error:
        return fail(error.failure)
    if set(payload) - {'code', 'p_correct'}:
        return fail('unexpected_fields')
    code = payload.get('code')
    if 'code' not in payload:
        return fail('missing_code')
    if not isinstance(code, str):
        return fail('invalid_code_type')
    if not code.strip():
        return fail('empty_code')
    if 'p_correct' not in payload:
        return GeneratorParse(code, None, 'code_extracted', None, 'missing')
    p, status = _probability_status(payload['p_correct'])
    return GeneratorParse(code, p, 'code_extracted', None, status)


@dataclass(frozen=True)
class VerifierParse:
    p_correct: float | None
    failure: str | None


def parse_verifier_response(text: str, *, truncated: bool = False) -> VerifierParse:
    if truncated:
        return VerifierParse(None, 'truncated_output')
    try:
        payload = strict_json_object(text)
    except StrictJSONError as error:
        return VerifierParse(None, error.failure)
    if set(payload) != {'p_correct'}:
        return VerifierParse(None, 'unexpected_or_missing_fields')
    p, status = _probability_status(payload['p_correct'])
    return VerifierParse(p, None if status == 'valid' else f'confidence_{status}')
