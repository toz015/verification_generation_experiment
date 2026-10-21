"""Generator prompt and strict reply parser for the math trial (fixed before any model call).

The reply must be one JSON object with exactly reasoning, answer and p_correct; provider
JSON mode is requested. Nothing is repaired or re-requested; a reply that does not parse
counts as a generator format failure (scored as incorrect).
"""
from __future__ import annotations

from dataclasses import dataclass

from vgx.quantum_katas.prompt import StrictJSONError, _probability_status, strict_json_object

PROMPT_VERSION = 'math_trial_v1_json_reasoning_answer'

GENERATOR_SYSTEM = ('You solve competition mathematics problems and report a calibrated probability '
                    'that your final answer is correct. Reply with one JSON object only, never with prose.')

GENERATOR_TEMPLATE = """\
Solve the following mathematics problem.

Problem (verbatim from the benchmark):
<<<PROBLEM
{problem}
PROBLEM>>>

Reply with exactly one JSON object and nothing else:
{{"reasoning": "<your complete step-by-step solution>", "answer": "<the final answer only>", "p_correct": <number from 0 to 1>}}
Write the final answer in LaTeX without surrounding $ signs or \\boxed{{}}, in simplest exact form
unless the problem asks otherwise. It is compared with the reference answer by a fixed
automatic checker for mathematical equivalence. p_correct is your probability that this
final answer is correct."""


def build_generator_prompt(problem: str) -> str:
    return GENERATOR_TEMPLATE.format(problem=problem)


@dataclass(frozen=True)
class GeneratorParse:
    answer: str | None
    p_correct: float | None
    failure: str | None
    confidence_status: str


def parse_generator_response(text: str, *, truncated: bool = False) -> GeneratorParse:
    def fail(reason):
        return GeneratorParse(None, None, reason, 'not_applicable')
    if truncated:
        return fail('truncated_output')
    try:
        payload = strict_json_object(text)
    except StrictJSONError as error:
        return fail(error.failure)
    if set(payload) - {'reasoning', 'answer', 'p_correct'}:
        return fail('unexpected_fields')
    answer = payload.get('answer')
    if not isinstance(answer, str) or not answer.strip():
        return fail('missing_or_empty_answer')
    if 'p_correct' not in payload:
        return GeneratorParse(answer, None, None, 'missing')
    p, status = _probability_status(payload['p_correct'])
    return GeneratorParse(answer, p, None, status)
