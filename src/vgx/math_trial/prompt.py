"""Generator prompt and strict reply parser for the math trial (fixed before any model call).

The reply must be one JSON object with exactly reasoning, answer and p_correct; provider
JSON mode is requested. Nothing is repaired or re-requested; a reply that does not parse
counts as a generator format failure (scored as incorrect).
"""
from __future__ import annotations

from dataclasses import dataclass

from vgx.quantum_katas.prompt import StrictJSONError, _probability_status, strict_json_object

PROMPT_VERSION = 'math_trial_v1_json_reasoning_answer'
# For generators with provider-side extended thinking (Sonnet 5.5): the reply asks only for the
# answer and p_correct. Asking such a model to write its reasoning into the reply is refused by the
# provider ("reasoning_extraction", 2026-10-21); the model still reasons in its own thinking.
PROMPT_VERSION_ANSWER_ONLY = 'math_trial_v1_json_answer_only'
PROMPT_VERSIONS = (PROMPT_VERSION, PROMPT_VERSION_ANSWER_ONLY)

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


ANSWER_ONLY_REPLY = ('{"reasoning": "<your complete step-by-step solution>", "answer": "<the final answer only>", '
                     '"p_correct": <number from 0 to 1>}', '{"answer": "<the final answer only>", "p_correct": <number from 0 to 1>}')


def build_generator_prompt(problem: str, version: str = PROMPT_VERSION) -> str:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f'unknown prompt version: {version}')
    text = GENERATOR_TEMPLATE.format(problem=problem)
    if version == PROMPT_VERSION_ANSWER_ONLY:
        old, new = ANSWER_ONLY_REPLY
        assert old in text
        text = text.replace(old, new)
    return text


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


VERIFIER_SYSTEM = ('You evaluate whether a proposed final answer to a competition mathematics problem is correct '
                   'and report a calibrated probability. Reply with one JSON object only, never with prose.')

VERIFIER_TEMPLATE = """\
Evaluate a fixed candidate final answer to the following mathematics problem. The candidate
is frozen and will be submitted exactly as shown. Do not suggest a replacement answer.

Problem (verbatim from the benchmark):
<<<PROBLEM
{problem}
PROBLEM>>>

Candidate final answer (frozen):
<<<ANSWER
{answer}
ANSWER>>>

The candidate is compared with the reference answer by a fixed automatic checker for
mathematical equivalence. Report the probability that the candidate final answer is correct
as a number from 0 to 1; intermediate values are allowed.
Reply with exactly one JSON object and nothing else: {{"p_correct": <number from 0 to 1>}}"""


def build_verifier_prompt(problem: str, answer: str) -> str:
    return VERIFIER_TEMPLATE.format(problem=problem, answer=answer)
