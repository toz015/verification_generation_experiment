"""Prompt building and strict response parsing for the GPQA pilot."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from vgx.gpqa.load import GPQAItem, LETTERS

SYSTEM = "You are answering a graduate-level multiple-choice science question."

_JSON = re.compile(r"\{.*?\}", re.DOTALL)
@dataclass(frozen=True)
class GeneratorResponse:
    answer: str | None
    p_correct: float | None
    ok: bool
    failure: str | None = None


@dataclass(frozen=True)
class VerifierResponse:
    p_correct: float | None
    ok: bool
    failure: str | None = None


def _payload(text: str) -> dict | None:
    for candidate in [text.strip(), *[m.group(0) for m in _JSON.finditer(text)]]:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        return None
    return probability


def build_generator_prompt(item: GPQAItem) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, item.choices))
    return (
        f"Subject: {item.subject}\n\nQuestion: {item.question}\n\n{options}\n\n"
        "Choose the best answer. Report your probability that your selected answer is correct.\n"
        'Reply with JSON only: {"answer": "A"|"B"|"C"|"D", "p_correct": 0.0}'
    )


def build_verifier_prompt(item: GPQAItem, candidate: str) -> str:
    if candidate not in LETTERS:
        raise ValueError(f"candidate must be one of {LETTERS}, got {candidate!r}")
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, item.choices))
    return (
        f"Subject: {item.subject}\n\nQuestion: {item.question}\n\n{options}\n\n"
        f"Fixed candidate answer: {candidate}. {item.choices[LETTERS.index(candidate)]}\n\n"
        "Assess whether this fixed answer is correct. Do not suggest a replacement answer.\n"
        'Reply with JSON only: {"p_correct": 0.0}'
    )


def parse_generator_response(text: str) -> GeneratorResponse:
    payload = _payload(text)
    if payload is None:
        return GeneratorResponse(None, None, False, "invalid_json")
    answer = payload.get("answer")
    answer = answer.strip().upper() if isinstance(answer, str) else None
    probability = _probability(payload.get("p_correct"))
    if answer not in LETTERS:
        return GeneratorResponse(None, probability, False, "invalid_answer")
    if probability is None:
        return GeneratorResponse(answer, None, False, "invalid_confidence")
    return GeneratorResponse(answer, probability, True)


def parse_verifier_response(text: str) -> VerifierResponse:
    payload = _payload(text)
    if payload is None:
        return VerifierResponse(None, False, "invalid_json")
    probability = _probability(payload.get("p_correct"))
    if probability is None:
        return VerifierResponse(None, False, "invalid_confidence")
    return VerifierResponse(probability, True)
