"""Prompt construction and parsing for PubMedQA.

The task: a research question plus the abstract's non-conclusion sections,
answered `yes`, `no`, or `maybe`. This is the standard PubMedQA reasoning-
required setting, so results sit alongside published numbers.

`maybe` carries the weight for this project. It is the closest thing in the
three datasets to a supervised abstention label: the correct answer when the
evidence genuinely does not settle the question. The prompt therefore states
what `maybe` means rather than leaving models to infer it, since a model that
never emits `maybe` because it did not realise it was allowed would be a
prompting artefact rather than a finding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

LABELS = ("yes", "no", "maybe")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_ALIASES = {
    "YES": "yes",
    "TRUE": "yes",
    "NO": "no",
    "FALSE": "no",
    "MAYBE": "maybe",
    "UNCERTAIN": "maybe",
    "UNCLEAR": "maybe",
    "INCONCLUSIVE": "maybe",
    "NOT ENOUGH INFO": "maybe",
    "INSUFFICIENT": "maybe",
}

SYSTEM = (
    "You are a biomedical research assistant. You answer questions about "
    "research findings strictly from the abstract excerpts provided."
)

INSTRUCTIONS = """Answer the question using only the excerpts above.

- yes: the evidence supports an affirmative answer.
- no: the evidence supports a negative answer.
- maybe: the evidence is genuinely inconclusive, mixed, or insufficient to
  decide either way.

Reply with JSON and nothing else:

{"answer": "yes" | "no" | "maybe"}"""


def build_prompt(question: str, context: str) -> str:
    return f"Question: {question}\n\n{context}\n\n{INSTRUCTIONS}"


@dataclass
class Answer:
    label: str | None
    ok: bool = True
    failure: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _extract_json(text: str) -> dict | None:
    cleaned = _THINK.sub("", text).strip()
    candidates = []
    fenced = _FENCE.search(cleaned)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalise(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    return _ALIASES.get(raw.strip().upper().replace("-", " ").replace("_", " "))


def parse_response(text: str) -> Answer:
    """Parse a completion into yes/no/maybe.

    Forgiving about form, strict about content - the same principle as the
    SciFact parser. A bare word answer is accepted as a fallback, because
    rejecting it would measure JSON compliance rather than the model's verdict,
    and that would bias hardest against whichever model formats worst.
    """
    if not text or not text.strip():
        return Answer(label=None, ok=False, failure="empty_response")

    payload = _extract_json(text)
    if payload is not None:
        label = _normalise(payload.get("answer"))
        if label:
            return Answer(label=label)
        return Answer(label=None, ok=False, failure="bad_label")

    # Fallback: an unfenced bare answer, e.g. "maybe" or "Answer: yes".
    #
    # Whichever label appears *earliest* wins, rather than a fixed label order.
    # Scanning in a fixed order would resolve "the answer is not yes, it is
    # maybe" to `yes`, silently inverting the model's verdict.
    cleaned = _THINK.sub("", text).strip().lower()
    hits = [
        (match.start(), label)
        for label in LABELS
        if (match := re.search(rf"\b{label}\b", cleaned))
    ]
    if hits:
        _, label = min(hits)
        return Answer(label=label, notes=("recovered_without_json",))

    return Answer(label=None, ok=False, failure="no_json")
