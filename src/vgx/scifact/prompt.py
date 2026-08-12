"""Prompt construction and response parsing for SciFact claim verification.

The task as posed to the model: given a claim and one or more abstracts with
numbered sentences, return a label and the sentence indices that justify it.
That is the standard SciFact framing, so the resulting numbers sit alongside
how zero-shot LLM baselines are normally reported.

Parsing is deliberately forgiving about *form* and strict about *content*. A
model that wraps its JSON in prose or a markdown fence has still answered, and
rejecting that would measure formatting compliance rather than verification
ability. But a cited sentence index that does not exist, or a document that was
never shown, is a substantive error and is recorded as one.

Nothing is silently dropped. Every failure carries a reason and is counted, so
the report can state a parse-failure rate per model instead of quietly
excluding the cases where a model did worst.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from vgx.scifact.load import Abstract

LABELS = ("SUPPORT", "CONTRADICT", "NEI")

# Qwen3 emits reasoning inside <think> tags by default. Left in place it swallows
# the JSON, so it is stripped before parsing as well as disabled at generation
# time via chat_template_kwargs.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_LABEL_ALIASES = {
    "SUPPORT": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "CONTRADICTS": "CONTRADICT",
    "CONTRADICTED": "CONTRADICT",
    "REFUTE": "CONTRADICT",
    "REFUTES": "CONTRADICT",
    "NEI": "NEI",
    "NOT ENOUGH INFO": "NEI",
    "NOTENOUGHINFO": "NEI",
    "NOT_ENOUGH_INFO": "NEI",
    "NOT ENOUGH INFORMATION": "NEI",
    "INSUFFICIENT": "NEI",
}

SYSTEM = (
    "You are a scientific claim verification assistant. You judge whether "
    "abstracts from the research literature support or contradict a claim, and "
    "you cite the specific sentences that justify your judgement."
)

INSTRUCTIONS = """Decide how the abstracts bear on the claim:

- SUPPORT: an abstract states evidence that supports the claim.
- CONTRADICT: an abstract states evidence that refutes the claim.
- NEI: the abstracts do not settle the claim either way.

If the label is SUPPORT or CONTRADICT, cite the sentences that justify it using
the bracketed indices shown, grouped under the document id they came from. If
the label is NEI, cite nothing.

Reply with JSON and nothing else:

{"label": "SUPPORT" | "CONTRADICT" | "NEI", "evidence": {"<doc_id>": [<sentence indices>]}}"""


def render_abstract(abstract: Abstract) -> str:
    """One abstract with per-sentence indices the model can cite by number."""
    header = f"Document {abstract.doc_id}: {abstract.title}"
    body = "\n".join(f"[{i}] {s}" for i, s in enumerate(abstract.sentences))
    return f"{header}\n{body}"


def build_prompt(claim_text: str, abstracts: list[Abstract]) -> str:
    blocks = "\n\n".join(render_abstract(a) for a in abstracts)
    return f"Claim: {claim_text}\n\n{blocks}\n\n{INSTRUCTIONS}"


# --- parsing ----------------------------------------------------------------


@dataclass
class Answer:
    """A parsed model response.

    `ok=False` means no usable label was recovered. `notes` records substantive
    problems that were repaired rather than fatal - a hallucinated document id,
    an out-of-range sentence - so the report can quantify them instead of
    letting them vanish into the score.
    """

    label: str | None
    evidence: dict[int, tuple[int, ...]] = field(default_factory=dict)
    ok: bool = True
    failure: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def cited_sentence_count(self) -> int:
        return sum(len(v) for v in self.evidence.values())


def _extract_json(text: str) -> dict | None:
    """Find the JSON object in a response that may also contain prose."""
    cleaned = _THINK.sub("", text).strip()

    fenced = _FENCE.search(cleaned)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(cleaned)

    # Fall back to the outermost brace pair, for replies like
    # "Here is my answer: {...} Hope this helps."
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


def _normalise_label(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    return _LABEL_ALIASES.get(raw.strip().upper().replace("-", " "))


def parse_response(
    text: str,
    abstracts: list[Abstract],
) -> Answer:
    """Turn a raw completion into an Answer, validated against what was shown."""
    if not text or not text.strip():
        return Answer(label=None, ok=False, failure="empty_response")

    payload = _extract_json(text)
    if payload is None:
        return Answer(label=None, ok=False, failure="no_json")

    label = _normalise_label(payload.get("label"))
    if label is None:
        return Answer(label=None, ok=False, failure="bad_label")

    shown = {a.doc_id: len(a.sentences) for a in abstracts}
    evidence: dict[int, tuple[int, ...]] = {}
    notes: list[str] = []

    raw_evidence = payload.get("evidence") or {}
    if not isinstance(raw_evidence, dict):
        notes.append("evidence_not_an_object")
        raw_evidence = {}

    for raw_doc, raw_sentences in raw_evidence.items():
        try:
            doc_id = int(raw_doc)
        except (TypeError, ValueError):
            notes.append(f"uncastable_doc_id:{raw_doc!r}")
            continue

        if doc_id not in shown:
            # A citation to a document the model was never given. Substantive
            # error, not a formatting one, so it is counted and discarded
            # rather than scored.
            notes.append(f"unshown_doc:{doc_id}")
            continue

        if not isinstance(raw_sentences, (list, tuple)):
            raw_sentences = [raw_sentences]

        kept: list[int] = []
        for raw_index in raw_sentences:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                notes.append(f"uncastable_sentence:{raw_index!r}")
                continue
            if not 0 <= index < shown[doc_id]:
                notes.append(f"sentence_out_of_range:{doc_id}:{index}")
                continue
            kept.append(index)

        if kept:
            evidence[doc_id] = tuple(sorted(set(kept)))

    # A label of SUPPORT or CONTRADICT with no usable rationale is not a parse
    # failure: it is precisely the pathology this study measures, so it is
    # preserved and flagged.
    if label in ("SUPPORT", "CONTRADICT") and not evidence:
        notes.append("label_without_rationale")
    if label == "NEI" and evidence:
        notes.append("nei_with_rationale")
        evidence = {}

    return Answer(label=label, evidence=evidence, ok=True, notes=tuple(notes))
