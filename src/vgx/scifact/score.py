"""Scoring: bridge from parsed model answers to the official SciFact evaluator.

The metrics themselves come from `third_party/scifact`, vendored unmodified
from the benchmark authors at a pinned revision. This module only converts our
records into the format that code expects, and computes the diagnostics the
official scorer does not provide.

The conversion is the part worth being careful about. The official format
expresses NEI by *omitting* the abstract entirely rather than labelling it, so
a naive mapping that emits an explicit NEI entry raises inside the loader.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.scifact.data import GoldDataset, PredictedDataset  # noqa: E402
from third_party.scifact.metrics import compute_metrics  # noqa: E402

from vgx.scifact.load import DATA_DIR, Claim  # noqa: E402
from vgx.scifact.prompt import Answer  # noqa: E402

# The official evaluator's cap: only the first three cited sentences per
# abstract count toward abstract-level credit.
MAX_ABSTRACT_SENTS = 3


def to_prediction(claim_id: int, answer: Answer) -> dict:
    """Convert one parsed answer into the official prediction record.

    NEI is represented by an empty `evidence` object. The official loader calls
    `make_label(..., allow_NEI=False)`, so emitting an explicit NEI label would
    raise rather than score.
    """
    if answer.label is None or answer.label == "NEI":
        return {"id": claim_id, "evidence": {}}

    evidence = {
        str(doc_id): {"label": answer.label, "sentences": list(sentences)}
        for doc_id, sentences in answer.evidence.items()
    }
    return {"id": claim_id, "evidence": evidence}


def official_metrics(
    predictions: list[dict],
    claims_path: Path | None = None,
    corpus_path: Path | None = None,
) -> dict:
    """Run the vendored evaluator over a list of prediction records."""
    inner = DATA_DIR / "data"
    claims_path = claims_path or inner / "claims_dev.jsonl"
    corpus_path = corpus_path or inner / "corpus.jsonl"

    with TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "predictions.jsonl"
        with pred_path.open("w") as fh:
            for record in predictions:
                fh.write(json.dumps(record) + "\n")

        gold = GoldDataset(str(corpus_path), str(claims_path))
        predicted = PredictedDataset(gold, str(pred_path))
        return compute_metrics(predicted).to_dict()


# --- diagnostics ------------------------------------------------------------


@dataclass
class Diagnostics:
    """The measurements the official scorer does not provide.

    Each maps to a specific claim in the project proposal, and none requires an
    extra generation run.
    """

    n: int
    parse_failures: Counter
    nei_recall: float | None
    nei_precision: float | None
    label_confusion: Counter
    cited_total: int
    gold_total: int
    over_cited_claims: int
    citations_beyond_cap: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "parse_failures": dict(self.parse_failures),
            "abstention": {"nei_recall": self.nei_recall, "nei_precision": self.nei_precision},
            "over_citation": {
                "cited_sentences_total": self.cited_total,
                "gold_sentences_total": self.gold_total,
                "ratio": (self.cited_total / self.gold_total) if self.gold_total else None,
                "claims_citing_more_than_gold": self.over_cited_claims,
                "citations_discarded_by_cap": self.citations_beyond_cap,
            },
            "label_confusion": {f"{g}->{p}": c for (g, p), c in self.label_confusion.items()},
        }


def diagnostics(
    claims: list[Claim],
    answers: dict[int, Answer],
    shown_docs: dict[int, list[int]],
) -> Diagnostics:
    """Compute abstention, over-citation and confusion over one run."""
    failures: Counter = Counter()
    confusion: Counter = Counter()
    cited_total = gold_total = over_cited = beyond_cap = 0
    nei_tp = nei_fn = nei_fp = 0

    for claim in claims:
        answer = answers.get(claim.id)
        if answer is None or not answer.ok:
            failures[answer.failure if answer else "missing"] += 1
            continue

        confusion[(claim.label, answer.label)] += 1

        if claim.label == "NEI" and answer.label == "NEI":
            nei_tp += 1
        elif claim.label == "NEI":
            nei_fn += 1
        elif answer.label == "NEI":
            nei_fp += 1

        cited = answer.cited_sentence_count
        gold = sum(len(claim.gold_sentences(d)) for d in shown_docs.get(claim.id, []))
        cited_total += cited
        gold_total += gold
        if cited > gold:
            over_cited += 1
        # Citations the official metric silently truncates: cited for free, but
        # carrying no abstract-level credit either way.
        beyond_cap += sum(
            max(0, len(s) - MAX_ABSTRACT_SENTS) for s in answer.evidence.values()
        )

    return Diagnostics(
        n=len(claims),
        parse_failures=failures,
        nei_recall=(nei_tp / (nei_tp + nei_fn)) if (nei_tp + nei_fn) else None,
        nei_precision=(nei_tp / (nei_tp + nei_fp)) if (nei_tp + nei_fp) else None,
        label_confusion=confusion,
        cited_total=cited_total,
        gold_total=gold_total,
        over_cited_claims=over_cited,
        citations_beyond_cap=beyond_cap,
    )


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Every reported proportion carries one: at n=50 the interval is roughly
    +/-14pp, and a bare point estimate would invite over-reading.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((centre - margin) / denom, (centre + margin) / denom)
