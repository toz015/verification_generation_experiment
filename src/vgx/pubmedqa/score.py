"""PubMedQA scoring.

The official evaluation is `sklearn.accuracy_score` plus
`f1_score(average="macro")` over the 500-item test set - see
https://github.com/pubmedqa/pubmedqa/blob/master/evaluation.py. Both are
computed here with the same sklearn calls rather than reimplemented.

Macro-F1 is the metric that matters. Accuracy is dominated by `yes` at 55% of
the data, so a model that never emits `maybe` can look respectable on accuracy
while scoring zero on the one class this project cares about. Macro-F1 averages
the per-class F1s and so exposes exactly that.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from vgx.pubmedqa.load import LABELS, Item
from vgx.pubmedqa.prompt import Answer

# A parse failure still has to score. Mapping it to a real label would invent an
# answer the model did not give, so it is scored as this sentinel, which can
# never match gold and therefore counts as wrong for whichever class was right.
UNPARSED = "__unparsed__"


@dataclass
class Result:
    n: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: Counter
    parse_failures: Counter

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion": {f"{g}->{p}": c for (g, p), c in sorted(self.confusion.items())},
            "parse_failures": dict(self.parse_failures),
        }


def score(items: list[Item], answers: dict[str, Answer]) -> Result:
    """Official accuracy and macro-F1, plus per-class detail and confusion."""
    truth: list[str] = []
    preds: list[str] = []
    confusion: Counter = Counter()
    failures: Counter = Counter()

    for item in items:
        answer = answers.get(item.pubid)
        truth.append(item.decision)
        if answer is None or not answer.ok or answer.label is None:
            failures[answer.failure if answer else "missing"] += 1
            preds.append(UNPARSED)
            confusion[(item.decision, UNPARSED)] += 1
        else:
            preds.append(answer.label)
            confusion[(item.decision, answer.label)] += 1

    accuracy = accuracy_score(truth, preds)
    # labels= pins the averaged set to the three real classes, so an unparsed
    # sentinel drags the score down as a miss rather than entering as a fourth
    # class with its own F1.
    macro = f1_score(truth, preds, average="macro", labels=list(LABELS), zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, preds, labels=list(LABELS), zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }

    return Result(
        n=len(items),
        accuracy=float(accuracy),
        macro_f1=float(macro),
        per_class=per_class,
        confusion=confusion,
        parse_failures=failures,
    )


def majority_baseline(items: list[Item]) -> Result:
    """Always answer `yes`: the free floor.

    PubMedQA's analogue of SciFact's cite-everything gate. It encodes no
    reasoning at all, yet accuracy tracks the `yes` prevalence of ~55%. The gap
    between this accuracy and its macro-F1 is the clearest demonstration that
    accuracy alone hides the `maybe` failure.
    """
    return score(items, {i.pubid: Answer(label="yes") for i in items})
