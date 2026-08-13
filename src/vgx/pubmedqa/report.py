"""PubMedQA exploration report.

Reads the sweep logs, re-parses, scores with the official metrics, and writes
`docs/pubmedqa-report.md`.

The question this dataset answers for the project is narrow and specific: can
8B models produce `maybe` when the evidence genuinely does not settle the
question? PubMedQA is the only one of the three datasets with a supervised
label for that, which makes it the cleanest test of the abstention half of the
proposed mechanism.
"""

from __future__ import annotations

from pathlib import Path

from vgx.common.llm import CallLog
from vgx.pubmedqa.load import LABELS, load_items, sample_items
from vgx.pubmedqa.prompt import Answer, parse_response
from vgx.pubmedqa.score import majority_baseline, score
from vgx.scifact.score import wilson

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "pubmedqa"
OUT_PATH = REPO_ROOT / "docs" / "pubmedqa-report.md"

MODELS = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]

# From the project proposal's slide 7, recorded as literature context to verify
# against the papers rather than as reproduction targets.
PUBLISHED = [
    ("BioBERT, multi-phase fine-tuning", "~68%", "—"),
    ("single human annotator", "~78%", "—"),
    ("large instruction-tuned / medical LLMs", "~79–81%", "—"),
]


def short(model: str) -> str:
    return model.split("/")[-1]


def load_run(model: str) -> dict[str, Answer]:
    path = RESULTS_DIR / f"{model.replace('/', '_')}.jsonl"
    if not path.exists():
        return {}
    return {
        record["meta"]["pubid"]: parse_response(record["response"])
        for record in CallLog(path).records()
        if not record.get("error")
    }


def build() -> str:
    items = sample_items(load_items("test"))
    runs = {m: load_run(m) for m in MODELS}
    available = [m for m, answers in runs.items() if answers]
    if not available:
        raise SystemExit(f"no run logs found in {RESULTS_DIR}")

    n_maybe = sum(i.decision == "maybe" for i in items)
    baseline = majority_baseline(items)

    lines = [
        "# PubMedQA — exploration report",
        "",
        f"Pinned stratified sample of **{len(items)}** items from the official "
        f"500-item test set "
        f"(yes {sum(i.decision == 'yes' for i in items)}, "
        f"no {sum(i.decision == 'no' for i in items)}, "
        f"maybe {n_maybe}). Greedy decoding, bf16, no retrieval.",
        "",
        "Metrics are the official ones — `accuracy_score` and",
        "`f1_score(average='macro')`, the same calls as the authors'",
        "`evaluation.py`.",
        "",
        "## Results",
        "",
        "| model | accuracy | 95% CI | macro-F1 |",
        "|---|---|---|---|",
    ]

    results = {}
    for model in available:
        result = score(items, runs[model])
        results[model] = result
        correct = round(result.accuracy * result.n)
        lo, hi = wilson(correct, result.n)
        lines.append(
            f"| {short(model)} | {result.accuracy:.1%} | [{lo:.0%}, {hi:.0%}] "
            f"| {result.macro_f1:.3f} |"
        )

    lines += [
        f"| _always-yes baseline_ | {baseline.accuracy:.1%} | — "
        f"| {baseline.macro_f1:.3f} |",
        "",
        "The always-yes row is the free floor. It encodes no reasoning at all, yet",
        "its accuracy tracks the `yes` prevalence. The distance between its accuracy",
        "and its macro-F1 is why macro-F1 is the metric that matters here: accuracy",
        "alone cannot distinguish reasoning from guessing the majority class.",
        "",
        "### Published reference points",
        "",
        "| system | accuracy | macro-F1 |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {acc} | {maf} |" for name, acc, maf in PUBLISHED]
    lines += [
        "",
        "Taken from the project proposal and **not yet verified against the source",
        "papers**. They are on the same 500-item test set, so they are comparable in",
        "principle, but our sample is 50 of those 500 and carries a much wider",
        "interval.",
        "",
        "## Per-class detail",
        "",
        "| model | class | precision | recall | F1 | support |",
        "|---|---|---|---|---|---|",
    ]
    for model in available:
        for label in LABELS:
            c = results[model].per_class[label]
            lines.append(
                f"| {short(model)} | {label} | {c['precision']:.2f} | {c['recall']:.2f} "
                f"| {c['f1']:.2f} | {c['support']} |"
            )

    lines += [
        "",
        "## The `maybe` class",
        "",
        "| model | maybe recall | 95% CI | maybe predicted |",
        "|---|---|---|---|",
    ]
    for model in available:
        c = results[model].per_class["maybe"]
        recovered = round(c["recall"] * n_maybe)
        lo, hi = wilson(recovered, n_maybe)
        predicted = sum(1 for a in runs[model].values() if a.ok and a.label == "maybe")
        lines.append(
            f"| {short(model)} | {c['recall']:.0%} | [{lo:.0%}, {hi:.0%}] | {predicted} |"
        )

    lines += [
        "",
        f"**Read these as directional only.** With {n_maybe} `maybe` items the",
        "interval spans most of the unit line. Raising `n` to 500 in",
        "`configs/pubmedqa_experiment.json` uses the full official test set and gives",
        "55 — about two minutes of GPU time, and the right move before any of this",
        "is quoted.",
        "",
        "## Confusion",
        "",
    ]
    for model in available:
        lines += [f"### {short(model)}", "", "| gold \\ predicted | " +
                  " | ".join(LABELS) + " | unparsed |", "|---|" + "---|" * (len(LABELS) + 1)]
        for gold in LABELS:
            row = [str(results[model].confusion.get((gold, p), 0)) for p in LABELS]
            unparsed = sum(
                c for (g, p), c in results[model].confusion.items()
                if g == gold and p not in LABELS
            )
            lines.append(f"| **{gold}** | " + " | ".join(row) + f" | {unparsed} |")
        lines.append("")

    lines += [
        "## Parse health",
        "",
        "| model | parsed | failures |",
        "|---|---|---|",
    ]
    for model in available:
        r = results[model]
        failed = sum(r.parse_failures.values())
        detail = ", ".join(f"{k}={v}" for k, v in r.parse_failures.items()) or "—"
        lines.append(f"| {short(model)} | {r.n - failed}/{r.n} | {detail} |")

    lines += ["", "## Verdict", "", "_See the analysis committed alongside this run._", ""]
    return "\n".join(lines)


def main() -> None:
    text = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
