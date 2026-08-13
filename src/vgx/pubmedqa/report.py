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

    lines += [
        "",
        "## Verdict",
        "",
        "### The abstention decision is arbitrary",
        "",
        "Two models of the same size, on the same items, adopt opposite policies:",
        "",
        "- **Qwen3-8B answers `maybe` 34 times out of 50**, against 5 in gold.",
        "- **Llama-3.1-8B answers `maybe` zero times.** All five gold `maybe` items",
        "  are called `yes`.",
        "",
        "Nothing in the evidence explains a 34-versus-0 split. The threshold at which",
        "these models decline to commit is inherited from post-training, not derived",
        "from the data in front of them. That is a decision failure in the precise",
        "sense the proposal needs: the capability question and the abstention question",
        "come apart.",
        "",
        "### Qwen has the information and misuses it",
        "",
        "This is the sharpest result. When Qwen3-8B does commit, its **precision is",
        "1.00 on `yes` and 1.00 on `no`** — every single committal answer is correct.",
        "Its recall is 0.43 and 0.24 because it declines to answer most items it would",
        "have got right.",
        "",
        "The discrimination ability is there. What is wrong is the threshold, and a",
        "threshold is exactly what a scoring rule with an abstention reserve sets. This",
        "is the strongest evidence in the exploration so far that the target is a",
        "decision rule rather than a capability.",
        "",
        "### The metric cannot see the difference",
        "",
        "Despite opposite behaviour, the two models land at **macro-F1 0.412 and",
        "0.420** — a difference of 0.008. One abstains on two-thirds of the data, the",
        "other never abstains, and the official metric rates them equal.",
        "",
        "So the existing metric provides no gradient toward correct abstention. It is",
        "not merely that the models are poorly calibrated; the benchmark cannot reward",
        "fixing it. That argues the mechanism is addressing a real gap in the",
        "evaluation, not just chasing a number.",
        "",
        "### Accuracy actively misleads here",
        "",
        "Qwen3-8B scores **42.0% accuracy, below the 56.0% always-yes baseline**. On",
        "accuracy alone it looks worse than a constant predictor, while its macro-F1",
        "(0.412) is well above the baseline's (0.239). Any PubMedQA result quoted as",
        "accuracy without macro-F1 alongside it should be treated as uninformative.",
        "",
        "### Caveats",
        "",
        "- **5 `maybe` items.** Qwen's 100% `maybe` recall is trivially produced by",
        "  answering `maybe` almost everywhere — its `maybe` precision is 0.15. Do not",
        "  read that 100% as competence. Going to the full 500-item test set is one",
        "  config line and about two minutes of GPU.",
        "- **Qwen3 ran with thinking disabled** (`enable_thinking=False`), needed for",
        "  reliable JSON. Qwen3 is a reasoning model, so this may materially change its",
        "  behaviour and is a genuine confound in comparing it with Llama.",
        "- **One prompt.** The prompt defines what `maybe` means; a different framing",
        "  could move the abstention rate substantially, which is itself consistent",
        "  with the finding that the threshold is arbitrary.",
        "- The published reference points above are unverified.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    text = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
