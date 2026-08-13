"""S6: turn the sweep logs into the headroom report.

Reads the JSONL call logs, re-parses every response, scores with the vendored
official evaluator, computes the three diagnostics, and writes
`docs/scifact-report.md`.

Scoring is separate from generation, so this can be re-run freely as the
analysis changes without touching the GPU.

The report ends in an explicit verdict about which of three failure modes
dominates - retrieval, capability, or decision - because only the third is
addressable by mechanism design. A finding of little headroom is reported as
such rather than argued around.
"""

from __future__ import annotations

from pathlib import Path

from vgx.common.llm import CallLog
from vgx.scifact.load import Abstract, Claim, load_claims, load_corpus, sample_claims
from vgx.scifact.prompt import Answer, parse_response
from vgx.scifact.score import diagnostics, official_metrics, to_prediction, wilson

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "scifact"
OUT_PATH = REPO_ROOT / "docs" / "scifact-report.md"

MODELS = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]
MODES = ["oracle", "bm25"]


def short(model: str) -> str:
    return model.split("/")[-1]


def load_run(
    model: str, mode: str, corpus: dict[int, Abstract]
) -> tuple[dict[int, Answer], dict[int, list[int]]]:
    """Re-parse one run's log into answers keyed by claim id."""
    path = RESULTS_DIR / f"{model.replace('/', '_')}__{mode}.jsonl"
    answers: dict[int, Answer] = {}
    shown: dict[int, list[int]] = {}
    if not path.exists():
        return answers, shown

    for record in CallLog(path).records():
        if record.get("error"):
            continue
        claim_id = record["meta"]["claim_id"]
        doc_ids = record["meta"]["shown_docs"]
        shown[claim_id] = doc_ids
        answers[claim_id] = parse_response(record["response"], [corpus[d] for d in doc_ids])
    return answers, shown


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def ci(successes: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(successes, n)
    return f"[{lo:.0%}, {hi:.0%}]"


def build() -> str:
    corpus = load_corpus()
    claims = sample_claims(load_claims("dev"))
    by_id = {c.id: c for c in claims}

    runs: dict[tuple[str, str], tuple[dict[int, Answer], dict[int, list[int]]]] = {}
    for model in MODELS:
        for mode in MODES:
            runs[(model, mode)] = load_run(model, mode, corpus)

    available = [key for key, (answers, _) in runs.items() if answers]
    if not available:
        raise SystemExit(f"no run logs found in {RESULTS_DIR}")

    lines: list[str] = [
        "# SciFact — exploration report",
        "",
        f"Pinned sample of **{len(claims)}** dev claims "
        f"(SUPPORT {sum(c.label == 'SUPPORT' for c in claims)}, "
        f"CONTRADICT {sum(c.label == 'CONTRADICT' for c in claims)}, "
        f"NEI {sum(c.label == 'NEI' for c in claims)}), greedy decoding, bf16.",
        "",
        "Official metrics come from the SciFact evaluator vendored at a pinned",
        "revision; see `third_party/scifact/PROVENANCE.md`. Every proportion",
        "carries a 95% Wilson interval — at n=50 that is roughly ±14pp, so this",
        "sample can show a large label-versus-rationale gap but cannot rank two",
        "models against each other.",
        "",
        "## Official metrics",
        "",
        "| model | mode | label-only F1 | label+rationale F1 | gap |",
        "|---|---|---|---|---|",
    ]

    metric_rows: dict[tuple[str, str], dict] = {}
    for model, mode in available:
        answers, _ = runs[(model, mode)]
        preds = [to_prediction(cid, answers[cid]) for cid in sorted(answers)]
        result = official_metrics(preds)
        metric_rows[(model, mode)] = result
        label_only = result["abstract_label_only"]["f1"]
        rationalized = result["abstract_rationalized"]["f1"]
        lines.append(
            f"| {short(model)} | {mode} | {label_only:.3f} | {rationalized:.3f} "
            f"| **{label_only - rationalized:+.3f}** |"
        )

    lines += [
        "",
        "The gap column is the headline. Label-only credits getting the verdict",
        "right; label+rationale credits it only when the cited sentences are also",
        "right. A large positive gap means the models know *what* is true but not",
        "*where* it is written — a decision failure about citation, not a",
        "capability failure about entailment.",
        "",
        "## Reference points",
        "",
        "| strategy | label+rationale F1 |",
        "|---|---|",
        "| gold annotations replayed | 1.000 |",
        "| cite sentences 0–2, gold labels and documents | 0.220 |",
        "| cite sentence 0, gold labels and documents | 0.053 |",
        "",
        "The 0.220 row is the free floor: it encodes no evidence selection at all,",
        "yet earns credit because the official metric caps rationales at three",
        "sentences and grants credit when a gold rationale is a subset of what was",
        "cited. Any model scoring near it has demonstrated no citation ability.",
        "",
        "## Diagnostics",
        "",
        "### Abstention (NEI)",
        "",
        "| model | mode | NEI recall | 95% CI | NEI precision |",
        "|---|---|---|---|---|",
    ]

    diag_rows = {}
    for model, mode in available:
        answers, shown = runs[(model, mode)]
        d = diagnostics(claims, answers, shown)
        diag_rows[(model, mode)] = d
        n_nei = sum(c.label == "NEI" for c in claims)
        recovered = round((d.nei_recall or 0) * n_nei)
        lines.append(
            f"| {short(model)} | {mode} | {pct(d.nei_recall)} | "
            f"{ci(recovered, n_nei)} | {pct(d.nei_precision)} |"
        )

    lines += [
        "",
        "### Over-citation",
        "",
        "| model | mode | cited | gold | ratio | claims over-citing | beyond the cap |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, mode in available:
        d = diag_rows[(model, mode)]
        ratio = f"{d.cited_total / d.gold_total:.2f}x" if d.gold_total else "n/a"
        lines.append(
            f"| {short(model)} | {mode} | {d.cited_total} | {d.gold_total} | {ratio} "
            f"| {d.over_cited_claims} | {d.citations_beyond_cap} |"
        )

    lines += [
        "",
        "`beyond the cap` counts cited sentences the official metric silently",
        "discards — emitted for free, scored neither way.",
        "",
        "### Retrieval isolation (oracle − BM25)",
        "",
        "| model | label-only Δ | label+rationale Δ |",
        "|---|---|---|",
    ]
    for model in MODELS:
        if (model, "oracle") in metric_rows and (model, "bm25") in metric_rows:
            o, b = metric_rows[(model, "oracle")], metric_rows[(model, "bm25")]
            lines.append(
                f"| {short(model)} "
                f"| {o['abstract_label_only']['f1'] - b['abstract_label_only']['f1']:+.3f} "
                f"| {o['abstract_rationalized']['f1'] - b['abstract_rationalized']['f1']:+.3f} |"
            )

    lines += [
        "",
        "BM25 hit@5 on full dev is 89.9%, and recall saturates near 96% at any k,",
        "so roughly 4% of claims have gold abstracts that are unreachable by lexical",
        "retrieval. That is a floor on this difference which no model can close.",
        "",
        "### Parse health",
        "",
        "| model | mode | parsed | failures |",
        "|---|---|---|---|",
    ]
    for model, mode in available:
        d = diag_rows[(model, mode)]
        failed = sum(d.parse_failures.values())
        detail = ", ".join(f"{k}={v}" for k, v in d.parse_failures.items()) or "—"
        lines.append(f"| {short(model)} | {mode} | {d.n - failed}/{d.n} | {detail} |")

    lines += [
        "",
        "Parse failures are reported rather than dropped: excluding them would",
        "flatter whichever model fails most.",
        "",
        "## Verdict",
        "",
        "### The predicted failure is not the observed one",
        "",
        "The project proposal predicts `label >> rationale` for zero-shot LLMs:",
        "models that reach the right verdict while citing the wrong sentences.",
        "**The data does not support that at this sample size.** Gaps are 3.5 to",
        "9.3 points, not the tens of points predicted. Read as a ratio, between",
        "**84% and 91%** of label-correct abstracts also carry a correct rationale",
        "(0.888 / 0.840 / 0.909 / 0.857 across the four runs).",
        "When these models know the answer, they mostly know where it is written.",
        "",
        "Both models also sit far above the 0.220 free floor in oracle mode, so",
        "they are demonstrating genuine evidence selection rather than harvesting",
        "the metric's cap.",
        "",
        "### Retrieval failure dominates",
        "",
        "The oracle-minus-BM25 difference is **+0.23 for Qwen3-8B and +0.38 for",
        "Llama-3.1-8B** — three to ten times the label-versus-rationale gap. The",
        "largest single source of error is that the right abstract is not in front",
        "of the model, and roughly 4 points of that is BM25's hard ceiling rather",
        "than anything a model or a mechanism could fix.",
        "",
        "For a mechanism-design paper this is the important caveat: measured under",
        "realistic retrieval, most of the loss is upstream of any citation",
        "decision. Any claimed improvement has to be shown net of it.",
        "",
        "### Where headroom does exist",
        "",
        "Two of the three diagnostics show real weakness, and both are decision",
        "failures rather than capability failures:",
        "",
        "**Over-citation, and it scales with uncertainty.** In oracle mode the",
        "models cite 1.24x and 1.32x the gold sentence count. Under BM25 that rises",
        "to **2.80x and 6.59x**. Llama emits 303 cited sentences against 46 gold,",
        "**120 of them past the cap where the metric discards them silently** —",
        "cited at no cost and no benefit. When evidence is weaker, citation volume",
        "goes up rather than down. That is exactly the behaviour a proper scoring",
        "rule with an abstention reserve is meant to price.",
        "",
        "**Abstention is weak.** NEI recall is 47-68%, so between a third and a half",
        "of claims with no supporting evidence still receive a confident verdict.",
        "Llama's NEI precision falls to 55.6% under BM25, meaning its abstentions",
        "become unreliable in both directions at once. SciFact has ground truth for",
        "abstention and these models are not using it.",
        "",
        "### Answer to the headroom question",
        "",
        "There is room for mechanism design on SciFact, but **not in the place the",
        "proposal aims at**. The label-versus-rationale gap is small. The exploitable",
        "gaps are abstention and citation volume under uncertainty, and both appear",
        "specifically when retrieval is realistic rather than oracular.",
        "",
        "The practical implication is to run the mechanism against BM25 retrieval,",
        "not oracle abstracts. Oracle mode compresses precisely the pathologies the",
        "mechanism targets: over-citation is 2-5x lower and the abstention problem",
        "is milder.",
        "",
        "### What would change this conclusion",
        "",
        "- **n=50.** Every proportion carries roughly ±14pp. The label-rationale gap",
        "  is small enough that full dev (300 claims) could move it, though not,",
        "  plausibly, from 9 points to tens of points.",
        "- **Llama's BM25 run had 4 parse failures**, scored as empty evidence. That",
        "  depresses its BM25 row by an unknown amount and makes its dramatic",
        "  collapse there partly a formatting artefact rather than a reasoning one.",
        "- **Llama predicted CONTRADICT once in 50 under BM25** against 11 in gold.",
        "  A near-total collapse of one class is worth understanding before drawing",
        "  conclusions from that row.",
        "- **One prompt format.** These numbers describe this prompt. A different",
        "  framing could shift the label-rationale balance.",
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
