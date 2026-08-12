"""Run one model over the pinned SciFact sample in one retrieval mode.

Used twice: with `--limit 10` as the S3 smoke test, and unrestricted as the S5
sweep. One code path either way, so the smoke test exercises exactly what the
real run does.

Generation only. Scoring is separate (S4) and reads the JSONL log, so metrics
can be recomputed or corrected without paying for generation again.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vgx.common.llm import BatchRunner, CallLog, Request
from vgx.scifact.load import Abstract, Claim, load_claims, load_corpus, sample_claims
from vgx.scifact.prompt import SYSTEM, build_prompt, parse_response
from vgx.scifact.retrieve import Retriever, oracle_docs

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "scifact_experiment.json"
RESULTS_DIR = REPO_ROOT / "results" / "scifact"

# Qwen3 needs thinking explicitly disabled; other families ignore the flag, so
# passing it unconditionally would risk a template error on models that reject
# unknown kwargs.
THINKING_MODELS = ("qwen3",)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def context_for(
    claim: Claim,
    mode: str,
    corpus: dict[int, Abstract],
    retriever: Retriever | None,
    k: int,
) -> list[Abstract]:
    """The abstracts shown to the model for this claim under this mode."""
    if mode == "oracle":
        return [corpus[d] for d in oracle_docs(claim) if d in corpus]
    if mode == "bm25":
        assert retriever is not None
        return [corpus[d] for d in retriever.search(claim.text, k)]
    raise ValueError(f"unknown mode {mode!r}")


def summarise(claims: list[Claim], contexts: dict[int, list[Abstract]], responses: dict[str, str],
              model: str, mode: str) -> str:
    """Report parse health and label behaviour without scoring correctness.

    Parse failures are reported as a rate rather than dropped: excluding them
    would flatter whichever model fails most, which is the opposite of what
    this study needs.
    """
    failures: Counter = Counter()
    labels: Counter = Counter()
    notes: Counter = Counter()
    cited: list[int] = []
    missing = 0

    for claim in claims:
        key = f"{model}|{mode}|{claim.id}"
        raw = responses.get(key)
        if raw is None:
            missing += 1
            continue
        answer = parse_response(raw, contexts[claim.id])
        if not answer.ok:
            failures[answer.failure] += 1
            continue
        labels[answer.label] += 1
        for note in answer.notes:
            notes[note.split(":")[0]] += 1
        cited.append(answer.cited_sentence_count)

    n = len(claims)
    parsed = sum(labels.values())
    lines = [
        f"model={model}  mode={mode}  claims={n}",
        f"  generated      {n - missing}/{n}",
        f"  parsed         {parsed}/{n - missing}"
        + (f"  ({parsed / max(n - missing, 1):.0%})" if n - missing else ""),
    ]
    if failures:
        lines.append("  parse failures " + ", ".join(f"{k}={v}" for k, v in failures.most_common()))
    if labels:
        lines.append("  labels         " + ", ".join(f"{k}={v}" for k, v in labels.most_common()))
    if notes:
        lines.append("  flags          " + ", ".join(f"{k}={v}" for k, v in notes.most_common()))
    if cited:
        lines.append(
            f"  cited sentences per claim: mean {sum(cited) / len(cited):.2f}, "
            f"max {max(cited)}, zero-citation {sum(c == 0 for c in cited)}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("oracle", "bm25"), default="oracle")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test with N claims")
    ap.add_argument("--k", type=int, default=None, help="override retrieval k")
    args = ap.parse_args()

    cfg = load_config()
    k = args.k or cfg["retrieval"]["k"]
    gen = cfg["generation"]

    corpus = load_corpus()
    claims = sample_claims(load_claims("dev"))
    if args.limit:
        claims = claims[: args.limit]

    retriever = Retriever(corpus) if args.mode == "bm25" else None
    contexts = {c.id: context_for(c, args.mode, corpus, retriever, k) for c in claims}

    requests = [
        Request(
            key=f"{args.model}|{args.mode}|{c.id}",
            prompt=build_prompt(c.text, contexts[c.id]),
            system=SYSTEM,
            meta={"claim_id": c.id, "mode": args.mode, "gold_label": c.label,
                  "shown_docs": [a.doc_id for a in contexts[c.id]]},
        )
        for c in claims
    ]

    slug = args.model.replace("/", "_")
    log = CallLog(RESULTS_DIR / f"{slug}__{args.mode}.jsonl")

    template_kwargs = (
        {"enable_thinking": False}
        if any(t in args.model.lower() for t in THINKING_MODELS)
        else None
    )
    runner = BatchRunner(
        model=args.model,
        max_tokens=gen["max_tokens"],
        max_model_len=gen["max_model_len"],
        dtype=gen["dtype"],
        chat_template_kwargs=template_kwargs,
    )
    responses = runner.run(requests, log)

    print()
    print(summarise(claims, contexts, responses, args.model, args.mode))
    print(f"\nlog: {log.path}")


if __name__ == "__main__":
    main()
