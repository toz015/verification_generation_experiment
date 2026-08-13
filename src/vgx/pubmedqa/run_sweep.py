"""Run one model over the pinned PubMedQA sample.

Generation only; scoring is separate so metrics can be recomputed from the
JSONL log without paying for generation again. Same shape as the SciFact
runner, and it reuses the same BatchRunner.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vgx.common.llm import BatchRunner, CallLog, Request
from vgx.pubmedqa.load import Item, load_items, sample_items
from vgx.pubmedqa.prompt import SYSTEM, build_prompt, parse_response

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "pubmedqa_experiment.json"
RESULTS_DIR = REPO_ROOT / "results" / "pubmedqa"

THINKING_MODELS = ("qwen3",)


def summarise(items: list[Item], responses: dict[str, str], model: str) -> str:
    labels: Counter = Counter()
    failures: Counter = Counter()
    notes: Counter = Counter()
    missing = 0

    for item in items:
        raw = responses.get(f"{model}|{item.pubid}")
        if raw is None:
            missing += 1
            continue
        answer = parse_response(raw)
        if not answer.ok:
            failures[answer.failure] += 1
            continue
        labels[answer.label] += 1
        for note in answer.notes:
            notes[note] += 1

    n = len(items)
    parsed = sum(labels.values())
    lines = [
        f"model={model}  items={n}",
        f"  generated      {n - missing}/{n}",
        f"  parsed         {parsed}/{max(n - missing, 1)}",
        "  gold           " + ", ".join(f"{k}={v}" for k, v in
                                        Counter(i.decision for i in items).most_common()),
    ]
    if labels:
        lines.append("  predicted      " + ", ".join(f"{k}={v}" for k, v in labels.most_common()))
    if failures:
        lines.append("  parse failures " + ", ".join(f"{k}={v}" for k, v in failures.most_common()))
    if notes:
        lines.append("  flags          " + ", ".join(f"{k}={v}" for k, v in notes.most_common()))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test with N items")
    args = ap.parse_args()

    cfg = json.loads(CONFIG_PATH.read_text())
    gen = cfg["generation"]

    items = sample_items(load_items("test"))
    if args.limit:
        items = items[: args.limit]

    requests = [
        Request(
            key=f"{args.model}|{i.pubid}",
            prompt=build_prompt(i.question, i.context_text()),
            system=SYSTEM,
            meta={"pubid": i.pubid, "gold": i.decision},
        )
        for i in items
    ]

    log = CallLog(RESULTS_DIR / f"{args.model.replace('/', '_')}.jsonl")
    runner = BatchRunner(
        model=args.model,
        max_tokens=gen["max_tokens"],
        max_model_len=gen["max_model_len"],
        dtype=gen["dtype"],
        max_num_seqs=gen["max_num_seqs"],
        chat_template_kwargs=(
            {"enable_thinking": False}
            if any(t in args.model.lower() for t in THINKING_MODELS)
            else None
        ),
    )
    responses = runner.run(requests, log)

    print()
    print(summarise(items, responses, args.model))
    print(f"\nlog: {log.path}")


if __name__ == "__main__":
    main()
