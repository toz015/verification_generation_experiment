"""PubMedQA loading, on the paper's official test split.

Two sources are joined:

* `qiaojin/PubMedQA`, config `pqa_labeled` - the 1000 expert-annotated items,
  which HuggingFace ships as one undifferentiated split.
* `test_ground_truth.json` from the authors' GitHub - the 500 pubids that make
  up the published test set.

The join matters. Published numbers (BioBERT ~68%, instruction-tuned LLMs
~79-81%) are reported on those 500 items, so scoring on all 1000, or on a
private split, silently breaks comparability.

Unlike SciFact there is no retrieval stage: the context ships with each item.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PARQUET_URL = (
    "https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/main/"
    "pqa_labeled/train-00000-of-00001.parquet"
)
TEST_IDS_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "pubmedqa"
SAMPLE_PATH = REPO_ROOT / "configs" / "pubmedqa_sample_50.json"

LABELS = ("yes", "no", "maybe")

# Verified against the released data. A mismatch means upstream changed and
# every downstream number would be silently incomparable.
EXPECTED = {"labeled": 1000, "test": 500}
EXPECTED_BALANCE = {"yes": 552, "no": 338, "maybe": 110}


@dataclass(frozen=True)
class Item:
    pubid: str
    question: str
    contexts: tuple[str, ...]
    sections: tuple[str, ...]  # BACKGROUND / METHODS / RESULTS ...
    decision: str  # yes | no | maybe

    def context_text(self) -> str:
        """Sections labelled where headings exist, matching how they appear in the abstract."""
        if len(self.sections) == len(self.contexts):
            return "\n\n".join(f"{s}: {c}" for s, c in zip(self.sections, self.contexts))
        return "\n\n".join(self.contexts)


# --- acquisition ------------------------------------------------------------


def ensure_data(data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet = data_dir / "pqa_labeled.parquet"
    test_ids = data_dir / "test_ground_truth.json"

    if not parquet.exists():
        print(f"downloading {PARQUET_URL}")
        urllib.request.urlretrieve(PARQUET_URL, parquet)
    if not test_ids.exists():
        print(f"downloading {TEST_IDS_URL}")
        urllib.request.urlretrieve(TEST_IDS_URL, test_ids)
    return parquet, test_ids


def load_items(split: str = "test", data_dir: Path = DATA_DIR) -> list[Item]:
    """Load PQA-L. `split="test"` restricts to the official 500-item test set."""
    import pyarrow.parquet as pq

    parquet, test_ids_path = ensure_data(data_dir)
    table = pq.read_table(parquet)

    if table.num_rows != EXPECTED["labeled"]:
        raise ValueError(f"pqa_labeled has {table.num_rows} rows, expected {EXPECTED['labeled']}")

    pubids = [str(x) for x in table.column("pubid").to_pylist()]
    questions = table.column("question").to_pylist()
    contexts = table.column("context").to_pylist()
    decisions = table.column("final_decision").to_pylist()

    ground_truth = json.loads(test_ids_path.read_text())
    if len(ground_truth) != EXPECTED["test"]:
        raise ValueError(f"test set has {len(ground_truth)} ids, expected {EXPECTED['test']}")

    items = []
    for pubid, question, context, decision in zip(pubids, questions, contexts, decisions):
        if split == "test" and pubid not in ground_truth:
            continue
        if decision not in LABELS:
            raise ValueError(f"pubid {pubid}: unexpected decision {decision!r}")
        # The authors' ground-truth file is authoritative for the test split.
        if pubid in ground_truth and ground_truth[pubid] != decision:
            raise ValueError(f"pubid {pubid}: HF says {decision!r}, ground truth says "
                             f"{ground_truth[pubid]!r}")
        items.append(
            Item(
                pubid=pubid,
                question=question,
                # `long_answer` is deliberately never read: it is the withheld
                # CONCLUSIONS section and states the answer outright.
                contexts=tuple(context["contexts"]),
                sections=tuple(context.get("labels") or ()),
                decision=decision,
            )
        )

    expected_n = EXPECTED["test"] if split == "test" else EXPECTED["labeled"]
    if len(items) != expected_n:
        raise ValueError(f"{split} split has {len(items)} items, expected {expected_n}")
    return items


# --- sampling ---------------------------------------------------------------


def stratified_sample(items: list[Item], n: int = 50, seed: int = 0) -> list[str]:
    """Pick `n` pubids preserving the label distribution.

    Integer arithmetic throughout, with ties broken toward the smaller class -
    the same rule as SciFact, and for the same reason: floating-point
    remainders let rounding noise decide which class loses an item, and `maybe`
    is both the smallest class and the one this dataset exists to test.
    """
    by_label: dict[str, list[str]] = {label: [] for label in LABELS}
    for item in items:
        by_label[item.decision].append(item.pubid)

    total = len(items)
    sizes = {label: len(ids) for label, ids in by_label.items()}
    quota = {label: sizes[label] * n // total for label in LABELS}
    remainder = {label: sizes[label] * n % total for label in LABELS}

    order = sorted(LABELS, key=lambda label: (-remainder[label], sizes[label]))
    for label in order[: n - sum(quota.values())]:
        quota[label] += 1

    rng = random.Random(seed)
    picked: list[str] = []
    for label in LABELS:
        picked.extend(rng.sample(sorted(by_label[label]), quota[label]))
    return sorted(picked)


def write_sample(pubids: list[str], path: Path = SAMPLE_PATH, seed: int = 0) -> None:
    """Persist the sample. Refuses to overwrite, so results stay comparable."""
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Delete it explicitly to re-draw the sample; "
            "overwriting would make existing results incomparable."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": "pubmedqa",
                "split": "official test (500)",
                "n": len(pubids),
                "seed": seed,
                "stratified_by": "final_decision (yes/no/maybe)",
                "pubids": pubids,
            },
            indent=2,
        )
        + "\n"
    )


def load_sample(path: Path = SAMPLE_PATH) -> list[str]:
    return json.loads(path.read_text())["pubids"]


def sample_items(items: list[Item], path: Path = SAMPLE_PATH) -> list[Item]:
    wanted = set(load_sample(path))
    picked = [i for i in items if i.pubid in wanted]
    if len(picked) != len(wanted):
        missing = wanted - {i.pubid for i in picked}
        raise ValueError(f"sample references pubids absent from the split: {sorted(missing)}")
    return picked


# --- fact sheet -------------------------------------------------------------


def fact_sheet() -> str:
    full = load_items("all")
    test = load_items("test")
    full_dist = Counter(i.decision for i in full)
    test_dist = Counter(i.decision for i in test)
    n_sec = [len(i.contexts) for i in test]

    lines = [
        "# PubMedQA — dataset anatomy",
        "",
        f"Items: `qiaojin/PubMedQA` config `pqa_labeled` ({len(full)} expert-annotated).",
        f"Split: official {len(test)}-item test set, from the authors' "
        "`test_ground_truth.json`.",
        "",
        "No retrieval stage — the context ships with each item.",
        "",
        "## Label distribution",
        "",
        "| label | PQA-L (1000) | official test (500) |",
        "|---|---|---|",
    ]
    for label in LABELS:
        lines.append(
            f"| {label} | {full_dist[label]} ({full_dist[label] / len(full):.1%}) "
            f"| {test_dist[label]} ({test_dist[label] / len(test):.1%}) |"
        )

    if SAMPLE_PATH.exists():
        picked = sample_items(test)
        dist = Counter(i.decision for i in picked)
        lines += ["", f"## Pinned sample (n={len(picked)})", "", "| label | items |", "|---|---|"]
        for label in LABELS:
            lines.append(f"| {label} | {dist[label]} |")
        lines += [
            "",
            f"`maybe` has only {dist['maybe']} items here. That is the class this dataset",
            "exists to test, so its recall carries a very wide interval; treat it as",
            "directional. Raising `n` to 500 in the config gives 55.",
        ]

    lines += [
        "",
        "## Structure",
        "",
        f"Context sections per item: {min(n_sec)}–{max(n_sec)} "
        f"(mean {sum(n_sec) / len(n_sec):.1f}), labelled BACKGROUND / METHODS / RESULTS.",
        "",
        "```",
        "pubid          str",
        "question       str",
        "context        {contexts: list[str], labels: list[str], meshes: list[str]}",
        "long_answer    str   <- the withheld CONCLUSIONS section",
        "final_decision yes | no | maybe",
        "```",
        "",
        "**`long_answer` must never enter the prompt.** It is the abstract's",
        "conclusion and states the answer outright; including it turns the task into",
        "copying. The loader does not even read the field.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    test = load_items("test")
    if not SAMPLE_PATH.exists():
        write_sample(stratified_sample(test, n=50, seed=0))
        print(f"wrote {SAMPLE_PATH}")
    else:
        print(f"sample already pinned at {SAMPLE_PATH}")

    sheet = fact_sheet()
    out = REPO_ROOT / "docs" / "pubmedqa-anatomy.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet)
    print(sheet)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
