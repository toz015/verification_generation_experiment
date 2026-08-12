"""SciFact loading, from the official release tarball.

Deliberately not the HuggingFace `allenai/scifact` dataset. Two reasons:

1. It is a loading-script dataset, which `datasets>=3.0` refuses to execute.
2. Its claim splits are 1261/450/300 rather than the paper's 809/300/300,
   because the script emits one row per (claim, evidence-document) pair.
   Evaluating on that silently over-weights multi-document claims.

Every field name is asserted on load rather than assumed, so a schema change
fails here with a clear message instead of producing wrong scores later.
"""

from __future__ import annotations

import json
import random
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "scifact"
SAMPLE_PATH = REPO_ROOT / "configs" / "scifact_sample_50.json"

# Verified against the official release. Deviations mean the upstream data
# changed and every downstream number would be silently incomparable.
EXPECTED = {"corpus": 5183, "train": 809, "dev": 300, "test": 300}
LABELS = ("SUPPORT", "CONTRADICT", "NEI")

CORPUS_KEYS = {"doc_id", "title", "abstract", "structured"}
CLAIM_KEYS = {"id", "claim", "evidence", "cited_doc_ids"}
TEST_CLAIM_KEYS = {"id", "claim"}  # labels are withheld for the test split


@dataclass(frozen=True)
class Abstract:
    doc_id: int
    title: str
    sentences: tuple[str, ...]
    structured: bool

    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass(frozen=True)
class Evidence:
    """One rationale: a set of sentence indices in one abstract, plus a label."""

    doc_id: int
    sentences: tuple[int, ...]
    label: str


@dataclass(frozen=True)
class Claim:
    id: int
    text: str
    evidence: tuple[Evidence, ...]
    cited_doc_ids: tuple[int, ...]

    @property
    def is_nei(self) -> bool:
        """No evidence at all. This is SciFact's abstention ground truth."""
        return not self.evidence

    @property
    def label(self) -> str:
        if self.is_nei:
            return "NEI"
        labels = {e.label for e in self.evidence}
        if len(labels) > 1:
            # Never occurs in the released dev split. If upstream ever changes,
            # a single claim-level label stops being well defined and the
            # metrics below would quietly misreport it.
            raise ValueError(f"claim {self.id} mixes labels {sorted(labels)}")
        return labels.pop()

    @property
    def evidence_doc_ids(self) -> tuple[int, ...]:
        return tuple(sorted({e.doc_id for e in self.evidence}))

    def gold_sentences(self, doc_id: int) -> set[int]:
        return {s for e in self.evidence if e.doc_id == doc_id for s in e.sentences}


# --- acquisition ------------------------------------------------------------


def ensure_data(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract the release tarball if it is not already present."""
    inner = data_dir / "data"
    if (inner / "corpus.jsonl").exists():
        return inner

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "data.tar.gz"
    if not archive.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, archive)

    with tarfile.open(archive) as tar:
        # Guard against path traversal in archive members.
        for member in tar.getmembers():
            target = (data_dir / member.name).resolve()
            if not str(target).startswith(str(data_dir.resolve())):
                raise ValueError(f"unsafe path in archive: {member.name}")
        tar.extractall(data_dir)
    return inner


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _check_keys(row: dict, expected: set[str], where: str) -> None:
    actual = set(row)
    if actual != expected:
        raise ValueError(
            f"{where}: schema mismatch.\n"
            f"  expected: {sorted(expected)}\n"
            f"  actual:   {sorted(actual)}\n"
            "The upstream release changed; downstream metrics would be wrong."
        )


# --- loading ----------------------------------------------------------------


def load_corpus(data_dir: Path = DATA_DIR) -> dict[int, Abstract]:
    inner = ensure_data(data_dir)
    rows = _read_jsonl(inner / "corpus.jsonl")
    _check_keys(rows[0], CORPUS_KEYS, "corpus.jsonl")

    if len(rows) != EXPECTED["corpus"]:
        raise ValueError(f"corpus has {len(rows)} abstracts, expected {EXPECTED['corpus']}")

    return {
        int(r["doc_id"]): Abstract(
            doc_id=int(r["doc_id"]),
            title=r["title"],
            sentences=tuple(r["abstract"]),
            structured=bool(r["structured"]),
        )
        for r in rows
    }


def load_claims(split: str = "dev", data_dir: Path = DATA_DIR) -> list[Claim]:
    if split not in ("train", "dev", "test"):
        raise ValueError(f"unknown split {split!r}")

    inner = ensure_data(data_dir)
    rows = _read_jsonl(inner / f"claims_{split}.jsonl")
    _check_keys(rows[0], TEST_CLAIM_KEYS if split == "test" else CLAIM_KEYS, f"claims_{split}")

    if len(rows) != EXPECTED[split]:
        raise ValueError(f"{split} has {len(rows)} claims, expected {EXPECTED[split]}")

    claims = []
    for r in rows:
        evidence = []
        # Evidence keys are strings while cited_doc_ids are ints; normalising
        # to int here prevents a join that silently matches nothing.
        for doc_id, entries in r.get("evidence", {}).items():
            for entry in entries:
                evidence.append(
                    Evidence(
                        doc_id=int(doc_id),
                        sentences=tuple(entry["sentences"]),
                        label=entry["label"],
                    )
                )
        claims.append(
            Claim(
                id=int(r["id"]),
                text=r["claim"],
                evidence=tuple(evidence),
                cited_doc_ids=tuple(int(d) for d in r.get("cited_doc_ids", [])),
            )
        )
    return claims


# --- sampling ---------------------------------------------------------------


def stratified_sample(claims: list[Claim], n: int = 50, seed: int = 0) -> list[int]:
    """Pick `n` claim ids, preserving the label distribution of the split.

    Stratified rather than uniform: a uniform draw of 50 could leave only a
    handful of NEI claims and gut the abstention diagnostic. Allocation uses
    largest-remainder so the parts sum to exactly `n`.
    """
    by_label: dict[str, list[int]] = {label: [] for label in LABELS}
    for claim in claims:
        by_label[claim.label].append(claim.id)

    total = len(claims)
    sizes = {label: len(ids) for label, ids in by_label.items()}

    # Integer arithmetic throughout. Computing remainders in floating point
    # makes the allocation depend on rounding noise: on the real dev split all
    # three classes sit at exactly 2/3 above their quota, and float error
    # alone decided which class lost the tie.
    quota = {label: sizes[label] * n // total for label in LABELS}
    remainder = {label: sizes[label] * n % total for label in LABELS}

    # Ties are broken toward the smaller class. Minority classes carry the
    # least statistical power, so an extra claim is worth most there - and NEI
    # and CONTRADICT are the diagnostically interesting cells.
    order = sorted(LABELS, key=lambda label: (-remainder[label], sizes[label]))
    for label in order[: n - sum(quota.values())]:
        quota[label] += 1

    rng = random.Random(seed)
    picked: list[int] = []
    for label in LABELS:
        ids = sorted(by_label[label])  # sort first so the draw is seed-stable
        picked.extend(rng.sample(ids, quota[label]))
    return sorted(picked)


def write_sample(claim_ids: list[int], path: Path = SAMPLE_PATH, seed: int = 0) -> None:
    """Persist the sample. Refuses to overwrite.

    Re-drawing between runs would make results silently incomparable, so
    regenerating has to be a deliberate act: delete the file first.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Delete it explicitly to re-draw the sample; "
            "overwriting would make existing results incomparable."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "scifact",
        "split": "dev",
        "n": len(claim_ids),
        "seed": seed,
        "stratified_by": "claim label (SUPPORT/CONTRADICT/NEI)",
        "claim_ids": claim_ids,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_sample(path: Path = SAMPLE_PATH) -> list[int]:
    return json.loads(path.read_text())["claim_ids"]


def sample_claims(claims: list[Claim], path: Path = SAMPLE_PATH) -> list[Claim]:
    wanted = set(load_sample(path))
    picked = [c for c in claims if c.id in wanted]
    if len(picked) != len(wanted):
        missing = wanted - {c.id for c in picked}
        raise ValueError(f"sample references claim ids absent from the split: {sorted(missing)}")
    return picked


# --- fact sheet -------------------------------------------------------------


def _distribution(claims: list[Claim]) -> Counter:
    return Counter(c.label for c in claims)


def fact_sheet() -> str:
    corpus = load_corpus()
    dev = load_claims("dev")
    train = load_claims("train")
    test = load_claims("test")

    n_sent = [len(a.sentences) for a in corpus.values()]
    dev_dist = _distribution(dev)

    lines = [
        "# SciFact — dataset anatomy",
        "",
        f"Source: {DATA_URL}",
        "",
        "## Splits",
        "",
        "| split | claims |",
        "|---|---|",
        f"| train | {len(train)} |",
        f"| dev | {len(dev)} |",
        f"| test | {len(test)} (labels withheld) |",
        "",
        f"Corpus: **{len(corpus)}** abstracts, "
        f"{min(n_sent)}–{max(n_sent)} sentences each "
        f"(mean {sum(n_sent) / len(n_sent):.1f}), "
        f"{sum(a.structured for a in corpus.values())} structured.",
        "",
        "## Dev label distribution",
        "",
        "| label | claims | share |",
        "|---|---|---|",
    ]
    for label in LABELS:
        count = dev_dist[label]
        lines.append(f"| {label} | {count} | {count / len(dev):.1%} |")

    if SAMPLE_PATH.exists():
        picked = sample_claims(dev)
        sample_dist = _distribution(picked)
        lines += [
            "",
            f"## Pinned sample (n={len(picked)}, {SAMPLE_PATH.name})",
            "",
            "| label | claims | share |",
            "|---|---|---|",
        ]
        for label in LABELS:
            count = sample_dist[label]
            lines.append(f"| {label} | {count} | {count / len(picked):.1%} |")

    lines += [
        "",
        "## Schema",
        "",
        "```",
        "corpus.jsonl   doc_id:int  title:str  abstract:list[str]  structured:bool",
        "claims_*.jsonl id:int  claim:str  cited_doc_ids:list[int]",
        "               evidence: {doc_id_as_STRING: [{sentences:list[int], label:str}]}",
        "```",
        "",
        "Note the asymmetry: `evidence` keys are strings while `cited_doc_ids` are ints.",
        "A claim with an empty `evidence` object is NEI — the abstention ground truth.",
        "`cited_doc_ids` is populated even for NEI claims, so it is not a label proxy.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    dev = load_claims("dev")
    if not SAMPLE_PATH.exists():
        write_sample(stratified_sample(dev, n=50, seed=0))
        print(f"wrote {SAMPLE_PATH}")
    else:
        print(f"sample already pinned at {SAMPLE_PATH}")

    sheet = fact_sheet()
    out = REPO_ROOT / "docs" / "scifact-anatomy.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet)
    print(sheet)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
