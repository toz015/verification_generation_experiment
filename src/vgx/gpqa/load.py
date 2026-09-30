"""Local GPQA loading, deterministic sampling, and answer-choice shuffling.

The source dataset is intentionally not downloaded here. Supply a local CSV or
Parquet file obtained under the dataset's access terms. GPQA text and manifests
belong under ``data/`` (gitignored by this repository).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

LETTERS = ("A", "B", "C", "D")
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 20260928


@dataclass(frozen=True)
class GPQAItem:
    item_id: str
    question: str
    subject: str
    choices: tuple[str, str, str, str]
    correct_index: int

    @property
    def correct_letter(self) -> str:
        return LETTERS[self.correct_index]


@dataclass(frozen=True)
class PilotSplit:
    sample: tuple[GPQAItem, ...]
    calibration: tuple[GPQAItem, ...]
    evaluation: tuple[GPQAItem, ...]
    seed: int


class DuplicateChoicesError(ValueError):
    """A valid question row whose answer options are not four distinct choices."""


def _stable_id(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]


def _normalise_subject(value: object) -> str:
    subject = str(value).strip()
    return subject or "unknown"


def _row_to_item(row: dict[str, object]) -> GPQAItem:
    required = (
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    )
    missing = [column for column in required if column not in row]
    if missing:
        raise ValueError(f"GPQA source is missing required columns: {missing}")

    question = str(row["Question"]).strip()
    if not question:
        raise ValueError("GPQA source contains an empty question")
    choices = tuple(
        str(row[column]).strip()
        for column in (
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        )
    )
    if any(not choice for choice in choices):
        raise ValueError(f"GPQA item {_stable_id(question)} has an empty choice")
    if len(set(choices)) != 4:
        raise DuplicateChoicesError(f"GPQA item {_stable_id(question)} has duplicate choices")
    return GPQAItem(
        item_id=_stable_id(question),
        question=question,
        subject=_normalise_subject(row.get("High-level domain", row.get("Subdomain", "unknown"))),
        choices=choices,  # source puts the correct option first; shuffle before inference
        correct_index=0,
    )


def _read_rows(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"GPQA local data file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    elif suffix in {".parquet", ".pq"}:
        rows = pd.read_parquet(source).to_dict(orient="records")
    else:
        raise ValueError(f"expected a .csv or .parquet GPQA file, got {source.name!r}")

    return rows


def load_items(path: str | Path) -> list[GPQAItem]:
    """Read GPQA Main strictly; any invalid row fails validation."""
    items = [_row_to_item(row) for row in _read_rows(path)]
    if not items:
        raise ValueError(f"GPQA source is empty: {Path(path).expanduser()}")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("GPQA source contains duplicate question text / item IDs")
    return items


def load_items_excluding_duplicate_choices(
    path: str | Path,
) -> tuple[list[GPQAItem], list[dict[str, str]]]:
    """Load valid rows, recording hashed IDs for rows with duplicate choices.

    Empty choices, missing fields, and duplicate question IDs remain fatal. This
    narrowly scoped exclusion is suitable for the known GPQA source anomalies.
    """
    items: list[GPQAItem] = []
    exclusions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row in _read_rows(path):
        # Validate/hash the question before processing choices, so IDs are stable
        # and duplicate source questions cannot silently disappear.
        if "Question" not in row:
            _row_to_item(row)  # raises the standard missing-column error
        question = str(row["Question"]).strip()
        item_id = _stable_id(question)
        if item_id in seen_ids:
            raise ValueError("GPQA source contains duplicate question text / item IDs")
        seen_ids.add(item_id)
        try:
            items.append(_row_to_item(row))
        except DuplicateChoicesError:
            exclusions.append({"item_id": item_id, "reason": "duplicate_answer_choices"})
    if not items:
        raise ValueError(f"GPQA source has no valid items after exclusions: {path}")
    return items, sorted(exclusions, key=lambda entry: entry["item_id"])


def _allocate_quotas(groups: dict[object, list[GPQAItem]], n: int) -> dict[object, int]:
    total = sum(map(len, groups.values()))
    if not 0 <= n <= total:
        raise ValueError(f"sample size must be between 0 and {total}, got {n}")
    quota = {key: (len(rows) * n) // total for key, rows in groups.items()}
    remainder = {key: (len(rows) * n) % total for key, rows in groups.items()}
    order = sorted(groups, key=lambda key: (-remainder[key], len(groups[key]), str(key)))
    for key in order[: n - sum(quota.values())]:
        quota[key] += 1
    return quota


def stratified_sample(
    items: Iterable[GPQAItem], n: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> list[GPQAItem]:
    """Sample by subject, using integer largest-remainder quotas."""
    groups: dict[str, list[GPQAItem]] = {}
    for item in items:
        groups.setdefault(item.subject, []).append(item)
    if not groups:
        raise ValueError("cannot sample an empty GPQA collection")
    quota = _allocate_quotas(groups, n)
    picked: list[GPQAItem] = []
    for subject in sorted(groups):
        ordered = sorted(groups[subject], key=lambda item: item.item_id)
        picked.extend(random.Random(f"{seed}:{subject}").sample(ordered, quota[subject]))
    return sorted(picked, key=lambda item: item.item_id)


def shuffle_choices(item: GPQAItem, seed: int = DEFAULT_SEED) -> GPQAItem:
    """Deterministically permute choices for this item and update its key."""
    order = list(range(len(item.choices)))
    random.Random(f"{seed}:{item.item_id}:choices").shuffle(order)
    choices = tuple(item.choices[index] for index in order)
    return replace(item, choices=choices, correct_index=order.index(item.correct_index))


def _split_60_40(items: list[GPQAItem], seed: int) -> tuple[list[GPQAItem], list[GPQAItem]]:
    groups: dict[tuple[str, int], list[GPQAItem]] = {}
    for item in items:
        groups.setdefault((item.subject, item.correct_index), []).append(item)
    calibration: list[GPQAItem] = []
    evaluation: list[GPQAItem] = []
    target_calibration = (3 * len(items)) // 5
    quota = _allocate_quotas(groups, target_calibration)
    for stratum, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: item.item_id)
        random.Random(f"{seed}:{stratum}").shuffle(rows)
        n_cal = quota[stratum]
        calibration.extend(rows[:n_cal])
        evaluation.extend(rows[n_cal:])
    return calibration, evaluation


def prepare_pilot(items: Iterable[GPQAItem], n: int = 50, seed: int = DEFAULT_SEED) -> PilotSplit:
    """Create a subject-stratified sample, shuffle its choices, then split it."""
    sample = [shuffle_choices(item, seed) for item in stratified_sample(items, n, seed)]
    calibration, evaluation = _split_60_40(sample, seed)
    return PilotSplit(
        sample=tuple(sample),
        calibration=tuple(sorted(calibration, key=lambda item: item.item_id)),
        evaluation=tuple(sorted(evaluation, key=lambda item: item.item_id)),
        seed=seed,
    )


def write_manifest(
    split: PilotSplit, path: str | Path, source_sha256: str | None = None,
    exclusions: list[dict[str, str]] | None = None,
) -> None:
    """Write IDs and correct positions only; choose a path under ignored data/."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite {destination}; remove it explicitly to redraw the sample"
        )
    payload = {
        "dataset": "gpqa_main",
        "n": len(split.sample),
        "seed": split.seed,
        "source_sha256": source_sha256,
        "excluded_items": sorted(exclusions or [], key=lambda entry: entry["item_id"]),
        "stratified_by": ["subject", "correct_option_position after shuffle"],
        "sample_ids": [item.item_id for item in split.sample],
        "calibration_ids": [item.item_id for item in split.calibration],
        "evaluation_ids": [item.item_id for item in split.evaluation],
        "correct_option_positions": {item.item_id: item.correct_index for item in split.sample},
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def restore_pilot(
    items: Iterable[GPQAItem],
    path: str | Path,
    source_sha256: str | None = None,
    exclusions: list[dict[str, str]] | None = None,
) -> PilotSplit:
    """Recreate a previously pinned split and choice order from its local manifest."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "gpqa_main":
        raise ValueError(f"{manifest_path} is not a GPQA Main manifest")
    if source_sha256 is not None and manifest.get("source_sha256") != source_sha256:
        raise ValueError("GPQA source file changed since the local sample was pinned")
    if manifest.get("excluded_items", []) != sorted(exclusions or [], key=lambda entry: entry["item_id"]):
        raise ValueError("GPQA source exclusions differ from the pinned manifest")
    by_id = {item.item_id: item for item in items}
    wanted = set(manifest.get("sample_ids", []))
    if len(wanted) != manifest.get("n"):
        raise ValueError("GPQA manifest sample_ids do not match its declared n")
    missing = wanted - by_id.keys()
    if missing:
        raise ValueError(f"GPQA manifest references missing items: {sorted(missing)}")

    seed = int(manifest["seed"])
    sample = [shuffle_choices(by_id[item_id], seed) for item_id in sorted(wanted)]
    by_shuffled_id = {item.item_id: item for item in sample}
    calibration_ids = set(manifest.get("calibration_ids", []))
    evaluation_ids = set(manifest.get("evaluation_ids", []))
    if calibration_ids & evaluation_ids or calibration_ids | evaluation_ids != wanted:
        raise ValueError("GPQA manifest calibration/evaluation IDs must partition the sample")
    positions = manifest.get("correct_option_positions", {})
    if any(positions.get(item.item_id) != item.correct_index for item in sample):
        raise ValueError("GPQA choice shuffle no longer matches the pinned manifest")
    return PilotSplit(
        sample=tuple(sample),
        calibration=tuple(by_shuffled_id[item_id] for item_id in sorted(calibration_ids)),
        evaluation=tuple(by_shuffled_id[item_id] for item_id in sorted(evaluation_ids)),
        seed=seed,
    )
