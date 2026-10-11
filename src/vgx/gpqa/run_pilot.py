"""Historical pilot helpers and the frozen-candidate workflow entry point.

This CLI now requires an existing validated frozen bundle. It never collects
new generator answers. The injected collect_records helper remains available
for synthetic regression tests and historical code readers.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import subprocess

from vgx.common.llm import BatchRunner, Request
from vgx.gpqa.load import (DEFAULT_SEED, PilotSplit, load_items_excluding_duplicate_choices,
                           prepare_pilot, restore_pilot, write_manifest)
from vgx.gpqa.prompt import SYSTEM, build_generator_prompt, build_verifier_prompt, parse_generator_response, parse_verifier_response

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "gpqa_experiment.json"
RESULTS_DIR = REPO_ROOT / "results" / "gpqa"
THINKING_MODELS = ("qwen3",)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_split(config: dict) -> PilotSplit:
    source = REPO_ROOT / config["source_file"]
    manifest = REPO_ROOT / config["manifest"]
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    items, exclusions = load_items_excluding_duplicate_choices(source)
    split = (restore_pilot(items, manifest, source_sha256, exclusions) if manifest.exists()
             else prepare_pilot(items, config["sample_size"], config.get("seed", DEFAULT_SEED)))
    if (len(split.sample) != config["sample_size"]
            or len(split.calibration) != config["calibration_size"]
            or split.seed != config.get("seed", DEFAULT_SEED)):
        raise ValueError("GPQA manifest sample size, calibration size or seed differs from config")
    if not manifest.exists():
        write_manifest(split, manifest, source_sha256, exclusions)
    return split


def validate_model_revisions(config: dict) -> None:
    """Never silently identify mutable Hugging Face 'main' as a frozen model."""
    revisions = config.get("model_revisions", {})
    for model in {config["models"]["generator"], *config["models"]["verifiers"]}:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revisions.get(model) or ""):
            raise ValueError(f"pin model_revisions[{model!r}] to an immutable 40-character commit before collection")


def _runner(model: str, generation: dict, revision: str | None = None) -> BatchRunner:
    return BatchRunner(
        model=model, revision=revision, tokenizer_revision=revision,
        max_tokens=generation["max_tokens"], max_model_len=generation["max_model_len"],
        dtype=generation["dtype"], max_num_seqs=generation["max_num_seqs"],
        temperature=generation["temperature"], seed=generation["seed"],
        top_p=generation.get("top_p", 1.0),
        gpu_memory_utilization=generation.get("gpu_memory_utilization", .90),
        chat_template_kwargs=({"enable_thinking": False}
                              if any(name in model.lower() for name in THINKING_MODELS) else None),
    )


def run_identity(config: dict, split: PilotSplit, limit: int | None = None) -> tuple[str, dict]:
    """Historical analysis identity, retained for manifest/regression tooling.

    Hash source content rather than writing question text into the run manifest.
    This identity no longer selects the provider-response cache directory.
    Shared provider request identities live in vgx.common.api independently.
    """
    files = [*Path(__file__).parent.glob("*.py"), Path(__file__).parents[1]/"common"/"llm.py",
             Path(__file__).parents[1]/"common"/"vertex.py",
             Path(__file__).parents[1]/"common"/"billing.py"]
    code_hashes = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(files)}
    libraries = {}
    for name in ("vllm", "transformers", "torch", "numpy", "scikit-learn", "requests"):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    project_hash = None
    if config.get("inference", {}).get("provider") == "vertex_ai":
        project_result = subprocess.run(
            ["gcloud", "config", "get-value", "project"], capture_output=True,
            text=True, check=False,
        )
        project = project_result.stdout.strip() if project_result.returncode == 0 else ""
        if project and project != "(unset)":
            project_hash = hashlib.sha256(project.encode()).hexdigest()
    payload = {
        "schema": 2, "config": config, "limit": limit,
        "sample_content_sha256": hashlib.sha256(json.dumps(
            [asdict(item) for item in split.sample], sort_keys=True).encode()).hexdigest(),
        "calibration_ids": [i.item_id for i in split.calibration],
        "evaluation_ids": [i.item_id for i in split.evaluation], "split_seed": split.seed,
        "source_sha256": hashlib.sha256((REPO_ROOT/config["source_file"]).read_bytes()).hexdigest(),
        "vertex_project_sha256": project_hash,
        "code_sha256": code_hashes, "libraries": libraries,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return digest, payload


def collect_records(split: PilotSplit, config: dict, run_model, limit: int | None = None) -> list[dict]:
    """Collect with an injected model adapter; tests supply synthetic responses."""
    if limit is not None and not 1 <= limit <= len(split.sample):
        raise ValueError("--limit must be between 1 and the pinned sample size")
    sample = list(split.sample)[:limit]
    partition = {**{i.item_id: "calibration" for i in split.calibration},
                 **{i.item_id: "evaluation" for i in split.evaluation}}
    requests = [Request(
        key=f"generator|{i.item_id}", prompt=build_generator_prompt(i), system=SYSTEM,
        meta={"item_id": i.item_id, "partition": partition[i.item_id], "role": "generator"},
    ) for i in sample]
    responses = run_model(config["models"]["generator"], requests, "generator")
    parsed = {i.item_id: parse_generator_response(responses.get(f"generator|{i.item_id}", "")) for i in sample}
    outputs = {}
    for index, model in enumerate(config["models"]["verifiers"], 1):
        tag = f"verifier_{index}"
        requests = [Request(
            key=f"{tag}|{i.item_id}", prompt=build_verifier_prompt(i, parsed[i.item_id].answer), system=SYSTEM,
            meta={"item_id": i.item_id, "partition": partition[i.item_id], "role": "verifier",
                  "verifier_index": index, "candidate": parsed[i.item_id].answer},
        ) for i in sample if parsed[i.item_id].answer is not None]
        responses = run_model(model, requests, tag) if requests else {}
        outputs[tag] = {i.item_id: parse_verifier_response(responses.get(f"{tag}|{i.item_id}", ""))
                        for i in sample if parsed[i.item_id].answer is not None}
    records = []
    for item in sample:
        answer = parsed[item.item_id]
        records.append({
            "item_id": item.item_id, "partition": partition[item.item_id], "subject": item.subject,
            "correct_index": item.correct_index, "generator_answer": answer.answer,
            "generator_p_correct": answer.p_correct, "generator_ok": answer.ok,
            "generator_failure": answer.failure,
            "verifiers": {tag: (asdict(values[item.item_id]) if item.item_id in values else
                                {"p_correct": None, "ok": False, "failure": "generator_parse_failure"})
                          for tag, values in outputs.items()},
        })
    return records


def main() -> None:
    from vgx.gpqa.workflow import main as frozen_main
    frozen_main()


if __name__ == "__main__":
    main()
