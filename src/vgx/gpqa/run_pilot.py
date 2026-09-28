"""Run generator and verifier elicitation on the pinned local GPQA pilot.

Requires a local GPQA Main CSV at the path in ``configs/gpqa_experiment.json``.
The source file, prompts, and call logs remain under gitignored ``data/`` and
``results/`` directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from vgx.common.llm import BatchRunner, CallLog, Request
from vgx.gpqa.load import (
    DEFAULT_SEED,
    PilotSplit,
    load_items,
    prepare_pilot,
    restore_pilot,
    write_manifest,
)
from vgx.gpqa.prompt import (
    SYSTEM,
    build_generator_prompt,
    build_verifier_prompt,
    parse_generator_response,
    parse_verifier_response,
)
from vgx.gpqa.score import (
    binary_forecast_metrics,
    correctness_outcome,
    fit_verifier_likelihood,
    simulate_sequential_decision,
)

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
    items = load_items(source)
    if manifest.exists():
        split = restore_pilot(items, manifest, source_sha256)
        if len(split.sample) != config["sample_size"]:
            raise ValueError("existing GPQA manifest has a different sample size from config")
        return split

    split = prepare_pilot(items, config["sample_size"], config.get("seed", DEFAULT_SEED))
    if len(split.calibration) != config["calibration_size"]:
        raise ValueError(
            f"stratified split yielded {len(split.calibration)} calibration items, "
            f"expected {config['calibration_size']}"
        )
    write_manifest(split, manifest, source_sha256)
    return split


def _runner(model: str, generation: dict) -> BatchRunner:
    return BatchRunner(
        model=model,
        max_tokens=generation["max_tokens"],
        max_model_len=generation["max_model_len"],
        dtype=generation["dtype"],
        max_num_seqs=generation["max_num_seqs"],
        chat_template_kwargs=(
            {"enable_thinking": False}
            if any(name in model.lower() for name in THINKING_MODELS)
            else None
        ),
    )


def _partition_by_id(split: PilotSplit) -> dict[str, str]:
    return {
        **{item.item_id: "calibration" for item in split.calibration},
        **{item.item_id: "evaluation" for item in split.evaluation},
    }


def _run_model(model: str, requests: list[Request], tag: str, generation: dict) -> dict[str, str]:
    model_slug = model.replace("/", "_").replace(":", "_")
    log = CallLog(RESULTS_DIR / f"{tag}__{model_slug}.jsonl")
    return _runner(model, generation).run(requests, log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="run first N pinned sample items")
    args = parser.parse_args()

    config = load_config()
    split = load_split(config)
    sample = list(split.sample)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        sample = sample[: args.limit]

    models = config["models"]
    generation = config["generation"]
    partition = _partition_by_id(split)
    generator = models["generator"]
    gen_requests = [
        Request(
            key=f"generator|{item.item_id}",
            prompt=build_generator_prompt(item),
            system=SYSTEM,
            meta={"item_id": item.item_id, "partition": partition[item.item_id], "role": "generator"},
        )
        for item in sample
    ]
    gen_responses = _run_model(generator, gen_requests, "generator", generation)
    gen_parsed = {
        item.item_id: parse_generator_response(gen_responses.get(f"generator|{item.item_id}", ""))
        for item in sample
    }

    summary: dict[str, dict] = {}
    verifier_outputs: dict[str, dict[str, object]] = {}
    for index, verifier_model in enumerate(models["verifiers"], start=1):
        tag = f"verifier_{index}"
        requests = []
        for item in sample:
            candidate = gen_parsed[item.item_id].answer
            if candidate is None:
                continue
            requests.append(
                Request(
                    key=f"{tag}|{item.item_id}",
                    prompt=build_verifier_prompt(item, candidate),
                    system=SYSTEM,
                    meta={
                        "item_id": item.item_id,
                        "partition": partition[item.item_id],
                        "role": "verifier",
                        "verifier_index": index,
                        "candidate": candidate,
                    },
                )
            )
        responses = _run_model(verifier_model, requests, tag, generation) if requests else {}
        verifier_outputs[tag] = {
            item.item_id: parse_verifier_response(responses.get(f"{tag}|{item.item_id}", ""))
            for item in sample
            if gen_parsed[item.item_id].answer is not None
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_suffix = f"_limit_{args.limit}" if args.limit is not None else ""
    output_path = RESULTS_DIR / f"pilot_records{run_suffix}.jsonl"
    # Write a fresh aggregate file from resumable call logs. It contains no
    # question text, only hashed item IDs and parsed predictions.
    with output_path.open("w", encoding="utf-8") as stream:
        for item in sample:
            answer = gen_parsed[item.item_id]
            record = {
                "item_id": item.item_id,
                "partition": partition[item.item_id],
                "subject": item.subject,
                "correct_index": item.correct_index,
                "generator_answer": answer.answer,
                "generator_p_correct": answer.p_correct,
                "generator_ok": answer.ok,
                "generator_failure": answer.failure,
                "verifiers": {},
            }
            for tag, outputs in verifier_outputs.items():
                value = outputs.get(item.item_id)
                record["verifiers"][tag] = (
                    {"p_correct": value.p_correct, "ok": value.ok, "failure": value.failure}
                    if value
                    else {"p_correct": None, "ok": False, "failure": "generator_parse_failure"}
                )
            stream.write(json.dumps(record) + "\n")

    likelihoods = {}
    calibration_status = {}
    calibration_items = [item for item in sample if partition[item.item_id] == "calibration"]
    for tag, outputs in verifier_outputs.items():
        eligible = [
            item for item in calibration_items
            if item.item_id in outputs and outputs[item.item_id].p_correct is not None
        ]
        try:
            likelihoods[tag] = fit_verifier_likelihood(
                [correctness_outcome(gen_parsed[item.item_id].answer, item.correct_index)
                 for item in eligible],
                [outputs[item.item_id].p_correct for item in eligible],
            )
            calibration_status[tag] = {"fit": True, "n": len(eligible), "failure": None}
        except ValueError as error:
            calibration_status[tag] = {"fit": False, "n": len(eligible), "failure": str(error)}

    for subset in ("calibration", "evaluation"):
        subset_items = [item for item in sample if partition[item.item_id] == subset]
        valid_generator = [
            item for item in subset_items
            if gen_parsed[item.item_id].ok
        ]
        generator_outcomes = [
            correctness_outcome(gen_parsed[item.item_id].answer, item.correct_index)
            for item in valid_generator
        ]
        generator_probs = [gen_parsed[item.item_id].p_correct for item in valid_generator]
        counts = Counter(
            result.failure or "parsed"
            for result in (gen_parsed[item.item_id] for item in subset_items)
        )
        summary[subset] = {
            "n": len(subset_items),
            "generator_parse": dict(counts),
            "generator_answer_accuracy_all_items": (
                sum(generator_outcomes) / len(subset_items) if subset_items else None
            ),
            "generator_confidence_n": len(valid_generator),
            "generator_confidence": binary_forecast_metrics(
                generator_outcomes, generator_probs
            ).to_dict(),
            "verifiers": {},
            "verifier_likelihood_fit": calibration_status,
        }
        sequential_outcomes = []
        sequential_probabilities = []
        sequential_n = 0
        for tag, outputs in verifier_outputs.items():
            eligible = [
                item
                for item in subset_items
                if item.item_id in outputs and outputs[item.item_id].p_correct is not None
            ]
            verifier_metrics = binary_forecast_metrics(
                [correctness_outcome(gen_parsed[item.item_id].answer, item.correct_index)
                 for item in eligible],
                [outputs[item.item_id].p_correct for item in eligible],
            ).to_dict()
            posterior_items = [
                item for item in eligible
                if gen_parsed[item.item_id].p_correct is not None and tag in likelihoods
            ]
            posterior_probs = [
                likelihoods[tag].posterior(
                    gen_parsed[item.item_id].p_correct,
                    outputs[item.item_id].p_correct,
                )
                for item in posterior_items
            ]
            verifier_metrics["posterior_n"] = len(posterior_items)
            verifier_metrics["posterior_from_generator_prior"] = binary_forecast_metrics(
                [correctness_outcome(gen_parsed[item.item_id].answer, item.correct_index)
                 for item in posterior_items],
                posterior_probs,
            ).to_dict()
            summary[subset]["verifiers"][tag] = verifier_metrics

        if subset == "evaluation" and likelihoods:
            for item in subset_items:
                answer = gen_parsed[item.item_id]
                if answer.p_correct is None or answer.answer is None:
                    continue
                posterior = answer.p_correct
                available = True
                for tag in verifier_outputs:
                    observation = verifier_outputs[tag].get(item.item_id)
                    if tag not in likelihoods or observation is None or observation.p_correct is None:
                        available = False
                        break
                    posterior = likelihoods[tag].posterior(posterior, observation.p_correct)
                if available:
                    sequential_n += 1
                    sequential_outcomes.append(
                        correctness_outcome(answer.answer, item.correct_index)
                    )
                    sequential_probabilities.append(posterior)
            summary[subset]["sequential_independence_n"] = sequential_n
            summary[subset]["sequential_independence_posterior"] = binary_forecast_metrics(
                sequential_outcomes, sequential_probabilities
            ).to_dict()

            scenario_results = {}
            for scenario in config.get("routing_scenarios", []):
                costs = scenario["verifier_costs"]
                if len(costs) != len(verifier_outputs) or any(
                    tag not in likelihoods for tag in verifier_outputs
                ):
                    scenario_results[scenario["name"]] = {
                        "available": False,
                        "reason": "not all verifier likelihood models were estimable",
                    }
                    continue

                r_correct = float(scenario["correct_reward"])
                l_incorrect = float(scenario["incorrect_loss"])
                threshold = l_incorrect / (r_correct + l_incorrect)
                evaluated = []
                for item in subset_items:
                    answer = gen_parsed[item.item_id]
                    observations = [
                        verifier_outputs[tag].get(item.item_id)
                        for tag in verifier_outputs
                    ]
                    if (
                        answer.answer is None
                        or answer.p_correct is None
                        or any(obs is None or obs.p_correct is None for obs in observations)
                    ):
                        continue

                    outcome = correctness_outcome(answer.answer, item.correct_index)
                    initial_action = "assert" if answer.p_correct >= threshold else "abstain"
                    initial_utility = (
                        (r_correct if outcome else -l_incorrect)
                        if initial_action == "assert"
                        else 0.0
                    )
                    posterior = answer.p_correct
                    for tag, observation in zip(verifier_outputs, observations):
                        posterior = likelihoods[tag].posterior(posterior, observation.p_correct)
                    fixed_action = "assert" if posterior >= threshold else "abstain"
                    fixed_utility = (
                        (r_correct if outcome else -l_incorrect)
                        if fixed_action == "assert"
                        else 0.0
                    ) - sum(costs)

                    first_tag = next(iter(verifier_outputs))
                    first_observation = verifier_outputs[first_tag][item.item_id]
                    first_posterior = likelihoods[first_tag].posterior(
                        answer.p_correct, first_observation.p_correct
                    )
                    one_action = "assert" if first_posterior >= threshold else "abstain"
                    one_utility = (
                        (r_correct if outcome else -l_incorrect)
                        if one_action == "assert"
                        else 0.0
                    ) - costs[0]

                    sequential = simulate_sequential_decision(
                        prior=answer.p_correct,
                        scores=[obs.p_correct for obs in observations],
                        likelihoods=[likelihoods[tag] for tag in verifier_outputs],
                        costs=costs,
                        correct_reward=r_correct,
                        incorrect_loss=l_incorrect,
                    )
                    sequential_utility = (
                        (r_correct if outcome else -l_incorrect)
                        if sequential.action == "assert"
                        else 0.0
                    ) - sum(costs[: sequential.verifiers_used])
                    evaluated.append(
                        {
                            "outcome": outcome,
                            "initial_action": initial_action,
                            "initial_utility": initial_utility,
                            "always_answer_utility": r_correct if outcome else -l_incorrect,
                            "always_abstain_utility": 0.0,
                            "one_verifier_action": one_action,
                            "one_verifier_utility": one_utility,
                            "fixed_action": fixed_action,
                            "fixed_utility": fixed_utility,
                            "sequential_action": sequential.action,
                            "sequential_utility": sequential_utility,
                            "sequential_queries": sequential.verifiers_used,
                        }
                    )

                n_evaluated = len(evaluated)

                def policy_stats(prefix: str, query_key: str | None = None) -> dict:
                    if not n_evaluated:
                        return {"n": 0, "mean_utility": None}
                    released = [row for row in evaluated if row[f"{prefix}_action"] == "assert"]
                    return {
                        "n": n_evaluated,
                        "release_coverage": len(released) / n_evaluated,
                        "accuracy_among_released": (
                            sum(row["outcome"] for row in released) / len(released)
                            if released else None
                        ),
                        "mean_utility": sum(row[f"{prefix}_utility"] for row in evaluated)
                        / n_evaluated,
                        "mean_queries": (
                            sum(row[query_key] for row in evaluated) / n_evaluated
                            if query_key else 0.0
                        ),
                    }

                scenario_results[scenario["name"]] = {
                    "n": n_evaluated,
                    "utility_units": "normalized sensitivity analysis, not measured monetary cost",
                    "always_answer": {
                        "mean_utility": (
                            sum(row["always_answer_utility"] for row in evaluated) / n_evaluated
                            if n_evaluated else None
                        ),
                        "release_coverage": 1.0 if n_evaluated else None,
                    },
                    "always_abstain": {
                        "mean_utility": 0.0 if n_evaluated else None,
                        "release_coverage": 0.0 if n_evaluated else None,
                    },
                    "confidence_only": policy_stats("initial"),
                    "query_one_verifier": policy_stats("one_verifier"),
                    "query_all_verifiers": policy_stats("fixed"),
                    "sequential_stopping": policy_stats("sequential", "sequential_queries"),
                    "actual_verifier_calls_collected_per_item": len(verifier_outputs),
                    "assumes_conditional_independence": True,
                }
            summary[subset]["routing_scenarios"] = scenario_results

    metrics_path = RESULTS_DIR / f"pilot_metrics{run_suffix}.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nrecords: {output_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
