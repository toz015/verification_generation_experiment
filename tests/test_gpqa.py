"""Entirely synthetic GPQA software checks; no benchmark/model access required."""
from copy import deepcopy
import csv
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest

from vgx.common.llm import BatchRunner, Call, CallLog, Request
from vgx.common.vertex import VertexBatchRunner
from vgx.gpqa.load import (GPQAItem, load_items, load_items_excluding_duplicate_choices,
                           prepare_pilot, restore_pilot, write_manifest)
from vgx.gpqa.prompt import build_verifier_prompt, parse_generator_response, parse_verifier_response
from vgx.gpqa.report import build_report, risk_coverage, rate_interval
from vgx.gpqa.run_pilot import collect_records, load_config, load_split, run_identity, validate_model_revisions, _runner
from vgx.gpqa.score import binary_forecast_metrics, fit_verifier_likelihood, simulate_sequential_decision, VerifierLikelihood


@pytest.fixture(autouse=True)
def forbid_real_inference(monkeypatch):
    # The dedicated cache test explicitly replaces this with a fake engine.
    monkeypatch.setitem(sys.modules, "vllm", None)


@pytest.fixture
def config():
    # Keep synthetic software fixtures pinned to the original 50-item pilot;
    # the live collection config may deliberately scale independently.
    value = load_config()
    value.update(manifest="data/gpqa/sample_50.json", sample_size=50, calibration_size=30)
    return value


@pytest.fixture
def split():
    items = [GPQAItem(f"synthetic-{i:03}", f"Synthetic question {i}",
                      ["biology", "chemistry", "physics"][i % 3],
                      (f"right-{i}", f"wrong-a-{i}", f"wrong-b-{i}", f"wrong-c-{i}"), 0)
             for i in range(90)]
    return prepare_pilot(items)


@pytest.fixture
def records(split, config):
    items = {i.item_id: i for i in split.sample}
    positions = {i.item_id: n for n, i in enumerate(split.sample)}

    def fake_model(model, requests, tag):
        responses = {}
        for request in requests:
            item = items[request.meta["item_id"]]
            index = positions[item.item_id]
            if tag == "generator":
                answer = item.correct_letter if index % 2 else "ABCD"[(item.correct_index+1) % 4]
                responses[request.key] = json.dumps({"answer": answer, "p_correct": .8 if index % 2 else .4})
            else:
                responses[request.key] = json.dumps({"p_correct": .7 if index % 2 else .3})
                assert "p_correct" not in request.meta
                assert request.meta["candidate"] in "ABCD"
        return responses
    return collect_records(split, config, fake_model)


def report(records, config):
    return build_report(records, config, bootstrap_repeats=12, synthetic=True)


def test_pinned_fifty_and_manifest_roundtrip(split, config, tmp_path):
    assert config["sample_size"] == len(split.sample) == 50
    assert config["calibration_size"] == len(split.calibration) == 30
    assert len(split.evaluation) == 20
    assert {i.item_id for i in split.calibration}.isdisjoint(i.item_id for i in split.evaluation)
    # Reconstruct the unshuffled source, then verify restoration exactly.
    original = [replace(i, choices=(f"right-{int(i.item_id[-3:])}", f"wrong-a-{int(i.item_id[-3:])}",
                                    f"wrong-b-{int(i.item_id[-3:])}", f"wrong-c-{int(i.item_id[-3:])}"), correct_index=0)
                for i in split.sample]
    path = tmp_path/"manifest.json"
    write_manifest(split, path, "source-sha")
    assert restore_pilot(original, path, "source-sha") == split
    assert all("right-" in i.choices[i.correct_index] for i in split.sample)
    assert prepare_pilot(original) == prepare_pilot(reversed(original))
    with pytest.raises(ValueError, match="source file changed"):
        restore_pilot(original, path, "different")
    with pytest.raises(FileExistsError):
        write_manifest(split, path)


def test_local_csv_loader_and_invalid_rows(tmp_path):
    path = tmp_path/"synthetic.csv"
    columns = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "High-level domain"]
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerow(["Synthetic example", "one", "two", "three", "four", "physics"])
    items = load_items(path)
    assert len(items) == 1 and items[0].correct_index == 0
    assert items[0].subject == "physics"
    with path.open("a") as stream:
        csv.writer(stream).writerow(["Synthetic example", "one", "two", "three", "four", "physics"])
    with pytest.raises(ValueError, match="duplicate"):
        load_items(path)


def test_duplicate_choice_rows_are_excluded_and_manifested(tmp_path, config):
    source, manifest = tmp_path/"gpqa.csv", tmp_path/"sample_50.json"
    columns = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2",
               "Incorrect Answer 3", "High-level domain"]
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i in range(90):
            writer.writerow([f"Synthetic valid {i}", f"right {i}", f"wrong a {i}",
                             f"wrong b {i}", f"wrong c {i}", ["biology", "physics", "chemistry"][i%3]])
        writer.writerow(["Known anomaly one", "same", "same", "other", "last", "physics"])
        writer.writerow(["Known anomaly two", "right", "wrong", "wrong", "last", "chemistry"])
        writer.writerow(["Known anomaly three", "right", "one", "two", "two", "biology"])
    original = source.read_bytes()
    config.update(source_file=str(source), manifest=str(manifest))
    items, exclusions = load_items_excluding_duplicate_choices(source)
    assert len(items) == 90
    assert exclusions == [
        {"item_id": __import__("hashlib").sha256(q.encode()).hexdigest()[:16],
         "reason": "duplicate_answer_choices"}
        for q in sorted(["Known anomaly one", "Known anomaly two", "Known anomaly three"],
                        key=lambda q: __import__("hashlib").sha256(q.encode()).hexdigest()[:16])
    ]
    split = load_split(config)
    payload = json.loads(manifest.read_text())
    assert len(split.sample) == 50 and len(split.calibration) == 30 and len(split.evaluation) == 20
    assert payload["excluded_items"] == exclusions
    assert len(payload["sample_ids"]) == 50
    assert source.read_bytes() == original
    assert load_split(config) == split
    payload["excluded_items"].pop()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exclusions differ"):
        load_split(config)


@pytest.mark.parametrize("value", [True, "0.8", -0.1, 1.1, None, float("nan"), float("inf")])
def test_invalid_confidence_keeps_valid_answer(value):
    parsed = parse_generator_response(json.dumps({"answer": "b", "p_correct": value}))
    assert parsed.answer == "B" and not parsed.ok and parsed.p_correct is None
    assert parsed.failure == "invalid_confidence"
    assert not parse_verifier_response(json.dumps({"p_correct": value})).ok


def test_answer_and_json_failures():
    assert parse_generator_response("not json").failure == "invalid_json"
    assert parse_generator_response('{"answer":"E","p_correct":0.6}').failure == "invalid_answer"
    assert parse_generator_response('{"answer":"A","p_correct":1}').ok
    assert parse_verifier_response('{"p_correct":0}').ok


def test_verifier_prompt_blinds_confidence_and_gold(split):
    item = split.sample[0]
    assert build_verifier_prompt(item, "A") == build_verifier_prompt(replace(item, correct_index=(item.correct_index+1)%4), "A")
    assert "generator_p_correct" not in build_verifier_prompt(item, "A")


def test_known_metrics_likelihood_and_degenerate_classes():
    metrics = binary_forecast_metrics([0, 1], [.25, .75])
    assert metrics.brier == pytest.approx(.0625)
    assert metrics.auroc == 1
    assert metrics.ece == pytest.approx(.25)
    assert binary_forecast_metrics([], []).n == 0
    assert binary_forecast_metrics([1, 1], [.4, .9]).auroc is None
    likelihood = fit_verifier_likelihood([0, 0, 1, 1], [.1, .1, .9, .9])
    assert likelihood.p_bin_if_correct == pytest.approx((.2, .2, .6))
    assert likelihood.posterior(.5, .9) == pytest.approx(.75)
    assert likelihood.posterior(0, .9) == 0
    assert likelihood.posterior(1, .1) == 1
    with pytest.raises(ValueError, match="both correct and incorrect"):
        fit_verifier_likelihood([1], [.9])


def test_stopping_and_missing_observation_cost():
    perfect = VerifierLikelihood((0, .5, 1), (0, 1), (1, 0))
    decision = simulate_sequential_decision(.5, [.9], [perfect], [.1], 1, 1)
    assert decision.action == "assert" and decision.verifiers_used == 1
    assert decision.expected_value_at_start == pytest.approx(.4)
    assert simulate_sequential_decision(.5, [.9], [perfect], [1], 1, 1).verifiers_used == 0
    missing = simulate_sequential_decision(.5, [None], [perfect], [.1], 1, 1)
    assert missing.action == "abstain" and missing.verifiers_used == 1
    assert missing.failure == "missing_verifier_score"
    # Missing unqueried observations must not suppress a valid immediate release.
    assert simulate_sequential_decision(1, [None], [perfect], [.1], 1, 1).action == "assert"


def test_risk_coverage_ties_and_full_denominator():
    curve = risk_coverage([1, 0, 1], [.9, .9, .2], 4)
    assert [p["coverage"] for p in curve["points"]] == [.5, .75]
    assert curve["aurc_observed_coverage"] == pytest.approx(.5*.5+.25/3)
    assert risk_coverage([], [], 0)["aurc_observed_coverage"] is None
    assert 0 < rate_interval([1]*20)["low"] < 1
    assert 0 < rate_interval([0]*20)["high"] < 1


def test_synthetic_pipeline_report(records, config):
    result = report(records, config)
    assert result["synthetic"] and "do not demonstrate" in result["interpretation"]
    assert result["complete_pilot"]
    assert [result[f"{s}_metrics"]["n"] for s in ("all", "calibration", "evaluation")] == [50, 30, 20]
    evaluation = result["evaluation_metrics"]
    assert evaluation["actual_verifier_calls_collected"] is None
    assert evaluation["full_collection_requests_if_uncached"] == 40
    assert evaluation["generator_confidence"]["ci95"]["brier"]["valid_resamples"] == 12
    assert evaluation["verifiers"]["verifier_1"]["posterior_minus_generator"]["n"] == 20
    assert evaluation["dependence"]["verifier_1|verifier_2"]["error_agreement_at_0.5"] == 1
    assert result["calibration"]["verifier_fits"]["verifier_1"]["bootstrap_stability"]["requested_resamples"] == 12
    for scenario in evaluation["routing_scenarios"].values():
        policies = scenario["policies"]
        assert scenario["routing_solver"]["method"] == "linear_grid_tables"
        assert policies["query_one_verifier"]["mean_queries"] == 1
        assert policies["query_all_verifiers"]["mean_queries"] == 2
        assert policies["query_all_verifiers"]["mean_verification_cost"] > 0
        assert "sequential_stopping_minus_confidence_only" in scenario["paired_differences"]
    json.dumps(result, allow_nan=False)


def test_no_evaluation_leakage_into_fit_or_base_rate(records, config):
    original = report(records, config)
    changed = deepcopy(records)
    for row in changed:
        if row["partition"] == "evaluation":
            row["correct_index"] = (row["correct_index"]+1)%4
            row["generator_p_correct"] = .99
            row["verifiers"]["verifier_1"]["p_correct"] = .01
    assert report(changed, config)["calibration"] == original["calibration"]


def test_failures_preserve_accuracy_denominators_and_charge_calls(records, config):
    evaluation = [r for r in records if r["partition"] == "evaluation"]
    row = evaluation[0]
    row.update(generator_answer="ABCD"[row["correct_index"]], generator_p_correct=None,
               generator_ok=False, generator_failure="invalid_confidence")
    evaluation[1]["verifiers"]["verifier_1"] = {"p_correct": None, "ok": False, "failure": "invalid_json"}
    result = report(records, config)["evaluation_metrics"]
    expected = sum(r["generator_answer"] == "ABCD"[r["correct_index"]] for r in evaluation)/20
    assert result["generator_answer_accuracy_all_items"] == expected
    assert result["generator_confidence"]["n"] == 19
    for scenario in result["routing_scenarios"].values():
        policies = scenario["policies"]
        assert all(p["n"] == 20 for p in policies.values())
        assert policies["query_one_verifier"]["mean_queries"] == 19/20
        assert policies["query_all_verifiers"]["mean_queries"] == 38/20
        assert policies["query_one_verifier"]["failure_counts"]["missing_verifier_score"] == 1
        assert policies["always_answer"]["release_coverage"] == 1
        assert policies["confidence_only"]["failure_counts"]["invalid_generator_answer_or_confidence"] == 1


def test_single_class_and_empty_partitions_are_explicit(records, config):
    for row in records:
        if row["partition"] == "calibration":
            row["generator_answer"] = "ABCD"[row["correct_index"]]
    result = report(records, config)
    assert not result["calibration"]["verifier_fits"]["verifier_1"]["fit"]
    policy = result["evaluation_metrics"]["routing_scenarios"]["balanced_low_query_cost"]["policies"]
    assert not policy["query_one_verifier"]["available"]
    assert policy["confidence_only"]["available"]
    empty = report([], config)
    assert empty["evaluation_metrics"]["n"] == 0
    assert not empty["complete_pilot"]
    json.dumps(empty, allow_nan=False)


def test_collector_skips_invalid_answer_but_verifies_invalid_confidence(split, config):
    calls = []
    def fake(model, requests, tag):
        calls.append((tag, len(requests)))
        if tag == "generator":
            return {r.key: ('{"answer":"A","p_correct":"bad"}' if n else 'bad')
                    for n, r in enumerate(requests)}
        return {r.key: '{"p_correct":0.6}' for r in requests}
    rows = collect_records(split, config, fake)
    assert calls == [("generator", 50), ("verifier_1", 49), ("verifier_2", 49)]
    assert rows[0]["verifiers"]["verifier_1"]["failure"] == "generator_parse_failure"
    assert rows[1]["generator_answer"] == "A" and not rows[1]["generator_ok"]


def test_cache_identity_changes_with_every_call_input():
    request = Request("key", "question", "system", {"candidate": "A"})
    original = BatchRunner("model", revision="a"*40)
    key = original.cache_key(request)
    assert key == BatchRunner("model", revision="a"*40).cache_key(request)
    for modified in (replace(request, prompt="other"), replace(request, system="other"),
                     replace(request, meta={"candidate": "B"})):
        assert original.cache_key(modified) != key
    for kwargs in ({"temperature": .2}, {"seed": 12}, {"max_tokens": 7}, {"dtype": "float16"},
                   {"max_model_len": 99}, {"top_p": .8}, {"chat_template_kwargs": {"enable_thinking": True}},
                   {"max_num_seqs": 3}, {"tokenizer_revision": "b"*40}):
        assert BatchRunner("model", revision="a"*40, **kwargs).cache_key(request) != key
    assert BatchRunner("model", revision="b"*40).cache_key(request) != key
    assert BatchRunner("different", revision="a"*40).cache_key(request) != key


def test_fake_engine_resume_ignores_legacy_and_stale_responses(tmp_path, monkeypatch):
    calls, engine_args = [], []
    class FakeLLM:
        def __init__(self, **kwargs):
            engine_args.append(kwargs)
        def chat(self, conversations, sampling, **kwargs):
            calls.append((conversations, sampling, kwargs))
            return [SimpleNamespace(outputs=[SimpleNamespace(text="fresh")]) for _ in conversations]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw))
    request = Request("same", "prompt", "system")
    log = CallLog(tmp_path/"calls.jsonl")
    log.append(Call("same", "model", "prompt", "legacy", {}, "bfloat16", 0, 0))
    runner = BatchRunner("model", temperature=.2, seed=42, revision="a"*40)
    assert runner.run([request], log) == {"same": "fresh"}
    assert runner.run([request], log) == {"same": "fresh"}
    assert len(calls) == 1
    runner.run([replace(request, prompt="changed")], log)
    BatchRunner("model", temperature=.3, seed=42, revision="a"*40).run([request], log)
    assert len(calls) == 3
    assert calls[0][1]["temperature"] == .2 and calls[0][1]["seed"] == 42
    assert engine_args[0]["revision"] == engine_args[0]["tokenizer_revision"] == "a"*40
    stored = list(log.records())[1]
    assert stored["meta"]["system"] == "system"
    assert stored["meta"]["runner_identity"]["revision"] == "a"*40


def test_run_identity_and_revision_preflight(config, split, tmp_path):
    source = tmp_path/"synthetic-source.csv"
    source.write_text("synthetic")
    config["source_file"] = str(source)
    first, manifest = run_identity(config, split)
    assert first == run_identity(config, split)[0]
    changed = deepcopy(config)
    changed["generation"]["temperature"] = .5
    assert first != run_identity(changed, split)[0]
    assert first != run_identity(config, split, limit=2)[0]
    source.write_text("changed synthetic source")
    assert first != run_identity(config, split)[0]
    assert "Synthetic question" not in json.dumps(manifest)
    config["model_revisions"] = {m: None for m in [config["models"]["generator"], *config["models"]["verifiers"]]}
    with pytest.raises(ValueError, match="immutable"):
        validate_model_revisions(config)
    config["model_revisions"] = {m: "a"*40 for m in config["model_revisions"]}
    validate_model_revisions(config)
    runner = _runner(config["models"]["generator"], {
        **config["generation"], "temperature": .25, "seed": 7,
        "max_model_len": 4096, "dtype": "bfloat16", "max_num_seqs": 2,
    }, "a"*40)
    assert runner.params["temperature"] == .25 and runner.params["seed"] == 7


def test_vertex_adapter_logs_usage_and_never_reuses_other_identity(tmp_path, monkeypatch):
    import vgx.common.vertex as vertex

    monkeypatch.setattr(vertex, "_get_adc_token", lambda project: "not-logged-token")
    requests_seen = []
    class FakeResponse:
        ok = True
        status_code = 200
        def json(self):
            return {"id": "synthetic-response", "model": "google/gemini-3.8-flash",
                    "choices": [{"message": {"content": '{"answer":"A","p_correct":0.8}'}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 8}}
    class FakeSession:
        def post(self, url, **kwargs):
            requests_seen.append((url, kwargs))
            return FakeResponse()
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(Session=FakeSession))
    runner = VertexBatchRunner("google/gemini-3.8-flash", "global", "synthetic-project", max_tokens=32)
    request = Request("generator|synthetic", "synthetic prompt", "synthetic system")
    path = tmp_path/"calls.jsonl"
    responses = runner.run([request], CallLog(path))
    assert responses[request.key] == '{"answer":"A","p_correct":0.8}'
    logged = list(CallLog(path).records())[0]
    assert logged["meta"]["usage"] == {"prompt_tokens": 11, "completion_tokens": 8}
    assert "not-logged-token" not in path.read_text()
    sent_body = requests_seen[0][1]["json"]
    assert sent_body["reasoning_effort"] == "low"
    assert "temperature" not in sent_body and "top_p" not in sent_body
    assert runner.cache_key(request) != VertexBatchRunner("google/gemini-3.7-flash", "global", "synthetic-project").cache_key(request)
    assert runner.cache_key(request) != VertexBatchRunner("google/gemini-3.8-flash", "us", "synthetic-project", max_tokens=32).cache_key(request)
    assert runner.cache_key(request) != VertexBatchRunner("google/gemini-3.8-flash", "global", "another-project", max_tokens=32).cache_key(request)


def test_restored_manifest_enforces_calibration_size_and_seed(config, tmp_path):
    source, manifest = tmp_path/"fixture.csv", tmp_path/"manifest.json"
    with source.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "High-level domain"])
        for i in range(90):
            writer.writerow([f"Synthetic local {i}", "one", "two", "three", "four", ["biology", "physics", "chemistry"][i%3]])
    config.update(source_file=str(source), manifest=str(manifest))
    assert load_split(config) == load_split(config)
    with pytest.raises(ValueError, match="differs from config"):
        load_split({**config, "calibration_size": 29})
    with pytest.raises(ValueError, match="differs from config"):
        load_split({**config, "seed": config["seed"]+1})


def test_exact_policy_failure_cost_and_paired_difference():
    from vgx.gpqa.report import _policy_report
    perfect = VerifierLikelihood((0, .5, 1), (0, 1), (1, 0))
    rows = [{"generator_answer": "A", "correct_index": 0, "generator_p_correct": prior,
             "verifiers": {"verifier_1": {"p_correct": None}, "verifier_2": {"p_correct": .9}}}
            for prior in (.5, 1.0)]
    scenario = {"correct_reward": 1, "incorrect_loss": 1, "verifier_costs": [.02, .02]}
    result = _policy_report(rows, ["verifier_1", "verifier_2"],
                            {"verifier_1": perfect, "verifier_2": perfect}, scenario, 12, 1)
    sequential = result["policies"]["sequential_stopping"]
    assert sequential["n"] == 2 and sequential["release_coverage"] == .5
    assert sequential["mean_queries"] == .5
    assert sequential["mean_verification_cost"] == .01
    assert sequential["utility"]["estimate"] == pytest.approx(.49)
    assert result["policies"]["query_one_verifier"]["utility"]["estimate"] == -.02
    assert result["policies"]["query_all_verifiers"]["utility"]["estimate"] == -.04
    paired = result["paired_differences"]["sequential_stopping_minus_confidence_only"]
    assert paired["deltas"]["utility"]["estimate"] == pytest.approx(-.51)


def test_offline_report_cli_uses_collection_config(records, config, tmp_path, monkeypatch):
    from vgx.gpqa import report as module
    path, output = tmp_path/"records.jsonl", tmp_path/"metrics.json"
    path.write_text("".join(json.dumps(row)+"\n" for row in records))
    (tmp_path/"run_manifest.json").write_text(json.dumps({"run_id": "synthetic-run", "config": config}))
    original = module.build_report
    # Keep CLI coverage fast while exercising the real scoring implementation.
    monkeypatch.setattr(module, "build_report", lambda records, config, **kw:
                        original(records, config, bootstrap_repeats=12, **kw))
    monkeypatch.setattr(sys, "argv", ["report", "--records", str(path), "--output", str(output), "--synthetic"])
    module.main()
    result = json.loads(output.read_text())
    assert result["synthetic"] and result["planned_sample_size"] == 50
    assert result["provenance"]["collection_run_id"] == "synthetic-run"
    assert result["provenance"]["config_differs_from_collection"] is False
    assert len(result["provenance"]["records_sha256"]) == 64
