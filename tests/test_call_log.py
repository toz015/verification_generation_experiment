"""CallLog is the resume index for the sweep, so its edge cases matter.

A wrong answer here either silently skips work that never happened or repeats
work that did, and both corrupt the results table quietly.
"""

import json

from vgx.common.llm import Call, CallLog


def make_call(key: str, response: str = "ok", error: str | None = None) -> Call:
    return Call(
        key=key,
        model="test-model",
        prompt="p",
        response=response,
        params={"temperature": 0.0},
        dtype="bfloat16",
        latency_s=0.1,
        created=0.0,
        error=error,
    )


def test_roundtrip_marks_key_done(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")
    assert not log.has("a")
    log.append(make_call("a"))
    assert log.has("a")


def test_resume_reads_keys_from_disk(tmp_path):
    path = tmp_path / "calls.jsonl"
    CallLog(path).append(make_call("a"))

    reopened = CallLog(path)
    assert reopened.has("a"), "a fresh CallLog must recover completed keys"


def test_errored_calls_are_not_treated_as_done(tmp_path):
    """An error is logged for the failure count, but must be retried on resume."""
    path = tmp_path / "calls.jsonl"
    log = CallLog(path)
    log.append(make_call("a", response="", error="timeout"))

    assert not log.has("a")
    assert not CallLog(path).has("a")


def test_truncated_final_line_is_tolerated(tmp_path):
    """A killed process can leave a half-written line; earlier keys must survive."""
    path = tmp_path / "calls.jsonl"
    log = CallLog(path)
    log.append(make_call("a"))
    with path.open("a") as fh:
        fh.write('{"key": "b", "resp')  # truncated mid-write

    reopened = CallLog(path)
    assert reopened.has("a")
    assert not reopened.has("b")


def test_records_yields_every_parseable_row(tmp_path):
    path = tmp_path / "calls.jsonl"
    log = CallLog(path)
    log.append(make_call("a"))
    log.append(make_call("b", response="", error="boom"))

    rows = list(CallLog(path).records())
    assert [r["key"] for r in rows] == ["a", "b"]
    assert rows[1]["error"] == "boom"


def test_appended_record_is_valid_json_with_expected_fields(tmp_path):
    path = tmp_path / "calls.jsonl"
    CallLog(path).append(make_call("a"))

    rec = json.loads(path.read_text().strip())
    for field in ("key", "model", "prompt", "response", "params", "dtype"):
        assert field in rec, f"{field} must be logged for re-scoring without regeneration"


def test_responses_excludes_errors(tmp_path):
    """Scoring reads responses(); a failed call must not enter as an empty answer."""
    path = tmp_path / "calls.jsonl"
    log = CallLog(path)
    log.append(make_call("a", response="yes"))
    log.append(make_call("b", response="", error="oom"))

    assert CallLog(path).responses() == {"a": "yes"}


def test_module_imports_without_cuda():
    """vllm is CUDA-only; the module must import on a workstation without it."""
    import vgx.common.llm as mod

    assert not hasattr(mod, "LLM"), "vllm must stay lazily imported inside run()"
