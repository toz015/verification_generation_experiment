"""Offline token reconciliation, usage-based estimates, and separate billing evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from vgx.common.llm import CallLog
from vgx.common.storage import atomic_json, digest, read_jsonl


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _count(mapping: dict, *names: str) -> int | None:
    values = [mapping[name] for name in names if name in mapping and mapping[name] is not None]
    if not values:
        return None
    if any(not _number(v) or int(v) != v for v in values) or len(set(values)) != 1:
        raise ValueError("invalid or conflicting token counts: " + "/".join(names))
    return int(values[0])


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    complete: bool
    basis: str
    issues: tuple[str, ...] = ()


def reconcile_usage(usage: Any, *, model: str = "", semantics: str | None = None) -> TokenUsage:
    """Never add nested reasoning counts without resolving inclusion semantics.

    Supported overrides: completion_includes_reasoning, completion_excludes_reasoning.
    An override is a documented provider contract, not a reason to ignore a
    conflicting total. Native Google candidate/thought counts are disjoint.
    """
    if semantics not in (None, "completion_includes_reasoning", "completion_excludes_reasoning"):
        raise ValueError("unknown completion token semantics")
    if not isinstance(usage, dict):
        return TokenUsage(None, None, None, None, False, "missing", ("missing_usage",))
    try:
        native = "promptTokenCount" in usage or "candidatesTokenCount" in usage
        prompt = _count(usage, "promptTokenCount" if native else "prompt_tokens", "input_tokens")
        completion = _count(usage, "candidatesTokenCount" if native else "completion_tokens", "output_tokens")
        total = _count(usage, "totalTokenCount" if native else "total_tokens")
        details = usage.get("completion_tokens_details") or {}
        input_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        if not isinstance(details, dict) or not isinstance(input_details, dict):
            raise ValueError("invalid token detail object")
        reasoning = _count(usage, "thoughtsTokenCount") if native else _count(details, "reasoning_tokens")
        if not native and "reasoning_tokens" in usage:
            top = _count(usage, "reasoning_tokens")
            if reasoning is not None and top != reasoning:
                raise ValueError("conflicting reasoning token fields")
            reasoning = top
        cached = _count(usage, "cachedContentTokenCount") if native else _count(input_details, "cached_tokens")
        cached = cached or 0
        if prompt is None or (completion is None and (native or total is None)):
            raise ValueError("missing input or output token count")
        if cached > prompt:
            raise ValueError("cached tokens exceed input tokens")
        if native:
            output = completion + (reasoning or 0)
            basis = "google_candidates_plus_thoughts"
        elif completion is None:
            # Some Gemini length-limited replies omit completion_tokens when
            # all output was internal reasoning. Total minus input still
            # identifies billable output; do not infer zero from empty text.
            output = total - prompt
            if output < 0 or (reasoning is not None and reasoning > output):
                raise ValueError("total tokens cannot contain reported reasoning")
            basis = "total_minus_input_completion_omitted"
        elif total is not None:
            output = total - prompt
            if output < 0:
                raise ValueError("total tokens below input tokens")
            if output == completion and (reasoning is None or reasoning <= completion):
                basis = "total_confirms_completion_includes_reasoning"
                if reasoning and semantics == "completion_excludes_reasoning":
                    raise ValueError("declared output semantics conflict with total")
            elif reasoning is not None and output == completion + reasoning:
                basis = "total_confirms_completion_excludes_reasoning"
                if reasoning and semantics == "completion_includes_reasoning":
                    raise ValueError("declared output semantics conflict with total")
            else:
                raise ValueError("unreconciled total/completion/reasoning counts")
        elif semantics == "completion_excludes_reasoning":
            if reasoning is None:
                raise ValueError("separate reasoning count missing")
            output, basis = completion + reasoning, "declared_completion_excludes_reasoning"
        elif semantics == "completion_includes_reasoning":
            if reasoning is not None and reasoning > completion:
                raise ValueError("reasoning exceeds inclusive completion count")
            output, basis = completion, "declared_completion_includes_reasoning"
        elif reasoning:
            raise ValueError("ambiguous reasoning inclusion: need total or documented semantics")
        elif "gemini-3" in model and reasoning is None:
            raise ValueError("reasoning-capable endpoint missing total/reasoning semantics")
        else:
            output, basis = completion, "reported_output_no_separate_reasoning"
        if total is not None and prompt + output != total:
            raise ValueError("input plus output does not reconcile to total")
        return TokenUsage(prompt, output, reasoning, cached, True, basis)
    except ValueError as error:
        return TokenUsage(None, None, None, None, False, "unresolved", (str(error),))


def estimate_call(row: dict, pricing: dict) -> dict:
    model, meta = row.get("model", ""), row.get("meta") or {}
    usage = reconcile_usage(meta.get("usage"), model=model,
                            semantics=pricing.get("usage_semantics", {}).get(model))
    rate = pricing.get("usd_per_million_tokens", {}).get(model)
    issues = list(usage.issues)
    usd = None
    if not isinstance(rate, dict) or not all(_number(rate.get(k)) for k in ("input", "output")):
        issues.append("missing_or_invalid_rate")
    elif usage.complete:
        cached = usage.cached_input_tokens or 0
        if cached and not _number(rate.get("cached_input")):
            issues.append("missing_cached_input_rate")
        else:
            usd = ((usage.input_tokens - cached) * rate["input"]
                   + cached * rate.get("cached_input", 0)
                   + usage.output_tokens * rate["output"]) / 1_000_000
    return {"usage": asdict(usage), "estimated_usd": usd, "issues": issues,
            "status": "estimated_from_usage" if usd is not None else "unpriced",
            "confirmed_billing_usd": None}


def execution_identity(row: dict) -> tuple[str, str]:
    meta = row.get("meta") or {}
    scope = {"model": row.get("model"), "runner": meta.get("runner_identity")}
    if meta.get("execution_id"):
        return "execution_id", digest({**scope, "execution_id": meta["execution_id"]})
    if meta.get("response_id"):
        return "provider_response_id", digest({**scope, "response_id": meta["response_id"]})
    if meta.get("legacy_row_fingerprint"):
        return "legacy_row_fingerprint", meta["legacy_row_fingerprint"]
    # Exact copied legacy rows collapse; two distinct executions of the same
    # request do not collapse merely because their logical cache keys match.
    return "legacy_row_fingerprint", digest(row)


def _confirmed(evidence: dict | None) -> dict:
    if evidence is None:
        return {"status": "not_provided", "usd": None}
    if (evidence.get("currency") != "USD" or not _number(evidence.get("usd"))
            or not all(isinstance(evidence.get(k), str) and evidence[k].strip()
                       for k in ("source", "reference", "scope"))):
        raise ValueError("billing evidence requires USD amount, source, reference and scope")
    return {**evidence, "status": "imported_billing_evidence",
            "note": "External evidence supplied by the caller; not inferred from API token usage or verified with a billing service."}


def summarize_vertex_usage(logs: dict[str, CallLog], pricing: dict | None = None,
                           *, confirmed_billing: dict | None = None,
                           operation_id: str | None = None) -> dict:
    """Provider-neutral despite the historical public function name.

    A unique executed response, not a logical request key, is the accounting unit.
    Known HTTP rejections and unresolved/lost responses are disclosed separately.
    """
    pricing = pricing or {}
    unique, attempts, duplicates, legacy = {}, {}, 0, 0
    for log in logs.values():
        for row in log.records():
            if operation_id is not None and row.get("meta", {}).get("operation_id") != operation_id:
                continue
            basis, identity = execution_identity(row)
            if identity in unique:
                # Imported copies can have a migrated cache key. Compare the
                # billed facts; incompatible usage for one execution is an error.
                previous = unique[identity]
                if (previous.get("response"), previous.get("error"), previous.get("meta", {}).get("usage")) != (
                        row.get("response"), row.get("error"), row.get("meta", {}).get("usage")):
                    raise ValueError("conflicting records for the same provider execution")
                duplicates += 1
                continue
            unique[identity] = row
            legacy += basis == "legacy_row_fingerprint"
        for event in read_jsonl(str(log.path) + ".attempts.jsonl", tolerate_tail=True):
            if operation_id is None or event.get("operation_id") == operation_id:
                attempts.setdefault(event["execution_id"], []).append(event)
    completed_ids = {r.get("meta", {}).get("execution_id") for r in unique.values() if r.get("error") is None}
    unresolved = sorted(key for key, events in attempts.items()
                  if key not in completed_ids and any(e["event"] == "started" for e in events)
                  and not any(e["event"] == "rejected" for e in events))
    groups, details = {}, []
    # Cache inventories may originate from sets. Stable execution order avoids
    # cross-process float-sum drift and differing detail-array order on resume.
    for identity, row in sorted(unique.items()):
        meta, model = row.get("meta") or {}, row.get("model", "unknown")
        if row.get("error") is not None:
            continue
        estimate = estimate_call(row, pricing)
        group_key = (meta.get("role", "unknown"), model, meta.get("partition", "unknown"))
        group = groups.setdefault(group_key, {"role": group_key[0], "model": model,
            "partition": group_key[2], "successful_calls": 0, "priced_calls": 0,
            "prompt_tokens": 0, "output_tokens_including_reasoning": 0,
            "estimated_usd_for_priced_calls": 0.0})
        group["successful_calls"] += 1
        if estimate["estimated_usd"] is not None:
            group["priced_calls"] += 1
            group["prompt_tokens"] += estimate["usage"]["input_tokens"]
            group["output_tokens_including_reasoning"] += estimate["usage"]["output_tokens"]
            group["estimated_usd_for_priced_calls"] += estimate["estimated_usd"]
        details.append({"execution_identity": identity, "request_key": row["key"],
                        "execution_id": meta.get("execution_id"), "model": model,
                        "operation_id": meta.get("operation_id"), **estimate})
    rows = sorted(groups.values(), key=lambda r: (r["role"], r["model"], r["partition"]))
    successes = sum(r["successful_calls"] for r in rows)
    priced = sum(r["priced_calls"] for r in rows)
    for row in rows:
        row["mean_estimated_usd_per_priced_call"] = (
            row["estimated_usd_for_priced_calls"] / row["priced_calls"] if row["priced_calls"] else None)
    return {"schema": 2, "currency": "USD", "pricing_basis": pricing,
            "scope": {"operation_id": operation_id, "kind": "observed_executions"},
            "successful_calls": successes, "priced_calls": priced,
            "all_calls_priced": successes == priced and not unresolved and not any(r.get("error") is not None for r in unique.values()),
            "estimate_complete_for_observed_successes": successes == priced,
            "estimated_usd_for_priced_calls": sum(r["estimated_usd_for_priced_calls"] for r in rows),
            "confirmed_billing": _confirmed(confirmed_billing),
            "rejected_attempt_records": sum(r.get("error") is not None for r in unique.values()),
            "unresolved_attempts": unresolved, "duplicate_execution_rows_ignored": duplicates,
            "legacy_execution_identity_rows": legacy,
            "limitations": ["Usage estimates are not confirmed billing.",
                            "Lost responses and executions absent from logs cannot be priced.",
                            "Legacy logs may omit retries; no historical completeness claim is made."],
            "by_role_model_partition": rows, "executions": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--confirmed-billing", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing = [str(path) for path in args.log if not path.is_file()]
    if missing:
        parser.error("missing usage logs: " + ", ".join(missing))
    config = json.loads(args.pricing.read_text())
    evidence = json.loads(args.confirmed_billing.read_text()) if args.confirmed_billing else None
    result = summarize_vertex_usage({str(p): CallLog(p) for p in args.log}, config.get("api_pricing", config),
                                    confirmed_billing=evidence)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
