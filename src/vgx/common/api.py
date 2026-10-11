"""Paid-request boundary: stable identity, durable response cache, uncertain recovery.

No claim of provider exactly-once execution is made. A lost response is blocked
on resume instead of being automatically sent again.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

from vgx.common.llm import Call, CallLog, Request
from vgx.common.storage import append_jsonl, digest, file_lock, read_jsonl


class OfflineCacheMiss(RuntimeError):
    pass


class UncertainRequestError(RuntimeError):
    pass


class ProviderRejectedError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'provider request failed with HTTP {status}; no automatic retry')


def request_key(identity: dict, request: Request) -> str:
    # Role, split, item IDs, prices, analysis code and logical labels do not
    # change the provider's input. Candidate text is already inside the prompt.
    return "sha256:" + digest({"schema": 1, "runner": identity,
                                "prompt": request.prompt, "system": request.system})


def execute_request(runner, request: Request, log: CallLog, send) -> Call:
    """send() returns (HTTP status, decoded JSON); no credentials enter the log."""
    key = runner.cache_key(request)
    attempts_path = Path(str(log.path) + ".attempts.jsonl")
    with file_lock(str(log.path) + ".request.lock"):
        successes = [r for r in log.records() if r.get("key") == key and r.get("error") is None]
        if successes:
            responses = {digest({"response": r["response"], "meta_usage": r.get("meta", {}).get("usage")}) for r in successes}
            if len(responses) != 1:
                raise ValueError("conflicting successful cache entries for one provider request")
            return Call(**successes[0])
        attempts = read_jsonl(attempts_path, tolerate_tail=True)
        started = {r["execution_id"] for r in attempts if r.get("key") == key and r["event"] == "started"}
        rejected = {r["execution_id"] for r in attempts if r.get("key") == key and r["event"] == "rejected"}
        if started - rejected:
            raise UncertainRequestError(
                f"request {key} has an unresolved attempt; inspect provider records or restore its response before retrying"
            )
        execution_id, started_at = str(uuid.uuid4()), time.time()
        base = {"key": key, "execution_id": execution_id, "operation_id": request.meta.get("operation_id")}
        append_jsonl(attempts_path, {**base, "event": "started", "created": started_at})
        try:
            status, result = send()
        except Exception:
            # A timeout/connection failure may occur after provider execution.
            # Do not record exception text: it may contain credentials or URLs.
            append_jsonl(attempts_path, {**base, "event": "uncertain", "created": time.time()})
            raise UncertainRequestError(f"request {key} failed with unknown provider outcome; automatic retry disabled") from None
        if not 200 <= status < 300:
            error_envelope = result[0] if isinstance(result, list) and result else result
            provider_error = error_envelope.get('error', error_envelope) if isinstance(error_envelope, dict) else {}
            provider_error = ({k: provider_error[k] for k in ('code', 'status', 'message') if k in provider_error}
                              if isinstance(provider_error, dict) else
                              {'message':provider_error[:2000]} if isinstance(provider_error,str) else {})
            log.append(Call(key, runner.model, request.prompt, "", runner.params, "managed_api",
                            time.time() - started_at, started_at, error=f"http_{status}",
                            meta={**request.meta, "execution_id": execution_id, "http_status": status,
                                  "provider_error": provider_error,
                                  "runner_identity": runner.identity, "system": request.system}))
            # HTTP error does not prove billing for every provider. The ledger
            # reports rejected attempts separately from usage-based estimates.
            append_jsonl(attempts_path, {**base, "event": "rejected", "http_status": status})
            raise ProviderRejectedError(status)
        result = result if isinstance(result, dict) else {"invalid_response": True}
        content = runner.response_text(result)
        call = Call(key, runner.model, request.prompt, content, runner.params, "managed_api",
                    round(time.time() - started_at, 3), started_at,
                    meta={**request.meta, "execution_id": execution_id, "http_status": status,
                          "logical_key": request.key, "system": request.system,
                          "runner_identity": runner.identity,
                          "provider_response_model": result.get("model"),
                          "model_version": result.get("modelVersion"),
                          "usage": result.get("usage", result.get("usageMetadata")),
                          "response_id": result.get("id"), "provider_response": result})
        log.append(call)  # durable before marking the attempt complete
        append_jsonl(attempts_path, {**base, "event": "completed", "created": time.time()})
        return call


class RequestCache:
    """One log per exact request, shared across all collection/analysis runs."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def log(self, key: str) -> CallLog:
        return CallLog(self.root / "calls" / f"{digest(key)}.jsonl")

    def get(self, runner, request: Request, *, allow_api=False) -> tuple[Call, bool]:
        key = runner.cache_key(request)
        log = self.log(key)
        with file_lock(self.root / "locks" / f"{digest(key)}.lock"):
            rows = [r for r in log.records() if r.get("key") == key and r.get("error") is None]
            if rows:
                if len({r["response"] for r in rows}) != 1:
                    raise ValueError("conflicting cached responses")
                return Call(**rows[0]), True
            if not allow_api:
                raise OfflineCacheMiss(f"no cached response for {key}; API collection is disabled")
            runner.run([request], log)
            rows = [r for r in log.records() if r.get("key") == key and r.get("error") is None]
            if not rows:
                raise RuntimeError("provider adapter returned without a durable response")
            return Call(**rows[0]), False

    def import_call(self, call: Call) -> None:
        """Import only after artifact validation; never overwrite a frozen response."""
        with file_lock(self.root / "locks" / f"{digest(call.key)}.lock"):
            log = self.log(call.key)
            rows = [r for r in log.records() if r.get("key") == call.key and r.get("error") is None]
            if rows:
                if any(r["response"] != call.response for r in rows):
                    raise ValueError("import conflicts with an existing frozen response")
                return
            log.append(call)

    def logs(self, keys=None) -> dict[str, CallLog]:
        if keys is not None:
            return {key: self.log(key) for key in set(keys)}
        return {p.stem: CallLog(p) for p in (self.root / "calls").glob("*.jsonl")
                if not p.name.endswith(".attempts.jsonl")}
