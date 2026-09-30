"""Small Vertex AI Chat Completions adapter using gcloud ADC credentials."""

from __future__ import annotations

import json
import time
from typing import Any

from vgx.common.llm import Call, CallLog, Request


def _get_adc_token(project: str) -> str:
    """Refresh ADC in-process; do not fall back to the VM's scoped gcloud identity."""
    import google.auth
    from google.auth.transport.requests import Request as AuthRequest

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(AuthRequest())
    if not credentials.token:
        raise RuntimeError(f"ADC did not provide an access token for project {project}")
    return credentials.token


class VertexBatchRunner:
    """Call one managed Vertex model and resume only identical requests."""

    def __init__(
        self,
        model: str,
        location: str,
        project: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning_effort: str = "low",
    ):
        self.model = model
        self.location = location
        self.project = project
        if not project:
            raise ValueError("Vertex project_id is required")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "provider": "vertex_ai_chat_completions",
            "model": self.model,
            "location": self.location,
            "project_sha256": __import__("hashlib").sha256(self.project.encode()).hexdigest(),
            "max_tokens": self.max_tokens,
            "temperature": None if self.model.startswith("google/gemini-3") else self.temperature,
            "top_p": None if self.model.startswith("google/gemini-3") else self.top_p,
            "reasoning_effort": self.reasoning_effort if self.model.startswith("google/gemini-3") else None,
        }

    def cache_key(self, request: Request) -> str:
        import hashlib
        from dataclasses import asdict

        payload = {"runner": self.identity, "request": asdict(request)}
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        return f"{request.key}|sha256:{digest}"

    def _endpoint(self) -> str:
        host = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1beta1/projects/{self.project}/locations/{self.location}"
            "/endpoints/openapi/chat/completions"
        )

    def run(self, requests_to_run: list[Request], log: CallLog) -> dict[str, str]:
        if len({r.key for r in requests_to_run}) != len(requests_to_run):
            raise ValueError("request keys must be unique within a batch")
        keyed = {r.key: self.cache_key(r) for r in requests_to_run}
        pending = [(r, keyed[r.key]) for r in requests_to_run if not log.has(keyed[r.key])]
        if pending:
            import requests

            # Keep the short-lived credential in memory only; never log or print it.
            token = _get_adc_token(self.project)
            session = requests.Session()
            for request, cache_key in pending:
                messages = ([{"role": "system", "content": request.system}] if request.system else [])
                messages.append({"role": "user", "content": request.prompt})
                body: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
                if not self.model.startswith("google/gemini-3"):
                    body.update(temperature=self.temperature, top_p=self.top_p)
                else:
                    body["reasoning_effort"] = self.reasoning_effort
                started = time.time()
                # Preserve completed calls in CallLog and retry only transient
                # transport/server/rate-limit failures. The request cache key
                # is unchanged across retries, so rerunning resumes safely.
                for attempt in range(6):
                    try:
                        response = session.post(
                            self._endpoint(),
                            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                            json=body,
                            timeout=300,
                        )
                    except requests.RequestException:
                        if attempt == 5:
                            raise
                        time.sleep(min(60, 2 ** attempt))
                        continue
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        break
                    if attempt == 5:
                        break
                    response.close()
                    time.sleep(min(60, 2 ** attempt))
                elapsed = time.time() - started
                if not response.ok:
                    raise RuntimeError(
                        f"Vertex request for {self.model} failed with HTTP {response.status_code}"
                    )
                result = response.json()
                try:
                    content = result["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as error:
                    raise RuntimeError(f"Vertex returned an unexpected response for {self.model}") from error
                if not isinstance(content, str):
                    raise RuntimeError(f"Vertex returned non-text content for {self.model}")
                params = {k: body[k] for k in ("max_tokens", "temperature", "top_p", "reasoning_effort") if k in body}
                log.append(Call(
                    key=cache_key,
                    model=self.model,
                    prompt=request.prompt,
                    response=content,
                    params=params,
                    dtype="managed_api",
                    latency_s=round(elapsed, 3),
                    created=started,
                    meta={
                        **request.meta,
                        "logical_key": request.key,
                        "system": request.system,
                        "runner_identity": self.identity,
                        "provider_response_model": result.get("model"),
                        "usage": result.get("usage"),
                        "response_id": result.get("id"),
                    },
                ))
        responses = log.responses()
        return {key: responses[cache_key] for key, cache_key in keyed.items() if cache_key in responses}
