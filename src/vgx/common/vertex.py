"""Small Vertex AI Chat Completions adapter using gcloud ADC credentials."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from vgx.common.llm import CallLog, Request
from vgx.common.api import execute_request, request_key


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

    def legacy_cache_key(self, request: Request) -> str:
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

    @property
    def params(self) -> dict:
        params = {"max_tokens": self.max_tokens}
        if self.model.startswith("google/gemini-3"):
            params["reasoning_effort"] = self.reasoning_effort
        else:
            params.update(temperature=self.temperature, top_p=self.top_p)
        return params

    def cache_key(self, request: Request) -> str:
        return request_key(self.identity, request)

    def request_body(self, request: Request) -> dict:
        messages = ([{"role": "system", "content": request.system}] if request.system else [])
        messages.append({"role": "user", "content": request.prompt})
        return {"model": self.model, "messages": messages, "stream": False, **self.params}

    @staticmethod
    def response_text(result: dict) -> str:
        try:
            content = result["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else ""
        except (KeyError, IndexError, TypeError):
            return ""

    def run(self, requests_to_run: list[Request], log: CallLog) -> dict[str, str]:
        if len({r.key for r in requests_to_run}) != len(requests_to_run):
            raise ValueError("request keys must be unique within a batch")
        outputs = {}
        for request in requests_to_run:
            cached = log.responses().get(self.cache_key(request))
            if cached is not None:
                outputs[request.key] = cached
                continue
            import requests
            # Resolve authentication before marking a provider attempt started.
            token = _get_adc_token(self.project)
            session = requests.Session()
            body = self.request_body(request)

            def send():
                response = session.post(
                    self._endpoint(), headers={"Authorization": f"Bearer {token}",
                                               "x-goog-user-project": self.project,
                                               "Content-Type": "application/json"},
                    json=body, timeout=300,
                )
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"invalid_response": True}
                return response.status_code, payload

            try:
                billed_request = replace(request, meta={**request.meta,
                    "resource_project": self.project, "quota_project": self.project})
                call = execute_request(self, billed_request, log, send)
                outputs[request.key] = call.response
            finally:
                if hasattr(session, "close"):
                    session.close()
        return outputs
