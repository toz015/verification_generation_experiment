"""Batched local inference over HuggingFace weights.

Models are loaded straight from HuggingFace and run on the local GPU via
vLLM's offline API. There is no server, no HTTP and no external service: the
whole sweep is one `LLM.chat(...)` call over a list of prompts.

`vllm` is imported lazily inside `BatchRunner.run` on purpose. It requires
CUDA and cannot be installed on macOS, so the loader, retriever, scorer and
their tests all have to remain importable on a workstation without a GPU.

Every generation is appended to a JSONL log with its prompt, response and
sampling parameters, so scoring can be redone without re-running the models,
and a crashed run resumes instead of regenerating from scratch.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Greedy decoding: results are deterministic, so a difference between two runs
# is a real difference and not a resample.
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0}


@dataclass(frozen=True)
class Request:
    """One prompt to run. `key` identifies it for logging and resume."""

    key: str
    prompt: str
    system: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Call:
    """One completed generation, as written to the log."""

    key: str
    model: str
    prompt: str
    response: str
    params: dict[str, Any]
    dtype: str
    latency_s: float
    created: float
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class CallLog:
    """Append-only JSONL log that doubles as the resume index.

    The sweep is small, but a model load is not, so losing completed
    generations to a crash means paying the load cost again for nothing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done = self._read_keys()

    def _read_keys(self) -> set[str]:
        keys: set[str] = set()
        for rec in self.records():
            if rec.get("error") is None:
                keys.add(rec["key"])
        return keys

    def has(self, key: str) -> bool:
        return key in self._done

    def append(self, call: Call) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(call)) + "\n")
        if call.error is None:
            self._done.add(call.key)

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line is expected if a run was killed
                    # mid-write. Skip it; that key simply gets regenerated.
                    continue

    def responses(self) -> dict[str, str]:
        return {r["key"]: r["response"] for r in self.records() if r.get("error") is None}


class BatchRunner:
    """Loads one HuggingFace model onto the GPU and runs a batch of prompts."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization

    @property
    def params(self) -> dict[str, Any]:
        return {**SAMPLING, "max_tokens": self.max_tokens}

    def run(self, requests: list[Request], log: CallLog) -> dict[str, str]:
        """Generate every pending request in one batch. Returns {key: response}.

        Requests already present in the log are skipped, so re-running after a
        crash costs only what was not finished.
        """
        pending = [r for r in requests if not log.has(r.key)]
        if not pending:
            return log.responses()

        # Imported here, not at module scope: vllm needs CUDA and is absent on
        # the development machine.
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=self.model,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        sampling = SamplingParams(max_tokens=self.max_tokens, **SAMPLING)

        conversations = [
            ([{"role": "system", "content": r.system}] if r.system else [])
            + [{"role": "user", "content": r.prompt}]
            for r in pending
        ]

        started = time.time()
        outputs = llm.chat(conversations, sampling)
        elapsed = time.time() - started

        # vLLM returns outputs in request order, so zip is safe here.
        per_call = round(elapsed / max(len(pending), 1), 3)
        for req, out in zip(pending, outputs):
            log.append(
                Call(
                    key=req.key,
                    model=self.model,
                    prompt=req.prompt,
                    response=out.outputs[0].text,
                    params=self.params,
                    dtype=self.dtype,
                    latency_s=per_call,
                    created=started,
                    meta=req.meta,
                )
            )

        return log.responses()
