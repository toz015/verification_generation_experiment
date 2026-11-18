# QuantumKatas experiments: how to run them yourself

This guide is for running your own QuantumKatas generation–verification experiments
with the `vgx.quantum_katas` package. A generator writes a complete Python module
plus a self-reported probability `r` that it passes the official checker; the code is
frozen; verifiers each give a probability for that frozen code; a sequential policy
decides to ask another verifier, accept, or abstain. Labels come only from the pinned
official QuantumKatas checker.

Results so far: `docs/quantum-katas-full-results-20261019.txt`,
`docs/quantum-katas-stopping-verifiers-sweep-20261019.txt` and the page
`reports/quantum_katas_full_20261019/results.html`.

## 1. One-time setup

### Two Python environments

The main project needs NumPy < 2.3; Qiskit needs NumPy 2.4. Keep them apart.

```bash
# Main environment (controller, analysis, tests)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"

# Separate quantum environment (only the official checker runs here)
uv venv tmp/quantum_katas_pilot_20261018/venv --python 3.12.13
uv pip install --python tmp/quantum_katas_pilot_20261018/venv/bin/python \
  -r configs/quantum_katas_pilot_requirements.txt
```

Never install Qiskit into `.venv`, and avoid `uv run` / `uv sync` on this project:
they re-sync the main environment.

### Pinned upstream data and checker

The loader refuses anything but the pinned files (SHA-256 checked).

```bash
git clone https://github.com/qiskit-community/Qiskit-QuantumKatas \
  tmp/quantum_four_dataset_review_20261018/Qiskit-QuantumKatas
git -C tmp/quantum_four_dataset_review_20261018/Qiskit-QuantumKatas \
  checkout 3790fb3e9892537b550c884ad54b65c73c13f1ad
```

Only `dataset/qiskit_quantumkatas.jsonl`, `benchmark/evaluator.py`, `uv.lock` and
`LICENSE` are used. The data is CC-BY-NC-SA-4.0; keep the attribution files that
each run directory writes.

### Google Cloud (only for real model calls)

```bash
gcloud auth application-default login
```

All existing configs use project `llm-applications-490420`. To use your own project,
change `project` in every model spec **and** `budget.project`, and make sure the
partner models (Claude, Mistral, Qwen, gpt-oss) are enabled in Model Garden.

### Check everything works (offline, free)

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_quantum_katas_*.py
PYTHONPATH=src .venv/bin/python -m vgx.quantum_katas.workflow dry-run \
  --root results/my_dryrun --checker macos_sandbox     # use reviewed_local on Linux
```

The dry run uses mock model replies and reviewed reference code. It takes about
30 seconds and makes no API calls.

## 2. What is fixed and what you choose

Fixed by protocol (do not change for comparable results):
- task text, tests, reference solutions, the official `evaluate_solution(..., timeout=30)`
  and its `passed` label; one official evaluation per frozen candidate, never re-run;
- the 28 development tasks (`configs/quantum_katas_pilot_20261018.json`) and the
  pre-registered 5-fold split of the other 322 (`configs/quantum_katas_split_20261019.json`);
- the strict JSON parser: a reply that is not exactly one JSON object (or one fenced
  JSON block) is a format failure. No repair, no regeneration.

You choose: generator model and settings, verifier models and settings, budget, and
whether the run is the 28-task development set or the 322-task split.

## 3. Run an experiment

### 3.1 Write a run config

Copy the closest existing config and edit it:

| Starting point | Tasks |
|---|---|
| `configs/quantum_katas_official_pilot_v2_qwen3_20261019.json` | 28 development tasks |
| `configs/quantum_katas_full_qwen3_20261019.json` | 322 split tasks (`"split"` key present) |

Fields you normally change:

```jsonc
"models": {
  "generator": {"id": "generator", "provider": "vertex_openai",        // or vertex_anthropic
                "model": "openai/gpt-oss-120b-maas", "location": "us-central1",
                "project": "...", "generation": {"max_tokens": 32768, "temperature": 0.0,
                "top_p": 1.0, "reasoning_effort": "medium", "response_format": {"type": "json_object"}}},
  "verifiers": [ /* 3 to 5 specs; ids must be unique */ ]
},
"budget": {"project": "...", "limit_usd": 30.0, "pricing": {"usd_per_million_tokens": {
  "openai/gpt-oss-120b-maas": {"input": 0.09, "output": 0.36, "cached_input": 0.09}}}},
"paid_collection_authorized": true,
"checker": {"backend": "macos_sandbox", "model_code_execution_approved": true, ...}
```

Providers:
- `vertex_openai`: Gemini, Qwen, gpt-oss and other OpenAI-compatible Vertex models.
  `response_format: {"type": "json_object"}` turns on JSON mode.
- `vertex_anthropic`: Claude with `thinking` / `output_config` passed through as written.
  For strict JSON use `output_config.format` with a `json_schema` (see the Sonnet full config).
- `vertex_partner`: Mistral (and Claude without thinking), as in the GPQA screens.

Every model in the run must have a price in `budget.pricing`. Take prices from
https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing and note
the date in `pricing_source`. Output prices must cover reasoning tokens.

### 3.2 Prepare, smoke-test, run

```bash
W() { PYTHONPATH=src .venv/bin/python -m vgx.quantum_katas.workflow "$@"; }   # bash or zsh
R=results/my_run_name

W prepare --config configs/my_run.json --output $R      # offline; freezes tasks, prompts, identities
W collect-generators --root $R --allow-api --limit 3 --continue-after-uncertain
W collect-verifiers  --root $R --allow-api --limit 3 --continue-after-uncertain
W score-official     --root $R --limit 3
# look at $R/frozen_candidates.jsonl, $R/verifier_signals.jsonl and $R/cache/budget.json;
# if formats, costs and checker results look sane, run everything:
W collect-generators --root $R --allow-api --continue-after-uncertain --workers 3
W collect-verifiers  --root $R --allow-api --continue-after-uncertain --workers 4
W score-official     --root $R
W report             --root $R                          # writes $R/report.txt and report.json
```

Every command can be re-run safely. Cached responses and finished official results
are reused; nothing is sent or evaluated twice.

When a command reports `"stage": "blocked"`:
- **HTTP 429** (rate limit): wait a minute and run the same command again. Rejected
  requests are not charged. Use fewer `--workers` for Qwen-type MaaS endpoints.
- **Unknown outcome** (timeout, connection loss): with `--continue-after-uncertain`
  the request is recorded as a service failure, counted in the denominators, and
  never re-sent. Without the flag the run stops so you can decide.
- **Budget**: raise `limit_usd` only by preparing a new run directory; a plan cannot
  be changed after `prepare`.

### 3.3 Analyse

For runs on the 322-task split, list them in an analysis config (copy
`configs/quantum_katas_crossfit_3gen_20261019.json`, edit `runs`), then:

```bash
PYTHONPATH=src .venv/bin/python -m vgx.quantum_katas.crossfit \
  --config configs/my_crossfit.json --output results/my_crossfit            # primary (raw prior)
PYTHONPATH=src .venv/bin/python -m vgx.quantum_katas.crossfit_extended \
  --config configs/my_crossfit.json --output results/my_crossfit_extended   # early stopping, verifier choice, loss sweep
PYTHONPATH=src .venv/bin/python -m vgx.quantum_katas.crossfit_calibrated \
  --config configs/my_crossfit.json --output results/my_crossfit_calibrated # calibrated prior, risk-coverage, matched coverage
```

Every number is out-of-fold: each fold's likelihoods, costs, verifier order (and
calibrated prior) are fitted on the other four folds only. Write the analysis config
before you look at labels, and commit it, so the analysis is pre-registered.

## 4. Where things are

```
results/<run>/                  (git-ignored; keep it, it is the raw record)
  plan.json, manifest.json       sealed config and identities
  public_tasks.jsonl             what models may see
  frozen_candidates.jsonl        frozen code, r, provenance
  verifier_signals.jsonl         each verifier's probability or failure
  cache/                         raw provider responses, usage, budget ledger
  private/official_results.jsonl official checker results (labels)
  collection/                    status of each collection pass, abandoned requests
  report.txt / report.json       per-run summary
reports/quantum_katas_full_20261019/   committed copies of reports and analyses
src/vgx/quantum_katas/           the code (load, prompt, runners, oracle, run_pilot,
                                 sequential, report, split, crossfit*)
```

## 5. Rules that keep results valid

- Collection commands never read `private/`; only `score-official` and `report` do.
  Verifiers never see `r`, model names, tests, reference code or other verifiers.
- Model-written code runs only in the macOS sandbox (no network, no home or repo
  reads). On Linux there is no sandbox backend yet: `reviewed_local` refuses
  unreviewed code, so real runs need a container backend first.
- Do not edit files in a run directory, change a prompt or parser for an existing
  run, or relabel anything. Change something by preparing a new run directory.
- Report format failures and service failures in the denominators.
- Cost numbers are usage-based estimates, not billing.

## 6. Adding a generator to the existing comparison

1. Copy `configs/quantum_katas_full_gptoss_20261019.json`, change the generator spec
   and its price, keep everything else.
2. `prepare`, smoke-test 3 tasks, run all 322 (section 3.2).
3. Add the run to a new analysis config and run the three analyses (section 3.3).

Questions about why a choice was made are usually answered in
`docs/quantum-katas-claude-handoff-20261018.txt` (the original protocol) and the
result docs listed at the top.
