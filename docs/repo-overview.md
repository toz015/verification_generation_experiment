# Repository overview: verification-generation experiment

Snapshot of 2026-10-19. GitHub: https://github.com/toz015/verification_generation_experiment

## What the project studies

A generator answers a question (or writes a program) and reports its own probability
`r` of being right. The answer is frozen. Verifier models then each give a probability
that the frozen answer is correct. A sequential policy (Green–Laffont style nested
stopping, the "NestedPlanner" / "ExactPlanner" code) decides, one verifier at a time,
whether to ask another verifier, accept the answer, or abstain, trading the cost of a
query against the reward for a correct answer and the loss L for a wrong one
(accept iff belief ≥ L/(1+L)). Labels are used only to fit the verifier likelihoods on
calibration data and to score results offline.

## Branches

```
main (2026-08-12)  SciFact + PubMedQA exploration
 └─ codex/gpqa-pilot                       PR #1  open   GPQA pilot → fresh Vertex → verifier screens (to 2026-10-11)
     ├─ claude/project-thread-k4jr14       PR #2  draft  side analysis, not merged anywhere
     ├─ claude/gpqa-risk-acceptance        PR #3  draft  side analysis, not merged anywhere
     └─ diamond-verifier-signals           no PR         + GPQA Diamond plan/score/cost + Sonnet 5.5 (to 2026-10-17)
         └─ claude/project-thread-m8td13   PR #4  open   + QuantumKatas (2026-10-18/19)
```

| Branch | PR | What it adds | Use it for |
|---|---|---|---|
| `main` | — | Scaffold, local vLLM inference on an A100 VM, **SciFact** (BM25 retrieval, label vs rationale F1) and **PubMedQA** explorations. | History only; everything newer branches from here. |
| `codex/gpqa-pilot` | #1 open, base `main` | **GPQA** pilot (200 Main questions, Gemini generator, Llama + Gemini verifiers), Vertex request cache and budget ledger (`src/vgx/common`), the fresh Vertex experiment (249 calibration + 196 Diamond), Jev comparison, frozen-candidate workflow, five-verifier screens, nested diagnostics, probability-only triples. | Fully contained in the two branches below. |
| `claude/project-thread-k4jr14` | #2 draft | Offline experiment: Gemini Flash likelihoods conditioned on the generator's original confidence (`configs/gpqa_conditioned_likelihood.json`, `reports/gpqa_conditioned_likelihood_20261009/`). | Reading that one result. Not in any other branch. |
| `claude/gpqa-risk-acceptance` | #3 draft | Offline risk-controlled acceptance workflow for GPQA with a sample-size plan (`reports/gpqa_risk_acceptance_20261009/`). | Reading that one result. Not in any other branch. |
| `diamond-verifier-signals` | none | Correlation-aware triples, training-fold prior calibration (diagnostic), verifier-subset comparison, the **sealed Diamond analysis plan and its scoring**, Diamond cost comparison (adaptive vs always 3 or 5 verifiers), Sonnet 5.5 as generator with sequential verification. | Latest GPQA state. |
| `claude/project-thread-m8td13` | #4 open, base `main` | Everything above except #2/#3, plus **QuantumKatas**: `src/vgx/quantum_katas`, official-checker adapter with macOS sandbox, 28-task pilot, pre-registered 5-fold split of 322 tasks, full runs with Qwen3-235B, Sonnet 5.5 and gpt-oss-120b, cross-fitting analyses, results page. | **Start here.** It is the superset of the GPQA and QuantumKatas work. |

Because PR #4 targets `main`, its diff also shows all GPQA commits from #1 and the
Diamond branch. Merging #4 makes #1 redundant. #2 and #3 would still need their own
decision (merge as side analyses or close).

## Experiments by dataset

| Dataset | Status | Where to read |
|---|---|---|
| SciFact | Done, exploration only (50 claims, 8B models on A100). Finding: large label vs rationale F1 gap. | `docs/scifact-report.md` |
| PubMedQA | Done, exploration only (50 items). Finding: the abstention threshold is arbitrary; accuracy tracks the "yes" base rate. | `docs/pubmedqa-report.md` |
| GPQA (Main pilot) | Done. 200 questions, 120 calibration / 80 evaluation. | `reports/gpqa_offline_validation_20261004.md`, `docs/gpqa-frozen-workflow.md` |
| GPQA (fresh Vertex) | Done. 249 calibration + 196 Diamond. Live policy released 158 answers, 155 correct, 266 verifier calls vs 390. | `reports/gpqa_fresh_vertex_20261004.md`, `docs/gpqa-fresh-vertex.md` |
| GPQA (verifier screens, Diamond plan) | Done. Five candidate verifiers screened; sealed Diamond plan scored; adaptive vs fixed cost comparison; Sonnet 5.5 generator. | `docs/gpqa-diamond-plan-score.md`, `reports/gpqa_diamond_*`, `docs/gpqa-sonnet55-*.txt` |
| GPQA (gpt-oss / DeepSeek replacement generators) | **In progress, local only** (see below). | untracked `docs/gpqa-gptoss-*.txt` |
| QuantumKatas | Done for Qwen3-235B and Sonnet 5.5 on 322 tasks; gpt-oss-120b running. | `docs/quantum-katas-guide.md`, `docs/quantum-katas-full-results-20261019.txt`, `reports/quantum_katas_full_20261019/results.html` |

The repository README describes the GPQA decisions in detail (raw `r` as the prior,
calibration deferred, reporting conventions).

## Code layout

```
src/vgx/common/          shared: Request/Call logs, Vertex runners, RequestCache with
                         uncertain-request blocking, BudgetedRequestCache, billing, storage
src/vgx/gpqa/            GPQA: loaders, prompts, frozen artifacts, planners (NestedPlanner,
                         ExactPlanner), sequential executor, many study modules
src/vgx/quantum_katas/   QuantumKatas adapter (see docs/quantum-katas-guide.md)
src/vgx/scifact/, src/vgx/pubmedqa/   early explorations
configs/                 one JSON per study; most are frozen inputs of a finished run
docs/                    protocols, guides, write-ups
reports/                 committed results (aggregates only)
tests/                   pytest; dataset-dependent tests skip when data is absent
```

## Where data and results live

- `reports/` (committed): aggregate results meant for review.
- `results/` (git-ignored, on the original Mac): raw run directories with provider
  responses, frozen candidates, private labels and budget ledgers. These are the
  primary record and are not on GitHub.
- `output/` (untracked): zipped delivery bundles of past GPQA reviews.
- `tmp/` (untracked): downloaded upstream data (QuantumKatas source) and the separate
  Qiskit virtual environment.
- GPQA data comes from the gated Hugging Face dataset; it is not stored in the repo.

## Work that exists only on the original Mac (not on GitHub)

As of this snapshot, the local checkout has uncommitted GPQA work from other sessions:

- `src/vgx/gpqa/diamond_evaluation*.py`, `configs/gpqa_diamond_evaluation.json`,
  `reports/gpqa_diamond_evaluation_20261012/`, `reports/gpqa_diamond_diagnosis_20261012/`
- the gpt-oss / DeepSeek replacement-generator series: `src/vgx/gpqa/gptoss_*.py`,
  `configs/gpqa_replacement_generators_20261014.json`, `docs/gpqa-gptoss-*.txt`,
  `docs/gpqa-replacement-generators-20261014.txt`, `reports/gpqa_gptoss_*`,
  `reports/gpqa_replacement_generators_20261014/`
- `docs/dataset-proposal-review-20261018.txt` (StabilizerBench / PDEAgent-Bench proposal review),
  `asqa_prompt.json`, `milestone_report/`

A colleague cloning the repo will not see these until they are committed.

## Conventions every study follows

- Freeze candidates before verification; never repair, regenerate or relabel.
- All model calls go through the request cache: cached responses are reused, a request
  with an unknown outcome is never re-sent automatically, and every paid call is
  reserved against a local budget ledger (estimates, not billing).
- Collection commands are offline unless `--allow-api` is given.
- Calibration (fitting) and evaluation data are kept apart; analysis plans are
  committed before evaluation labels are read.
- Paid runs need an explicit dollar budget from the project owner.

## For a new contributor

1. Check out `claude/project-thread-m8td13` (or `main` once PR #4 is merged).
2. Read the README (GPQA decisions) and `docs/quantum-katas-guide.md` (QuantumKatas how-to).
3. Run the tests: `PYTHONPATH=src .venv/bin/python -m pytest`.
4. Ask the project owner for Vertex project access and a budget before any `--allow-api` run.
