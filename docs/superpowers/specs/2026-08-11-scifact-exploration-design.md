# SciFact Exploration — Design

**Date:** 2026-08-11
**Status:** Approved, pending implementation plan
**Parent project:** Reference-Aware Selective Generation for Scientific QA (see `LLM Citation Benchmark Design.pdf`)

## 1. Context and scope

The parent project proposes treating citation as a reporting game: the agent holds private
information (how strongly its evidence supports a claim), the metric is the payment rule, and
a strictly proper scoring rule with an abstention reserve should induce truthful reporting.
Before any mechanism is designed, we need to establish that current 7–14B models actually
leave room for one.

This spec covers **SciFact only**. PubMedQA and ALCE are deliberately out of scope; each gets
its own spec after this one lands. There is no unified cross-dataset harness. If a pattern
repeats across all three explorations, it gets factored out then, with three real examples in
hand rather than zero.

### The question this exploration answers

Not "are the scores low." A low score has at least three explanations, and only one of them
is addressable by mechanism design:

1. **Retrieval failure** — the evidence was never surfaced.
2. **Capability failure** — the model cannot judge entailment at this scale.
3. **Decision failure** — the model can judge entailment but asserts and cites anyway,
   because nothing in the metric prices a wrong citation against silence.

The exploration is designed to separate these. If the gap is (1) or (2), mechanism design
does not close it and the parent project needs rethinking.

## 2. Why SciFact first

- Corpus is 5,183 abstracts, so BM25 is trivial and no index server is needed.
- Gold rationale annotations name specific sentences, which is the closest thing in the three
  datasets to citation ground truth.
- `NOT ENOUGH INFO` is a first-class label, so abstention has ground truth.
- It has a real retrieval stage, which is the hardest component. Learning it here informs the
  other two specs.

## 3. Environment and repository

`git init` in place. `uv` for dependency management — reproducible lockfile, and it installs
cleanly on the GCP VM without fighting the system Python's PEP 668 restrictions.

```
src/vgx/
  common/llm.py          # batched local inference; the only shared module
  scifact/
    load.py              # tarball -> typed records
    retrieve.py          # BM25 + oracle
    prompt.py            # prompt construction + response parsing
    score.py             # vendored official scorer + diagnostics
third_party/scifact/     # pinned official evaluation code
scripts/setup_vm.sh      # driver check, uv, vLLM, HF auth, weight prefetch
configs/
data/                    # gitignored
results/                 # gitignored
docs/                    # specs, fact-sheet, final report
```

### Execution model

**All model inference runs on the GCP A100 VM.** The laptop is used to write code, commit,
and push; it never serves a model. Loop: edit locally → push to remote → `ssh` to the VM →
`git pull` → run.

Weights are pulled **from HuggingFace** and run on the local GPU through vLLM's **offline
Python API** (`LLM.chat`) — no inference server, no HTTP, no external service. The sweep is a
fixed batch of 200 prompts, so a server buys nothing over one batched call, and skipping it
removes ports, tmux and a process lifecycle from the setup.

Weights are **bf16, unquantized** — an 8B model is roughly 16 GB, so the 40 GB A100 holds one
comfortably with room for KV cache. Models are loaded one at a time rather than co-resident.

Running unquantized on a single backend removes a confound that a laptop path would have
introduced: Q4 output is not the same as bf16 output, and mixing the two within one results
table would make differences unattributable. Every run still records its backend, model
revision, and dtype.

`scripts/setup_vm.sh` makes the VM reproducible from scratch: verify the NVIDIA driver and
CUDA are visible, install `uv`, sync the locked environment, authenticate to HuggingFace, and
prefetch both model weights before any experiment runs.

### Prerequisite requiring manual action

`meta-llama/Llama-3.1-8B-Instruct` is `gated=manual` on HuggingFace. Before S3, a human must
accept Meta's license on the model page and create a read token, exported as `HF_TOKEN` on the
VM. `setup_vm.sh` checks for a working token and both models' accessibility **up front**, so
this fails in setup with a clear message rather than midway through a sweep.

## 4. Data layer

### Source

Official tarball: `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`

**Not** the HuggingFace `allenai/scifact` dataset. Two verified reasons:

- It is a loading-script dataset (`scifact.py`), which `datasets>=3.0` refuses to execute.
- Its claim splits report `train 1261 / validation 450 / test 300`, which do not match the
  paper's `809 / 300 / 300`, because the script emits one row per (claim, evidence-document)
  pair. Evaluating on that naively over-weights multi-document claims.

The tarball preserves the nested structure the official scorer expects. Corpus size confirmed
at 5,183 abstracts.

### Records

Two files matter. Field names are **verified on download** by `load.py`, not assumed from
memory; the loader asserts the schema and fails loudly on mismatch.

- `corpus.jsonl` — one abstract per line: document id, title, sentence-split abstract body,
  and a flag for whether the abstract is structured.
- `claims_dev.jsonl` — one claim per line: claim id, claim text, an evidence mapping from
  document id to a list of `{sentences, label}` entries, and the set of cited document ids.

An **NEI claim is one whose evidence mapping is empty**. This is the abstention ground truth.

### Evaluation split and sample

Test labels are withheld by the benchmark, so all work is on **dev** (300 claims).

This exploration runs on a **stratified 50-claim sample** of dev, not the full split.

- **Stratified**, not random, across SUPPORT / CONTRADICT / NEI. A uniform random 50 could
  yield only a handful of NEI claims and destroy the abstention diagnostic.
- **Pinned.** The sample is drawn once with a fixed seed and the chosen claim ids are written
  to `configs/scifact_sample_50.json`, which is **committed to the repository**. Every model
  and every re-run scores the identical 50 claims, so numbers are comparable across runs and
  reproducible by anyone who clones the repo. The sample is never re-drawn silently.

The loader emits a class-balance summary (SUPPORT / CONTRADICT / NEI counts) for both full
dev and the sample, as part of the fact-sheet.

### Statistical power

At n = 50, a proportion has a standard error of at most ~7pp, so a 95% interval is roughly
**±14pp**. This is a deliberate trade: the exploration is sized to detect the effect it is
looking for, not to rank models.

- **In scope.** The label-only versus label+rationale gap. Slide 5 predicts "label >>
  rationale", i.e. tens of points. An effect that large is visible at n = 50.
- **Out of scope.** Ranking the two models against each other. A plausible few-point
  difference sits well inside the interval. The report states this explicitly rather than
  reading a ranking out of noise. What the second model *does* buy is a replication check:
  a pathology that shows up in both families is a property of the task, not of one model.

Every number in the report carries a Wilson score interval, so the reader cannot mistake a
point estimate for a precise one. Scaling to the full 300 dev claims is a one-line config
change if a result turns out to be borderline.

## 5. Retrieval

`rank_bm25` over concatenated title and abstract text. No index server, no dense retriever.

Two modes run for every experiment:

- **Oracle** — the gold evidence documents are placed in context directly.
- **BM25 top-k** — retrieved documents are placed in context, for k in a small sweep.

### Standalone first output: recall@k

Before any generation, `retrieve.py` produces a **recall@k curve** over **all 300 dev
claims**, not just the 50-claim sample. Retrieval costs no model calls, so there is no reason
to sample it, and the full-dev curve is the more trustworthy basis for fixing k. The curve is
also reported restricted to the sample, to confirm the sample is not retrieval-anomalous. The
fraction of claims whose gold evidence document appears in the top k. This is cheap and it
bounds everything downstream — no citation metric can exceed what retrieval surfaces. It also
fixes the k used in the main sweep.

The **oracle-minus-BM25 gap** in final metrics is the retrieval-failure term from Section 1,
isolated and subtractable.

## 6. Generation

### Client

One `BatchRunner` in `common/llm.py` wrapping vLLM's offline `LLM.chat`. Generations are
logged to disk with full prompt, response, sampling parameters, model id and dtype, so every
number is reproducible and re-scorable without regenerating.

`vllm` is imported lazily inside `run()` rather than at module scope, because it requires
CUDA and cannot be installed on macOS. That keeps the loader, retriever, scorer and their
tests importable on a GPU-less workstation, which is what lets S1, S2, S4 and S6 be developed
off the VM.

### Prompt condition

**One condition.** Claim plus abstract, output is a label and the supporting sentence
indices. This is the standard SciFact framing and the way zero-shot LLM baselines are
normally reported.

The parent proposal's structured record — explicit support score in [0, 1] and an action in
{assert, qualify, abstain} — is **deferred**. It is the object the mechanism acts on, not
something needed to measure whether headroom exists, and eliciting it would double generation
cost while adding a schema that small models may not reliably emit. It becomes a follow-on
once the native results establish that headroom is there. The cost of deferring is that this
exploration produces no calibration curve.

### Models

Two, matched at roughly 8B, drawn from different families:

- **`Qwen/Qwen3-8B`** — the most-downloaded model at this scale, ~15.2M/month at time of
  writing, and current-generation. Ungated.
- **`meta-llama/Llama-3.1-8B-Instruct`** — the most-cited open baseline in the literature,
  which makes results legible to readers. Gated; see Section 3 for the manual prerequisite.

Note there is no Llama 3 *7B*; Llama 3 and 3.1 ship at 8B and 70B, so 8B is the correct tier.

**Why matched-size cross-family rather than same-family 8B→14B.** At n = 50 the interval is
about ±14pp, which is wider than any plausible 8B-versus-14B difference, so a size sweep is
unresolvable at this sample size and would buy nothing. The purpose of the second model is
instead to check that any observed pathology **replicates across families** rather than being
one lab's post-training quirk. Matched-size cross-family answers that; same-family does not.
`Qwen/Qwen3-14B` can be added later as a third run for 100 extra calls if the size question
becomes interesting, ideally at full dev where it is measurable.

Decoding is greedy, so results are deterministic and differences are attributable to the
model rather than to sampling.

Total generation for the sweep: 2 models × 2 retrieval modes × 50 claims = **200 calls**.
That is minutes of A100 time. The dominant cost is not inference but weight download and
server startup, which is why models are served one at a time and each model's four runs
(2 retrieval modes × 50 claims) complete before the server is swapped.

Because the sweep is so small relative to the setup cost, `run_sweep.py` is **resumable**: it
writes one JSONL record per call and skips any (model, mode, claim id) already present. An
interrupted SSH session or a preempted VM costs the remaining calls, not the completed ones.

## 7. Scoring

### Official metrics

The official SciFact evaluation code is vendored into `third_party/scifact/` at a pinned
revision and used as the source of truth. **Two numbers**, both on dev, both abstract-level —
the headline metric SciFact results are normally reported against:

- Abstract-level F1, **label-only**
- Abstract-level F1, **label + rationale**

Sentence-level variants are dropped. Keeping both of these two is not redundancy: the
**gap between them is the citation signal**. Label-only credits getting the verdict right;
label+rationale credits it only when the cited sentences are also right. The parent
proposal's Slide 5 predicts "label >> rationale" for zero-shot LLMs, and that gap is the
single most important number this exploration produces.

### Validation gate

The parent proposal's Slide 5 baselines (VeriSci ≈ 39, VerT5erini ≈ 60, MultiVerS ≈ 67) are
*trained* systems. Reproducing them means training them, which is out of scope for Phase 0.
They are therefore treated as literature context to be verified against the papers, not as a
reproduction target.

The gate is instead a direct test of the scorer, which is cheaper and stronger:

1. **Identity test** — feed gold annotations in as predictions; assert every F1 is 1.0.
2. **Always-SUPPORT baseline** — assert label-only F1 tracks the SUPPORT class prevalence and
   label+rationale F1 is near zero.
3. **Random-sentence baseline** — assert rationale credit collapses while label credit is
   unaffected.
4. **Cite-everything baseline** — every sentence in the abstract cited. Under a
   precision/recall metric this must score non-trivially despite carrying no information.

**No model number is reported until all four pass.** Baseline (4) is not only a test: it is
the first hard evidence for the parent proposal's claim that citation-count-monotone metrics
reward over-citation, and it belongs in the final report as a result.

### Diagnostics

Three, each tied to a specific claim in the parent proposal. All three are computed from the
same 1,200 generations — none requires an extra run.

| Diagnostic | Measure | Proposal claim tested |
|---|---|---|
| Abstention | Recall and precision on NEI claims | Slide 5, abstention has ground truth here |
| Over-citation | Cited sentence count vs. gold count per claim; precision of cited sentences | Slide 2, "citing is free, so the model over-cites" |
| Retrieval isolation | Oracle metrics minus BM25 metrics, per model | Separates explanation (1) from (2) and (3) |

Calibration of a reported support score is deferred along with the structured condition
(Section 6).

## 8. Deliverable

`docs/scifact-report.md`, containing:

1. **Dataset anatomy** — structure, schemas, split sizes, class balance, worked examples, and
   the composition of the pinned 50-claim sample.
2. **How to load and use it** — the exact commands, the HF pitfalls above, and the retrieval setup.
3. **Recall@k curve** for BM25 over the corpus.
4. **Scorer validation results**, including the cite-everything baseline.
5. **Metrics table** — 2 metrics × 2 models × 2 retrieval modes, i.e. 8 numbers, each with a
   Wilson score interval, and the label-only versus label+rationale gap called out explicitly.
6. **The three diagnostics.**
7. **Headroom verdict** — an explicit statement of which of the three failure modes in
   Section 1 dominates, and therefore whether mechanism design has room to operate on SciFact.

The verdict is the point. A finding that headroom is small is a real result and gets reported
as such rather than argued around.

## 9. Milestones

| ID | Deliverable | Needs GPU |
|---|---|---|
| S0 | Repo scaffold, `uv` env, git remote, `common/llm.py`, `scripts/setup_vm.sh` | no |
| S1 | `load.py`, pinned stratified 50-claim sample, dataset anatomy fact-sheet | no |
| S2 | `retrieve.py` + recall@k curve over full dev | no |
| S3 | VM provisioned, vLLM installed, `prompt.py` + parser, 10-claim smoke test | **yes** |
| S4 | Vendored scorer wired, all four validation gates passing | no |
| S5 | Sweep: 2 models × 2 retrieval modes × 50 claims = 200 calls | **yes** |
| S6 | Diagnostics computed, `docs/scifact-report.md` written | no |

All inference happens on the VM. S1, S2, S4 and S6 make **no model calls at all** — the four
scorer gates in Section 7 are pure unit tests over gold and synthetic predictions — so they
can be developed and run without the GPU. This is deliberate: it keeps loader, retriever and
scorer work off the VM's clock, so GPU time is spent only on S3 and S5.

The manual HuggingFace license step in Section 3 must be done before S3.

## 10. Risks

- **Schema drift.** Field names are asserted at load time rather than assumed, so a mismatch
  fails at S1 with a clear error instead of producing silently wrong scores at S6.
- **Output parsing.** Even the native format requires a label and a list of sentence indices
  to come back parseably. `prompt.py` records a parse-failure rate per model; if it is high,
  that is a reported finding, and the fallback is constrained decoding via vLLM's guided
  generation. Parse failures are never silently dropped, since dropping them would bias every
  metric upward for exactly the weaker models being characterized.
- **Vendored scorer incompatibility.** The official code may target an older Python. If it
  cannot be run as-is, it is ported minimally with the identity test from Section 7 proving
  the port is faithful.
- **Small-sample power.** This is the largest risk and it is accepted deliberately. At n = 50
  every proportion carries roughly ±14pp, so only large effects are readable. Stratified
  sampling protects the NEI cell specifically, and Wilson intervals on every number keep the
  limitation visible in the report. The mitigation if a result lands borderline is to rerun on
  full dev, which is a config change and 1,200 calls.
- **Sample drift.** Re-drawing the sample between runs would make numbers silently
  incomparable. The claim ids are committed to the repo and the sampler refuses to overwrite
  an existing sample file; regenerating it is an explicit, deliberate act.
- **Gated weights.** Llama-3.1-8B needs a license acceptance and `HF_TOKEN`. `setup_vm.sh`
  verifies access to both models before anything else runs, so this surfaces during setup
  rather than after the Qwen half of the sweep has already completed.
- **VM environment drift.** CUDA, driver and vLLM versions on the VM are outside the
  lockfile's control. `setup_vm.sh` records driver, CUDA and vLLM versions into the run
  manifest, so a later discrepancy is diagnosable rather than mysterious.
- **Interrupted runs.** SSH drops and preemption are normal on cloud VMs. The sweep is
  resumable at (model, mode, claim id) granularity, and the server is run under `nohup` or
  `tmux` so it survives a disconnected session.
