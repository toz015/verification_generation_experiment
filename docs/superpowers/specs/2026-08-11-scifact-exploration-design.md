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
  common/llm.py          # OpenAI-compatible client; the only shared module
  scifact/
    load.py              # tarball -> typed records
    retrieve.py          # BM25 + oracle
    prompt.py            # prompt construction + response parsing
    score.py             # vendored official scorer + diagnostics
third_party/scifact/     # pinned official evaluation code
configs/
data/                    # gitignored
results/                 # gitignored
docs/                    # specs, fact-sheet, final report
```

Laptop (M2 Air, 24 GB) and the A100 VM run identical code. The only difference is the model
backend URL: `vllm serve` on the VM, Ollama's OpenAI-compatible endpoint locally. Work is
developed and smoke-tested locally, pushed to a remote, then pulled and run on the VM over SSH.

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

### Evaluation split

**Dev, 300 claims.** Test labels are withheld by the benchmark. All reported numbers are dev.

The loader emits a class-balance summary (SUPPORT / CONTRADICT / NEI counts) as part of the
fact-sheet, since NEI prevalence bounds how much the abstention diagnostics can show.

## 5. Retrieval

`rank_bm25` over concatenated title and abstract text. No index server, no dense retriever.

Two modes run for every experiment:

- **Oracle** — the gold evidence documents are placed in context directly.
- **BM25 top-k** — retrieved documents are placed in context, for k in a small sweep.

### Standalone first output: recall@k

Before any generation, `retrieve.py` produces a **recall@k curve** over the dev set: the
fraction of claims whose gold evidence document appears in the top k. This is cheap and it
bounds everything downstream — no citation metric can exceed what retrieval surfaces. It also
fixes the k used in the main sweep.

The **oracle-minus-BM25 gap** in final metrics is the retrieval-failure term from Section 1,
isolated and subtractable.

## 6. Generation

### Client

One OpenAI-compatible HTTP client in `common/llm.py`. vLLM and Ollama both speak this, so
there is no backend branching in experiment code. Requests are logged to disk with full
prompt, response, sampling parameters, and model identifier so every number is reproducible
and re-scorable without regeneration.

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

Two, both current-generation and ungated on HuggingFace:

- **Qwen3-8B** — the most-downloaded model at this scale (~15.2M/month at time of writing).
- **Qwen3-14B** — same family, covering the upper end of the 7–14B target range.

Holding the family fixed makes the 8B→14B difference a clean size effect rather than a
family confound. Llama-3.1-8B-Instruct is the obvious cross-family reference point if one is
wanted later, but it is `gated=manual` and so needs a license approval and an HF token on the
VM; it is not worth that friction for the exploration stage.

Decoding is greedy, so results are deterministic and differences are attributable to the
model rather than to sampling.

Total generation for the sweep: 2 models × 2 retrieval modes × 300 claims = **1,200 calls**.
Minutes on a batched A100.

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

1. **Dataset anatomy** — structure, schemas, split sizes, class balance, worked examples.
2. **How to load and use it** — the exact commands, the HF pitfalls above, and the retrieval setup.
3. **Recall@k curve** for BM25 over the corpus.
4. **Scorer validation results**, including the cite-everything baseline.
5. **Metrics table** — 2 metrics × 2 models × 2 retrieval modes, i.e. 8 numbers, with the
   label-only versus label+rationale gap called out explicitly.
6. **The three diagnostics.**
7. **Headroom verdict** — an explicit statement of which of the three failure modes in
   Section 1 dominates, and therefore whether mechanism design has room to operate on SciFact.

The verdict is the point. A finding that headroom is small is a real result and gets reported
as such rather than argued around.

## 9. Milestones

| ID | Deliverable | Where |
|---|---|---|
| S0 | Repo scaffold, `uv` env, git remote, `common/llm.py` with a passing smoke call | Laptop |
| S1 | `load.py` + dataset anatomy fact-sheet | Laptop |
| S2 | `retrieve.py` + recall@k curve | Laptop |
| S3 | `prompt.py`, 10-claim smoke test via Ollama | Laptop |
| S4 | Vendored scorer wired, all four validation gates passing | Laptop |
| S5 | VM setup, full sweep: 2 models × 2 retrieval modes × 300 claims = 1,200 calls | A100 VM |
| S6 | Diagnostics computed, `docs/scifact-report.md` written | Either |

S0–S4 need no GPU. S5 is the first step requiring the VM.

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
- **NEI sparsity.** If NEI claims are a small fraction of the 300-claim dev set, abstention
  diagnostics will have wide error bars. The class balance from S1 determines this, and if it
  is too sparse the report says so rather than over-reading the numbers.
