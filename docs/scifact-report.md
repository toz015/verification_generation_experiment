# SciFact — exploration report

Pinned sample of **50** dev claims (SUPPORT 20, CONTRADICT 11, NEI 19), greedy decoding, bf16.

Official metrics come from the SciFact evaluator vendored at a pinned
revision; see `third_party/scifact/PROVENANCE.md`. Every proportion
carries a 95% Wilson interval — at n=50 that is roughly ±14pp, so this
sample can show a large label-versus-rationale gap but cannot rank two
models against each other.

## Official metrics

| model | mode | label-only F1 | label+rationale F1 | gap |
|---|---|---|---|---|
| Qwen3-8B | oracle | 0.806 | 0.716 | **+0.090** |
| Qwen3-8B | bm25 | 0.581 | 0.488 | **+0.093** |
| Llama-3.1-8B-Instruct | oracle | 0.647 | 0.588 | **+0.059** |
| Llama-3.1-8B-Instruct | bm25 | 0.248 | 0.212 | **+0.035** |

The gap column is the headline. Label-only credits getting the verdict
right; label+rationale credits it only when the cited sentences are also
right. A large positive gap means the models know *what* is true but not
*where* it is written — a decision failure about citation, not a
capability failure about entailment.

## Reference points

| strategy | label+rationale F1 |
|---|---|
| gold annotations replayed | 1.000 |
| cite sentences 0–2, gold labels and documents | 0.220 |
| cite sentence 0, gold labels and documents | 0.053 |

The 0.220 row is the free floor: it encodes no evidence selection at all,
yet earns credit because the official metric caps rationales at three
sentences and grants credit when a gold rationale is a subset of what was
cited. Any model scoring near it has demonstrated no citation ability.

## Diagnostics

### Abstention (NEI)

| model | mode | NEI recall | 95% CI | NEI precision |
|---|---|---|---|---|
| Qwen3-8B | oracle | 68.4% | [46%, 85%] | 92.9% |
| Qwen3-8B | bm25 | 47.4% | [27%, 68%] | 90.0% |
| Llama-3.1-8B-Instruct | oracle | 52.6% | [32%, 73%] | 83.3% |
| Llama-3.1-8B-Instruct | bm25 | 52.6% | [32%, 73%] | 55.6% |

### Over-citation

| model | mode | cited | gold | ratio | claims over-citing | beyond the cap |
|---|---|---|---|---|---|---|
| Qwen3-8B | oracle | 73 | 59 | 1.24x | 13 | 8 |
| Qwen3-8B | bm25 | 157 | 56 | 2.80x | 24 | 46 |
| Llama-3.1-8B-Instruct | oracle | 78 | 59 | 1.32x | 20 | 12 |
| Llama-3.1-8B-Instruct | bm25 | 303 | 46 | 6.59x | 27 | 120 |

`beyond the cap` counts cited sentences the official metric silently
discards — emitted for free, scored neither way.

### Retrieval isolation (oracle − BM25)

| model | label-only Δ | label+rationale Δ |
|---|---|---|
| Qwen3-8B | +0.225 | +0.228 |
| Llama-3.1-8B-Instruct | +0.399 | +0.376 |

BM25 hit@5 on full dev is 89.9%, and recall saturates near 96% at any k,
so roughly 4% of claims have gold abstracts that are unreachable by lexical
retrieval. That is a floor on this difference which no model can close.

### Parse health

| model | mode | parsed | failures |
|---|---|---|---|
| Qwen3-8B | oracle | 50/50 | — |
| Qwen3-8B | bm25 | 50/50 | — |
| Llama-3.1-8B-Instruct | oracle | 50/50 | — |
| Llama-3.1-8B-Instruct | bm25 | 46/50 | no_json=4 |

Parse failures are reported rather than dropped: excluding them would
flatter whichever model fails most.

## Verdict

_Filled in from the numbers above once the sweep completes._
