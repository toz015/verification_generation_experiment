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

### The predicted failure is not the observed one

The project proposal predicts `label >> rationale` for zero-shot LLMs:
models that reach the right verdict while citing the wrong sentences.
**The data does not support that at this sample size.** Gaps are 3.5 to
9.3 points, not the tens of points predicted. Read as a ratio, between
**84% and 91%** of label-correct abstracts also carry a correct rationale
(0.888 / 0.840 / 0.909 / 0.857 across the four runs).
When these models know the answer, they mostly know where it is written.

Both models also sit far above the 0.220 free floor in oracle mode, so
they are demonstrating genuine evidence selection rather than harvesting
the metric's cap.

### Retrieval failure dominates

The oracle-minus-BM25 difference is **+0.23 for Qwen3-8B and +0.38 for
Llama-3.1-8B** — three to ten times the label-versus-rationale gap. The
largest single source of error is that the right abstract is not in front
of the model, and roughly 4 points of that is BM25's hard ceiling rather
than anything a model or a mechanism could fix.

For a mechanism-design paper this is the important caveat: measured under
realistic retrieval, most of the loss is upstream of any citation
decision. Any claimed improvement has to be shown net of it.

### Where headroom does exist

Two of the three diagnostics show real weakness, and both are decision
failures rather than capability failures:

**Over-citation, and it scales with uncertainty.** In oracle mode the
models cite 1.24x and 1.32x the gold sentence count. Under BM25 that rises
to **2.80x and 6.59x**. Llama emits 303 cited sentences against 46 gold,
**120 of them past the cap where the metric discards them silently** —
cited at no cost and no benefit. When evidence is weaker, citation volume
goes up rather than down. That is exactly the behaviour a proper scoring
rule with an abstention reserve is meant to price.

**Abstention is weak.** NEI recall is 47-68%, so between a third and a half
of claims with no supporting evidence still receive a confident verdict.
Llama's NEI precision falls to 55.6% under BM25, meaning its abstentions
become unreliable in both directions at once. SciFact has ground truth for
abstention and these models are not using it.

### Answer to the headroom question

There is room for mechanism design on SciFact, but **not in the place the
proposal aims at**. The label-versus-rationale gap is small. The exploitable
gaps are abstention and citation volume under uncertainty, and both appear
specifically when retrieval is realistic rather than oracular.

The practical implication is to run the mechanism against BM25 retrieval,
not oracle abstracts. Oracle mode compresses precisely the pathologies the
mechanism targets: over-citation is 2-5x lower and the abstention problem
is milder.

### What would change this conclusion

- **n=50.** Every proportion carries roughly ±14pp. The label-rationale gap
  is small enough that full dev (300 claims) could move it, though not,
  plausibly, from 9 points to tens of points.
- **Llama's BM25 run had 4 parse failures**, scored as empty evidence. That
  depresses its BM25 row by an unknown amount and makes its dramatic
  collapse there partly a formatting artefact rather than a reasoning one.
- **Llama predicted CONTRADICT once in 50 under BM25** against 11 in gold.
  A near-total collapse of one class is worth understanding before drawing
  conclusions from that row.
- **One prompt format.** These numbers describe this prompt. A different
  framing could shift the label-rationale balance.
