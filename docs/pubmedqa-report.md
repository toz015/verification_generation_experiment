# PubMedQA — exploration report

Pinned stratified sample of **50** items from the official 500-item test set (yes 28, no 17, maybe 5). Greedy decoding, bf16, no retrieval.

Metrics are the official ones — `accuracy_score` and
`f1_score(average='macro')`, the same calls as the authors'
`evaluation.py`.

## Results

| model | accuracy | 95% CI | macro-F1 |
|---|---|---|---|
| Qwen3-8B | 42.0% | [29%, 56%] | 0.412 |
| Llama-3.1-8B-Instruct | 66.0% | [52%, 78%] | 0.420 |
| _always-yes baseline_ | 56.0% | — | 0.239 |

The always-yes row is the free floor. It encodes no reasoning at all, yet
its accuracy tracks the `yes` prevalence. The distance between its accuracy
and its macro-F1 is why macro-F1 is the metric that matters here: accuracy
alone cannot distinguish reasoning from guessing the majority class.

### Published reference points

| system | accuracy | macro-F1 |
|---|---|---|
| BioBERT, multi-phase fine-tuning | ~68% | — |
| single human annotator | ~78% | — |
| large instruction-tuned / medical LLMs | ~79–81% | — |

Taken from the project proposal and **not yet verified against the source
papers**. They are on the same 500-item test set, so they are comparable in
principle, but our sample is 50 of those 500 and carries a much wider
interval.

## Per-class detail

| model | class | precision | recall | F1 | support |
|---|---|---|---|---|---|
| Qwen3-8B | yes | 1.00 | 0.43 | 0.60 | 28 |
| Qwen3-8B | no | 1.00 | 0.24 | 0.38 | 17 |
| Qwen3-8B | maybe | 0.15 | 1.00 | 0.26 | 5 |
| Llama-3.1-8B-Instruct | yes | 0.63 | 0.96 | 0.76 | 28 |
| Llama-3.1-8B-Instruct | no | 0.86 | 0.35 | 0.50 | 17 |
| Llama-3.1-8B-Instruct | maybe | 0.00 | 0.00 | 0.00 | 5 |

## The `maybe` class

| model | maybe recall | 95% CI | maybe predicted |
|---|---|---|---|
| Qwen3-8B | 100% | [57%, 100%] | 34 |
| Llama-3.1-8B-Instruct | 0% | [-0%, 43%] | 0 |

**Read these as directional only.** With 5 `maybe` items the
interval spans most of the unit line. Raising `n` to 500 in
`configs/pubmedqa_experiment.json` uses the full official test set and gives
55 — about two minutes of GPU time, and the right move before any of this
is quoted.

## Confusion

### Qwen3-8B

| gold \ predicted | yes | no | maybe | unparsed |
|---|---|---|---|---|
| **yes** | 12 | 0 | 16 | 0 |
| **no** | 0 | 4 | 13 | 0 |
| **maybe** | 0 | 0 | 5 | 0 |

### Llama-3.1-8B-Instruct

| gold \ predicted | yes | no | maybe | unparsed |
|---|---|---|---|---|
| **yes** | 27 | 1 | 0 | 0 |
| **no** | 11 | 6 | 0 | 0 |
| **maybe** | 5 | 0 | 0 | 0 |

## Parse health

| model | parsed | failures |
|---|---|---|
| Qwen3-8B | 50/50 | — |
| Llama-3.1-8B-Instruct | 50/50 | — |

## Verdict

### The abstention decision is arbitrary

Two models of the same size, on the same items, adopt opposite policies:

- **Qwen3-8B answers `maybe` 34 times out of 50**, against 5 in gold.
- **Llama-3.1-8B answers `maybe` zero times.** All five gold `maybe` items
  are called `yes`.

Nothing in the evidence explains a 34-versus-0 split. The threshold at which
these models decline to commit is inherited from post-training, not derived
from the data in front of them. That is a decision failure in the precise
sense the proposal needs: the capability question and the abstention question
come apart.

### Qwen has the information and misuses it

This is the sharpest result. When Qwen3-8B does commit, its **precision is
1.00 on `yes` and 1.00 on `no`** — every single committal answer is correct.
Its recall is 0.43 and 0.24 because it declines to answer most items it would
have got right.

The discrimination ability is there. What is wrong is the threshold, and a
threshold is exactly what a scoring rule with an abstention reserve sets. This
is the strongest evidence in the exploration so far that the target is a
decision rule rather than a capability.

### The metric cannot see the difference

Despite opposite behaviour, the two models land at **macro-F1 0.412 and
0.420** — a difference of 0.008. One abstains on two-thirds of the data, the
other never abstains, and the official metric rates them equal.

So the existing metric provides no gradient toward correct abstention. It is
not merely that the models are poorly calibrated; the benchmark cannot reward
fixing it. That argues the mechanism is addressing a real gap in the
evaluation, not just chasing a number.

### Accuracy actively misleads here

Qwen3-8B scores **42.0% accuracy, below the 56.0% always-yes baseline**. On
accuracy alone it looks worse than a constant predictor, while its macro-F1
(0.412) is well above the baseline's (0.239). Any PubMedQA result quoted as
accuracy without macro-F1 alongside it should be treated as uninformative.

### Caveats

- **5 `maybe` items.** Qwen's 100% `maybe` recall is trivially produced by
  answering `maybe` almost everywhere — its `maybe` precision is 0.15. Do not
  read that 100% as competence. Going to the full 500-item test set is one
  config line and about two minutes of GPU.
- **Qwen3 ran with thinking disabled** (`enable_thinking=False`), needed for
  reliable JSON. Qwen3 is a reasoning model, so this may materially change its
  behaviour and is a genuine confound in comparing it with Llama.
- **One prompt.** The prompt defines what `maybe` means; a different framing
  could move the abstention rate substantially, which is itself consistent
  with the finding that the threshold is arbitrary.
- The published reference points above are unverified.
