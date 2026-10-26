# Raw generator samples: answers, stated confidence, correctness

Built by `python -m vgx.raw_generator_export` (no API calls) from the raw collections in
`results/qwen_raw_20261021`, `results/raw_gptoss_20261022`, `results/raw_sonnet55_20261026`
and the offline labels in `results/raw_labels_20261026`. Snapshot of 2026-10-26 (evening); Sonnet sampling was still running, so later snapshots will have
more Sonnet rows and labels.

## Generators

| generator | model | sampling |
|---|---|---|
| `qwen3` | qwen3-235b-a22b-instruct-2507 (Vertex) | T=0.7, top_p=1, seeds 0–9 |
| `gptoss` | gpt-oss-120b (Vertex), reasoning effort medium | T=0.7, top_p=1, seeds 0–9 |
| `sonnet55` | claude-sonnet-5-5 (Vertex), adaptive thinking, effort high, 32768 max tokens | provider default temperature; samples 0–9 |

Prompts are the main-experiment generator prompts (GPQA: answer letter + `p_correct`; MATH-500:
reasoning + answer + `p_correct`, answer-only for Sonnet; QuantumKatas: v2 prompt, code + `p_correct`).

## Files

`gpqa.jsonl.gz`, `math500.jsonl.gz`, `qkatas.jsonl.gz`: one row per (generator, item_id, sample).

| field | meaning |
|---|---|
| `generator`, `item_id`, `sample` | sample = seed index 0–9 |
| `answer` | GPQA letter A–D, or MATH final answer string (`null` = no answer) |
| `code` | QuantumKatas only: the submitted Python/Qiskit code; `code_sha256` identifies it |
| `p_correct` | the model's own stated probability that its answer is correct |
| `correct` | offline label: GPQA vs. answer key; MATH with the pinned Omni-MATH-Rule `math_equal`; QuantumKatas with the official checker. `null` = not graded yet. No answer counts as `false`. |
| `finish_reason` | `length` / `max_tokens` = truncated |
| `answer_parse` | `direct`, or `recovered`: gpt-oss wrapped its JSON (`{"final": ...}`) and the answer was taken from inside the wrapper |

`gpqa_option_probs.jsonl.gz`: per item and generator, the model's probability for each option A–D
(`question: correct`, T=0); Sonnet also has "probability each option is incorrect" (`question: incorrect`).
`correct_letter` is the key. 286 of the 445 gpt-oss replies are broken JSON (e.g. `{"final{":":0}`) and have no usable probabilities.

`summary.json`: rows, labelled, correct and recovered counts per dataset and generator.

## Counts in this snapshot

| dataset | generator | items | samples | graded | correct |
|---|---|---|---|---|---|
| GPQA (445-item main-experiment set) | qwen3 | 445 | 4450 | 4450 | 2120 |
|  | gptoss | 445 | 4450 | 4251 | 2919 |
|  | sonnet55 | 445 | 2670 | 2641 | 2291 |
| MATH-500 | qwen3 | 500 | 4998 | 4998 | 4826 |
|  | gptoss | 500 | 5000 | 5000 | 4455 |
|  | sonnet55 | 500 | 2802 | 2802 | 2498 |
| QuantumKatas (322 tasks) | qwen3 | 322 | 3220 | 3219 | 1727 |
|  | gptoss | 322 | 3220 | 3168 | 2037 |
|  | sonnet55 | 322 | 1610 | 329 | 318 |

Item ids: GPQA ids are the sha256 ids of our GPQA copy (no question text is included here,
following the GPQA authors' request not to post examples in plain text); MATH-500 ids are
`math500:<original path>`; QuantumKatas ids are `<kata>/<task>`.

Full raw replies, prompts and token logprobs for Qwen3 are in `reports/qwen_raw_export_20261021/`.
