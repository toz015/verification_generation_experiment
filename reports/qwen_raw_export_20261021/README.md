# Raw Qwen3-235B data for baselines (collected 2026-10-21, before the endpoint retired)

Model `qwen/qwen3-235b-a22b-instruct-2507-maas` (Vertex, us-south1). Built by `python -m vgx.qwen_raw_export`
from the request caches in `results/qwen_raw_20261021/` (kept locally, not in git). One gzip JSONL per kind;
each row has `kind, item_id, sample, attempt, status, prompt, response, parsed fields, finish_reason, usage, logprobs`.
A job whose first attempt was a service failure and that was re-sent once appears twice (attempt 1 failed, attempt 2).

| kind | what | settings |
|---|---|---|
| gpqa_option_probs | probability of each option A–D (JSON) | T=0, top-5 logprobs |
| gpqa_letter | reply with the best letter only | 1 token, T=0, top-5 logprobs |
| gpqa_peg_gen_correct / _incorrect | PEG generator prompt ending "Answer:" / "Incorrect Answer:" (verbatim from toz015/neurips2025-repo@0ade29f) | 1 token, T=0, top-20 logprobs |
| gpqa_peg_disc | PEG discriminator "A. Correct / B. Incorrect" for each option (`sample` = option index) | 1 token, T=0, top-20 logprobs |
| gpqa_judge_p | Qwen as verifier: p_correct for each option, GPQA verifier wording (`sample` = option index) | T=0, top-5 logprobs |
| gpqa_samples | 10 samples of the GPQA generator prompt (`sample` = seed 0–9) | T=0.7, top-5 logprobs |
| math500_samples_nolp | 10 samples of the MATH prompt (reasoning, answer, p_correct) | T=0.7, no logprobs |
| math500_samples | first-round MATH samples with token logprobs (173 ok; the rest stalled, see nolp) | T=0.7, logprobs |
| qkatas_samples_nolp | 10 samples of the QuantumKatas v2 prompt (code, p_correct); grade offline with the official checker | T=0.7, no logprobs |
| math500_peg_disc / math500_judge_p | PEG A/B and verbal p_correct for every distinct candidate answer | T=0 |
| math500_peg_gen_correct / _incorrect | PEG generator distributions over the distinct candidates as options (problems with 2–20 candidates) | T=0, top-20 |
| qkatas_peg_disc / qkatas_judge_p | PEG A/B and verbal p_correct (QK verifier prompt) for every distinct code candidate | T=0 |

Candidate sets (`sample` index → answer/code and which samples or frozen generators produced it):
`stage2_candidates/` (from samples 0–4 + frozen Qwen3/gpt-oss/Sonnet answers) and `stage2b_candidates/`
(samples 0–9). MATH candidates are clustered with the official Omni-MATH-Rule `math_equal`.
