# Raw GPQA verifier scores

Built by `python -m vgx.raw_verifier_export` (no API calls) from `results/raw_verify_20261026/gpqa/signals.jsonl`.

Each of the five verifiers scored **every option A–D of all 445 GPQA questions** (249 calibration + 196
Diamond evaluation) with the main-experiment probability prompt ("probability this answer is correct",
max_tokens 4096). Join to a generator sample in `reports/raw_generator_export_20261026/gpqa.jsonl.gz`
on `item_id` and `candidate == answer`.

`gpqa_verifier_scores.jsonl.gz`: 8,900 rows, one per (item_id, candidate, verifier).

| field | meaning |
|---|---|
| `item_id`, `partition` | as in the generator export |
| `candidate` | option letter A–D being judged |
| `verifier`, `model` | gemini_flash (gemini-3.7-flash), gemini_lite (gemini-3.5-flash-lite), mistral_small (mistral-small-2503), mistral_medium (mistral-medium-3), claude_haiku (claude-haiku-4-5) |
| `score` | verifier's stated probability the candidate is correct; `null` when its reply was invalid JSON (3 rows: 2 gemini_lite, 1 gemini_flash) |
| `failure` | parse failure reason, else `null` |
| `candidate_is_key` | candidate equals the answer key |
