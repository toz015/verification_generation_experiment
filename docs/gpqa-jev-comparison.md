# Jev comparison on frozen GPQA candidates

This comparison adds Jev to the completed Vertex experiment without replacing
generator answers, changing original prompts or making new Vertex requests.
Its separate Jev budget is **USD 10**, authorized by the user. The source is
`results/gpqa_vertex_20261004/frozen`: 249 calibration questions and 196 filtered
Diamond evaluation questions, including three generator format failures.
The existing evaluation results have already been examined, so subsequent Jev
comparisons on this cohort are exploratory.

## Model, credentials and costs

- Pin `jev-1.13.0`, and require the response to return the same version.
- Keep `TYPESAFE_API_KEY` in the local ignored `.env` or process environment.
  The loader reads only that variable; it does not execute shell syntax or
  expand substitutions. Credentials are never part of inference cache keys.
- The [official model page](https://docs.typesafe.ai/models), checked on
  2026-10-04, lists USD 0.042 per million input tokens and free output tokens.
- The separate budget ledger reserves a full 64,000 input tokens per pending
  request. Recorded usage replaces the reservation after a successful response.
  Missing usage, over-reservation usage, rejected requests and unknown outcomes
  block new calls pending review. Jev errors are not assumed to be free.
- This is a local usage-estimate guard, not a provider account spending limit.
  No invoice or account-credit deduction is inferred from token counts.

## Signals and predeclared comparisons

**Choice:** ask which option answers the original multiple-choice question.
Keep the original option order and extract the probability assigned to the
frozen generator option, even if another option has a higher probability.

**Noul:** ask whether the fixed generator answer is correct; use the returned
yes probability. The generator confidence and answer key are hidden from both
modes. The Jev `confidence` field is not a correctness probability and is not used.
Choice and Noul are separate arms and are never chained as independent evidence.

The primary arm updates the generator prior using Choice likelihoods. Secondary
arms use Noul alone and Choice followed by the cached Gemini verifier. Each arm
uses the same reward +1, incorrect-release loss 19, abstention 0 and conversion
of 10 utility units/USD as the primary Vertex study. Query costs are the mean
calibration usage estimates and remain fixed throughout evaluation.

Offline forecasts compare raw Jev probabilities, the likelihood-updated
generator prior, and a score-only logistic calibrator fitted on calibration
data. A separate direct policy queries Jev for every valid candidate and releases
when the raw Jev probability reaches 0.95. This distinguishes treating Jev as a
probability forecast from treating it as an evidence channel for our algorithm.

## Collection sequence

1. Save the immutable configuration, analysis plan and source bundle identity.
2. Run Choice and Noul on the first 10 valid calibration candidates, selected
   from the existing deterministic order without using answer keys.
3. Check response parsing and observed costs; reuse these responses in the
   complete 247-valid-candidate calibration collection.
4. Fit and seal all policies from calibration observations and labels only.
5. Execute the primary Choice arm, then secondary arms, with durable item
   checkpoints. Each decision can access only a selected verifier response.
6. Collect extra evaluation Jev responses for direct and query-all baselines.
7. Open evaluation labels for scoring, paired forecast and utility comparisons,
   intervals, dependence diagnostics and live/replay consistency checks.

The primary Jev arm obtains new evaluation responses as it selects queries.
Secondary arms may reuse those responses and existing Vertex responses. Report
logical query costs separately from incremental new API spending. There are at
most 884 distinct successful Jev requests for the two modes on 442 valid frozen
answers. No replacement generator requests are available in this runner.

## Commands

Configuration: [`configs/gpqa_jev_comparison.json`](../configs/gpqa_jev_comparison.json).
Run the initial batch, then the full comparison:

```sh
.venv/bin/python -m vgx.gpqa.jev_experiment --phase pilot --allow-api
.venv/bin/python -m vgx.gpqa.jev_experiment --phase complete --allow-api
```

Resume entirely offline by omitting `--allow-api`:

```sh
.venv/bin/python -m vgx.gpqa.jev_experiment --phase complete
```

The default output is `results/gpqa_jev_20261004/`, containing the configuration,
analysis plan, pilot summary, calibration observations, sealed policies, live
item states, individual arm reports, summary report and Jev usage ledger.
These detailed artifacts remain ignored by Git. Only aggregate results should
be copied into `reports/`.

## Completed run and rounding sensitivity

The [completed comparison report](../reports/gpqa_jev_comparison_20261004.md)
records results, costs, validation and limitations. Eight Choice responses had
probabilities summing to 0.99 and were rejected by the predeclared strict parser.
Their full responses remain cached. A separate post hoc analysis accepts only
mass errors consistent with four two-decimal probabilities and evaluates both
the original marginals and renormalization. It preserves the strict live result:

```sh
.venv/bin/python -m vgx.gpqa.jev_rounding
```

This command is offline only. Its policies are counterfactual sensitivity
analyses, not live reruns or replacements of the sealed policies.
