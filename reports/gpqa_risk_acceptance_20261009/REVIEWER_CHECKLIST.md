# Reviewer checklist

Each item names where to look and how to check it. Tests are in `tests/test_risk_control.py`; run with `PYTHONPATH=src python -m pytest tests/test_risk_control.py -q`.

## Target and scope

- [ ] The certified quantity is P(W | A), conditional error among accepted answers, and nothing else drives selection. `risk_control.certify_bonferroni`, `select_certified`; utility appears only in `risk_exploration.curve` as `secondary_mean_utility_L19`.
- [ ] No model, provider or network call can happen. `test_export_reproduces_frozen_fits_and_decisions_offline` and `test_exploration_run_is_offline_and_refuses_a_changed_protocol` block sockets and the provider runner factory; `config.new_api_requests = 0`.
- [ ] Generator confidence is used raw: no calibration, replacement or clipping. `risk_control.score_generator_only`; priors passed unmodified into `decide_public` and `VerifierLikelihood.posterior`.
- [ ] No Diamond answer key is read. The export reads only the 249/199-question development cohort files (`export_observations`; listed inputs in `data/observations.json.gz` provenance).
- [ ] PR #2 (conditioned likelihoods) is untouched; its results are cited only as a historical reference.

## Data and denominators

- [ ] The export matches the source package: `manifest.json` records the package manifest check (81 files); `reproduce.sh OUT PACKAGE` rebuilds the export and byte-compares it.
- [ ] Every Phase I fold refit equals the frozen fold fit, and every B and C decision equals the frozen decision: `phase1/exploration.json` `checks` (60 fold refits, 4,480 always-query, 4,480 sequential, 4,480 replay matches).
- [ ] Invalid candidates and malformed Flash responses stay in N, are never accepted, and malformed responses are paid. `test_invalid_candidates_and_malformed_signals_stay_in_the_denominator`; `TABLES.md` section 1.
- [ ] Costs: B queries every valid candidate; C queries only where its policy does; expected cost = queries x fit-data mean usage. `test_costs_for_always_query_and_sequential_policies`.

## Leakage and order of operations

- [ ] Phase I likelihoods and costs come from training folds only (refit check in `risk_exploration.score_run`).
- [ ] Phase II fitting reads only fit-role labels. `test_fitting_reads_only_fit_role_labels`.
- [ ] Scores never read labels. `test_scores_are_computed_without_labels` (rows without a `correct` key).
- [ ] Selection reads calibration labels and never evaluation labels. `test_selection_reads_calibration_labels_but_never_evaluation_labels`.
- [ ] The selection is sealed before evaluation; evaluation rejects a changed selection or partition. `test_evaluation_needs_the_sealed_selection_and_same_partition`.
- [ ] The partition is by question ID, shared by both generators, deterministic and label-free. `test_partition_is_question_level_label_free_disjoint_and_deterministic`; `phase2/partition.json`.
- [ ] The sequential policy reads only the signal it selects; replay agrees with execution. `test_sequential_policy_reads_only_the_selected_signal`, `test_a_read_beyond_the_counted_queries_is_rejected`, `test_execution_and_replay_agree_and_a_mismatch_is_detected`.

## Statistics

- [ ] Bound: U = BetaQuantile(1 - delta/M; k + 1, m - k); U = 1 when k = m; m = 0 ineligible. `test_binomial_upper_bound_zero_errors_all_errors_and_empty`.
- [ ] Multiplicity: M = 26 counts every declared rule, including rules with m = 0. `test_bonferroni_counts_every_declared_rule_including_empty_ones`, `test_declared_family_matches_config_and_order`.
- [ ] No monotonicity is assumed: each rule is bounded on its own (`certify_bonferroni` loops over all rules).
- [ ] Ties at a threshold are accepted or rejected together; selection ties break by cost, then declared order, never by labels. `test_ties_at_a_threshold_are_accepted_or_rejected_together`, `test_selection_prefers_coverage_then_cost_then_declared_order_never_labels`.
- [ ] Nothing certified prints "No positive-coverage rule certified." and abstain-all is not called a certified 0% rule. `phase2/selection_alpha_*.json`, `phase2/evaluation_alpha_*.json` (`abstain_all.note`).
- [ ] The fixed-sequence alternative is reported separately and its familywise error stays within delta in simulation. `validation/procedure_validation.json`; `test_monte_carlo_familywise_error_is_within_delta`.
- [ ] Sample-size reference values 59 (M = 1) and 117 (M = 20). `test_reference_sample_sizes_and_vectorized_error_allowance`; `plan/sample_size_plan.json`.

## Claims

- [ ] Phase I statements are labeled exploratory; no threshold from a curve is called certified.
- [ ] The Phase II run is labeled a software dry run on development data, not a certification or confirmation; the CLI refuses to run without `--dry-run-on-development-data`.
- [ ] No method is declared the winner on development coverage.
- [ ] Measured results, statistical guarantees and interpretations are kept apart in `FINDINGS.md`.
- [ ] Missing inputs are listed, not substituted: `SAMPLE_SIZE_PLAN.md` "Blockers".
