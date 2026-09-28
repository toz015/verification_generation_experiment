# GPQA Confidence and Verification Pilot

**Date:** 2026-09-28
**Status:** Revised draft for a 50-question pilot
**Parent project:** Reference-Aware Selective Generation for Scientific QA

## 1. Goal

Test whether an LLM agent's initial confidence report and one or more verifier scores support useful selective decisions on a compact, answer-key-scored task. This is a pilot of the confidence, verifier-calibration, and sequential-routing pipeline described in `green-laffont.pdf`; it is not a test of the paper's incentive-compatibility theorem.

GPQA is a graduate-level science multiple-choice benchmark, not a math benchmark. The original benchmark has 448 Main questions in biology, chemistry, and physics. Use Main for the pilot so the benchmark's smaller Diamond subset is not used to tune routing thresholds. The benchmark requires agreement to its access conditions; do not commit its questions, answers, or downloaded data to this repository, and do not reproduce examples in public reports.

## 2. Research questions

1. Does the generator's reported probability of being correct predict correctness better than a constant/base-rate estimate?
2. Does a verifier's numeric score distinguish correct from incorrect fixed candidate answers on held-out questions?
3. Does sequential verification improve the accuracy/coverage/cost tradeoff over confidence-only, one-verifier, and fixed-budget baselines?
4. How much does performance change when the generator and verifier use different model families, and how correlated are their errors?

## 3. Scope and limits

- The candidate answer is fixed before verification. Verifiers assess that answer; they do not replace it during the primary experiment.
- The observable correctness target is exact match to the GPQA answer key after normalizing the selected option. Also record answer parsing failures as failures, not dropped items.
- This setting tests a binary correctness signal and answer-level selective release. It does not test citation grounding or evidence retrieval.
- Frozen-model prompts do not establish strategic truth-telling. Without a generator that responds to the proposed reward, the compensation and audit mechanism's IC guarantee remains untested.
- Results on GPQA do not justify transferring calibration parameters to SciFact or PubMedQA. Refit score models and thresholds on training/calibration data from each target dataset.

## 4. Dataset and split

Use the official `Idavidrein/gpqa` source and the Main split, but begin with a pinned sample of **50 questions**. Select the sample with a fixed seed and stratify across biology, chemistry, and physics; store item IDs locally. Record the source revision, access date, license/terms acceptance, and a cryptographic hash of the local source file in a private run manifest. Keep source data and any derived file containing question text outside Git; commit only code, configs without item text, and aggregate reports.

Deterministically shuffle the four options for each selected item and relabel them A-D. Show the same order to the generator and every verifier. Split the 50 questions into 30 calibration and 20 evaluation items, stratified by subject and correct-option position. Store the split and shuffle mappings locally. The small evaluation set is for feasibility and pipeline checks only; it is not enough to establish stable calibration, subtle performance differences, or generalizable mechanism gains. Do not tune on GPQA Diamond. Report the seed and counts by subject and answer class, not question text.

Thirty calibration questions will yield very few examples in some correctness-by-subject cells. Treat any fitted verifier likelihoods as rough pilot estimates, use strong smoothing or a low-parameter calibration model, and report uncertainty. If estimates are unstable, run only prespecified diagnostic baselines and defer claims about optimal sequential routing until a larger follow-up.

## 5. Experimental protocol

### 5.1 Generate and elicit confidence

For all 50 items, request a single answer option and a probability `p_correct` in `[0,1]` that this exact answer is correct. Keep the question/options, prompt, decoding parameters, model revision, and response in a private call log. Use one fixed generator/model and deterministic decoding for the primary pilot; add a second generator only as a replication if budget permits.

Ask for confidence after the answer is committed in the same structured response. The primary analysis treats this report as an observed forecast, not as truthful by assumption. Do not expose answer keys to the model.

### 5.2 Collect verifier signals

For each fixed candidate, obtain an independent numeric confidence score from at least two verifier prompts or models. Each verifier must return a probability that the candidate option is correct, plus a structured parseable response. The verifier sees the original question, choices, and fixed candidate answer, but not the answer key, generator confidence, other verifier output, or other partition outcomes.

Pre-register verifier identity, prompt, temperature, and any tool access. The primary pilot should use no external search so the signal is attributable to verifier reasoning rather than retrieval. If multiple verifier calls use the same model, explicitly treat their conditional independence as a hypothesis to measure, not as a default.

### 5.3 Fit verifier observation models

Using only the calibration partition, estimate each verifier's score distribution conditional on answer correctness (`correct` / `incorrect`). Begin with a small number of fixed score bins with Laplace smoothing, and compare against a logistic mapping from score to correctness. Check calibration and discrimination on the held-out evaluation partition before using the score in routing.

For multiple verifiers, estimate pairwise residual/error dependence conditional on correctness. Compare the sequential policy using a naive conditional-independence update with empirical joint or conservative dependence-aware estimates. Do not present an independence-based posterior as calibrated if verifier errors are materially correlated.

### 5.4 Routing and stopping policies

Freeze rewards/losses and per-call verifier costs before examining evaluation results. Express the system's decision value as `R` for a correct release, `-L` for an incorrect release, and `0` for abstention; charge each verifier call its declared cost. Evaluate a small preregistered grid of cost/loss ratios selected on the calibration split.

Compare these policies descriptively on the untouched 20-item evaluation split:

1. Always answer with the generator's candidate.
2. Always abstain.
3. Confidence-only release at a threshold selected on calibration data.
4. Query one verifier for every item, then release/abstain using its calibrated posterior.
5. Fixed verifier budget (one or two verifiers), then release/abstain.
6. Sequential policy using the nested verifier order and dynamic-programming stopping rule from `green-laffont.pdf`.

The verifier order must be fixed before evaluation. Query costs are incremental; an abstained or released answer is not changed by a post-decision audit. Do not implement an audit reward in this pilot because the frozen models do not create the strategic reporting setting needed to test its claim.

## 6. Metrics

Report aggregate metrics for all 50 questions and the held-out 20 questions separately. On the held-out set, show point estimates and wide uncertainty intervals; do not interpret small differences as reliable. Include subject-level results only as descriptive counts. Track:

- answer accuracy and parse-failure rate;
- confidence Brier score, log loss, reliability plot, and expected calibration error (ECE, with binning stated);
- verifier AUROC and Brier score, plus the same metrics for calibrated posterior estimates;
- selective risk versus coverage curve and area under the risk-coverage curve;
- release coverage, accuracy among released answers, abstention rate, verifier calls per item, and mean verification cost;
- realized system utility for each preregistered `(R,L,cost)` setting;
- paired utility and cost differences against confidence-only and one-verifier baselines;
- pairwise verifier error agreement/correlation, and observed deviation from the conditional-independence approximation.

Accuracy alone is insufficient: include cost and coverage so an always-abstain policy cannot appear successful.

## 7. Decision criteria

Use the 50-question pilot as a feasibility gate, not a formal success claim. Proceed to a larger or cross-dataset experiment only if:

1. Confidence and verifier scores parse reliably, with failures reported in denominators.
2. At least one verifier score adds held-out predictive value beyond the generator confidence alone.
3. The calibrated verifier model is reasonably stable under bootstrap resampling; otherwise simplify or gather more calibration data.
4. Sequential routing has a promising held-out utility direction over the confidence-only policy for at least one preregistered, plausible cost/loss setting, without gains coming solely from nearly universal abstention. Twenty held-out items cannot validate a small utility gain.
5. Dependence checks do not invalidate the posterior updates used for stopping, or the policy is revised to account for the measured dependence.

If criteria 2-4 fail, do not port the mechanism to larger datasets yet. Diagnose whether the limiting factor is poor confidence, weak verifier signal, correlation, or verification cost.

## 8. Implementation sequence

1. Add a GPQA loader that reads a user-provided local dataset path and never downloads implicitly.
2. Add a deterministic split manifest with item IDs only; keep it local/ignored.
3. Extend the existing `BatchRunner`/`CallLog` pattern for generator and verifier requests, preserving model and prompt metadata.
4. Implement strict answer/confidence parsing and scoring without exposing question text in aggregate reports.
5. Implement calibration, held-out diagnostics, and baseline policies before implementing sequential value tables.
6. Run format smoke tests on synthetic records, then the pinned 50-item pilot; inspect parse/latency/cost logs.
7. Review pilot feasibility and uncertainty before deciding whether to expand the sample. Freeze the next evaluation plan before collecting or inspecting any expansion results. Changes after inspecting the 20-item evaluation set are exploratory and require a fresh held-out sample for confirmatory claims.

### Current implementation

The first code slice now provides a local CSV/Parquet loader, deterministic 50-item sample and option shuffle, a 30/20 split, structured prompt parsers, forecast diagnostics, smoothed binned verifier likelihoods, and a counterfactual nested stopping evaluator. `configs/gpqa_experiment.json` specifies Qwen3-8B as generator and Llama-3.1-8B plus Qwen3-8B as verifiers. The repeated Qwen model is intentional for a same-model verifier comparison and must not be treated as an independent signal.

After placing an authorized GPQA Main CSV at `data/gpqa/gpqa_main.csv`, the generation command is `uv run python -m vgx.gpqa.run_pilot`. It writes its pinned sample manifest under ignored `data/gpqa/` and prompts/call logs/metrics under ignored `results/gpqa/`. The runner collects both verifier signals for every parseable candidate, then estimates the calibration likelihoods and counterfactual selective decisions. It does not save the compute cost of an adaptive policy yet because it deliberately gathers all signals for the pilot comparison. The optional `--limit N` flag is for small operational smoke runs; those partial runs must not be interpreted as evaluation results.

The linked notebook's next-token logit elicitation is documented as a comparison to add, not yet implemented. The current first slice uses explicit JSON confidence from the generator and scalar JSON correctness confidence from each verifier.

## 9. Reuse from the linked GPQA notebook

The notebook [`initial_policy_diffModel_GPQA.ipynb`](https://github.com/toz015/neurips2025-repo/blob/main/initial_policy/initial_policy_diffModel_GPQA.ipynb) in the linked repository is relevant as a reference for generating initial policies, not as a drop-in implementation of this paper's sequential mechanism.

Useful ideas to adapt cleanly:

- Load the official `Idavidrein/gpqa` `gpqa_main` configuration and map `Question`, `Correct Answer`, three `Incorrect Answer` fields, and `Subdomain` into a small typed record.
- Shuffle the four options before asking models and keep the correct-option mapping, but replace the notebook's unseeded `random.shuffle` with a deterministic per-item seed.
- Elicit a generator distribution over answer choices from next-token logits and an explicit binary correct/incorrect verifier distribution. These provide useful signal-collection baselines alongside natural-language confidence reports.
- Log model-specific initial distributions before applying any downstream mechanism, enabling a clean comparison of generator confidence, verifier score, and answer-key correctness.

Do not copy the notebook wholesale. Its method iteratively updates generator/discriminator policies using a peer-elicitation game; it does not implement the Green-Laffont sequential value recursion or independent terminal audit. Its discriminator helper also appears inconsistent in the published notebook: it forms a two-class A/B distribution, then maps over the four answer labels A-D. Reproduce only the intended signal-extraction ideas and add schema/shape checks. The notebook's option shuffle is unseeded, it hardcodes Hugging Face credentials in a cell (redacted in the public file), and its direct next-token letter scoring depends on tokenization; use environment-managed credentials and verify token IDs or sequence log-probabilities in our implementation.

For the 50-item pilot, collect both (a) explicit generator `p_correct` and (b) an optional normalized generator choice distribution, and both (a) a scalar verifier `p_correct` on the fixed candidate and (b) an optional binary correct/incorrect distribution. Keep the primary Green-Laffont routing input clearly defined as the probability the fixed candidate is correct. Treat the notebook-style outputs as comparative elicitation methods, not interchangeable confidence values.

## 10. Transfer to existing datasets

After GPQA demonstrates the pipeline:

- **SciFact:** use the existing claim-level gold labels, preserve BM25 and oracle contexts as separate conditions, and define verifier correctness against the fixed claim verdict and evidence support. Retrieval misses remain a separate failure source.
- **PubMedQA:** define correctness on yes/no/maybe decisions and evaluate abstention separately. The current pinned 50-item subset is exploratory; use a larger held-out split before fitting calibration parameters.
- Re-estimate verifier score distributions, confidence calibration, verifier dependence, and policy thresholds separately for each dataset. Carry over code and experimental procedure, not numeric calibration parameters.

## 11. References

- Rein et al., [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022); [official dataset card](https://huggingface.co/datasets/Idavidrein/gpqa).
- Chen et al., [Incentivizing Truthful Language Models via Peer Elicitation Games](https://github.com/toz015/neurips2025-repo), including the GPQA initial-policy notebook.
- `green-laffont.pdf`, Sections 2.4-2.5, for noise-corrected confidence elicitation and nested sequential verification.
