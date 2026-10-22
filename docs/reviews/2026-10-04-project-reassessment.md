# Project reassessment: theory, implementation, evidence, and next experiment

Reviewed on 2026-10-04. This is a review artifact, not an experiment or a runtime-code patch.

## Conclusion

The project has a useful foundation for studying **selective release of a fixed LLM answer under verification costs**. Its GPQA code implements a discrete observation model and a valid Bellman recursion for a fixed verifier order, subject to its statistical assumptions. It does not yet execute that stopping policy against live APIs or implement the paper's incentive mechanism. Existing results do not establish a reliable improvement from verification, truthful reporting, or real adaptive savings.

The next experiment should preserve the existing candidates, repair collection and accounting boundaries, evaluate additional verifier information, and then execute a frozen stopping policy live. Jev is a promising candidate for a cheaper verifier, but neither its probability output nor a different model family establishes truthful beliefs, calibration, or independence.

## 1. Scope and project state

I reviewed the available conversation back to its first request, repository history, the GPQA, SciFact and PubMedQA implementations, common inference/logging code, tests, design documents, and committed experiment reports. I read the seven-page `green-laffont.pdf`, the team member's twelve-page `online_learning_simulation_study_v5.pdf`, and the original citation-benchmark slides. Relevant algorithm and simulation pages were also inspected visually. I revisited the linked GPQA notebook and current TypeSafe/Jev documentation and research.

- Branch: `codex/gpqa-pilot`.
- Local HEAD and the remotely advertised branch HEAD both resolve to `2cea9e74b89634615f707f6da239a9e0c1265485`; no pull was needed.
- Existing local changes were preserved: `README.md`, `configs/gpqa_experiment.json`, `src/vgx/gpqa/run_pilot.py`, and the untracked `src/vgx/common/billing.py`. The preexisting `asqa_prompt.json` was untouched.
- GPQA raw data, the pinned sample manifest, and the reported run's `results/gpqa/` artifacts are **absent from this laptop checkout**. Metrics below are reported results, not an independent recomputation from raw responses.
- No model inference, paid API request, VM operation, commit, or push was performed for this review. Only this review document was added to the repository.

The project evolved from 50-item SciFact/PubMedQA explorations to a planned **50-question GPQA pilot, 30 calibration + 20 evaluation**. Later remote work produced the current **200-question, 120 + 80 Vertex experiment**. Preserve that completed experiment as a versioned artifact; do not silently reset its configuration or reinterpret it as the original 50-question pilot. The old L4/local-weight setup is historical infrastructure. Current managed-API inference can be orchestrated from the laptop.

The user's later instruction to focus on the two Green–Laffont algorithms supersedes the earlier tentative PEG comparison. PEG is not the next implementation target.

## 2. What the paper actually specifies

Source: local `green-laffont.pdf`, Sections 2.1–2.5 and Algorithms 1–2. The paper contains both a confidence-elicitation mechanism and a sequential verification policy. These have different implementation requirements.

### Fixed candidate and distinct quantities

| Quantity | Meaning | Current LLM mapping |
|---|---|---|
| `w=(x,o,H)` | Public question, fixed candidate, public history | GPQA question/options, chosen option, available history |
| `Y` | Whether that candidate is correct | Comparison with the answer key, used for calibration/evaluation |
| `p` | Generator's private subjective probability conditional on its information | Not directly observable in a hosted LLM |
| `r` or `hat p` | Generator's submitted report | JSON `p_correct` |
| `b` | Controller's current correctness belief | Raw report initially; Bayes-updated after observations |
| `V_j` | Verifier observation | Verifier score, currently placed in one of three bins |
| `P_{j,y}` | Observation law conditional on correctness and public context | Estimated pooled score-bin frequencies on calibration data |
| `R,L` | System reward for correct release and loss for wrong release | Configured utility values |
| `kappa_j`, `d_k` | Query cost and cost of the next layer | Configured normalized costs; not currently measured dollars |
| `Z` | Unbiased, noise-corrected terminal audit outcome | Not implemented; may lie outside `[0,1]` and is not a confidence |

An honest subjective probability need not be empirically calibrated. A well-calibrated forecast does not prove that an agent reported its private belief. A next-token probability, a verbal confidence, and a Jev option probability are different observable signals; none gives direct access to an LLM's unique “true belief.”

### Algorithm 1: plan the value of future verification

For clarity, call the immediate release value `a(b)=(R+L)b-L` and stopping value `S(b)=max(0,a(b))`. The paper does **not** call this expression `A(b)`. Its separate `A_y^Pi(r)` is the probability of release conditional on outcome class and report. Those notations must not be conflated.

At the last stage, `J_K(b)=S(b)`. Working backward:

```text
Q_k(b) = -d_(k+1) + E[J_(k+1)(updated belief)]
J_k(b) = max(S(b), Q_k(b))
```

`Q` is the expected value of paying for the next layer, including subsequent optimal decisions. `J` is the best value available at that stage. The expectation averages possible signals; it must not read the actual future response.

The paper explicitly constructs tables on a belief grid and interpolates their values. The current `simulate_sequential_decision` instead memoizes a recursion with belief keys rounded to `1e-9`. For the current small, discrete, one-verifier-per-layer problem, that is a reasonable numerical realization of the same recurrence. It is not an explicit reusable value-table implementation, and it rebuilds the cache for each item.

### Algorithm 2: execute the decisions

Start from `b=h(r,w)`, with the paper's baseline `h(r,w)=r`. At each stage:

1. Compare stopping value against continuation value.
2. Stop if `S(b) >= Q_k(b)`; ties stop.
3. When stopping, release if `b >= L/(R+L)`; otherwise abstain.
4. Otherwise call only the next layer, pay its cost, update the belief, and repeat.

Crossing the release threshold is not by itself the stopping rule. A high-confidence answer can still merit verification; a low-confidence answer can be abandoned immediately. The answer itself stays fixed. The paper shares signals with the generator, but does not require it to revise the answer or submit another report.

The optimality claim is within a specified nested hierarchy. It does not establish that the hierarchy, cheapest-first order, or choice of models is globally optimal. The current implementation supports singleton layers, not general multi-verifier layers queried in parallel.

The paper also specifies abstention when the observed signal has zero predictive likelihood. The current code instead retains the prior; a concrete discrepancy is reproduced below.

### Confidence incentives are a separate implementation

The paper gives the generator a release payoff and a terminal scoring transfer. A fresh audit is sampled after the release/abstain decision, independently of the decision transcript, with positive probability `epsilon`. With known audit sensitivity `s*` and false-positive rate `f*`:

```text
Z = (V* - f*) / (s* - f*)
ell(r,Z) = r^2 + (1 - 2r) Z
B_hat = release * [(R_G + L_G) Z - L_G]
T = -(B_hat + lambda * ell(r,Z)) / epsilon   if audited
T = 0                                       otherwise
```

The compensation term offsets the generator's release incentive. Under the stated assumptions, truthful reporting maximizes expected payoff, with a quadratic penalty for deviating from the private belief. This is a mathematical incentive result, not a claim that a frozen model will learn from an externally recorded number.

There is currently no terminal audit, transfer, reward-responsive training, or demonstrated strategic best response. A Vertex API call will not update its weights because our application records a reward. Feedback in future prompts or model training would be separate interventions. They are outside the immediate Algorithms 1–2 implementation.

## 3. How the online-learning study differs

Source: `/Users/wanghd/Desktop/online_learning_simulation_study_v5.pdf`, especially Sections 1.2–1.4 and the estimator appendix.

The simulation is useful for testing the interaction of learning, costs, and stopping, but it is not an empirical LLM implementation of the original mechanism:

| Aspect | Green–Laffont draft | Team simulation | Current GPQA |
|---|---|---|---|
| Signal laws | Assumed known | Learned without outcome labels from complete audit panels | Estimated with labeled calibration examples |
| Audit | Fresh terminal scoring observation | Fresh panel of three verifiers; probability decays as `t^(-1/3)` | None |
| Generator payoff | Release payoff plus compensation and proper scoring | Scoring payoff only; no additional release payoff | Frozen model prompted for confidence |
| Generator adaptation | Strategic best response in the theorem | Analytic best response in simulation | No reward-driven update |
| Planning | Belief-grid tables | 2,049-point grid, periodically rebuilt | Per-item memoized recursion |

The simulation explicitly omits the generator's extra release payoff, so its score-only reward does not need the same compensation term. Its stated reward also has no inverse-audit-probability factor. These are deliberate setting differences, not interchangeable implementations.

The unlabeled estimator uses three informative conditionally independent anchors and additional identification/orientation assumptions. Our two-verifier labeled pipeline cannot simply adopt it. Repeated calls to one model do not automatically supply independent anchors.

The million-round synthetic savings and quality figures are evidence within the simulation's assumed world. Binary and continuous settings use different rewards/losses, so comparing their absolute regret or quality as a signal-type ranking is inappropriate. Borrow the explicit timing, complete cost ledger, and oracle/control comparisons; do not transfer its numerical guarantees to a small GPQA sample.

## 4. What the GPQA evidence supports

The committed report records Gemini 3.8 Flash as generator, Llama 3.3 70B as verifier 1, and Gemini 3.7 Flash as verifier 2. The generator produced 176 correct answers over 200 items; four outputs were invalid JSON.

On evaluation, accuracy over all items was **69/80 = 86.25%**. Forecast metrics used **77** items with valid answers and confidence, of which **69/77 = 89.61%** were correct. These denominators answer different questions and should always accompany the metric.

| Forecast on the reported 77-item cohort | Brier | Log loss |
|---|---:|---:|
| Raw generator report | 0.0730 | 0.2575 |
| Updated with verifier 1 | 0.0737 | 0.2635 |
| Updated with verifier 2 | 0.0682 | 0.2542 |
| Updated with both | 0.0679 | 0.2592 |

The paired Brier difference for verifier 2 was `-0.00475`, with reported 95% interval `[-0.02293, 0.00964]`. Thus improvement was not established. The combined update slightly improves Brier and worsens log loss. The report's executive summary mentions paired intervals for the combined result, but its body explicitly says no combined paired interval exists, consistent with the code. That sentence needs correction.

Each verifier had only **12 incorrect calibration examples**. There were only **8 incorrect answers among the 77 valid evaluation forecasts**. Rare-error likelihoods, dependence diagnostics, and decisions near high-confidence thresholds therefore have substantial uncertainty.

Verifier 1 returned only zero or one on evaluation. Its scores scarcely separated correct from incorrect candidates. That is a result for this prompt and endpoint configuration, not proof that the model family is inherently unusable. The generator prompt explicitly requests a probability; the verifier prompt only says “Assess whether” and shows a `p_correct` JSON template. Its probability semantics should be made explicit in a new, separately versioned prompt arm.

The recorded 592 responses comprise 200 generator calls and 196 calls to each verifier. Both verifiers were collected before policy replay. These records support retrospective paired comparisons; they do not demonstrate actual API calls or dollars saved by adaptive stopping.

## 5. Statistical assumptions requiring attention

**Independence is stronger than using different model names.** The paper assumes independence of verifier observations conditional on `(Y,w)`, including independence from the generator's private information. The code fits pooled `P(score_bin | Y)`, losing variation across question context and generator confidence. One can double-count information even with only one verifier if its signal and the initial report share information not represented in the fitted model.

The reported residual correlation near `0.78` is a warning, but a pooled correlation of `(score - Y)` is not a test of conditional independence given `(Y,w)`. Even conditionally independent scores can have correlated residuals because of class-dependent means. Within-class diagnostics help but are sparse here and still do not condition on question context. Earlier language treating the pooled number as a direct contradiction of the paper's assumption was too strong.

The current likelihood fitter does not enforce the paper's monotone likelihood-ratio assumption. Laplace smoothing ensures nonzero bins, but with severe class imbalance an empty bin receives different class-conditional mass purely from pseudocount denominators. Inspect this sensitivity before treating the fitted channel as informative. Exact prior values zero and one also cannot be moved by ordinary Bayes updates.

A practical next comparison is raw generator confidence, a low-capacity calibrated generator forecast, and a joint forecast using generator confidence plus one verifier score. Fit and select these on calibration/development data only. A joint predictive model is a robustness baseline; using it in a Bellman planner additionally requires a coherent predictive law for future signals. It is not enough to substitute a classifier's output into the existing independence-based expectation.

The current bootstrap intervals condition on the fitted calibration model. Separate fit-stability diagnostics are useful, but do not propagate all calibration and policy-selection uncertainty into deployment outcomes.

Utility optimization is also not a distribution-free accuracy guarantee. `L=19,R=1` produces a nominal release threshold of `0.95`; an uncalibrated belief at that threshold does not guarantee 95% released accuracy. Report coverage, wrong-release risk, uncertainty, and false-accept rate explicitly.

## 6. Cost and stopping: what needs to connect

Keep three quantities distinct:

1. **Expected query cost available before deciding:** required by `Q_k(b)`.
2. **Observed usage and estimated dollars after a call:** required by the experiment's ledger.
3. **Total experiment spend:** includes calibration, all-response comparison collection, generator calls, and any audits or billed retries.

Current `.02` and `.10` verifier costs are utility sensitivity settings. They are not USD and are not automatically updated by the new billing module. A possible explicit mapping is `d_j = alpha * E[USD_j | information available before the query]`, with `alpha` fixed before evaluation. Add latency only if the objective explicitly values it.

Estimate expected output/reasoning usage from calibration, optionally conditional on observable input length. Do not let a policy inspect the actual unqueried evaluation response's token count or score. Realized costs belong in the subsequent ledger. A generator call already made is a sunk cost for the within-item stop comparison, but remains part of total experiment cost.

In the paper, a constant independent terminal-audit cost adds the same expected expense to every stopping action, so it cancels from the comparison. It still belongs in total utility if auditing is actually implemented. No audit charge should be claimed for the current unaudited experiment.

The new billing code only reads `prompt_tokens`/`input_tokens` and `completion_tokens`/`output_tokens`. It does not reconcile separately reported reasoning tokens. Google prices output including reasoning; whether an OpenAI-compatible `completion_tokens` value already includes reasoning must be checked against the actual endpoint response before adding anything. Blindly adding a detail field can double-count. [Google pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing)

Applying the configured rates to the report's prompt and completion totals gives **$0.15465921**. Adding its separately listed reasoning counts gives **$0.87204171 if those counts are disjoint**. The earlier approximately $0.87 estimate used the latter interpretation. Neither figure is a reconciled invoice; raw usage payloads are missing locally. The estimator can currently mark all calls priced without resolving this ambiguity.

The ledger also deduplicates successes by request key. That is appropriate for unique logical requests but cannot distinguish duplicated log lines from multiple genuinely billed executions. Log attempted requests and provider identifiers separately before calling the result actual spend.

## 7. Implementation findings and offline reproductions

Locations refer to the reviewed working tree, including the preexisting uncommitted billing patch.

| Priority | Finding | Location and consequence |
|---|---|---|
| Before another comparison run | Collector always regenerates candidates and collects every verifier | `src/vgx/gpqa/run_pilot.py:112`: no verifier-only augmentation of an immutable candidate artifact; no live stopping |
| Before adding Jev | Whole-run identity controls storage for reusable model responses | `run_pilot.py:72`, `:163`: changing verifier list, pricing, reporting code, or certain environment fields produces a new log directory even if the generator request is identical |
| Before using dollar-based conclusions | Reasoning usage and billed executions are not reconciled | `src/vgx/common/billing.py:34`, `:68`: observed logical-request estimate, not a complete bill |
| Fix resume robustness | Appending after a truncated final JSONL line loses the next successful row on reload | `src/vgx/common/llm.py:77`: reproduced with a temporary file |
| Fix paper discrepancy | Zero predictive likelihood returns the prior instead of abstaining | `src/vgx/gpqa/score.py:105`: reproduced; current positive smoothing masks it in ordinary pilot fits |
| Improve comparator fairness | Only the first verifier has a one-verifier policy baseline | `src/vgx/gpqa/report.py:189`: Gemini-only routing and alternate fixed orders are absent |
| Improve reproducibility | Hosted model aliases are not immutable revision pins | `src/vgx/common/vertex.py:50`: resolved response version is not enforced; old HF revision validation does not cover the live Vertex path |
| Correct reporting | Combined posterior has no paired delta interval; “actual calls” is inferred from row counts | `src/vgx/gpqa/report.py:330`, `:370`: read logs for observed calls and add the paired comparison |
| Fix legacy denominator | SciFact NEI recall excludes parse failures before its denominator | `src/vgx/scifact/score.py:123`: can inflate recall and downstream intervals |
| Fix legacy parsing | PubMedQA's fallback chooses the earliest label anywhere in prose | `src/vgx/pubmedqa/prompt.py:115`: negated mentions can invert the intended answer |

The identity behavior protects against stale reuse, but it is too broad for the intended comparison workflow. Separate immutable dataset/candidate identity, exact provider-request identity, and analysis identity. Altering prices or report code should not itself require paying for unchanged inference.

The current Vertex adapter deliberately omits temperature and top-p for Gemini 3 requests (`vertex.py:99`). Therefore, the config's `temperature: 0` is not evidence that those requests used temperature zero. Preserve the actual submitted parameters and returned provider model/version metadata when describing reproducibility.

Offline probes used synthetic items and temporary files only:

| Probe | Observed result |
|---|---|
| Append a valid successful call after a file ending in `{"key":` | In-memory cache says present; reloading finds zero valid rows and loses the successful key |
| Change only pricing metadata | Run ID changes |
| Append a new verifier while leaving generator request/settings fixed | Run ID changes; the generator's per-request cache key stays the same |
| Likelihoods `P(bin|correct)=(0,0,1)`, `P(bin|incorrect)=(1,0,0)`, prior `0.5`, cost `0.1`, observed middle bin | Queries once, then asserts at `0.5` with no failure; paper requires abstention on the impossible observation |
| Two true NEI items: one correct NEI response, one parse failure | Reported NEI recall `1.0`; full-cohort recall `0.5` |
| Parse `The answer is not yes, it is maybe.` | Returns `yes`, marked successful |
| Usage with 100 prompt, 10 completion and a separate reasoning detail of 90 | Estimator charges 110 tokens and says all calls priced, with no schema reconciliation |

The temporary reproducer was `/tmp/verification-audit-20261004/probes.py`; it changed no runtime code. These are additional probes, not claims that the existing test suite covers the cases.

## 8. Reassessment of the older datasets and previous advice

**PubMedQA “maybe” and SciFact NEI are task labels, not the mechanism's release/abstain action.** A system may confidently release a correct “maybe” or NEI answer. For mechanism experiments, represent `candidate_label`, `candidate_correct`, and `release_action` separately. The original conversation and older reports blurred this distinction. [PubMedQA task definition](https://pubmedqa.github.io/) · [SciFact task and evaluation](https://github.com/allenai/scifact)

The PubMedQA narrative's perfect precision on a selected yes/no subset does not tell us whether the model would have answered the remaining questions correctly. It cannot establish that the underlying issue is incentives rather than capability without a counterfactual intervention.

The SciFact narrative's retrieval ceiling “at any k” is unsupported by a sweep bounded at `k=100`. An oracle-versus-BM25 gap can also reflect distractor/context effects, not just missing evidence. Do not interpret the gap as a pure retrieval-miss effect without matched analyses.

Both legacy report generators contain fixed numerical conclusions embedded in source strings, and their ingestion is less strict about mixed configurations/cohorts than the GPQA pipeline. Regenerating them on new logs does not guarantee their prose describes the new run. Treat the old reports as versioned exploratory results, and repair these issues before extending the verification mechanism to those datasets.

**Correction to my earlier notebook review:** the discriminator helper computes two logits and uses `zip` with a four-label list. `zip` stops after two entries, so it maps A/B to `correct`/`incorrect`; it does not access C/D or cause the shape error I previously suggested. Explicit binary labels would be clearer, but the stated mismatch was not a demonstrated bug. The notebook's generator does normalize selected next-token answer-letter logits, and its subsequent policy dynamics are distinct from the fixed-candidate verification task. [Original notebook](https://github.com/toz015/neurips2025-repo/blob/main/initial_policy/initial_policy_diffModel_GPQA.ipynb)

## 9. Jev: revised assessment

Jev is worth evaluating as an additional low-cost signal. TypeSafe offers structured Choice, Noul and Score outputs through a different API schema from the current Vertex collector. A provider adapter is needed; changing only the model string will not work. [TypeSafe introduction](https://docs.typesafe.ai/introduction)

For GPQA, use either:

- **Choice:** obtain the four option probabilities, then extract the probability of the **frozen generator-selected option**. Do not substitute Jev's favorite option, or this ceases to be a fixed-candidate comparison.
- **Noul:** ask whether the fixed candidate is correct and use its yes probability as a verifier signal.

Choice can also select A–D and thus serve as a generator for this multiple-choice task. My earlier wording that it could not replace the generator was too categorical. It cannot generate an ordinary prose solution through these primitives, but our current generator only needs an option and confidence. Keeping Gemini fixed is an experimental-control recommendation, not a technical impossibility. [Choice documentation](https://docs.typesafe.ai/primitives/choice) · [Noul documentation](https://docs.typesafe.ai/primitives/noul)

**Do not map TypeSafe's `confidence` field to `p_correct`.** For `n` Choice options:

```text
confidence = (max(option_probability) - 1/n) / (1 - 1/n)
```

With four choices, top probability `0.85` produces confidence `0.80`. If the generator selected a different option, neither number is its correctness signal; use that option's entry from `probabilities`. Noul returns a yes probability without a separate confidence field. [Confidence definition](https://docs.typesafe.ai/confidence)

The official site currently advertises **$42 per billion input tokens**, or **$0.042 per million**. This supports testing its economics, not assuming a fixed per-question charge or billing terms without observing usage. [TypeSafe pricing](https://typesafe.ai/)

The independent September 2026 preprint evaluates many structured tasks and reports useful but task-dependent behavior and calibration. It does not establish GPQA fixed-candidate verifier calibration. The earlier cited personal GPQA experiment is a question-answering benchmark with option-order sensitivity; it is not evidence of marginal verifier value on Gemini's selected answers. TypeSafe also documents task weaknesses. [Independent benchmark preprint](https://arxiv.org/abs/2609.37647) · [Jev limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

Choosing a weaker model is not a theorem requirement. The useful criterion is its **expected improvement in decisions at its incremental cost**, given the information already available. A weaker solver may detect errors a stronger generator makes, but it may also miss exactly those hard errors. A stronger verifier may justify its cost on a small difficult subset. Different model families are a design variation, not an independence certificate.

Recommendation: preserve the current generator for the first paired comparison; collect Jev Choice and/or Noul signals for the same frozen candidates; choose a primary signal arm on calibration/development data; compare each existing verifier separately as well as any fixed hierarchy. Do not treat multiple Jev outputs from one request as conditionally independent evidence.

## 10. Concrete next implementation sequence

1. **Recover the existing private artifacts.** Obtain the pinned sample manifest, candidate records, raw provider usage, and run manifest from the original run. Validate IDs, hashes, choice order, split, and model metadata. Reproduce reported results locally before making a new scientific comparison. This can use private file transfer; raw questions and responses need not enter Git.

2. **Repair the software boundaries.** Add explicit immutable candidate records and a verifier-only collection command, separate request caching from analysis identity, repair truncated-log recovery, and reconcile token accounting. Preserve unknown billing fields and distinguish estimated charges from confirmed billing. This enables adding Jev without silently regenerating candidates.

3. **Finish the algorithm interface.** Fit observation models on calibration only; create reusable value tables or an explicitly documented equivalent planner; expose a `decide(state)` operation returning release, abstain, or next layer; save `b`, `S`, `Q`, expected cost, observations and stop reason. A live executor calls only the returned layer and resumes from logged state. Its interface must not expose future verifier scores or answer keys.

4. **Add fair offline comparisons.** Include each verifier alone, alternate prespecified fixed orders, calibrated confidence-only, all-verifier and no-verifier baselines. Compare both forecast quality and decision utility at matched cost/risk/coverage where possible. Include combined paired intervals, wrong-release rates, uncertainty from fitting, and actual collection spend separately from replay cost.

5. **Run a small frozen live feasibility evaluation.** Keep one selected policy and model/prompt version fixed. Fully observed offline replay is valuable for comparison; live execution demonstrates that the implementation really avoids calls. Validate live/replay decisions under the same recorded signals before interpreting savings. For real API comparisons, log timestamps and provider versions because changing hosted aliases can change behavior.

The 80 evaluation items have now informed model and design discussion. They remain useful for exploratory paired diagnostics, but repeatedly optimizing against them weakens a new confirmatory claim. Use calibration-only selection/cross-fitting where feasible and reserve genuinely unused items for confirmation. Main and Diamond overlap; changing the variant name does not create an independent holdout. Increasing the number of examples should be guided by the number of generator errors and the precision needed for the target release-risk claim, not merely total question count.

Keep generator incentives and online unlabeled learning as separate later work packages. Implementing them would require different data collection, reward-responsive behavior, fresh audit assumptions, and a separate evaluation protocol.

## 11. Validation and limits

The existing local suite completed with **118 passed, 4 skipped, 1 warning** in approximately 12 seconds using `.venv/bin/python -m pytest -q`. The warning concerns the official SciFact evaluator's truncation of long rationales. Additional synthetic probes reproduced the behaviors listed above. No persistent tests or runtime implementation were added in this review.

Passing software tests does not validate real calibration, causal independence, truthful reporting, or deployed savings. Raw GPQA artifacts were unavailable locally, so I could not independently verify the recorded 200-item outcomes, reconstruct its exact likelihood fits, inspect endpoint reasoning-token semantics, or reconcile its bill. Those limitations are reasons to recover the existing artifacts before paying to regenerate them.
