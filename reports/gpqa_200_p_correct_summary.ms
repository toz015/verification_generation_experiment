.TL
GPQA Model p_correct Summary
.AU
Verification Generation Experiment
.DA
September 30, 2026
.AB
Aggregate summary of raw p_correct values returned by the generator and two verifiers in the 200-question GPQA Main run. No question text is reproduced. Raw values are model-reported probabilities, not calibrated probabilities.
.AE
.NH 1
How to read this summary
.PP
The generator's p_correct is its stated probability that its selected answer is correct. A verifier's p_correct is its stated probability that the same fixed generator answer is correct. Verifiers did not choose replacement answers. Correctness is checked after inference against GPQA's answer key.
.PP
The full sample has 200 items: 120 calibration and 80 held-out evaluation. Counts below include only valid p_correct values. Invalid/missing values are disclosed. The held-out accuracy figures use the items with valid scores for the corresponding model. These raw score distributions are separate from Bayesian-updated posteriors in the full experiment report.
.NH 1
Generator - Gemini 3.8 Flash
.PP
Valid raw p_correct values: 196 of 200. Mean 0.943; median 0.98. Correct selected answers among these 196: 176 (89.8%). Four responses were invalid JSON and had no usable p_correct.
.PP
Full-sample value counts: 0.40: 2; 0.45: 1; 0.55: 1; 0.70: 1; 0.80: 1; 0.85: 24; 0.90: 10; 0.95: 46; 0.98: 45; 0.99: 62; 1.00: 3.
.PP
Held-out evaluation: valid p_correct values 77 of 80; mean 0.940, median 0.98. Correct answers 69 of 77 (89.6%). Value counts: 0.45: 1; 0.55: 1; 0.80: 1; 0.85: 10; 0.90: 3; 0.95: 21; 0.98: 15; 0.99: 25.
.NH 1
Verifier 1 - Llama 3.3 70B Instruct MaaS
.PP
Valid raw verifier scores: 196 of 200. Mean 0.663; median 1.00. Four generator answers were invalid and could not be sent to the verifier. All 196 verifier responses parsed.
.PP
The model returned only endpoint scores: 0.00 on 66 candidates and 1.00 on 130. Among score-0 candidates, 59 of 66 (89.4%) were actually correct. Among score-1 candidates, 117 of 130 (90.0%) were correct. The similar correctness rates show little separation by its raw score in this run.
.PP
Held-out evaluation: 77 valid scores, mean 0.636. Score 0.00: 28 candidates, 24 correct. Score 1.00: 49 candidates, 45 correct. Thus the verifier often reported zero even when the candidate was correct, and its binary output barely distinguished candidate correctness.
.NH 1
Verifier 2 - Gemini 3.7 Flash
.PP
Valid raw verifier scores: 195 of 200. Mean 0.940; median 1.00. Four generator answers were invalid and not sent to the verifier; one verifier response was invalid JSON.
.PP
Full-sample value counts: 0.00: 10; 0.15: 1; 0.25: 1; 0.90: 1; 0.95: 1; 1.00: 181. Among score-0 candidates, 5 of 10 were correct. Among score-1 candidates, 167 of 181 were correct.
.PP
Held-out evaluation: 77 valid scores, mean 0.934. Score 0.00: 5 candidates, 2 correct. Score 0.95: 1 candidate, correct. Score 1.00: 71 candidates, 66 correct. The score distribution is heavily concentrated at 1.00, so this run provides limited resolution among high-confidence cases.
.NH 1
Interpretation
.PP
These are the values the models literally reported before calibration. In particular, a verifier score of 0 or 1 should not be mistaken for verified certainty. The calibration step estimates how each score range related to correctness on the calibration partition; the held-out outcomes then assess that mapping. Sample sizes, especially the small number of incorrect answers, make these reliability estimates uncertain. See gpqa_200_question_experiment_report.pdf for Brier/log-loss results and the confidence-update procedure.
