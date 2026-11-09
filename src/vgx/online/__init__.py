"""Online learning replay for the generation-verification mechanism (QuantumKatas cached outputs).

Modules: views (learner/evaluator data isolation), calibrator (r_phi), likelihood (3-bin verifier
models, constrained M-step, full-history EM, recursive online EM), routing (DesignRouting under
selective-risk and budget constraints, NestedVerify), runner (one replay run per arm), evaluate
(metrics from logs and labels), simulate (synthetic checks), experiment (grid driver).
"""
