# Vendored SciFact evaluator

Source:   https://github.com/allenai/scifact
Revision: 68b98a56d93e0f9da0d2aab4e6c3294699a0f72e
Path:     verisci/evaluate/lib/{data.py,metrics.py}
Vendored: 2026-08-12

Copied **unmodified**. Only `__init__.py` was added so the two files import as
a package (`metrics.py` already uses `from .data import Label`).

Vendored rather than reimplemented so that label-only and label+rationale F1
are computed by the benchmark authors' own code. The headroom claim in this
project is "published baselines score X, current 8B models score Y"; if the
scorer were mine, a reader could reasonably ask whether the gap is real or an
artefact of my implementation.

## Two behaviours worth knowing

`MAX_ABSTRACT_SENTS = 3` — abstract-level evaluation uses only the first three
predicted rationale sentences, and credit requires a gold rationale set to be a
**subset** of what was cited. Citing three sentences rather than one therefore
weakly increases the chance of containing a gold rationale at no abstract-level
cost. This is the concrete form of "citing is free".

Sentence-level evaluation behaves differently: `retrieved` accumulates every
predicted sentence, so extra citations do enter the precision denominator and
wrong ones are penalised. The metric family is not uniformly monotone in
citation count, and the distinction should be stated precisely rather than
claimed for both.

## Prediction format

    {"id": 3, "evidence": {"14717500": {"label": "SUPPORT", "sentences": [2, 5]}}}

An NEI prediction is expressed by emitting no abstract entry at all; the loader
rejects an explicit NEI label via `make_label(..., allow_NEI=False)`.
