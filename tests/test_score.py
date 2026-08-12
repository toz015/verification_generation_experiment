"""The S4 validation gates.

The published SciFact baselines are trained systems, so reproducing them is out
of scope for this exploration. Instead the scorer itself is validated directly,
which is cheaper and a stronger check on the wiring: feed it inputs whose
correct score is known a priori and confirm it agrees.

No model number is reported until every gate here passes.

Gate 4 is not only a test. Citing three sentences everywhere carries no
information, and whatever it scores is the free floor any real system must be
judged against - the concrete form of "citing is free" from the proposal.
"""

import pytest

from vgx.scifact.load import DATA_DIR, load_claims
from vgx.scifact.prompt import Answer
from vgx.scifact.score import (
    MAX_ABSTRACT_SENTS,
    diagnostics,
    official_metrics,
    to_prediction,
    wilson,
)


def has_data() -> bool:
    return (DATA_DIR / "data" / "corpus.jsonl").exists()


needs_data = pytest.mark.skipif(not has_data(), reason="SciFact release not downloaded")


# --- format conversion ------------------------------------------------------


def test_nei_is_an_empty_evidence_object():
    """The official loader rejects an explicit NEI label, so NEI must omit it."""
    record = to_prediction(7, Answer(label="NEI"))
    assert record == {"id": 7, "evidence": {}}


def test_parse_failure_scores_as_no_evidence():
    """A failure must still produce a record, or the claim vanishes from the denominator."""
    record = to_prediction(7, Answer(label=None, ok=False, failure="no_json"))
    assert record == {"id": 7, "evidence": {}}


def test_supported_claim_maps_to_official_shape():
    answer = Answer(label="SUPPORT", evidence={143: (1, 2)})
    assert to_prediction(3, answer) == {
        "id": 3,
        "evidence": {"143": {"label": "SUPPORT", "sentences": [1, 2]}},
    }


# --- Gate 1: identity -------------------------------------------------------


def gold_predictions(claims) -> list[dict]:
    """Replay the gold annotations back as predictions."""
    records = []
    for claim in claims:
        evidence: dict[str, dict] = {}
        for ev in claim.evidence:
            key = str(ev.doc_id)
            if key not in evidence:
                evidence[key] = {"label": ev.label, "sentences": []}
            evidence[key]["sentences"].extend(ev.sentences)
        records.append({"id": claim.id, "evidence": evidence})
    return records


@needs_data
def test_gate1_gold_predictions_score_perfectly():
    """Feeding gold back in must give F1 = 1.0 everywhere.

    If this fails, the harness is mis-wired and every other number is
    meaningless. It also proves any future port of the vendored code is
    faithful.
    """
    claims = load_claims("dev")
    result = official_metrics(gold_predictions(claims))

    for level in ("abstract_label_only", "abstract_rationalized",
                  "sentence_selection", "sentence_label"):
        assert result[level]["f1"] == pytest.approx(1.0), f"{level} != 1.0: {result[level]}"


# --- Gate 2: always-SUPPORT -------------------------------------------------


@needs_data
def test_gate2_always_support_gets_label_credit_without_rationale_credit():
    """Label-only credit survives; rationale credit must collapse.

    Cites sentence 0 of the cited abstract and always says SUPPORT. That earns
    label credit whenever the claim really is SUPPORT, but sentence 0 is rarely
    a gold rationale, so rationalized F1 must be far lower.
    """
    claims = load_claims("dev")
    records = [
        {
            "id": c.id,
            "evidence": {str(c.cited_doc_ids[0]): {"label": "SUPPORT", "sentences": [0]}},
        }
        for c in claims
        if c.cited_doc_ids
    ]
    result = official_metrics(records)

    label_only = result["abstract_label_only"]["f1"]
    rationalized = result["abstract_rationalized"]["f1"]

    assert label_only > 0, "always-SUPPORT should earn some label credit"
    assert rationalized < label_only, (
        "rationale credit must be strictly harder than label credit; "
        f"got label_only={label_only:.3f} rationalized={rationalized:.3f}"
    )


# --- Gate 3: random sentences -----------------------------------------------


@needs_data
def test_gate3_correct_labels_with_wrong_sentences_keep_label_credit_only():
    """Gold labels and gold documents, but deliberately wrong sentences.

    Isolates the rationale component: label credit should be near perfect while
    rationalized credit collapses. This is the pattern the proposal predicts for
    zero-shot LLMs - "label >> rationale".
    """
    claims = [c for c in load_claims("dev") if not c.is_nei]
    records = []
    for claim in claims:
        evidence = {}
        for ev in claim.evidence:
            gold = set(claim.gold_sentences(ev.doc_id))
            # Pick indices that are definitely not gold.
            wrong = [i for i in range(200) if i not in gold][:2]
            evidence[str(ev.doc_id)] = {"label": ev.label, "sentences": wrong}
        records.append({"id": claim.id, "evidence": evidence})

    result = official_metrics(records)
    assert result["abstract_label_only"]["f1"] > 0.9, "gold labels should score highly"
    assert result["abstract_rationalized"]["f1"] < 0.2, (
        "wrong sentences must destroy rationale credit"
    )


# --- Gate 4: cite-everything ------------------------------------------------


@needs_data
def test_gate4_citing_the_cap_scores_nontrivially_without_information():
    """The free floor: gold labels, gold documents, first MAX_ABSTRACT_SENTS sentences.

    This strategy encodes no evidence-selection ability at all, yet the abstract
    metric caps rationales at 3 and grants credit whenever a gold rationale is a
    subset of what was cited. Whatever this scores is the floor any real system
    must beat, and it quantifies "citing is free".
    """
    claims = [c for c in load_claims("dev") if not c.is_nei]
    records = []
    for claim in claims:
        evidence = {}
        for ev in claim.evidence:
            evidence[str(ev.doc_id)] = {
                "label": ev.label,
                "sentences": list(range(MAX_ABSTRACT_SENTS)),
            }
        records.append({"id": claim.id, "evidence": evidence})

    result = official_metrics(records)
    floor = result["abstract_rationalized"]["f1"]

    assert floor > 0, (
        "citing the first 3 sentences with no evidence selection must still "
        "score above zero - that is the whole point of the gate"
    )
    print(f"\ncite-first-{MAX_ABSTRACT_SENTS} floor: "
          f"abstract_rationalized F1 = {floor:.3f}, "
          f"abstract_label_only F1 = {result['abstract_label_only']['f1']:.3f}")


# --- diagnostics ------------------------------------------------------------


def test_nei_recall_counts_only_gold_nei_claims():
    from vgx.scifact.load import Claim, Evidence

    def claim(cid, ev=()):
        return Claim(id=cid, text="c", evidence=tuple(ev), cited_doc_ids=(1,))

    claims = [
        claim(1),  # NEI
        claim(2),  # NEI
        claim(3, [Evidence(1, (0,), "SUPPORT")]),
    ]
    answers = {
        1: Answer(label="NEI"),
        2: Answer(label="SUPPORT", evidence={1: (0,)}),
        3: Answer(label="NEI"),
    }
    d = diagnostics(claims, answers, {1: [1], 2: [1], 3: [1]})

    assert d.nei_recall == pytest.approx(0.5)  # 1 of 2 gold NEI recovered
    assert d.nei_precision == pytest.approx(0.5)  # 1 of 2 NEI predictions right


def test_over_citation_counts_against_gold():
    from vgx.scifact.load import Claim, Evidence

    claims = [Claim(id=1, text="c", evidence=(Evidence(1, (0,), "SUPPORT"),), cited_doc_ids=(1,))]
    answers = {1: Answer(label="SUPPORT", evidence={1: (0, 1, 2, 3, 4)})}
    d = diagnostics(claims, answers, {1: [1]})

    assert d.cited_total == 5
    assert d.gold_total == 1
    assert d.over_cited_claims == 1
    assert d.citations_beyond_cap == 2  # 5 cited, cap is 3


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson(45, 50)
    assert lo < 0.9 < hi
    assert hi - lo > 0.1, "at n=50 the interval should be wide enough to be honest"


def test_wilson_handles_zero_denominator():
    assert wilson(0, 0) == (0.0, 0.0)
