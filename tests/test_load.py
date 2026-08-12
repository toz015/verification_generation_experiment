"""Loader tests.

These run against the real release, since the whole point of the loader is to
catch upstream drift. If the data is not downloaded, they skip rather than
fail, so the suite still runs offline.
"""

import json

import pytest

from vgx.scifact.load import (
    EXPECTED,
    LABELS,
    Claim,
    Evidence,
    load_claims,
    load_corpus,
    stratified_sample,
    write_sample,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def has_data() -> bool:
    from vgx.scifact.load import DATA_DIR

    return (DATA_DIR / "data" / "corpus.jsonl").exists()


needs_data = pytest.mark.skipif(not has_data(), reason="SciFact release not downloaded")


# --- label logic (no data required) -----------------------------------------


def make_claim(evidence: tuple[Evidence, ...], claim_id: int = 1) -> Claim:
    return Claim(id=claim_id, text="c", evidence=evidence, cited_doc_ids=(7,))


def test_empty_evidence_is_nei():
    claim = make_claim(())
    assert claim.is_nei
    assert claim.label == "NEI"


def test_cited_doc_ids_do_not_make_a_claim_non_nei():
    """NEI claims still carry cited_doc_ids, so it must not be used as a label proxy."""
    claim = make_claim(())
    assert claim.cited_doc_ids == (7,)
    assert claim.label == "NEI"


def test_label_reads_from_evidence():
    claim = make_claim((Evidence(14, (2, 5), "SUPPORT"), Evidence(14, (7,), "SUPPORT")))
    assert claim.label == "SUPPORT"
    assert claim.evidence_doc_ids == (14,)
    assert claim.gold_sentences(14) == {2, 5, 7}


def test_mixed_labels_raise_rather_than_pick_one():
    claim = make_claim((Evidence(1, (0,), "SUPPORT"), Evidence(2, (3,), "CONTRADICT")))
    with pytest.raises(ValueError, match="mixes labels"):
        _ = claim.label


def test_gold_sentences_are_per_document():
    claim = make_claim((Evidence(1, (0, 1), "SUPPORT"), Evidence(2, (9,), "SUPPORT")))
    assert claim.gold_sentences(1) == {0, 1}
    assert claim.gold_sentences(2) == {9}
    assert claim.gold_sentences(3) == set()


# --- sampling ---------------------------------------------------------------


def synthetic_claims() -> list[Claim]:
    """124 SUPPORT / 64 CONTRADICT / 112 NEI, matching the real dev split."""
    claims, next_id = [], 1
    for label, count in (("SUPPORT", 124), ("CONTRADICT", 64), ("NEI", 112)):
        for _ in range(count):
            ev = () if label == "NEI" else (Evidence(1, (0,), label),)
            claims.append(make_claim(ev, claim_id=next_id))
            next_id += 1
    return claims


def test_sample_size_is_exact():
    assert len(stratified_sample(synthetic_claims(), n=50)) == 50


def test_sample_preserves_label_proportions():
    claims = synthetic_claims()
    by_id = {c.id: c for c in claims}
    picked = [by_id[i] for i in stratified_sample(claims, n=50)]

    counts = {label: sum(c.label == label for c in picked) for label in LABELS}
    # 124/64/112 of 300, scaled to 50, is 20.7/10.7/18.7 -> 21/11/18.
    assert counts == {"SUPPORT": 21, "CONTRADICT": 11, "NEI": 18}
    assert sum(counts.values()) == 50


def test_sample_is_deterministic_for_a_seed():
    claims = synthetic_claims()
    assert stratified_sample(claims, n=50, seed=0) == stratified_sample(claims, n=50, seed=0)


def test_different_seeds_give_different_samples():
    claims = synthetic_claims()
    assert stratified_sample(claims, n=50, seed=0) != stratified_sample(claims, n=50, seed=1)


def test_write_sample_refuses_to_overwrite(tmp_path):
    """Re-drawing silently would make previously computed results incomparable."""
    path = tmp_path / "sample.json"
    write_sample([1, 2, 3], path=path)
    with pytest.raises(FileExistsError, match="Delete it explicitly"):
        write_sample([4, 5, 6], path=path)

    assert json.loads(path.read_text())["claim_ids"] == [1, 2, 3]


# --- real data --------------------------------------------------------------


@needs_data
def test_counts_match_the_paper_not_the_flattened_hf_version():
    assert len(load_corpus()) == EXPECTED["corpus"] == 5183
    assert len(load_claims("dev")) == 300
    assert len(load_claims("train")) == 809


@needs_data
def test_dev_distribution_is_stable():
    dev = load_claims("dev")
    counts = {label: sum(c.label == label for c in dev) for label in LABELS}
    assert counts == {"SUPPORT": 124, "CONTRADICT": 64, "NEI": 112}


@needs_data
def test_evidence_doc_ids_resolve_into_the_corpus():
    """evidence keys are strings and cited_doc_ids are ints; both must join."""
    corpus = load_corpus()
    for claim in load_claims("dev"):
        for doc_id in claim.evidence_doc_ids:
            assert doc_id in corpus, f"claim {claim.id} cites missing doc {doc_id}"


@needs_data
def test_gold_sentence_indices_are_in_range():
    corpus = load_corpus()
    for claim in load_claims("dev"):
        for ev in claim.evidence:
            n = len(corpus[ev.doc_id].sentences)
            assert all(0 <= s < n for s in ev.sentences), (
                f"claim {claim.id} cites sentence out of range in doc {ev.doc_id}"
            )


@needs_data
def test_test_split_has_no_labels():
    """Test labels are withheld, which is why all reporting is on dev."""
    for claim in load_claims("test"):
        assert claim.evidence == ()
