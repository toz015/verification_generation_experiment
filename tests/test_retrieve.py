"""Retrieval tests.

The recall numbers are an upper bound on every downstream citation metric, so
an error here silently mis-sets the ceiling for the whole exploration.
"""

import pytest

from vgx.scifact.load import Abstract, Claim, Evidence
from vgx.scifact.retrieve import (
    RecallRow,
    Retriever,
    oracle_docs,
    recall_at_k,
    tokenize,
)


def has_data() -> bool:
    from vgx.scifact.load import DATA_DIR

    return (DATA_DIR / "data" / "corpus.jsonl").exists()


needs_data = pytest.mark.skipif(not has_data(), reason="SciFact release not downloaded")


# --- tokenizer --------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Vitamin-D reduces influenza risk.") == [
        "vitamin",
        "d",
        "reduces",
        "influenza",
        "risk",
    ]


def test_tokenize_keeps_digits():
    """Claims reference figures like '1,000 genomes' and 'CD8+ T cells'."""
    assert tokenize("1,000 genomes project") == ["1", "000", "genomes", "project"]


# --- oracle selection -------------------------------------------------------


def make_claim(evidence=(), cited=(7,), claim_id=1) -> Claim:
    return Claim(id=claim_id, text="c", evidence=tuple(evidence), cited_doc_ids=tuple(cited))


def test_oracle_uses_evidence_docs_when_present():
    claim = make_claim(evidence=(Evidence(14, (2,), "SUPPORT"),), cited=(14, 99))
    assert oracle_docs(claim) == (14,)


def test_oracle_falls_back_to_cited_docs_for_nei():
    """NEI has no evidence, so the model must judge the inspected abstract.

    Handing it no abstract at all would make NEI trivially guessable.
    """
    claim = make_claim(evidence=(), cited=(31715818,))
    assert claim.is_nei
    assert oracle_docs(claim) == (31715818,)


# --- recall -----------------------------------------------------------------


def tiny_corpus() -> dict[int, Abstract]:
    return {
        1: Abstract(1, "vitamin d influenza", ("vitamin d reduces influenza risk",), False),
        2: Abstract(2, "aspirin stroke", ("aspirin lowers stroke incidence",), False),
        3: Abstract(3, "unrelated geology", ("basalt forms from cooled lava",), False),
    }


def test_search_ranks_the_matching_abstract_first():
    r = Retriever(tiny_corpus())
    assert r.search("vitamin d influenza", k=1) == [1]
    assert r.search("aspirin stroke", k=1) == [2]


def test_search_respects_k():
    r = Retriever(tiny_corpus())
    assert len(r.search("vitamin", k=2)) == 2
    assert len(r.search("vitamin", k=3)) == 3


def test_recall_excludes_nei_claims_for_evidence_gold():
    """Including NEI would deflate recall, since they have nothing to retrieve."""
    claims = [
        make_claim(evidence=(Evidence(1, (0,), "SUPPORT"),), claim_id=1),
        make_claim(evidence=(), claim_id=2),  # NEI
    ]
    rankings = {1: [1], 2: [3]}
    rows = recall_at_k(claims, rankings, ks=(1,), gold="evidence")
    # Only claim 1 is scored, and its gold doc is found.
    assert rows[0] == RecallRow(k=1, hit_rate=1.0, micro_recall=1.0)


def test_recall_cited_mode_includes_nei_claims():
    claims = [
        make_claim(evidence=(Evidence(1, (0,), "SUPPORT"),), cited=(1,), claim_id=1),
        make_claim(evidence=(), cited=(3,), claim_id=2),
    ]
    rankings = {1: [1], 2: [3]}
    rows = recall_at_k(claims, rankings, ks=(1,), gold="cited")
    assert rows[0].hit_rate == 1.0


def test_hit_rate_and_micro_recall_differ_on_multi_doc_claims():
    """A claim with two gold abstracts, only one retrieved: hit but partial recall."""
    claims = [
        make_claim(
            evidence=(Evidence(1, (0,), "SUPPORT"), Evidence(2, (0,), "SUPPORT")),
            claim_id=1,
        )
    ]
    rankings = {1: [1, 9]}
    rows = recall_at_k(claims, rankings, ks=(2,), gold="evidence")
    assert rows[0].hit_rate == 1.0
    assert rows[0].micro_recall == 0.5


def test_recall_is_monotone_in_k():
    claims = [make_claim(evidence=(Evidence(5, (0,), "SUPPORT"),), claim_id=1)]
    rankings = {1: [9, 8, 5]}
    rows = recall_at_k(claims, rankings, ks=(1, 2, 3), gold="evidence")
    assert [r.hit_rate for r in rows] == [0.0, 0.0, 1.0]
    assert all(b.hit_rate >= a.hit_rate for a, b in zip(rows, rows[1:]))


def test_unknown_gold_mode_raises():
    with pytest.raises(ValueError, match="unknown gold"):
        recall_at_k([], {}, ks=(1,), gold="nonsense")


# --- real data --------------------------------------------------------------


@needs_data
def test_oracle_recall_is_one_by_construction():
    """Sanity: oracle mode cannot miss, so it is the ceiling BM25 is measured against."""
    from vgx.scifact.load import load_claims

    dev = [c for c in load_claims("dev") if not c.is_nei]
    rankings = {c.id: list(oracle_docs(c)) for c in dev}
    rows = recall_at_k(dev, rankings, ks=(5,), gold="evidence")
    assert rows[0].hit_rate == 1.0
    assert rows[0].micro_recall == 1.0


@needs_data
def test_bm25_recall_is_monotone_on_real_dev():
    from vgx.scifact.load import load_claims, load_corpus

    dev = load_claims("dev")
    r = Retriever(load_corpus())
    rankings = r.rankings(dev, 20)
    rows = recall_at_k(dev, rankings, ks=(1, 3, 5, 10, 20), gold="evidence")
    assert all(b.micro_recall >= a.micro_recall for a, b in zip(rows, rows[1:]))
    assert rows[-1].hit_rate > rows[0].hit_rate, "recall must improve with k"
