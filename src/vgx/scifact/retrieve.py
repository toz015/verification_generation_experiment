"""Abstract retrieval over the SciFact corpus.

Two modes feed the experiment:

* **oracle** — the gold abstract is handed to the model directly.
* **bm25** — the model sees the top-k BM25 hits instead.

The gap between them isolates retrieval failure. Without it, a low
label+rationale score is ambiguous: the model may have cited badly, or the
evidence may never have been in front of it.

Recall@k is computed before any generation because it is an upper bound. No
citation metric can beat what retrieval surfaces, so this fixes k and tells us
what the ceiling is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from vgx.scifact.load import Abstract, Claim, load_claims, load_corpus, sample_claims

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KS = (1, 2, 3, 5, 10, 20, 50, 100)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    No stemming and no stopword list. The point of this baseline is to
    establish the retrieval ceiling with an unsurprising, standard setup; a
    tuned retriever would confound "how hard is this corpus" with "how good is
    my index".
    """
    return _TOKEN.findall(text.lower())


def oracle_docs(claim: Claim) -> tuple[int, ...]:
    """The abstracts a model should see in oracle mode.

    For SUPPORT and CONTRADICT this is the gold evidence abstract. NEI claims
    have no evidence by definition, so they fall back to `cited_doc_ids`: the
    abstracts annotators actually inspected and judged insufficient. That is
    what makes NEI a real decision rather than a trick question — the model
    must decline while looking at a plausible but inadequate abstract.
    """
    return claim.evidence_doc_ids or claim.cited_doc_ids


@dataclass
class Retriever:
    """BM25 over title plus abstract body."""

    corpus: dict[int, Abstract]

    def __post_init__(self) -> None:
        self._doc_ids = sorted(self.corpus)
        self._bm25 = BM25Okapi(
            [tokenize(f"{self.corpus[d].title} {self.corpus[d].text()}") for d in self._doc_ids]
        )

    def search(self, query: str, k: int) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._doc_ids[i] for i in ranked]

    def rankings(self, claims: list[Claim], k: int) -> dict[int, list[int]]:
        return {c.id: self.search(c.text, k) for c in claims}


# --- recall -----------------------------------------------------------------


@dataclass(frozen=True)
class RecallRow:
    k: int
    hit_rate: float  # >=1 gold abstract in the top k
    micro_recall: float  # gold abstracts found / gold abstracts total


def recall_at_k(
    claims: list[Claim],
    rankings: dict[int, list[int]],
    ks: tuple[int, ...] = DEFAULT_KS,
    gold: str = "evidence",
) -> list[RecallRow]:
    """Recall over claims that have something to retrieve.

    `gold="evidence"` scores only SUPPORT/CONTRADICT claims: NEI claims have no
    evidence abstract, so "did retrieval find it" is undefined for them and
    including them would silently deflate every number.

    `gold="cited"` scores all 300 claims against `cited_doc_ids`, which answers
    a different question — can BM25 surface the abstract an annotator thought
    was relevant, even when it turned out not to support the claim.
    """
    if gold == "evidence":
        scored = [c for c in claims if not c.is_nei]
        want = {c.id: set(c.evidence_doc_ids) for c in scored}
    elif gold == "cited":
        scored = [c for c in claims if c.cited_doc_ids]
        want = {c.id: set(c.cited_doc_ids) for c in scored}
    else:
        raise ValueError(f"unknown gold={gold!r}")

    rows = []
    total_gold = sum(len(want[c.id]) for c in scored)
    for k in ks:
        hits = 0
        found = 0
        for claim in scored:
            top = set(rankings[claim.id][:k])
            overlap = want[claim.id] & top
            found += len(overlap)
            hits += bool(overlap)
        rows.append(
            RecallRow(
                k=k,
                hit_rate=hits / len(scored),
                micro_recall=found / total_gold,
            )
        )
    return rows


def _table(rows: list[RecallRow], n_claims: int, n_gold: int) -> list[str]:
    out = [
        f"n claims = {n_claims}, gold abstracts = {n_gold}",
        "",
        "| k | hit rate | micro recall |",
        "|---|---|---|",
    ]
    out += [f"| {r.k} | {r.hit_rate:.1%} | {r.micro_recall:.1%} |" for r in rows]
    return out


def report() -> str:
    corpus = load_corpus()
    dev = load_claims("dev")
    retriever = Retriever(corpus)

    max_k = max(DEFAULT_KS)
    rankings = retriever.rankings(dev, max_k)

    with_ev = [c for c in dev if not c.is_nei]
    dev_rows = recall_at_k(dev, rankings, gold="evidence")
    cited_rows = recall_at_k(dev, rankings, gold="cited")

    picked = sample_claims(dev)
    sample_rows = recall_at_k(picked, rankings, gold="evidence")
    sample_with_ev = [c for c in picked if not c.is_nei]

    lines = [
        "# SciFact — BM25 retrieval",
        "",
        f"BM25Okapi over title + abstract of all {len(corpus)} abstracts.",
        "Lowercase alphanumeric tokens, no stemming, no stopword list.",
        "",
        "## Recall of gold evidence abstracts (full dev)",
        "",
        "NEI claims are excluded: they have no evidence abstract, so retrieval",
        "recall is undefined for them and including them would deflate every row.",
        "",
    ]
    lines += _table(dev_rows, len(with_ev), sum(len(c.evidence_doc_ids) for c in with_ev))

    lines += [
        "",
        "## Recall of cited abstracts (all dev claims)",
        "",
        "A different question: can BM25 surface the abstract an annotator judged",
        "relevant, including the NEI cases where it turned out not to support the claim.",
        "",
    ]
    lines += _table(
        cited_rows,
        len([c for c in dev if c.cited_doc_ids]),
        sum(len(set(c.cited_doc_ids)) for c in dev if c.cited_doc_ids),
    )

    lines += [
        "",
        f"## Pinned sample (n={len(picked)})",
        "",
        "Reported to confirm the sample is not retrieval-anomalous relative to full dev.",
        "",
    ]
    lines += _table(
        sample_rows,
        len(sample_with_ev),
        sum(len(c.evidence_doc_ids) for c in sample_with_ev),
    )

    # Context budgeting for S3: one abstract in this corpus is enormous, and
    # silently truncating it would destroy rationale scoring for that claim.
    sizes = sorted(((len(a.sentences), a.doc_id) for a in corpus.values()), reverse=True)
    lines += [
        "",
        "## Context budget",
        "",
        f"Longest abstracts: " + ", ".join(f"doc {d} ({n} sentences)" for n, d in sizes[:5]) + ".",
        f"Median is {sizes[len(sizes) // 2][0]} sentences.",
        "",
        "Relevant to S3: a top-k prompt concatenates k abstracts, so k and the",
        "context limit have to be chosen against the tail, not the median.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    text = report()
    out = REPO_ROOT / "docs" / "scifact-retrieval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
