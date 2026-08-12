"""Parser tests.

The parser sits between the model and every metric, so its failure modes are
the ones most likely to produce a wrong number that looks reasonable. Two
principles under test: forgiving about form, strict about content, and never
silently dropping anything.
"""

import pytest

from vgx.scifact.load import Abstract
from vgx.scifact.prompt import build_prompt, parse_response, render_abstract


def abstracts() -> list[Abstract]:
    return [
        Abstract(143, "Vitamin D and influenza", ("s0", "s1", "s2", "s3"), False),
        Abstract(999, "Unrelated", ("t0", "t1"), False),
    ]


# --- prompt construction ----------------------------------------------------


def test_sentences_are_indexed_from_zero():
    rendered = render_abstract(abstracts()[0])
    assert "[0] s0" in rendered
    assert "[3] s3" in rendered
    assert "[4]" not in rendered


def test_prompt_contains_claim_and_every_abstract():
    prompt = build_prompt("Vitamin D reduces influenza risk.", abstracts())
    assert "Vitamin D reduces influenza risk." in prompt
    assert "Document 143" in prompt
    assert "Document 999" in prompt


# --- form: forgiving --------------------------------------------------------


def test_plain_json():
    answer = parse_response('{"label": "SUPPORT", "evidence": {"143": [1, 2]}}', abstracts())
    assert answer.ok
    assert answer.label == "SUPPORT"
    assert answer.evidence == {143: (1, 2)}


def test_markdown_fence_is_unwrapped():
    raw = '```json\n{"label": "CONTRADICT", "evidence": {"143": [0]}}\n```'
    answer = parse_response(raw, abstracts())
    assert answer.ok
    assert answer.label == "CONTRADICT"


def test_surrounding_prose_is_tolerated():
    """Rejecting this would measure instruction-following, not verification."""
    raw = 'Sure! Here is my answer:\n{"label": "SUPPORT", "evidence": {"143": [2]}}\nHope that helps.'
    answer = parse_response(raw, abstracts())
    assert answer.ok
    assert answer.evidence == {143: (2,)}


def test_qwen_think_block_is_stripped():
    """Qwen3 emits <think> by default; left in place it swallows the JSON."""
    raw = (
        "<think>The abstract mentions vitamin D, so this looks supported.</think>\n"
        '{"label": "SUPPORT", "evidence": {"143": [1]}}'
    )
    answer = parse_response(raw, abstracts())
    assert answer.ok
    assert answer.label == "SUPPORT"
    assert answer.evidence == {143: (1,)}


def test_label_aliases_normalise():
    for raw_label, expected in [
        ("supports", "SUPPORT"),
        ("Refutes", "CONTRADICT"),
        ("NOT ENOUGH INFO", "NEI"),
        ("not_enough_info", "NEI"),
        ("nei", "NEI"),
    ]:
        answer = parse_response(f'{{"label": "{raw_label}", "evidence": {{}}}}', abstracts())
        assert answer.ok, raw_label
        assert answer.label == expected, raw_label


def test_string_indices_are_cast():
    answer = parse_response('{"label": "SUPPORT", "evidence": {"143": ["1", "2"]}}', abstracts())
    assert answer.evidence == {143: (1, 2)}


def test_scalar_instead_of_list_is_accepted():
    answer = parse_response('{"label": "SUPPORT", "evidence": {"143": 2}}', abstracts())
    assert answer.evidence == {143: (2,)}


def test_duplicate_indices_are_deduped_and_sorted():
    answer = parse_response('{"label": "SUPPORT", "evidence": {"143": [2, 1, 2]}}', abstracts())
    assert answer.evidence == {143: (1, 2)}


# --- content: strict --------------------------------------------------------


def test_citation_to_an_unshown_document_is_recorded_not_scored():
    """The model cannot legitimately cite an abstract it never saw."""
    answer = parse_response('{"label": "SUPPORT", "evidence": {"555": [0]}}', abstracts())
    assert answer.ok
    assert 555 not in answer.evidence
    assert any(n.startswith("unshown_doc:555") for n in answer.notes)


def test_sentence_index_beyond_the_abstract_is_recorded_not_scored():
    answer = parse_response('{"label": "SUPPORT", "evidence": {"143": [1, 99]}}', abstracts())
    assert answer.evidence == {143: (1,)}
    assert any("sentence_out_of_range:143:99" in n for n in answer.notes)


def test_negative_sentence_index_is_rejected():
    answer = parse_response('{"label": "SUPPORT", "evidence": {"143": [-1]}}', abstracts())
    assert answer.evidence == {}
    assert any("sentence_out_of_range" in n for n in answer.notes)


def test_label_without_rationale_is_kept_and_flagged():
    """This is the pathology under study, so it must not be discarded."""
    answer = parse_response('{"label": "SUPPORT", "evidence": {}}', abstracts())
    assert answer.ok
    assert answer.label == "SUPPORT"
    assert answer.evidence == {}
    assert "label_without_rationale" in answer.notes


def test_nei_with_rationale_is_normalised_and_flagged():
    answer = parse_response('{"label": "NEI", "evidence": {"143": [0]}}', abstracts())
    assert answer.label == "NEI"
    assert answer.evidence == {}
    assert "nei_with_rationale" in answer.notes


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", "empty_response"),
        ("   ", "empty_response"),
        ("I cannot answer that.", "no_json"),
        ('{"label": "MAYBE", "evidence": {}}', "bad_label"),
        ('{"evidence": {"143": [0]}}', "bad_label"),
        ("[1, 2, 3]", "no_json"),
    ],
)
def test_failures_carry_a_reason(raw, expected):
    answer = parse_response(raw, abstracts())
    assert not answer.ok
    assert answer.failure == expected
    assert answer.label is None


def test_cited_sentence_count_spans_documents():
    """Used by the over-citation diagnostic."""
    raw = '{"label": "SUPPORT", "evidence": {"143": [0, 1], "999": [1]}}'
    answer = parse_response(raw, abstracts())
    assert answer.cited_sentence_count == 3
