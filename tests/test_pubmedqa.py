"""PubMedQA loader, parser and scorer tests, including the validation gates.

Same discipline as SciFact: the scorer is validated against inputs whose
correct score is known a priori, before any model number is reported.
"""

import pytest

from vgx.pubmedqa.load import DATA_DIR, EXPECTED_BALANCE, LABELS, Item, load_items, stratified_sample
from vgx.pubmedqa.prompt import Answer as PromptAnswer
from vgx.pubmedqa.prompt import build_prompt, parse_response
from vgx.pubmedqa.score import UNPARSED, majority_baseline, score


def has_data() -> bool:
    return (DATA_DIR / "pqa_labeled.parquet").exists()


needs_data = pytest.mark.skipif(not has_data(), reason="PubMedQA not downloaded")


def item(pubid: str, decision: str) -> Item:
    return Item(
        pubid=pubid,
        question="Does X work?",
        contexts=("Background text.", "Results were mixed."),
        sections=("BACKGROUND", "RESULTS"),
        decision=decision,
    )


# --- prompt -----------------------------------------------------------------


def test_context_text_labels_sections():
    assert item("1", "yes").context_text() == (
        "BACKGROUND: Background text.\n\nRESULTS: Results were mixed."
    )


def test_prompt_contains_question_and_context_but_defines_maybe():
    prompt = build_prompt("Does X work?", item("1", "yes").context_text())
    assert "Does X work?" in prompt
    assert "RESULTS: Results were mixed." in prompt
    # A model that never emits `maybe` because it did not know it was allowed
    # would be a prompting artefact, not a finding.
    assert "maybe" in prompt


# --- parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"answer": "yes"}', "yes"),
        ('{"answer": "no"}', "no"),
        ('{"answer": "maybe"}', "maybe"),
        ('```json\n{"answer": "maybe"}\n```', "maybe"),
        ('<think>hmm</think>{"answer": "no"}', "no"),
        ('{"answer": "MAYBE"}', "maybe"),
        ('{"answer": "inconclusive"}', "maybe"),
        ('{"answer": "Uncertain"}', "maybe"),
    ],
)
def test_parses_expected_forms(raw, expected):
    answer = parse_response(raw)
    assert answer.ok and answer.label == expected


def test_bare_word_answer_is_recovered():
    """Rejecting this would measure JSON compliance, not the model's verdict."""
    answer = parse_response("maybe")
    assert answer.ok and answer.label == "maybe"
    assert "recovered_without_json" in answer.notes


def test_fallback_picks_the_earliest_label_not_a_fixed_order():
    """Regression: fixed-order scanning inverted 'not yes, it is maybe'."""
    answer = parse_response("The evidence is inconclusive, so maybe rather than yes.")
    assert answer.label == "maybe"


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_response_fails(raw):
    answer = parse_response(raw)
    assert not answer.ok and answer.failure == "empty_response"


def test_unrecognisable_answer_fails_loudly():
    answer = parse_response('{"answer": "purple"}')
    assert not answer.ok and answer.failure == "bad_label"


# --- scoring gates ----------------------------------------------------------


def synthetic(n_yes: int, n_no: int, n_maybe: int) -> list[Item]:
    items, i = [], 0
    for label, count in (("yes", n_yes), ("no", n_no), ("maybe", n_maybe)):
        for _ in range(count):
            items.append(item(str(i), label))
            i += 1
    return items


def test_gate1_perfect_predictions_score_one():
    """Identity: gold in must give accuracy and macro-F1 of exactly 1.0."""
    items = synthetic(28, 17, 5)
    result = score(items, {i.pubid: PromptAnswer(label=i.decision) for i in items})
    assert result.accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)


def test_gate2_majority_baseline_accuracy_tracks_yes_prevalence():
    """Always-yes: accuracy equals the yes share, macro-F1 collapses.

    This is the free floor. The gap between the two numbers is why macro-F1 is
    the metric that matters: accuracy alone hides a total failure on `maybe`.
    """
    items = synthetic(28, 17, 5)
    result = majority_baseline(items)

    assert result.accuracy == pytest.approx(28 / 50)
    assert result.per_class["maybe"]["f1"] == 0.0
    assert result.per_class["no"]["f1"] == 0.0
    assert result.macro_f1 < result.accuracy / 2, (
        "macro-F1 must expose the two dead classes that accuracy conceals"
    )


def test_gate3_never_predicting_maybe_zeroes_that_class():
    """A model that never abstains scores 0 recall on maybe, whatever its accuracy."""
    items = synthetic(28, 17, 5)
    answers = {
        i.pubid: PromptAnswer(label=("yes" if i.decision != "no" else "no")) for i in items
    }
    result = score(items, answers)
    assert result.per_class["maybe"]["recall"] == 0.0
    assert result.per_class["yes"]["recall"] == 1.0


def test_parse_failures_count_as_wrong_not_as_dropped():
    """Dropping failures would flatter whichever model formats worst."""
    items = synthetic(2, 0, 0)
    answers = {
        items[0].pubid: PromptAnswer(label="yes"),
        items[1].pubid: PromptAnswer(label=None, ok=False, failure="no_json"),
    }
    result = score(items, answers)

    assert result.n == 2
    assert result.accuracy == pytest.approx(0.5)
    assert result.parse_failures["no_json"] == 1
    assert result.confusion[("yes", UNPARSED)] == 1


def test_unparsed_does_not_become_a_fourth_class():
    items = synthetic(1, 0, 0)
    result = score(items, {items[0].pubid: PromptAnswer(label=None, ok=False, failure="no_json")})
    assert set(result.per_class) == set(LABELS)


# --- real data --------------------------------------------------------------


@needs_data
def test_official_test_split_matches_the_paper():
    test = load_items("test")
    assert len(test) == 500
    from collections import Counter

    dist = Counter(i.decision for i in test)
    assert dist == {"yes": 276, "no": 169, "maybe": 55}


@needs_data
def test_full_labeled_balance_is_stable():
    from collections import Counter

    dist = Counter(i.decision for i in load_items("all"))
    assert dist == EXPECTED_BALANCE


@needs_data
def test_sample_is_deterministic_and_stratified():
    test = load_items("test")
    first = stratified_sample(test, n=50, seed=0)
    assert first == stratified_sample(test, n=50, seed=0)
    assert len(first) == 50

    by_id = {i.pubid: i for i in test}
    from collections import Counter

    dist = Counter(by_id[p].decision for p in first)
    assert sum(dist.values()) == 50
    assert dist["maybe"] >= 5, "maybe must not be squeezed out of the sample entirely"


@needs_data
def test_long_answer_is_never_loaded():
    """It is the withheld CONCLUSIONS section and would leak the answer."""
    for it in load_items("test")[:20]:
        assert not hasattr(it, "long_answer")
