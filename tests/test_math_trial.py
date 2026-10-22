from vgx.math_trial import data, prompt


def test_parser_accepts_exact_fields_and_rejects_others():
    ok = prompt.parse_generator_response('{"reasoning": "r", "answer": "\\\\frac12", "p_correct": 0.7}')
    assert ok.answer == '\\frac12' and ok.p_correct == 0.7 and ok.failure is None
    assert prompt.parse_generator_response('{"answer": "3", "p_correct": 1.2}').confidence_status == 'out_of_range'
    assert prompt.parse_generator_response('{"answer": "3", "extra": 1}').failure == 'unexpected_fields'
    assert prompt.parse_generator_response('Answer: 3').failure == 'invalid_json'
    assert prompt.parse_generator_response('{"answer": " "}').failure == 'missing_or_empty_answer'
    assert prompt.parse_generator_response('{"answer": "3"}', truncated=True).failure == 'truncated_output'


def test_stratified_sample_is_proportional_and_deterministic():
    items = [{'item_id': f'x{i}', 'stratum': 'a' if i < 75 else 'b'} for i in range(100)]
    s1, alloc = data.stratified_sample(items, 20, 1)
    s2, _ = data.stratified_sample(items, 20, 1)
    assert alloc == {'a': 15, 'b': 5} and s1 == s2 and len({x['item_id'] for x in s1}) == 20


def test_pinned_sources_load():
    for name, n in (('math500', 500), ('omnimath_rule', 2821)):
        items, meta = data.load(name)
        assert meta['n'] == n and all(it['answer'] is not None for it in items)
