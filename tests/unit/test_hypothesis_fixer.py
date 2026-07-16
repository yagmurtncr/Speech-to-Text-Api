from hypothesis_fixer import extract_text_from_hypothesis as ext


def test_extracts_text_field():
    assert ext("Hypothesis(text='Merhaba', score=0.98)") == "Merhaba"


def test_returns_input_when_no_match():
    assert ext("plain text without the pattern") == "plain text without the pattern"


def test_preserves_inner_whitespace():
    assert ext("Hypothesis(text='  Ali gel  ')") == "  Ali gel  "
