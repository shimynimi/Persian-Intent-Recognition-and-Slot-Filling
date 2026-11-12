from persian_intent.text_normalization import normalize_persian_text


def test_normalizes_arabic_yeh_and_kaf_and_removes_diacritics():
    assert normalize_persian_text("كِتابٌ يَكْ") == "کتاب یک"


def test_removes_tatweel_and_collapses_whitespace():
    assert normalize_persian_text("  ســلام\n\tدنیا  ") == "سلام دنیا"


def test_preserves_zwnj_and_punctuation():
    text = "می‌روم، خانه!"
    assert normalize_persian_text(text) == text
