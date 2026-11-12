"""Lightweight, conservative normalization for Persian utterances."""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک"})
_WHITESPACE = re.compile(r"\s+")


def normalize_persian_text(text: str) -> str:
    """Normalize common Persian/Arabic character variants conservatively.

    Applies Unicode NFC, maps Arabic Yeh and Kaf to Persian forms,
    removes tatweel and Arabic vowel marks, and collapses whitespace. ZWNJ and
    punctuation are intentionally preserved.
    """
    if not isinstance(text, str):
        text = str(text)
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.translate(_ARABIC_TO_PERSIAN)
    normalized = normalized.replace("\u0640", "")
    normalized = _ARABIC_DIACRITICS.sub("", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return normalized
