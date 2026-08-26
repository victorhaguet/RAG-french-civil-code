"""Detecting a query's language (French/English, French fallback) via Lingua."""

from __future__ import annotations

from typing import Any, Literal

QueryLanguage = Literal["fr", "en"]

_detector: Any | None = None


def _get_detector() -> Any:
    global _detector
    if _detector is None:
        # Imported lazily so tests that don't need real detection never pay
        # for building Lingua's language models.
        from lingua import Language, LanguageDetectorBuilder

        _detector = LanguageDetectorBuilder.from_languages(
            Language.FRENCH, Language.ENGLISH
        ).build()
    return _detector


def detect_query_language(text: str) -> QueryLanguage:
    """Detect whether a query is French or English, defaulting to French.

    Detection is restricted to French/English via Lingua; anything else
    (including an inconclusive detection) falls back to French.

    Args:
        text (str): query text to detect the language of

    Returns:
        QueryLanguage: "en" when English is detected, "fr" otherwise
    """
    detected = _get_detector().detect_language_of(text)
    # Compared by name (rather than re-importing the `Language` enum here)
    # since `_get_detector()` already lazily imports it once.
    if detected is not None and detected.name == "ENGLISH":
        return "en"
    return "fr"
