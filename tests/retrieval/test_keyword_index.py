"""Tests for the BM25 Keyword Index, exercised against real rank_bm25 and nltk."""

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nltk
import pytest

from src.ingestion.dataset import to_article
from src.retrieval.keyword_index import _NLTK_DOWNLOAD_TIMEOUT, KeywordIndex, _get_stopwords, tokenize
from src.storage.article_store import ArticleStore
from tests.factories import raw_row

# BM25's IDF collapses toward zero on a tiny corpus (a term present in half
# of just 2 documents carries no discriminative signal), so every fixture
# here mirrors a realistically small corpus rather than a minimal one.
_DISTRACTORS = [
    raw_row(ref="D1", texte="Le contrat de vente transfère la propriété du vendeur.", etat="VIGUEUR"),
    raw_row(ref="D2", texte="Le bail commercial est conclu pour une durée de neuf ans.", etat="VIGUEUR"),
    raw_row(ref="D3", texte="La prescription acquisitive permet l'acquisition d'un bien.", etat="VIGUEUR"),
    raw_row(ref="D4", texte="Le mariage est dissous par le divorce légalement prononcé.", etat="VIGUEUR"),
]


def _store(tmp_path: Path, *rows: dict[str, Any]) -> ArticleStore:
    store = ArticleStore(str(tmp_path / "articles.db"))
    store.replace_all(to_article(row) for row in rows)
    return store


def test_tokenize_lowercases_before_splitting() -> None:
    assert tokenize("Le Mariage") == tokenize("le mariage")


def test_tokenize_stems_plural_and_singular_forms_to_the_same_token() -> None:
    assert tokenize("la tutelle") == tokenize("les tutelles")


def test_tokenize_drops_french_stopwords() -> None:
    tokens = tokenize("le contrat de la vente")

    assert "le" not in tokens
    assert "de" not in tokens
    assert "la" not in tokens
    assert "contrat" in tokens


def test_search_ranks_the_article_containing_the_query_term_first(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        *_DISTRACTORS,
        raw_row(ref="A2", texte="La tutelle est ouverte pour un majeur protégé.", etat="VIGUEUR"),
    )
    index = KeywordIndex(store)

    refs = index.search("Quelles sont les conditions de la tutelle ?", k=5)

    assert refs[0] == "A2"


def test_search_matches_an_inflected_query_term_via_stemming(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        *_DISTRACTORS,
        raw_row(ref="A2", texte="Les majeurs protégés bénéficient d'une représentation.", etat="VIGUEUR"),
    )
    index = KeywordIndex(store)

    # Singular query term "majeur" should still match the plural "majeurs" in A2.
    refs = index.search("Le majeur protégé", k=5)

    assert "A2" in refs


def test_search_excludes_articles_sharing_no_term_with_the_query(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        *_DISTRACTORS,
        raw_row(ref="A2", texte="La tutelle est ouverte pour un majeur protégé.", etat="VIGUEUR"),
    )
    index = KeywordIndex(store)

    refs = index.search("succession testament héritier", k=5)

    assert refs == []


def test_search_truncates_to_k(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        raw_row(ref="A1", texte="La tutelle du majeur protégé.", etat="VIGUEUR"),
        raw_row(ref="A2", texte="La tutelle et la curatelle du majeur.", etat="VIGUEUR"),
        raw_row(ref="A3", texte="La tutelle légale du majeur incapable.", etat="VIGUEUR"),
        *_DISTRACTORS,
    )
    index = KeywordIndex(store)

    refs = index.search("tutelle majeur", k=2)

    assert len(refs) == 2


def test_search_on_an_empty_store_returns_no_refs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    index = KeywordIndex(store)

    assert index.search("tutelle", k=5) == []


@pytest.fixture
def _uncached_stopwords() -> Iterator[None]:
    previous_timeout = socket.getdefaulttimeout()
    _get_stopwords.cache_clear()
    yield
    _get_stopwords.cache_clear()
    socket.setdefaulttimeout(previous_timeout)


@pytest.mark.usefixtures("_uncached_stopwords")
def test_stopwords_download_is_bounded_by_a_socket_timeout_then_restores_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts_during_download: list[float | None] = []

    def _missing(resource: str) -> None:
        raise LookupError(resource)

    def _download(*_args: object, **_kwargs: object) -> None:
        timeouts_during_download.append(socket.getdefaulttimeout())

    monkeypatch.setattr(nltk.data, "find", _missing)
    monkeypatch.setattr(nltk, "download", _download)
    socket.setdefaulttimeout(None)

    _get_stopwords()

    assert timeouts_during_download == [_NLTK_DOWNLOAD_TIMEOUT]
    assert socket.getdefaulttimeout() is None


@pytest.mark.usefixtures("_uncached_stopwords")
def test_stopwords_are_loaded_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _get_stopwords()
    monkeypatch.setattr(nltk.data, "find", lambda _resource: pytest.fail("reloaded"))

    assert _get_stopwords() is first
