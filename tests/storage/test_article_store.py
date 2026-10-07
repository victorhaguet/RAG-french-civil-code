from pathlib import Path

from src.storage.article_store import ArticleStore
from tests import factories


def _store(tmp_path: Path) -> ArticleStore:
    return ArticleStore(str(tmp_path / "articles.db"))


def test_get_returns_none_for_an_unknown_ref(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.get("MISSING") is None


def test_replace_all_then_get_returns_the_full_article(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = factories.article(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.")

    store.replace_all([original])

    assert store.get("A1") == original


def test_replace_all_makes_every_article_resolvable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    articles = [
        factories.article(ref="A1", texte="Premier texte."),
        factories.article(ref="A2", texte="Second texte."),
    ]

    store.replace_all(articles)

    first = store.get("A1")
    second = store.get("A2")
    assert first is not None and first["texte"] == "Premier texte."
    assert second is not None and second["texte"] == "Second texte."


def test_replace_all_rebuilds_from_scratch_dropping_previous_articles(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.replace_all([factories.article(ref="A1"), factories.article(ref="A2")])

    store.replace_all([factories.article(ref="A1")])

    assert store.get("A1") is not None
    assert store.get("A2") is None


def test_all_returns_every_stored_article(tmp_path: Path) -> None:
    store = _store(tmp_path)
    articles = [
        factories.article(ref="A1", texte="Premier texte."),
        factories.article(ref="A2", texte="Second texte."),
    ]
    store.replace_all(articles)

    assert {article["ref"] for article in store.all()} == {"A1", "A2"}


def test_all_returns_an_empty_list_when_the_store_is_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.all() == []


def test_reopening_the_same_path_sees_previously_persisted_articles(tmp_path: Path) -> None:
    path = str(tmp_path / "articles.db")
    ArticleStore(path).replace_all([factories.article(ref="A1", texte="Persisted text.")])

    reopened = ArticleStore(path)

    article = reopened.get("A1")
    assert article is not None and article["texte"] == "Persisted text."
