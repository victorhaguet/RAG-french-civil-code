from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api.app import _ranked_refs_from_chunks, _resolve_articles, app
from src.api.dependencies import get_article_store, get_bm25_index, get_chat_model, get_store
from src.ingestion.dataset import to_article
from src.ingestion.pipeline import run_ingestion
from src.retrieval.embeddings import FIXED_PREFIX_MODEL, MultilingualE5Embeddings
from src.retrieval.keyword_index import KeywordIndex
from src.storage.article_store import ArticleStore
from tests.factories import raw_row
from tests.fakes import FakeChatModel, FakeModel


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def _populate(tmp_path: Path, model: FakeModel, rows: list[dict] | None = None):
    rows = rows or [
        raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR"),
        raw_row(ref="A2", texte="Repealed provision, no longer applicable.", etat="ABROGE_DIFF"),
    ]
    chroma_store = run_ingestion(
        raw_rows=rows,
        # Pinned to the fixed-prefix model regardless of config.EMBEDDING_MODEL:
        # these tests exercise the API's wiring, not embedding-prefix behavior
        # (covered by tests/retrieval/test_embeddings.py), and a fixed prefix
        # keeps them deterministic without exercising real language detection.
        embeddings=MultilingualE5Embeddings(model=model, model_name=FIXED_PREFIX_MODEL),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
        sqlite_path=str(tmp_path / "articles.db"),
    )
    article_store = ArticleStore(str(tmp_path / "articles.db"))
    return chroma_store, article_store


def _client_for(store, article_store: ArticleStore, chat_model: FakeChatModel) -> TestClient:
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_article_store] = lambda: article_store
    app.dependency_overrides[get_chat_model] = lambda: chat_model
    app.dependency_overrides[get_bm25_index] = lambda: KeywordIndex(article_store)
    return TestClient(app)


def _client(tmp_path: Path, model: FakeModel, chat_model: FakeChatModel) -> TestClient:
    store, article_store = _populate(tmp_path, model)
    return _client_for(store, article_store, chat_model)


def test_health_returns_a_liveness_check() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_the_generated_answer_and_the_retrieved_articles(
    tmp_path: Path,
) -> None:
    model = FakeModel()
    chat_model = FakeChatModel(answer="Voici la réponse.")
    client = _client(tmp_path, model, chat_model)

    response = client.post("/query", json={"question": "Quand une loi entre-t-elle en vigueur ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Voici la réponse."
    assert body["articles"] == [
        {
            "ref": "A1",
            "sectionParentTitre": (
                "Titre préliminaire : De la publication, des effets et de "
                "l'application des lois en général"
            ),
        }
    ]
    # Only the VIGUEUR article was ingested, so it's the only possible result.
    [prompt] = chat_model.invoke_calls
    assert "Quand une loi entre-t-elle en vigueur ?" in prompt
    assert "Les lois s'appliquent dès leur entrée en vigueur." in prompt
    assert "A1" in prompt


def test_query_only_ever_retrieves_from_the_in_force_article(tmp_path: Path) -> None:
    model = FakeModel()
    chat_model = FakeChatModel()
    client = _client(tmp_path, model, chat_model)

    response = client.post("/query", json={"question": "Quelle est la loi applicable ?"})

    [article] = response.json()["articles"]
    assert article["ref"] == "A1"


def test_query_embeds_the_question_with_the_query_prefix(tmp_path: Path) -> None:
    model = FakeModel()
    client = _client(tmp_path, model, FakeChatModel())

    client.post("/query", json={"question": "Quand une loi entre-t-elle en vigueur ?"})

    [prefixed_text] = model.encode_calls[-1]
    assert prefixed_text == "query: Quand une loi entre-t-elle en vigueur ?"


def test_query_surfaces_via_the_keyword_index_an_article_vector_search_alone_would_miss(
    tmp_path: Path,
) -> None:
    model = FakeModel()
    question = "Quelles sont les conditions de mise sous tutelle d'un majeur protégé ?"
    # FakeModel embeds by prefixed-text length, so Chroma's L2 distance ranks
    # Chunks by how close their length is to the question's. The distractors
    # are sized close to the question; TUTELLE is deliberately much longer,
    # so it ranks last (outside top_k) on vector similarity alone. Only
    # TUTELLE's text contains "tutelle", so the Keyword Index side surfaces
    # it via an exact statutory-term match.
    rows = [
        raw_row(
            ref="D1",
            texte="Le mariage est dissous par la mort de l'un des époux ou par le divorce légalement prononcé.",
            etat="VIGUEUR",
        ),
        raw_row(
            ref="D2",
            texte="Le bail commercial est conclu pour une durée minimale de neuf années sauf convention contraire.",
            etat="VIGUEUR",
        ),
        raw_row(
            ref="D3",
            texte="La prescription acquisitive permet l'acquisition de la propriété par la possession prolongée d'un bien.",
            etat="VIGUEUR",
        ),
        raw_row(
            ref="D4",
            texte="Le contrat de vente transfère la propriété du vendeur à l'acheteur dès l'accord sur la chose et sur le prix.",
            etat="VIGUEUR",
        ),
        raw_row(
            ref="TUTELLE",
            texte=(
                "La tutelle est ouverte lorsque l'altération des facultés mentales ou "
                "corporelles de la personne majeure empêche l'expression de sa volonté "
                "et nécessite une représentation continue dans les actes de la vie "
                "civile courante et administrative de cette personne protégée par la loi."
            ),
            etat="VIGUEUR",
        ),
    ]
    store, article_store = _populate(tmp_path, model, rows=rows)
    client = _client_for(store, article_store, FakeChatModel())

    # Confirm the premise: TUTELLE ranks outside top_k on vector search alone.
    vector_only_refs = [
        chunk.metadata["ref"] for chunk in store.similarity_search(question, k=2)
    ]
    assert "TUTELLE" not in vector_only_refs

    response = client.post("/query", json={"question": question, "top_k": 2})

    assert response.status_code == 200
    refs = [article["ref"] for article in response.json()["articles"]]
    assert "TUTELLE" in refs


def _populate_with_two_in_force_articles(tmp_path: Path, model: FakeModel):
    rows = [
        raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR"),
        raw_row(
            ref="A3",
            texte=(
                "Le mariage est un acte solennel entre deux personnes qui "
                "s'engagent mutuellement à une communauté de vie, avec des "
                "droits et devoirs réciproques prévus par la loi civile."
            ),
            etat="VIGUEUR",
        ),
    ]
    return _populate(tmp_path, model, rows=rows)


def test_query_uses_a_default_top_k_when_none_is_supplied(tmp_path: Path) -> None:
    model = FakeModel()
    store, article_store = _populate_with_two_in_force_articles(tmp_path, model)
    client = _client_for(store, article_store, FakeChatModel())

    response = client.post("/query", json={"question": "Quelle est la loi applicable ?"})

    # Two VIGUEUR articles exist and the default top_k is above that, so
    # both come back — proving the default doesn't truncate below the corpus.
    assert response.status_code == 200
    assert len(response.json()["articles"]) == 2


def test_query_respects_a_caller_supplied_top_k(tmp_path: Path) -> None:
    model = FakeModel()
    store, article_store = _populate_with_two_in_force_articles(tmp_path, model)
    client = _client_for(store, article_store, FakeChatModel())

    response = client.post(
        "/query", json={"question": "Quelle est la loi applicable ?", "top_k": 1}
    )

    # A top_k lower than the number of matching Chunks actually truncates.
    assert response.status_code == 200
    assert len(response.json()["articles"]) == 1


def test_query_rejects_a_non_positive_top_k(tmp_path: Path) -> None:
    model = FakeModel()
    client = _client(tmp_path, model, FakeChatModel())

    response = client.post(
        "/query", json={"question": "Quelle est la loi applicable ?", "top_k": 0}
    )

    assert response.status_code == 422


def test_no_ingest_route_is_exposed() -> None:
    client = TestClient(app)

    response = client.post("/ingest")

    assert response.status_code == 404


def test_get_article_returns_the_full_article(tmp_path: Path) -> None:
    model = FakeModel()
    store, article_store = _populate(tmp_path, model)
    client = _client_for(store, article_store, FakeChatModel())

    response = client.get("/articles/A1")

    assert response.status_code == 200
    body = response.json()
    assert body["ref"] == "A1"
    assert body["texte"] == "Les lois s'appliquent dès leur entrée en vigueur."


def test_get_article_returns_404_for_an_unknown_ref(tmp_path: Path) -> None:
    model = FakeModel()
    store, article_store = _populate(tmp_path, model)
    client = _client_for(store, article_store, FakeChatModel())

    response = client.get("/articles/UNKNOWN")

    assert response.status_code == 404


def _seeded_article_store(tmp_path: Path, *rows: dict) -> ArticleStore:
    store = ArticleStore(str(tmp_path / "articles.db"))
    store.replace_all(to_article(row) for row in rows)
    return store


def test_ranked_refs_from_chunks_deduplicates_chunks_from_the_same_article() -> None:
    chunks = [
        Document(page_content="Texte un, partie 1.", metadata={"ref": "A1"}),
        Document(page_content="Texte deux.", metadata={"ref": "A2"}),
        Document(page_content="Texte un, partie 2.", metadata={"ref": "A1"}),
    ]

    refs = _ranked_refs_from_chunks(chunks)

    assert refs == ["A1", "A2"]


def test_resolve_articles_returns_the_articles_full_text_not_the_chunks(tmp_path: Path) -> None:
    article_store = _seeded_article_store(tmp_path, raw_row(ref="A1", texte="Texte complet."))

    [article] = _resolve_articles(["A1"], article_store)

    assert article["texte"] == "Texte complet."
