"""Shared wiring for tests that drive the real FastAPI app over a temporary corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_chroma import Chroma

from src.api.app import app
from src.api.dependencies import (
    get_article_store,
    get_bm25_index,
    get_chat_model,
    get_reranker,
    get_store,
)
from src.ingestion.pipeline import run_ingestion
from src.retrieval.embeddings import FIXED_PREFIX_MODEL, MultilingualE5Embeddings
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from src.storage.article_store import ArticleStore
from tests.fakes import FakeChatModel, FakeModel, PassthroughCrossEncoder


def ingest(
    tmp_path: Path, rows: list[dict[str, Any]], model: FakeModel | None = None
) -> tuple[Chroma, ArticleStore]:
    """Run the real ingestion over `rows` into `tmp_path`, with fake embeddings."""
    sqlite_path = str(tmp_path / "articles.db")
    chroma_store = run_ingestion(
        raw_rows=rows,
        # Pinned to the fixed-prefix model regardless of config.EMBEDDING_MODEL:
        # these tests exercise the API's wiring, not embedding-prefix behavior
        # (covered by tests/retrieval/test_embeddings.py), and a fixed prefix
        # keeps them deterministic without exercising real language detection.
        embeddings=MultilingualE5Embeddings(
            model=model or FakeModel(), model_name=FIXED_PREFIX_MODEL
        ),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
        sqlite_path=sqlite_path,
    )
    return chroma_store, ArticleStore(sqlite_path)


def client_for(
    store: Chroma,
    article_store: ArticleStore,
    chat_model: FakeChatModel,
    reranker: Reranker | None = None,
) -> TestClient:
    """A client for the app, with every `/query` dependency overridden.

    The reranker defaults to a passthrough, so retrieval order is unchanged.
    `tests/conftest.py` clears the overrides after each test.
    """
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_article_store] = lambda: article_store
    app.dependency_overrides[get_chat_model] = lambda: chat_model
    app.dependency_overrides[get_bm25_index] = lambda: KeywordIndex(article_store)
    app.dependency_overrides[get_reranker] = lambda: reranker or Reranker(
        model=PassthroughCrossEncoder()
    )
    return TestClient(app)
