"""End-to-end ingestion: load, filter, chunk, embed, and store Articles."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src import config
from src.ingestion.chunking import build_documents
from src.ingestion.dataset import load_articles
from src.retrieval.embeddings import MultilingualE5Embeddings
from src.storage.article_store import ArticleStore


def run_ingestion(
    *,
    raw_rows: Iterable[dict] | None = None,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
    collection_name: str | None = None,
    sqlite_path: str | None = None,
) -> Chroma:
    """Rebuild the Chroma collection and the Article store from scratch.

    Pass `raw_rows` and/or `embeddings` to run against a fixture dataset and
    a fake embedder in tests, bypassing the network and the real model.

    Args:
        raw_rows (Iterable[dict] | None, optional): Set custom raw rows for test purposes. Defaults to None.
        embeddings (Embeddings | None, optional): Set custom embeddings for test purposes. Defaults to None.
        persist_directory (str | None, optional): Directory to save the chroma.db vectorstore. Defaults to None.
        collection_name (str | None, optional): Name of the collection. Defaults to None.
        sqlite_path (str | None, optional): Path to the Article store's SQLite file. Defaults to None.

    Returns:
        Chroma: the rebuilt collection, containing every ingested Chunk.
    """
    articles = load_articles(raw_rows=raw_rows)
    documents = build_documents(articles)

    article_store = ArticleStore(sqlite_path or config.ARTICLES_DB_PATH)
    article_store.replace_all(articles)

    store = Chroma(
        collection_name=collection_name or config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings or MultilingualE5Embeddings(),
        persist_directory=persist_directory or config.CHROMA_PERSIST_DIR,
    )
    store.reset_collection()
    if documents:
        store.add_documents(documents=documents, ids=[doc.id for doc in documents])
    return store
