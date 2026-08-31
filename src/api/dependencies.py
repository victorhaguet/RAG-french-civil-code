"""FastAPI dependencies for the query API: the vectorstore and chat model.

Both are cached singletons in production, and are the seams tests override
via `app.dependency_overrides` to inject a pre-populated test Chroma
collection and a fake chat model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma

from src import config
from src.retrieval.embeddings import MultilingualE5Embeddings
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from src.generation.chat import build_chat_model
from src.storage.article_store import ArticleStore


@lru_cache
def get_store() -> Chroma:
    """The Chroma collection built by the ingestion pipeline.

    Note: Use the @lru_cache to avoid reloading the sentence-transformer
    model every time the function is called.
    """
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=MultilingualE5Embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


@lru_cache
def get_article_store() -> ArticleStore:
    """The Article store built by the ingestion pipeline.

    Note: Use the @lru_cache to reuse the same SQLite connection across
    requests instead of reopening the database file every time.
    """
    return ArticleStore(config.ARTICLES_DB_PATH)


@lru_cache
def get_bm25_index() -> KeywordIndex:
    """The BM25 Keyword Index, built lazily in memory from the Article store.

    Note: Use the @lru_cache so the underlying BM25 index (itself lazily
    built on first `.search()` call) is built at most once per process.
    """
    return KeywordIndex(get_article_store())


@lru_cache
def get_reranker() -> Reranker:
    """The cross-encoder Reranker used to reorder Hybrid Retrieval's Candidate Articles.

    Note: Use the @lru_cache to avoid reloading the cross-encoder model every
    time the function is called.
    """
    return Reranker()


@lru_cache
def get_chat_model() -> Any:
    """The OpenAI-compatible chat model used for answer generation.

    Typed as `Any` (a `ChatOpenAI` at runtime): `langchain_openai` is
    imported lazily inside `build_chat_model` to keep it off every request
    that overrides this dependency with a fake in tests, so its type can't
    be named here without eagerly importing it.
    """
    return build_chat_model()
