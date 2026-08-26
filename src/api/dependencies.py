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


@lru_cache
def get_store() -> Chroma:
    """The Chroma collection built by the ingestion pipeline."""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=MultilingualE5Embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


@lru_cache
def get_chat_model() -> Any:
    """The OpenAI-compatible chat model used for answer generation.

    Typed as `Any` (a `ChatOpenAI` at runtime): `langchain_openai` is
    imported lazily inside `build_chat_model` to keep it off every request
    that overrides this dependency with a fake in tests, so its type can't
    be named here without eagerly importing it.
    """
    from src.generation.chat import build_chat_model

    return build_chat_model()
