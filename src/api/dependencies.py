"""FastAPI dependencies for the query API.

Each component (vectorstore, chat model, Article store, Keyword Index,
Reranker) has its own provider: a cached singleton in production, and the
seam tests override via `app.dependency_overrides` to inject a pre-populated
test Chroma collection, a fake chat model, and so on. `get_query_pipeline`
bundles them for `/query`.

`require_access_token` guards the protected routes: it checks the bearer
token with the cached `TokenVerifier`, which tests override to trust a fixed
key pair instead of Keycloak's. `require_query_role`, built on it, also
requires the `QUERY_ROLE` realm role, for the routes that spend the OpenAI key.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_chroma import Chroma

from src import config
from src.auth.tokens import QUERY_ROLE, Identity, JwksKeyProvider, TokenVerifier
from src.retrieval.embeddings import MultilingualE5Embeddings
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from src.generation.chat import build_chat_model
from src.pipeline import QueryPipeline
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


def get_query_pipeline(
    vector_store: Chroma = Depends(get_store),
    chat_model: Any = Depends(get_chat_model),
    article_store: ArticleStore = Depends(get_article_store),
    keyword_index: KeywordIndex = Depends(get_bm25_index),
    reranker: Reranker = Depends(get_reranker),
) -> QueryPipeline:
    """Bundle `/query`'s dependencies, each resolved through its own provider.

    Not cached: resolving through the individual providers keeps each one
    overridable on its own via `app.dependency_overrides`.
    """
    return QueryPipeline(
        vector_store=vector_store,
        chat_model=chat_model,
        article_store=article_store,
        keyword_index=keyword_index,
        reranker=reranker,
    )


def build_query_pipeline() -> QueryPipeline:
    """Bundle the cached providers outside FastAPI's dependency injection.

    Used by the MCP server, which resolves it on each tool call: the first
    call loads every model, and the providers' caches keep them warm after.
    """
    return get_query_pipeline(
        vector_store=get_store(),
        chat_model=get_chat_model(),
        article_store=get_article_store(),
        keyword_index=get_bm25_index(),
        reranker=get_reranker(),
    )


@lru_cache
def get_token_verifier() -> TokenVerifier:
    """The Access Token verifier, trusting the keys Keycloak publishes.

    Note: Use the @lru_cache to keep one JWKS client, and so its cached
    signing keys, across requests instead of refetching them every time.
    """
    return TokenVerifier(
        JwksKeyProvider(config.OAUTH_JWKS_URL),
        issuer=config.OAUTH_ISSUER,
        audience=config.OAUTH_AUDIENCE,
    )


# auto_error=False: a missing token must get the same 401 as an invalid one,
# not FastAPI's own error response.
_bearer_scheme = HTTPBearer(auto_error=False)


def require_access_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    verifier: TokenVerifier = Depends(get_token_verifier),
) -> Identity:
    """The caller's Identity, from a valid `Authorization: Bearer` Access Token.

    Raises:
        HTTPException: 401 with `WWW-Authenticate: Bearer` when the token is
            missing or invalid, without saying why
    """
    identity = verifier.verify(credentials.credentials) if credentials else None
    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return identity


def require_query_role(identity: Identity = Depends(require_access_token)) -> Identity:
    """The caller's Identity, if it holds the `QUERY_ROLE` realm role.

    Raises:
        HTTPException: 401 as `require_access_token` does, or 403 when the
            token is valid but lacks the role
    """
    if QUERY_ROLE not in identity.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required role: {QUERY_ROLE}.",
        )
    return identity
