"""FastAPI application: query the Code civil corpus and get a grounded answer."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.api.dependencies import (
    get_article_store,
    get_bm25_index,
    get_chat_model,
    get_reranker,
    get_store,
)
from src.api.schemas import ArticleDetailOut, ArticleOut, QueryRequest, QueryResponse
from src.generation.prompt import render_prompt
from src.ingestion.dataset import Article
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from src.storage.article_store import ArticleStore

app = FastAPI()


def _ranked_refs_from_chunks(chunks: list[Document]) -> list[str]:
    """Dedupe retrieved Chunks to their Article refs, in first-seen (highest-relevance) order.

    Several Chunks can match the same Article; each ref is kept once.

    Args:
        chunks (list[Document]): retrieved chunks

    Returns:
        list[str]: Article refs obtained from the retrieved chunks (no duplication)
    """
    # Use a set for checks to reduce complexity (instead of using the list refs)
    seen_refs: set[str] = set()
    refs: list[str] = []
    for chunk in chunks:
        ref = chunk.metadata["ref"]
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        refs.append(ref)
    return refs


def _resolve_articles(refs: list[str], article_store: ArticleStore) -> list[Article]:
    """Resolve Article refs to their full records, preserving order.

    Args:
        refs (list[str]): Article refs retrieved
        article_store (ArticleStore): store containing all the articles

    Returns:
        list[Article]: Articles retrieved
    """
    articles: list[Article] = []
    for ref in refs:
        article = article_store.get(ref)
        # Every ref reaching here came from either the vectorstore or the
        # Keyword Index, both sourced from the same ingestion run that
        # populated the Article store, so it always resolves.
        assert article is not None
        articles.append(article)
    return articles


def _to_article_out(article: Article) -> ArticleOut:
    """Pydantic schema of the Article

    Args:
        article (Article): Article object

    Returns:
        ArticleOut: Article information stored in a usable pydantic schema
    """
    return ArticleOut(ref=article["ref"], sectionParentTitre=article["sectionParentTitre"])


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    store: Chroma = Depends(get_store),
    chat_model: Any = Depends(get_chat_model),
    article_store: ArticleStore = Depends(get_article_store),
    keyword_index: KeywordIndex = Depends(get_bm25_index),
    reranker: Reranker = Depends(get_reranker),
) -> QueryResponse:
    """Retrieve the most relevant Articles via Hybrid Retrieval, rerank, and generate a grounded answer.

    Args:
        request (QueryRequest): Query request received from the user
        store (Chroma, optional): Chroma vectorstore. Defaults to Depends(get_store).
        chat_model (Any, optional): LLM. Defaults to Depends(get_chat_model).
        article_store (ArticleStore, optional): SQL article store. Defaults to Depends(get_article_store).
        keyword_index (KeywordIndex, optional): BM25 keyword index. Defaults to Depends(get_bm25_index).
        reranker (Reranker, optional): cross-encoder Reranker. Defaults to Depends(get_reranker).

    Returns:
        QueryResponse: the generated answer and the Retrieved Articles it cites
    """
    fetch_k = max(config.FETCH_K_MULTIPLIER * request.top_k, config.MIN_FETCH_K)

    chunks = store.similarity_search(request.question, k=fetch_k)
    vector_refs = _ranked_refs_from_chunks(chunks)
    keyword_refs = keyword_index.search(request.question, k=fetch_k)

    candidate_refs = reciprocal_rank_fusion(
        [keyword_refs, vector_refs],
        weights=[config.RRF_WEIGHT_BM25, config.RRF_WEIGHT_VECTOR],
        k=config.RRF_K,
    )
    candidate_articles = _resolve_articles(candidate_refs, article_store)
    articles = reranker.rerank(request.question, candidate_articles)[: request.top_k]

    prompt = render_prompt(question=request.question, articles=articles)
    answer = chat_model.invoke(prompt).content

    return QueryResponse(
        answer=answer,
        articles=[_to_article_out(article) for article in articles],
    )


@app.get("/articles/{ref}", response_model=ArticleDetailOut)
def get_article(
    ref: str, article_store: ArticleStore = Depends(get_article_store)
) -> ArticleDetailOut:
    """Resolve a `ref` to its full Article."""
    article = article_store.get(ref)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found.")
    return ArticleDetailOut(**article)
