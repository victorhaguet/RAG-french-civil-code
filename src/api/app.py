"""FastAPI application: query the Code civil corpus and get a grounded answer."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.documents import Document

from src import config
from src.api.dependencies import QueryPipeline, get_article_store, get_query_pipeline
from src.api.schemas import ArticleOut, QueryRequest, QueryResponse
from src.generation.prompt import render_prompt
from src.ingestion.dataset import Article
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.language import detect_query_language
from src.storage.article_store import ArticleStore

app = FastAPI()

logger = logging.getLogger(__name__)

# Fixed, English-only, and deliberately not generated: the corpus and the
# whole pipeline are French-only, so a non-French question is redirected
# before any retrieval, embedding or LLM call.
NON_FRENCH_REFUSAL = (
    "This tool only answers questions in French, about the Code civil. "
    "Please rephrase your question in French."
)


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
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> QueryResponse:
    """Retrieve the most relevant Articles via Hybrid Retrieval, rerank, and generate a grounded answer.

    Args:
        request (QueryRequest): Query request received from the user
        pipeline (QueryPipeline, optional): the components the query runs
            through. Defaults to Depends(get_query_pipeline).

    Returns:
        QueryResponse: the generated answer and the Retrieved Articles it cites
    """
    logger.info("Query: %r", request.question)
    if detect_query_language(request.question) != "fr":
        logger.info("Non-French query refused before retrieval")
        return QueryResponse(answer=NON_FRENCH_REFUSAL, articles=[])

    fetch_k = max(config.FETCH_K_MULTIPLIER * request.top_k, config.MIN_FETCH_K)

    chunks = pipeline.vector_store.similarity_search(request.question, k=fetch_k)
    vector_refs = _ranked_refs_from_chunks(chunks)
    logger.info("Vector search: %d chunk(s) -> %d article(s)", len(chunks), len(vector_refs))

    keyword_refs = pipeline.keyword_index.search(request.question, k=fetch_k)
    logger.info("BM25 search: %d article(s)", len(keyword_refs))

    candidate_refs = reciprocal_rank_fusion(
        [keyword_refs, vector_refs],
        weights=[config.RRF_WEIGHT_BM25, config.RRF_WEIGHT_VECTOR],
        k=config.RRF_K,
    )
    logger.info("Fused to %d candidate(s)", len(candidate_refs))

    candidate_articles = _resolve_articles(candidate_refs, pipeline.article_store)
    articles = pipeline.reranker.rerank(request.question, candidate_articles)[: request.top_k]
    logger.info("Reranked to %d article(s)", len(articles))

    prompt = render_prompt(question=request.question, articles=articles)
    logger.info("Calling chat model...")
    answer = pipeline.chat_model.invoke(prompt).content
    logger.info("Chat model responded (%d chars)", len(answer))

    return QueryResponse(
        answer=answer,
        articles=[_to_article_out(article) for article in articles],
    )


@app.get("/articles/{ref}", response_model=Article)
def get_article(
    ref: str, article_store: ArticleStore = Depends(get_article_store)
) -> Article:
    """Resolve a `ref` to its full Article."""
    article = article_store.get(ref)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found.")
    return article
