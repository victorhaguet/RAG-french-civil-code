"""The query pipeline: from a question to a grounded answer and its Retrieved Articles.

Framework-free, so every interface (the REST API, the MCP Server) runs the exact
same flow: Query Language gate, Hybrid Retrieval, reranking and generation.

The flow is split in two steps a caller can run on their own: `retrieve` (gate,
Hybrid Retrieval, reranking, prompt rendering) and `generate` (the chat model
call), so the Retrieved Articles can be sent before the answer is generated.
`answer_query` runs both in sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.generation.prompt import render_prompt
from src.ingestion.dataset import Article
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.language import detect_query_language
from src.retrieval.reranker import Reranker
from src.storage.article_store import ArticleStore

logger = logging.getLogger(__name__)

# Fixed, English-only, and deliberately not generated: the corpus and the
# whole pipeline are French-only, so a non-French question is redirected
# before any retrieval, embedding or LLM call.
NON_FRENCH_REFUSAL = (
    "This tool only answers questions in French, about the Code civil. "
    "Please rephrase your question in French."
)


@dataclass(frozen=True)
class QueryPipeline:
    """Everything a query needs to be answered, bundled as one value."""

    vector_store: Chroma
    chat_model: Any
    article_store: ArticleStore
    keyword_index: KeywordIndex
    reranker: Reranker


@dataclass(frozen=True)
class QueryResult:
    """The generated answer and the Retrieved Articles, in final order."""

    answer: str
    articles: list[Article]


@dataclass(frozen=True)
class Retrieval:
    """The Retrieved Articles, in final order, and the generation prompt grounded in them."""

    articles: list[Article]
    prompt: str


@dataclass(frozen=True)
class Refusal:
    """A question refused before retrieval, with the fixed answer to give back instead."""

    answer: str


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


def retrieve(question: str, top_k: int, pipeline: QueryPipeline) -> Retrieval | Refusal:
    """Gate on the Query Language, retrieve via Hybrid Retrieval, rerank, and render the generation prompt.

    Args:
        question (str): the user's question
        top_k (int): how many Retrieved Articles ground the answer
        pipeline (QueryPipeline): the components the query runs through

    Returns:
        Retrieval | Refusal: the Retrieved Articles and the generation prompt
            grounded in them, or the fixed refusal for a non-French question
    """
    logger.info("Query: %r", question)
    if detect_query_language(question) != "fr":
        logger.info("Non-French query refused before retrieval")
        return Refusal(answer=NON_FRENCH_REFUSAL)

    fetch_k = max(config.FETCH_K_MULTIPLIER * top_k, config.MIN_FETCH_K)

    chunks = pipeline.vector_store.similarity_search(question, k=fetch_k)
    vector_refs = _ranked_refs_from_chunks(chunks)
    logger.info("Vector search: %d chunk(s) -> %d article(s)", len(chunks), len(vector_refs))

    keyword_refs = pipeline.keyword_index.search(question, k=fetch_k)
    logger.info("BM25 search: %d article(s)", len(keyword_refs))

    candidate_refs = reciprocal_rank_fusion(
        [keyword_refs, vector_refs],
        weights=[config.RRF_WEIGHT_BM25, config.RRF_WEIGHT_VECTOR],
        k=config.RRF_K,
    )
    logger.info("Fused to %d candidate(s)", len(candidate_refs))

    candidate_articles = _resolve_articles(candidate_refs, pipeline.article_store)
    articles = pipeline.reranker.rerank(question, candidate_articles)[:top_k]
    logger.info("Reranked to %d article(s)", len(articles))

    return Retrieval(articles=articles, prompt=render_prompt(question=question, articles=articles))


def generate(prompt: str, pipeline: QueryPipeline) -> str:
    """Generate the answer to a rendered generation prompt with the chat model.

    Args:
        prompt (str): the generation prompt, as rendered by `retrieve`
        pipeline (QueryPipeline): the components the query runs through

    Returns:
        str: the chat model's answer
    """
    logger.info("Calling chat model...")
    answer: str = pipeline.chat_model.invoke(prompt).content
    logger.info("Chat model responded (%d chars)", len(answer))
    return answer


def answer_query(question: str, top_k: int, pipeline: QueryPipeline) -> QueryResult:
    """Answer a question in one shot: `retrieve`, then `generate` unless the question was refused.

    Args:
        question (str): the user's question
        top_k (int): how many Retrieved Articles ground the answer
        pipeline (QueryPipeline): the components the query runs through

    Returns:
        QueryResult: the generated answer and the Retrieved Articles it was
            grounded in (the fixed refusal and no Articles for a non-French question)
    """
    retrieval = retrieve(question, top_k, pipeline)
    if isinstance(retrieval, Refusal):
        return QueryResult(answer=retrieval.answer, articles=[])

    return QueryResult(answer=generate(retrieval.prompt, pipeline), articles=retrieval.articles)
