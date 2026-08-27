"""FastAPI application: query the Code civil corpus and get a grounded answer."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.api.dependencies import get_article_store, get_chat_model, get_store
from src.api.schemas import ArticleDetailOut, ArticleOut, QueryRequest, QueryResponse
from src.generation.prompt import render_prompt
from src.ingestion.dataset import Article
from src.storage.article_store import ArticleStore

app = FastAPI()


def _unique_articles(chunks: list[Document], article_store: ArticleStore) -> list[Article]:
    """Resolve retrieved Chunks to their Articles, deduplicated by `ref`.

    Several Chunks can match the same Article; each Article is kept once,
    in first-seen (highest-relevance) order.
    """
    seen_refs: set[str] = set()
    articles: list[Article] = []
    for chunk in chunks:
        ref = chunk.metadata["ref"]
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        article = article_store.get(ref)
        # Every Chunk in the vectorstore was written by the same ingestion
        # run that populated the Article store, so its ref always resolves.
        assert article is not None
        articles.append(article)
    return articles


def _to_article_out(article: Article) -> ArticleOut:
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
) -> QueryResponse:
    """Retrieve the most relevant Articles and generate a grounded answer."""
    chunks = store.similarity_search(request.question, k=request.top_k)
    articles = _unique_articles(chunks, article_store)
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
