"""FastAPI application: query the Code civil corpus and get a grounded answer."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from src.api.dependencies import get_article_store, get_query_pipeline
from src.api.schemas import ArticleOut, QueryRequest, QueryResponse
from src.ingestion.dataset import Article
from src.pipeline import QueryPipeline, answer_query
from src.storage.article_store import ArticleStore

app = FastAPI()


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
    result = answer_query(request.question, request.top_k, pipeline)
    return QueryResponse(
        answer=result.answer,
        articles=[_to_article_out(article) for article in result.articles],
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
