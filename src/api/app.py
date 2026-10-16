"""FastAPI application: query the Code civil corpus and get a grounded answer.

The same app serves the MCP server over Streamable HTTP at `/mcp`, built with
the same cached providers as the REST routes, so both share one set of loaded
models.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from mcp.server.transport_security import TransportSecuritySettings

from src.api.dependencies import (
    build_query_pipeline,
    get_article_store,
    get_query_pipeline,
    require_access_token,
)
from src.api.schemas import ArticleOut, QueryRequest, QueryResponse
from src.code_civil_mcp.server import create_server
from src.ingestion.dataset import Article
from src.pipeline import QueryPipeline, answer_query
from src.storage.article_store import ArticleStore

# DNS-rebinding protection: a web page can't drive the unauthenticated local
# server from a browser, since only localhost Host/Origin values get through.
MCP_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
)

router = APIRouter()


def _to_article_out(article: Article) -> ArticleOut:
    """Pydantic schema of the Article

    Args:
        article (Article): Article object

    Returns:
        ArticleOut: Article information stored in a usable pydantic schema
    """
    return ArticleOut(ref=article["ref"], sectionParentTitre=article["sectionParentTitre"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@router.post(
    "/query", response_model=QueryResponse, dependencies=[Depends(require_access_token)]
)
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


@router.get(
    "/articles/{ref}", response_model=Article, dependencies=[Depends(require_access_token)]
)
def get_article(
    ref: str, article_store: ArticleStore = Depends(get_article_store)
) -> Article:
    """Resolve a `ref` to its full Article."""
    article = article_store.get(ref)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found.")
    return article


def create_app() -> FastAPI:
    """Build the app: the REST routes, plus the MCP server mounted at `/mcp`.

    A factory because the MCP session manager runs once per instance: each
    app (and so each lifespan) needs its own.

    Returns:
        FastAPI: the app, ready to serve
    """
    mcp_server = create_server(build_query_pipeline, get_article_store)
    # Mounted at the root, the MCP app's own `/mcp` route is served at `/mcp`
    # exactly (a mount at `/mcp` would only match `/mcp/...`). Its router only
    # handles `/mcp`, and the REST routes are matched before it.
    mcp_app = mcp_server.streamable_http_app(transport_security=MCP_TRANSPORT_SECURITY)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # A mounted app's lifespan never runs, so the session manager is
        # started here, as the SDK requires.
        async with mcp_server.session_manager.run():
            yield

    fastapi_app = FastAPI(lifespan=lifespan)
    fastapi_app.include_router(router)
    fastapi_app.mount("/", mcp_app)
    return fastapi_app


app = create_app()
