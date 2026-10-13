"""The MCP server and its tools, built through a factory that receives its components.

The factory takes *providers* (callables), not components: each one is
resolved on every tool call, so building the server loads nothing and the
first tool call pays for model loading. Production passes the cached
providers, which keep the models warm after that; tests pass fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from src import config
from src.api.schemas import ArticleOut
from src.ingestion.dataset import DATASET_AS_OF, Article
from src.pipeline import QueryPipeline, answer_query
from src.storage.article_store import ArticleStore

SERVER_NAME = "code-civil"

ASK_CODE_CIVIL_DESCRIPTION = (
    "Answer a question about French civil law, grounded in the Articles of the "
    "Code civil retrieved for it. The corpus covers the Code civil only "
    "(no other code, case law or doctrine), and only Articles in force, "
    f"as of the {DATASET_AS_OF} snapshot. Returns the generated answer and the "
    "full text of every Retrieved Article it was grounded in, so the answer can "
    "be checked against the sources."
)

GET_ARTICLE_DESCRIPTION = (
    "Get the full record of one Article of the Code civil by its `ref`, to "
    "follow a reference an answer makes to an Article that was not among its "
    "Retrieved Articles. Only Articles in force, as of the "
    f"{DATASET_AS_OF} snapshot, can be found."
)


class ArticleText(ArticleOut):
    """A Retrieved Article, with its full text so the host can check the answer."""

    texte: str


class AskCodeCivilResult(BaseModel):
    """The generated answer and the Retrieved Articles it was grounded in."""

    answer: str
    articles: list[ArticleText]


def create_server(
    query_pipeline: Callable[[], QueryPipeline],
    article_store: Callable[[], ArticleStore],
) -> MCPServer:
    """Build the MCP server.

    Args:
        query_pipeline (Callable[[], QueryPipeline]): provider of the
            components a query runs through, resolved on each tool call
        article_store (Callable[[], ArticleStore]): provider of the Article
            store `get_article` reads from, resolved on each tool call

    Returns:
        MCPServer: the server, ready to run on any transport
    """
    server = MCPServer(SERVER_NAME)

    @server.tool(description=ASK_CODE_CIVIL_DESCRIPTION)
    def ask_code_civil(
        question: Annotated[
            str,
            Field(
                description=(
                    "The legal question. Must be written in French: translate it "
                    "first if needed, otherwise it is refused."
                )
            ),
        ],
        top_k: Annotated[
            int,
            Field(gt=0, description="How many Retrieved Articles ground the answer."),
        ] = config.DEFAULT_TOP_K,
    ) -> AskCodeCivilResult:
        result = answer_query(question, top_k, query_pipeline())
        return AskCodeCivilResult(
            answer=result.answer,
            articles=[
                ArticleText(
                    ref=article["ref"],
                    sectionParentTitre=article["sectionParentTitre"],
                    texte=article["texte"],
                )
                for article in result.articles
            ],
        )

    @server.tool(description=GET_ARTICLE_DESCRIPTION)
    def get_article(
        ref: Annotated[
            str,
            Field(description="The Article's `ref`, e.g. LEGIARTI000006419287."),
        ],
    ) -> Article:
        article = article_store().get(ref)
        if article is None:
            raise ToolError("Article not found.")
        return article

    return server
