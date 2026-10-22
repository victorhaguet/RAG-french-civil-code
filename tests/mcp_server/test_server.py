from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
from mcp.client.client import Client
from mcp.server.mcpserver import MCPServer

from src.code_civil_mcp.server import create_server
from src.ingestion.dataset import Article
from src.pipeline import NON_FRENCH_REFUSAL, QueryPipeline
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from src.storage.article_store import ArticleStore
from tests.app_wiring import ingest
from tests.factories import article, raw_row
from tests.fakes import FakeChatModel, PassthroughCrossEncoder


_ROWS = [
    raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR"),
    raw_row(ref="A2", texte="Repealed provision, no longer applicable.", etat="ABROGE_DIFF"),
]


def _providers(
    tmp_path: Path, chat_model: FakeChatModel
) -> tuple[Callable[[], QueryPipeline], Callable[[], ArticleStore]]:
    store, article_store = ingest(tmp_path, _ROWS)
    pipeline = QueryPipeline(
        vector_store=store,
        chat_model=chat_model,
        article_store=article_store,
        keyword_index=KeywordIndex(article_store),
        reranker=Reranker(model=PassthroughCrossEncoder()),
    )
    return lambda: pipeline, lambda: article_store


def _server(tmp_path: Path, chat_model: FakeChatModel | None = None) -> MCPServer:
    # No token in memory, as over stdio: no role required.
    return create_server(
        *_providers(tmp_path, chat_model or FakeChatModel()), require_query_role=False
    )


def _call_tool(server: MCPServer, arguments: dict[str, Any], name: str = "ask_code_civil") -> Any:
    async def _run() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(_run)


def _list_tools(server: MCPServer) -> Any:
    async def _run() -> Any:
        async with Client(server) as client:
            return await client.list_tools()

    return anyio.run(_run)


def _listed_tool(server: MCPServer, name: str) -> Any:
    [tool] = [tool for tool in _list_tools(server).tools if tool.name == name]
    return tool


def test_tool_listing_exposes_ask_code_civil_with_its_parameters(tmp_path: Path) -> None:
    tool = _listed_tool(_server(tmp_path), "ask_code_civil")

    assert tool.name == "ask_code_civil"
    assert "Code civil" in (tool.description or "")
    properties = tool.input_schema["properties"]
    assert "French" in properties["question"]["description"]
    assert properties["top_k"]["exclusiveMinimum"] == 0
    assert tool.input_schema["required"] == ["question"]


def test_ask_code_civil_returns_the_answer_and_the_retrieved_articles_in_full(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, FakeChatModel(answer="Voici la réponse."))

    result = _call_tool(server, {"question": "Quand une loi entre-t-elle en vigueur ?"})

    assert not result.is_error
    assert result.structured_content == {
        "answer": "Voici la réponse.",
        "articles": [
            {
                "ref": "A1",
                "sectionParentTitre": raw_row()["sectionParentTitre"],
                "texte": "Les lois s'appliquent dès leur entrée en vigueur.",
            }
        ],
    }


def test_ask_code_civil_sends_the_question_and_article_text_to_the_chat_model(
    tmp_path: Path,
) -> None:
    chat_model = FakeChatModel()
    server = _server(tmp_path, chat_model)

    _call_tool(server, {"question": "Quand une loi entre-t-elle en vigueur ?"})

    [prompt] = chat_model.invoke_calls
    assert "Quand une loi entre-t-elle en vigueur ?" in prompt
    assert "Les lois s'appliquent dès leur entrée en vigueur." in prompt


def test_ask_code_civil_refuses_a_non_french_question_without_generation(
    tmp_path: Path,
) -> None:
    chat_model = FakeChatModel()
    server = _server(tmp_path, chat_model)

    result = _call_tool(server, {"question": "When does a law enter into force?"})

    assert not result.is_error
    assert result.structured_content == {"answer": NON_FRENCH_REFUSAL, "articles": []}
    assert not chat_model.invoke_calls


def test_ask_code_civil_rejects_a_non_positive_top_k(tmp_path: Path) -> None:
    chat_model = FakeChatModel()
    server = _server(tmp_path, chat_model)

    result = _call_tool(server, {"question": "Quelle est la loi applicable ?", "top_k": 0})

    assert result.is_error
    assert not chat_model.invoke_calls


def test_building_the_server_resolves_the_providers_only_on_a_tool_call(
    tmp_path: Path,
) -> None:
    provide_pipeline, provide_article_store = _providers(tmp_path, FakeChatModel())
    provider_calls: list[str] = []

    def recording_pipeline_provider() -> QueryPipeline:
        provider_calls.append("pipeline")
        return provide_pipeline()

    def recording_article_store_provider() -> ArticleStore:
        provider_calls.append("article_store")
        return provide_article_store()

    server = create_server(
        recording_pipeline_provider, recording_article_store_provider, require_query_role=False
    )
    _list_tools(server)
    assert not provider_calls

    _call_tool(server, {"question": "Quelle est la loi applicable ?"})
    assert provider_calls == ["pipeline"]

    _call_tool(server, {"ref": "A1"}, name="get_article")
    assert provider_calls == ["pipeline", "article_store"]


def test_tool_listing_exposes_get_article_with_a_ref_parameter(tmp_path: Path) -> None:
    tool = _listed_tool(_server(tmp_path), "get_article")

    assert "Article" in (tool.description or "")
    assert tool.input_schema["required"] == ["ref"]
    assert tool.input_schema["properties"]["ref"]["type"] == "string"


def test_get_article_returns_the_full_article_of_an_ingested_ref(tmp_path: Path) -> None:
    result = _call_tool(_server(tmp_path), {"ref": "A1"}, name="get_article")

    assert not result.is_error
    expected: Article = article(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.")
    assert result.structured_content == expected


def test_get_article_on_an_unknown_ref_returns_a_tool_error(tmp_path: Path) -> None:
    result = _call_tool(_server(tmp_path), {"ref": "UNKNOWN"}, name="get_article")

    assert result.is_error
    assert result.structured_content is None
    [content] = result.content
    assert "Article not found" in content.text
