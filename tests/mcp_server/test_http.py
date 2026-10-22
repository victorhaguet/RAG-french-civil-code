"""The MCP Server over Streamable HTTP, as mounted in the real app at `/mcp`."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from mcp.shared.exceptions import MCPDeprecationWarning

from src import config
from src.api.app import create_app
from src.auth.tokens import FixedKeyProvider, TokenVerifier
from src.code_civil_mcp.server import SERVER_NAME
from src.pipeline import QueryPipeline
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from tests.app_wiring import ingest
from tests.factories import raw_row
from tests.fakes import FakeChatModel, PassthroughCrossEncoder
from tests.tokens import INVALID_TOKENS, TokenFactory


_PROTOCOL_VERSION = "2025-06-18"

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0"},
    },
}

_HEADERS = {"Accept": "application/json, text/event-stream"}


@pytest.fixture(name="make_client")
def _make_client(trusted_key_tokens: TokenFactory) -> Any:
    """Builds a client for a fresh app that trusts the test key pair.

    A fresh app per test: the MCP session manager runs once per app lifespan.
    """
    verifier = TokenVerifier(
        FixedKeyProvider(trusted_key_tokens.public_key),
        issuer=config.OAUTH_ISSUER,
        audience=config.OAUTH_AUDIENCE,
    )

    def make(host: str = "localhost:8000") -> TestClient:
        return TestClient(create_app(token_verifier=lambda: verifier), base_url=f"http://{host}")

    return make


def _sse_message(body: str) -> Any:
    [data] = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
    return json.loads(data)


def _initialize(client: TestClient, token: str | None) -> Any:
    headers = {**_HEADERS, "Authorization": f"Bearer {token}"} if token else _HEADERS
    return client.post("/mcp", json=_INITIALIZE, headers=headers)


def test_mcp_initialize_over_http_succeeds_with_a_valid_token(
    make_client: Any, trusted_key_tokens: TokenFactory
) -> None:
    with make_client() as client:
        response = _initialize(client, trusted_key_tokens.sign())

    assert response.status_code == 200
    message = _sse_message(response.text)
    assert message["id"] == 1
    assert message["result"]["serverInfo"]["name"] == SERVER_NAME


def _resource_metadata_url(response: Any) -> str:
    match = re.search(r'resource_metadata="([^"]+)"', response.headers["WWW-Authenticate"])
    assert match, response.headers["WWW-Authenticate"]
    return match.group(1)


def test_mcp_rejects_a_request_without_a_token(make_client: Any) -> None:
    with make_client() as client:
        response = _initialize(client, None)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Bearer ")


@pytest.mark.parametrize("make_token", INVALID_TOKENS.values(), ids=INVALID_TOKENS.keys())
def test_mcp_rejects_an_invalid_token(
    make_client: Any,
    trusted_key_tokens: TokenFactory,
    untrusted_tokens: TokenFactory,
    make_token: Callable[[TokenFactory, TokenFactory], str],
) -> None:
    with make_client() as client:
        response = _initialize(client, make_token(trusted_key_tokens, untrusted_tokens))

    assert response.status_code == 401


def test_the_401_points_to_public_protected_resource_metadata(make_client: Any) -> None:
    with make_client() as client:
        metadata_url = _resource_metadata_url(_initialize(client, None))
        # Fetched at exactly the advertised URL (the MCP app is mounted at `/`
        # inside FastAPI), with no token.
        response = client.get(metadata_url)

    assert urlsplit(metadata_url).netloc == "localhost:8000"
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["resource"] == config.OAUTH_AUDIENCE
    assert metadata["authorization_servers"] == [config.OAUTH_ISSUER]


def test_mcp_rejects_a_non_localhost_host_header(
    make_client: Any, trusted_key_tokens: TokenFactory
) -> None:
    # A valid token: the 401 check runs before DNS-rebinding protection.
    with make_client("evil.example.com") as client:
        response = _initialize(client, trusted_key_tokens.sign())

    assert response.status_code == 421


def test_building_the_app_raises_no_mcp_deprecation_warning() -> None:
    # E.g. the one `AuthSettings` raises while `validate_token_resource` is unset.
    with warnings.catch_warnings():
        warnings.simplefilter("error", MCPDeprecationWarning)
        create_app()


_ROWS = [raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR")]


@pytest.fixture(name="fake_corpus")
def _fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the app's MCP Server at a temporary corpus and a fake chat model."""
    store, article_store = ingest(tmp_path, _ROWS)
    pipeline = QueryPipeline(
        vector_store=store,
        chat_model=FakeChatModel(answer="Voici la réponse."),
        article_store=article_store,
        keyword_index=KeywordIndex(article_store),
        reranker=Reranker(model=PassthroughCrossEncoder()),
    )
    monkeypatch.setattr("src.api.app.build_query_pipeline", lambda: pipeline)
    monkeypatch.setattr("src.api.app.get_article_store", lambda: article_store)


def _call_tool(client: TestClient, token: str, name: str, arguments: dict[str, Any]) -> Any:
    """Opens an MCP session with `token`, then calls tool `name` in it."""
    initialized = _initialize(client, token)
    headers = {
        **_HEADERS,
        "Authorization": f"Bearer {token}",
        "Mcp-Session-Id": initialized.headers["mcp-session-id"],
        "Mcp-Protocol-Version": _PROTOCOL_VERSION,
    }
    client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers
    )
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers=headers,
    )
    assert response.status_code == 200
    return _sse_message(response.text)["result"]


_WITHOUT_QUERY_ROLE = {"realm_access": {"roles": ["other"]}}


@pytest.mark.usefixtures("fake_corpus")
def test_ask_code_civil_over_http_answers_a_caller_with_the_query_role(
    make_client: Any, trusted_key_tokens: TokenFactory
) -> None:
    with make_client() as client:
        result = _call_tool(
            client,
            trusted_key_tokens.sign(),
            "ask_code_civil",
            {"question": "Quand une loi entre-t-elle en vigueur ?"},
        )

    assert not result.get("isError")
    assert result["structuredContent"]["answer"] == "Voici la réponse."


@pytest.mark.usefixtures("fake_corpus")
def test_ask_code_civil_over_http_refuses_a_caller_without_the_query_role(
    make_client: Any, trusted_key_tokens: TokenFactory
) -> None:
    with make_client() as client:
        result = _call_tool(
            client,
            trusted_key_tokens.sign(**_WITHOUT_QUERY_ROLE),
            "ask_code_civil",
            {"question": "Quand une loi entre-t-elle en vigueur ?"},
        )

    assert result["isError"] is True
    assert "code-civil-query" in result["content"][0]["text"]


@pytest.mark.usefixtures("fake_corpus")
def test_get_article_over_http_needs_no_query_role(
    make_client: Any, trusted_key_tokens: TokenFactory
) -> None:
    with make_client() as client:
        result = _call_tool(
            client, trusted_key_tokens.sign(**_WITHOUT_QUERY_ROLE), "get_article", {"ref": "A1"}
        )

    assert not result.get("isError")
    assert result["structuredContent"]["ref"] == "A1"
