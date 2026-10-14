import json
from typing import Any

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.code_civil_mcp.server import SERVER_NAME


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0"},
    },
}

_HEADERS = {"Accept": "application/json, text/event-stream"}


def _client(host: str) -> TestClient:
    # A fresh app per test: the MCP session manager runs once per app lifespan.
    return TestClient(create_app(), base_url=f"http://{host}")


def _sse_message(body: str) -> Any:
    [data] = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
    return json.loads(data)


def test_mcp_initialize_over_http_succeeds() -> None:
    with _client("localhost:8000") as client:
        response = client.post("/mcp", json=_INITIALIZE, headers=_HEADERS)

    assert response.status_code == 200
    message = _sse_message(response.text)
    assert message["id"] == 1
    assert message["result"]["serverInfo"]["name"] == SERVER_NAME


def test_mcp_rejects_a_non_localhost_host_header() -> None:
    with _client("evil.example.com") as client:
        response = client.post("/mcp", json=_INITIALIZE, headers=_HEADERS)

    assert response.status_code == 421
