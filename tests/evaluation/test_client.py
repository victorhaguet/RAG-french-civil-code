import httpx
from fastapi.testclient import TestClient

from src.evaluation.client import build_query_client


def test_defaults_to_an_in_process_test_client() -> None:
    client = build_query_client(None)

    assert isinstance(client, TestClient)

    response = client.get("/health")
    assert response.status_code == 200


def test_a_base_url_builds_a_live_http_client() -> None:
    client = build_query_client("http://localhost:8000")

    assert isinstance(client, httpx.Client)
    assert not isinstance(client, TestClient)
    assert str(client.base_url) == "http://localhost:8000"


def test_the_live_http_client_uses_a_longer_than_default_timeout() -> None:
    """httpx.Client's own 5s default is too short for /query's real embedding
    + reranking + OpenAI round-trip -- see src/evaluation/client.py."""
    client = build_query_client("http://localhost:8000")

    assert isinstance(client, httpx.Client)
    assert client.timeout.read == 60.0
