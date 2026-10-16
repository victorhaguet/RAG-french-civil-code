from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src import config
from src.evaluation.client import MissingAccessTokenError, build_query_client
from tests.app_wiring import client_for, ingest
from tests.factories import raw_row
from tests.fakes import FakeChatModel


@pytest.fixture(name="live_token")
def _live_token(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(config, "EVAL_ACCESS_TOKEN", "the-eval-token")
    return "the-eval-token"


def test_defaults_to_an_in_process_test_client() -> None:
    client = build_query_client(None)

    assert isinstance(client, TestClient)

    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.usefixtures("tokens")
def test_the_in_process_client_reaches_query_without_a_token(tmp_path: Path) -> None:
    """`tokens` makes the app's verifier trust only the test key pair, so a 200
    here can only come from the in-process client's trusted identity."""
    rows = [raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR")]
    # Overrides every /query dependency; its own client (without a token) is unused.
    client_for(*ingest(tmp_path, rows), FakeChatModel(answer="Voici la réponse."))
    client = build_query_client(None)

    response = client.post("/query", json={"question": "Quand une loi entre-t-elle en vigueur ?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Voici la réponse."
    assert isinstance(client, TestClient)
    assert "Authorization" not in client.headers


@pytest.mark.usefixtures("live_token")
def test_a_base_url_builds_a_live_http_client() -> None:
    client = build_query_client("http://localhost:8000")

    assert isinstance(client, httpx.Client)
    assert not isinstance(client, TestClient)
    assert str(client.base_url) == "http://localhost:8000"


@pytest.mark.usefixtures("live_token")
def test_the_live_http_client_uses_a_longer_than_default_timeout() -> None:
    """httpx.Client's own 5s default is too short for /query's real embedding
    + reranking + OpenAI round-trip -- see src/evaluation/client.py."""
    client = build_query_client("http://localhost:8000")

    assert isinstance(client, httpx.Client)
    assert client.timeout.read == 60.0


def test_the_live_http_client_sends_the_eval_access_token_as_a_bearer_token(
    live_token: str,
) -> None:
    client = build_query_client("http://localhost:8000")

    assert isinstance(client, httpx.Client)
    assert client.headers["Authorization"] == f"Bearer {live_token}"


@pytest.mark.parametrize("missing", [None, ""], ids=["unset", "empty"])
def test_a_live_run_without_an_eval_access_token_fails_fast(
    monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    monkeypatch.setattr(config, "EVAL_ACCESS_TOKEN", missing)

    with pytest.raises(MissingAccessTokenError, match="EVAL_ACCESS_TOKEN"):
        build_query_client("http://localhost:8000")
