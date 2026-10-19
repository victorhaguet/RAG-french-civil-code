"""The REST routes accept only a valid Access Token, checked through the real app over HTTP."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from tests.app_wiring import client_for, ingest
from tests.factories import raw_row
from tests.fakes import FakeChatModel
from tests.tokens import INVALID_TOKENS, TokenFactory

_ROWS = [raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR")]

_QUESTION = {"question": "Quand une loi entre-t-elle en vigueur ?"}


@pytest.fixture(name="client")
def _client(tmp_path: Path, tokens: TokenFactory) -> TestClient:
    # No default token: each test sends its own Authorization header, or none.
    del tokens  # requested only so the app trusts the test key pair
    return client_for(*ingest(tmp_path, _ROWS), FakeChatModel(answer="Voici la réponse."))


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _query(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post("/query", json=_QUESTION, headers=headers)


def _article(client: TestClient, headers: dict[str, str]) -> Any:
    return client.get("/articles/A1", headers=headers)


_ROUTES: list[Callable[[TestClient, dict[str, str]], Any]] = [_query, _article]


def test_a_valid_token_gets_a_grounded_answer(client: TestClient, valid_token: str) -> None:
    response = _query(client, _bearer(valid_token))

    assert response.status_code == 200
    assert response.json()["answer"] == "Voici la réponse."
    assert [article["ref"] for article in response.json()["articles"]] == ["A1"]


def test_a_valid_token_gets_the_article(client: TestClient, valid_token: str) -> None:
    response = _article(client, _bearer(valid_token))

    assert response.status_code == 200
    assert response.json()["ref"] == "A1"


def _assert_rejected(response: Any) -> None:
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("route", _ROUTES, ids=["POST /query", "GET /articles/{ref}"])
def test_a_request_without_a_token_is_rejected(
    client: TestClient, route: Callable[[TestClient, dict[str, str]], Any]
) -> None:
    _assert_rejected(route(client, {}))


@pytest.mark.parametrize("route", _ROUTES, ids=["POST /query", "GET /articles/{ref}"])
def test_a_non_bearer_authorization_header_is_rejected(
    client: TestClient, route: Callable[[TestClient, dict[str, str]], Any]
) -> None:
    _assert_rejected(route(client, {"Authorization": "Basic ZGVtbzpkZW1v"}))


@pytest.mark.parametrize("route", _ROUTES, ids=["POST /query", "GET /articles/{ref}"])
@pytest.mark.parametrize("make_token", INVALID_TOKENS.values(), ids=INVALID_TOKENS.keys())
def test_an_invalid_token_is_rejected_without_echoing_it(
    client: TestClient,
    tokens: TokenFactory,
    untrusted_tokens: TokenFactory,
    route: Callable[[TestClient, dict[str, str]], Any],
    make_token: Callable[[TokenFactory, TokenFactory], str],
) -> None:
    token = make_token(tokens, untrusted_tokens)

    response = route(client, _bearer(token))

    _assert_rejected(response)
    assert response.json() == {"detail": "Invalid or missing access token."}
    assert token not in response.text


@pytest.mark.parametrize("realm_access", ["not-a-dict", {"roles": "not-a-list"}])
def test_a_valid_token_with_malformed_roles_is_still_authenticated(
    client: TestClient, tokens: TokenFactory, realm_access: object
) -> None:
    response = _article(client, _bearer(tokens.sign(realm_access=realm_access)))

    assert response.status_code == 200


def test_health_needs_no_token() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
