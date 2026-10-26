"""`POST /query/stream`, driven through the real app over HTTP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.api.app import STREAM_ERROR_MESSAGE
from src.pipeline import NON_FRENCH_REFUSAL
from tests.app_wiring import client_for, ingest
from tests.factories import raw_row
from tests.fakes import FakeChatModel, FakeModel
from tests.tokens import TokenFactory

_ROWS = [raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR")]

_TWO_ROWS = [
    raw_row(ref="A1", texte="La loi applicable est la loi en vigueur.", etat="VIGUEUR"),
    raw_row(ref="A2", texte="La loi applicable au mariage est la loi civile.", etat="VIGUEUR"),
]

_QUESTION = "Quand une loi entre-t-elle en vigueur ?"


def _client(tmp_path: Path, token: str, chat_model: FakeChatModel) -> TestClient:
    return client_for(*ingest(tmp_path, _ROWS), chat_model, token=token)


def _messages(response: Any) -> list[dict[str, Any]]:
    """The streamed response's messages, one JSON object per line."""
    return [json.loads(line) for line in response.iter_lines() if line]


def test_stream_sends_the_articles_then_the_answer_in_pieces_then_done(
    tmp_path: Path, valid_token: str
) -> None:
    chat_model = FakeChatModel(answer="Voici la réponse, en plusieurs morceaux.")
    client = _client(tmp_path, valid_token, chat_model)

    response = client.post("/query/stream", json={"question": _QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    messages = _messages(response)
    assert [message["type"] for message in messages] == (
        ["articles"] + ["delta"] * (len(messages) - 2) + ["done"]
    )
    deltas = [message["text"] for message in messages if message["type"] == "delta"]
    assert len(deltas) > 1
    assert "".join(deltas) == "Voici la réponse, en plusieurs morceaux."
    assert messages[0]["articles"] == client.post("/query", json={"question": _QUESTION}).json()[
        "articles"
    ]


def test_stream_refuses_an_english_question_without_retrieval_or_generation(
    tmp_path: Path, valid_token: str
) -> None:
    model = FakeModel()
    chat_model = FakeChatModel()
    client = client_for(*ingest(tmp_path, _ROWS, model), chat_model, token=valid_token)
    encodes_before = len(model.encode_calls)

    response = client.post("/query/stream", json={"question": "When does a law enter into force?"})

    assert response.status_code == 200
    assert _messages(response) == [
        {"type": "articles", "articles": []},
        {"type": "delta", "text": NON_FRENCH_REFUSAL},
        {"type": "done"},
    ]
    assert not chat_model.invoke_calls
    assert not chat_model.stream_calls
    assert len(model.encode_calls) == encodes_before


def test_stream_ends_with_an_error_when_the_chat_model_fails_midway(
    tmp_path: Path, valid_token: str
) -> None:
    chat_model = FakeChatModel(answer="Voici la réponse, en plusieurs morceaux.", fail_after=2)
    client = _client(tmp_path, valid_token, chat_model)

    response = client.post("/query/stream", json={"question": _QUESTION})

    assert response.status_code == 200
    messages = _messages(response)
    assert [message["type"] for message in messages] == ["articles", "delta", "delta", "error"]
    assert [message["text"] for message in messages[1:3]] == ["Voici ", "la "]
    assert messages[-1] == {"type": "error", "message": STREAM_ERROR_MESSAGE}
    assert FakeChatModel.FAILURE_DETAIL not in response.text


def test_stream_ends_with_an_error_when_the_chat_model_fails_before_any_piece(
    tmp_path: Path, valid_token: str
) -> None:
    client = _client(tmp_path, valid_token, FakeChatModel(fail_after=0))

    response = client.post("/query/stream", json={"question": _QUESTION})

    assert response.status_code == 200
    assert [message["type"] for message in _messages(response)] == ["articles", "error"]


def test_stream_without_a_token_is_rejected_before_streaming(
    tmp_path: Path, tokens: TokenFactory
) -> None:
    del tokens  # requested only so the app trusts the test key pair
    client = client_for(*ingest(tmp_path, _ROWS), FakeChatModel())

    response = client.post("/query/stream", json={"question": _QUESTION})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_stream_with_a_token_without_the_query_role_is_forbidden(
    tmp_path: Path, tokens: TokenFactory
) -> None:
    chat_model = FakeChatModel()
    client = _client(tmp_path, tokens.sign(realm_access={"roles": ["other"]}), chat_model)

    response = client.post("/query/stream", json={"question": _QUESTION})

    assert response.status_code == 403
    assert not chat_model.stream_calls


def test_stream_rejects_a_non_positive_top_k(tmp_path: Path, valid_token: str) -> None:
    client = _client(tmp_path, valid_token, FakeChatModel())

    response = client.post("/query/stream", json={"question": _QUESTION, "top_k": 0})

    assert response.status_code == 422


def test_stream_uses_a_default_top_k_when_none_is_supplied(tmp_path: Path, valid_token: str) -> None:
    client = client_for(*ingest(tmp_path, _TWO_ROWS), FakeChatModel(), token=valid_token)

    default = _messages(client.post("/query/stream", json={"question": "Quelle est la loi applicable ?"}))
    truncated = _messages(
        client.post("/query/stream", json={"question": "Quelle est la loi applicable ?", "top_k": 1})
    )

    # The default top_k is above the corpus size, so both Articles come back,
    # while a caller-supplied top_k still truncates.
    assert len(default[0]["articles"]) == 2
    assert len(truncated[0]["articles"]) == 1
