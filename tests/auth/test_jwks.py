"""`JwksKeyProvider` against a real HTTP server publishing a JWKS, as Keycloak does."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from src import config
from src.auth.tokens import JwksKeyProvider, TokenVerifier
from tests.tokens import TokenFactory, generate_key_pair


class _JwksServer:
    """Serves a JWKS file at `url`, and counts how often it's fetched."""

    def __init__(self, directory: Path) -> None:
        self._file = directory / "certs"
        self.fetches = 0
        self.keys = []
        jwks_server = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=str(directory), **kwargs)

            def log_message(self, *args: Any) -> None:
                # Called once per request served: count it, keep the output quiet.
                del args
                jwks_server.fetches += 1

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._http.server_port}/certs"

    @property
    def keys(self) -> list[dict[str, Any]]:
        return list(json.loads(self._file.read_text())["keys"])

    @keys.setter
    def keys(self, keys: list[dict[str, Any]]) -> None:
        self._file.write_text(json.dumps({"keys": keys}))

    def __enter__(self) -> _JwksServer:
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._http.shutdown()
        self._http.server_close()


@pytest.fixture(name="jwks_server")
def _jwks_server(tmp_path: Path) -> Iterator[_JwksServer]:
    with _JwksServer(tmp_path) as server:
        yield server


@pytest.fixture(name="key_1", scope="module")
def _key_1() -> TokenFactory:
    return TokenFactory(generate_key_pair(), kid="key-1")


@pytest.fixture(name="key_2", scope="module")
def _key_2() -> TokenFactory:
    return TokenFactory(generate_key_pair(), kid="key-2")


def _verifier(url: str) -> TokenVerifier:
    return TokenVerifier(
        JwksKeyProvider(url), issuer=config.OAUTH_ISSUER, audience=config.OAUTH_AUDIENCE
    )


def test_a_token_is_checked_against_the_published_key_its_kid_names(
    jwks_server: _JwksServer, key_1: TokenFactory, key_2: TokenFactory
) -> None:
    jwks_server.keys = [key_1.jwk(), key_2.jwk()]
    verifier = _verifier(jwks_server.url)

    identity = verifier.verify(key_2.sign(sub="user-2"))

    assert identity is not None
    assert identity.subject == "user-2"


def test_the_published_keys_are_fetched_once_and_cached(
    jwks_server: _JwksServer, key_1: TokenFactory
) -> None:
    jwks_server.keys = [key_1.jwk()]
    verifier = _verifier(jwks_server.url)

    assert verifier.verify(key_1.sign()) is not None
    assert verifier.verify(key_1.sign()) is not None

    assert jwks_server.fetches == 1


def test_an_unknown_kid_right_after_a_fetch_does_not_refetch_the_keys(
    jwks_server: _JwksServer, key_1: TokenFactory, key_2: TokenFactory
) -> None:
    """PyJWKClient's cooldown: tokens with made-up `kid`s can't hammer Keycloak."""
    jwks_server.keys = [key_1.jwk()]
    verifier = _verifier(jwks_server.url)
    assert verifier.verify(key_1.sign()) is not None
    jwks_server.keys = [key_1.jwk(), key_2.jwk()]

    assert verifier.verify(key_2.sign()) is None
    assert jwks_server.fetches == 1


def test_a_key_rotated_in_is_picked_up_once_the_cooldown_has_passed(
    jwks_server: _JwksServer,
    key_1: TokenFactory,
    key_2: TokenFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jwks_server.keys = [key_1.jwk()]
    verifier = _verifier(jwks_server.url)
    assert verifier.verify(key_1.sign()) is not None
    jwks_server.keys = [key_1.jwk(), key_2.jwk()]

    # PyJWKClient's refetch cooldown is 30s by default: move its clock past it.
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 31)

    assert verifier.verify(key_2.sign()) is not None
    assert jwks_server.fetches == 2


def test_a_token_whose_kid_is_not_published_is_rejected(
    jwks_server: _JwksServer, key_1: TokenFactory, key_2: TokenFactory
) -> None:
    jwks_server.keys = [key_1.jwk()]

    assert _verifier(jwks_server.url).verify(key_2.sign()) is None


def test_a_token_signed_by_another_key_under_a_published_kid_is_rejected(
    jwks_server: _JwksServer, key_1: TokenFactory
) -> None:
    jwks_server.keys = [key_1.jwk()]
    impostor = TokenFactory(generate_key_pair(), kid="key-1")

    assert _verifier(jwks_server.url).verify(impostor.sign()) is None


def test_an_unreachable_jwks_url_rejects_the_token_instead_of_crashing(
    tmp_path: Path, key_1: TokenFactory
) -> None:
    with _JwksServer(tmp_path) as server:
        url = server.url
    # The server is shut down: nothing listens on its port any more.

    assert _verifier(url).verify(key_1.sign()) is None
