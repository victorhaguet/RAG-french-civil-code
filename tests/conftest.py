from collections.abc import Iterator

import pytest

from src import config
from src.api.app import app
from src.api.dependencies import get_token_verifier
from src.auth.tokens import FixedKeyProvider, TokenVerifier
from tests.tokens import TokenFactory, generate_key_pair


@pytest.fixture(autouse=True)
def _clear_dependency_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture(name="trusted_key_tokens", scope="session")
def _trusted_key_tokens() -> TokenFactory:
    # Session-scoped: generating an RSA key pair is slow enough to matter per test.
    return TokenFactory(generate_key_pair())


@pytest.fixture(name="untrusted_tokens", scope="session")
def _untrusted_tokens() -> TokenFactory:
    """Signs tokens with a second key pair the verifier doesn't trust."""
    return TokenFactory(generate_key_pair())


@pytest.fixture(name="tokens")
def _tokens(trusted_key_tokens: TokenFactory) -> TokenFactory:
    """Signs tokens the app trusts: its verifier checks them against this key pair."""
    verifier = TokenVerifier(
        FixedKeyProvider(trusted_key_tokens.public_key),
        issuer=config.OAUTH_ISSUER,
        audience=config.OAUTH_AUDIENCE,
    )
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    return trusted_key_tokens


@pytest.fixture(name="valid_token")
def _valid_token(tokens: TokenFactory) -> str:
    return tokens.sign()
