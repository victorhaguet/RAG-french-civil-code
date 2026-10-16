"""Signing test Access Tokens with a locally generated RSA key pair, instead of Keycloak."""

from __future__ import annotations

import time
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

from src import config


def generate_key_pair() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class TokenFactory:
    """Signs JWTs with `private_key`, defaulting to claims the verifier accepts."""

    def __init__(self, private_key: RSAPrivateKey) -> None:
        self.private_key = private_key

    @property
    def public_key(self) -> RSAPublicKey:
        return self.private_key.public_key()

    def claims(self, **overrides: Any) -> dict[str, Any]:
        """Valid claims, with `overrides` applied (a `None` value drops the claim)."""
        claims: dict[str, Any] = {
            "sub": "user-123",
            "preferred_username": "demo",
            "iss": config.OAUTH_ISSUER,
            "aud": config.OAUTH_AUDIENCE,
            "exp": int(time.time()) + 300,
            "realm_access": {"roles": ["code-civil-query"]},
        }
        claims.update(overrides)
        return {name: value for name, value in claims.items() if value is not None}

    def sign(self, **overrides: Any) -> str:
        return jwt.encode(self.claims(**overrides), self.private_key, algorithm="RS256")
