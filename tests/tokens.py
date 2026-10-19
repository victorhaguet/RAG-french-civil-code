"""Signing test Access Tokens with a locally generated RSA key pair, instead of Keycloak."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

from src import config


def generate_key_pair() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class TokenFactory:
    """Signs JWTs with `private_key`, defaulting to claims the verifier accepts.

    `kid`, when given, goes in each token's header and in the public `jwk()`,
    as Keycloak does, so a JWKS client can pick the key to verify with.
    """

    def __init__(self, private_key: RSAPrivateKey, kid: str | None = None) -> None:
        self.private_key = private_key
        self.kid = kid

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
        headers = {"kid": self.kid} if self.kid else None
        return jwt.encode(
            self.claims(**overrides), self.private_key, algorithm="RS256", headers=headers
        )

    def jwk(self) -> dict[str, Any]:
        """The public key as a JWK, as published in a JWKS."""
        jwk: dict[str, Any] = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.public_key))
        jwk.update(kid=self.kid, alg="RS256", use="sig")
        return jwk


def _hs256_token(tokens: TokenFactory) -> str:
    """Algorithm confusion: HMAC-signed with the *public* key as the shared secret.

    Built by hand, since PyJWT itself refuses to use a PEM key as an HMAC secret.
    """
    public_pem = tokens.public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )

    def segment(data: bytes) -> bytes:
        return base64.urlsafe_b64encode(data).rstrip(b"=")

    signing_input = (
        segment(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        + b"."
        + segment(json.dumps(tokens.claims()).encode())
    )
    signature = hmac.new(public_pem, signing_input, hashlib.sha256).digest()
    return (signing_input + b"." + segment(signature)).decode()


def _unsigned_token(tokens: TokenFactory) -> str:
    return jwt.encode(tokens.claims(), "", algorithm="none")


# Each builds a token the verifier must reject, from the trusted and an
# untrusted TokenFactory.
INVALID_TOKENS: dict[str, Callable[[TokenFactory, TokenFactory], str]] = {
    "garbage": lambda tokens, untrusted: "not-a-jwt",
    "expired": lambda tokens, untrusted: tokens.sign(exp=int(time.time()) - 60),
    "signed by another key": lambda tokens, untrusted: untrusted.sign(),
    "wrong iss": lambda tokens, untrusted: tokens.sign(iss="https://evil.example/realms/x"),
    "wrong aud": lambda tokens, untrusted: tokens.sign(aud="https://another.api.example"),
    "missing exp": lambda tokens, untrusted: tokens.sign(exp=None),
    "missing iss": lambda tokens, untrusted: tokens.sign(iss=None),
    "missing aud": lambda tokens, untrusted: tokens.sign(aud=None),
    "alg none": lambda tokens, untrusted: _unsigned_token(tokens),
    "HS256-signed": lambda tokens, untrusted: _hs256_token(tokens),
}
