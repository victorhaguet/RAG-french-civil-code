"""Verifying Access Tokens: JWTs issued by the authorization server (Keycloak).

Framework-agnostic, so the REST routes and the MCP Server share one verifier.
Its single seam is the signing-key provider: in production a `PyJWKClient` on
the authorization server's JWKS URL, in tests a fixed RSA public key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

# RS256 only: accepting any other algorithm opens the door to `alg: none`
# tokens, or to HS256 tokens "signed" with the public key as a shared secret.
_ALGORITHMS = ["RS256"]

_REQUIRED_CLAIMS = ["exp", "iss", "aud", "sub"]


@dataclass(frozen=True)
class Identity:
    """The authenticated caller, as the Access Token describes them."""

    subject: str
    username: str | None
    roles: frozenset[str]
    expires_at: int


class SigningKeyProvider(Protocol):
    """Finds the public key a token's signature must be checked against."""

    def get_signing_key(self, token: str) -> Any: ...


class FixedKeyProvider:
    """Always the same public key, whatever the token's `kid`."""

    def __init__(self, public_key: RSAPublicKey) -> None:
        self._public_key = public_key

    def get_signing_key(self, token: str) -> RSAPublicKey:
        del token  # the same key, whatever the token
        return self._public_key


class JwksKeyProvider:
    """The authorization server's published keys, picked by the token's `kid`.

    `PyJWKClient` caches the key set for 5 minutes. On an unknown `kid` (e.g.
    after a key rotation) it fetches the set again, but at most once every
    30 seconds, so tokens with made-up `kid`s can't flood Keycloak.
    """

    def __init__(self, jwks_url: str) -> None:
        self._client = jwt.PyJWKClient(jwks_url)

    def get_signing_key(self, token: str) -> Any:
        return self._client.get_signing_key_from_jwt(token).key


class TokenVerifier:
    """Checks an Access Token's signature, expiry, issuer and audience."""

    def __init__(
        self,
        key_provider: SigningKeyProvider,
        issuer: str,
        audience: str,
    ) -> None:
        self._key_provider = key_provider
        self._issuer = issuer
        self._audience = audience

    def verify(self, token: str) -> Identity | None:
        """The caller's Identity, or `None` for any invalid token.

        Args:
            token (str): the raw bearer token

        Returns:
            Identity | None: `None` whatever the reason, so no detail about the
                token can leak to the caller
        """
        try:
            claims = jwt.decode(
                token,
                self._key_provider.get_signing_key(token),
                algorithms=_ALGORITHMS,
                issuer=self._issuer,
                audience=self._audience,
                options={"require": _REQUIRED_CLAIMS},
            )
        # PyJWKClientError (unknown `kid`, unreachable JWKS URL) is a PyJWTError too.
        except jwt.PyJWTError:
            return None
        return Identity(
            subject=claims["sub"],
            username=claims.get("preferred_username"),
            roles=_realm_roles(claims),
            expires_at=claims["exp"],
        )


def _realm_roles(claims: dict[str, Any]) -> frozenset[str]:
    """Keycloak's realm roles (`realm_access.roles`), or none if malformed.

    Read defensively: a validly signed token with an unexpected shape there
    must not crash the request, just carry no role.
    """
    realm_access = claims.get("realm_access")
    roles = realm_access.get("roles") if isinstance(realm_access, dict) else None
    if not isinstance(roles, list):
        return frozenset()
    return frozenset(role for role in roles if isinstance(role, str))
