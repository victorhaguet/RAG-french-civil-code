"""Adapts the shared Access Token verifier to the MCP SDK's `TokenVerifier` protocol.

The SDK's bearer middleware calls it on every HTTP request to the MCP Server;
the checks themselves (signature, expiry, issuer, audience) stay in
`src.auth.tokens`, shared with the REST routes.
"""

from __future__ import annotations

from collections.abc import Callable

import anyio
from mcp.server.auth.provider import AccessToken

from src.auth.tokens import TokenVerifier


class McpTokenVerifier:
    """Verifies bearer tokens for the MCP Server with the shared `TokenVerifier`.

    Takes a *provider* of the verifier, resolved on each request, like the
    server's other components: building the app touches no key set.
    """

    def __init__(self, verifier: Callable[[], TokenVerifier]) -> None:
        self._verifier = verifier

    async def verify_token(self, token: str) -> AccessToken | None:
        # In a worker thread: on a cache miss, verifying fetches Keycloak's
        # JWKS over the network, which must not block the event loop.
        identity = await anyio.to_thread.run_sync(self._verifier().verify, token)
        if identity is None:
            return None
        # `resource` stays unset: the verifier already checked the audience
        # (see `validate_token_resource` in `src.api.app`). Realm roles aren't
        # OAuth scopes, so none are reported.
        return AccessToken(
            token=token,
            client_id=identity.subject,
            subject=identity.subject,
            scopes=[],
            expires_at=identity.expires_at,
        )
