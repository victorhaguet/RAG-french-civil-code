# OAuth resource server with Keycloak

Status: accepted

Nothing asked who was calling the API. `POST /query` and the `ask_code_civil` tool both spend the
owner's OpenAI key, and only the Compose port's `127.0.0.1` binding kept other machines out: open
the port on a VPS or a LAN, and anyone on the network could run up the bill. There was also no
notion of a user, so the server couldn't decide who may run the paid operations. We added OAuth2 /
OpenID Connect, and these decisions shaped it.

**1. The API is a resource server only, and Keycloak is the authorization server.** Users log in
to Keycloak, never to the API. Keycloak issues short-lived signed JWT Access Tokens, and the API
only checks them: it never sees a password and never issues a token. Keycloak runs as a service in
the Compose stack, with its realm (clients, role, audience mapper, demo users) imported from
`keycloak/code-civil-realm.json` on every start, so `docker compose up` gives a working login with
no setup in the admin console. One verifier, `src.auth.tokens.TokenVerifier`, serves both the REST
routes and the MCP Server, which only adapts it to the MCP SDK's `TokenVerifier` protocol.

**2. Clients are pre-registered public clients, using authorization code + PKCE.** The realm
declares `claude-code` (for Claude Code's MCP login) and `swagger-ui` (for the "Authorize" button
on `/docs`). Neither has a secret, since a CLI or a browser page can't keep one: PKCE protects the
code exchange instead. Each client's redirect URIs are fixed in the realm, and each carries an
audience mapper that puts the API's resource URL in the tokens' `aud`.

**3. Authorization is a realm role, `code-civil-query` (the Query Role).** The operations that
spend the OpenAI key (`POST /query`, `ask_code_civil`) require it. The free, read-only ones
(`GET /articles/{ref}`, `get_article`) only require a valid Access Token. A missing or invalid
token gets a `401`; a valid token without the role gets a `403` on REST, or a tool error over MCP.
The demo users show the difference: `demo` holds the role, `reader` doesn't.

**4. There is no switch to turn authentication off.** A fresh clone, a test run against a real
server, and a deployment are all protected by default; there's no flag to forget in production.
`/health` (the Docker healthcheck) and the Protected Resource Metadata (which a client reads
before it has a token) stay public, as they must, and so does Swagger UI (`/docs`,
`/openapi.json`), which only describes the API and logs in through Keycloak.

**5. Tokens are checked locally, not introspected.** The API checks each token's RS256 signature
against Keycloak's published keys (JWKS, cached by `PyJWKClient`), plus `exp`, `iss`, `aud` and
`sub`. It never calls Keycloak per request, so a token stays valid until it expires even if the
user logs out or loses the role. That's acceptable because Access Tokens live 5 minutes: a revoked
user keeps access for at most that long, and refresh tokens, which Keycloak does check, can't
renew it. Claude Code renews tokens silently with its refresh token; Swagger asks the user to
authorize again.

**6. The MCP SDK's `validate_token_resource` is off.** The SDK's own audience check expects RFC
8707 resource indicators, which Keycloak doesn't put in its tokens by default. The shared verifier
already checks `aud` against `OAUTH_AUDIENCE`, the same URL `/mcp` announces as its resource in its
Protected Resource Metadata, so the SDK's check would add nothing but a login that always fails.

**7. The stdio MCP Server and the in-process evaluation stay unauthenticated.** Neither has a
network listener or a token: the stdio server is a subprocess of the user's own Claude Code, and
an in-process `scripts/evaluate.py` run calls the app inside its own process, as a fixed trusted
identity holding the Query Role. Only a live evaluation (`--base-url`) sends a token, read from
`EVAL_ACCESS_TOKEN`.

## Considered options

- **The MCP SDK's built-in authorization server (the API issues its own tokens).** Rejected: the
  API would then store users and passwords, render a login page and manage signing keys, all
  security-critical code a dedicated authorization server already provides. A resource server
  only checks tokens, and the same tokens work for REST and MCP.
- **A hosted provider (Auth0, Okta, Entra ID, ...).** Rejected: it needs an account and network
  access before the stack can even start, and ties the repo to one vendor. Keycloak runs in the
  Compose stack, offline, with its whole configuration committed.
- **Dynamic client registration (RFC 7591) for Claude Code.** Rejected: Keycloak's anonymous
  registration is restricted by client registration policies (trusted hosts, allowed redirect
  URIs), so it fails or needs policies loosened to accept any caller. A pre-registered public
  client works reliably, at the cost of a fixed callback port (`33418`) the user passes to
  `claude mcp add`.
- **OAuth scopes instead of a realm role.** Rejected: a scope says what a *client* may ask for on
  the user's behalf, not what the *user* may do. Here the question is whether this user may spend
  the key, which is a role, and it is granted per user in Keycloak. The MCP verifier reports no
  scopes and carries the roles in their own field.
- **An environment variable to disable authentication for local development.** Rejected: it is
  one forgotten setting away from an open deployment. Local development logs in to the local
  Keycloak like anyone else, and tests trust a fixed key pair through the verifier's key-provider
  seam.
- **Token introspection (RFC 7662) on every request.** Rejected: one network call to Keycloak per
  request, and the API goes down whenever Keycloak does. With 5-minute tokens, local checks only
  delay a revocation by at most that long.

## Follow-ups

- **TLS for deployment.** The stack serves plain HTTP on localhost. Anywhere else, tokens and
  logins need HTTPS in front of both the API and Keycloak, with public URLs replacing `localhost`
  in `KC_HOSTNAME`, `OAUTH_ISSUER`, `OAUTH_JWKS_URL`, `OAUTH_AUDIENCE`, the clients' redirect URIs
  and audience mapper, and the MCP transport's localhost-only `Host`/`Origin` check.
- **Per-user quotas or rate limiting.** The Query Role decides who may spend the key, not how
  much: a user holding it can still run up the bill.
- **A Keycloak client for the planned UI**, a public client with authorization code + PKCE like
  `swagger-ui`, with the UI's own redirect URIs and the audience mapper.
