/**
 * Where the Web UI logs in and what it calls. Localhost only, like the rest
 * of the Compose stack (see README, "Deploying beyond localhost").
 */

/** Keycloak, the Authorization Server, as the browser reaches it. */
export const KEYCLOAK_URL = "http://localhost:8080";
export const KEYCLOAK_REALM = "code-civil";
/** The realm's public client for this UI: authorization code + PKCE, no secret. */
export const KEYCLOAK_CLIENT_ID = "web-ui";

/** Where the API serves the Web UI: Keycloak sends the browser back here after logout. */
export const WEB_UI_PATH = "/ui/";

/** The Streamed Answer endpoint, on the same origin that serves the UI. */
export const QUERY_STREAM_URL = "/query/stream";

/**
 * Refresh the Access Token before an API call if it expires within this
 * many seconds, so it can't expire while the answer streams in.
 */
export const TOKEN_MIN_VALIDITY_SECONDS = 30;
