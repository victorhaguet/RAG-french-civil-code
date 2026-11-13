# Streamed Answer as NDJSON on a separate endpoint

Status: accepted

`POST /query` runs the whole pipeline (Query Language gate, Hybrid Retrieval, reranking,
generation) before it returns anything, so a client waits several seconds with nothing to show.
Most of that time is generation, and the chat model can stream its answer. We added
`POST /query/stream`, which sends a Streamed Answer, and these decisions shaped it.

**1. A separate endpoint, and `/query` is unchanged.** `/query`'s one-shot JSON response is what
existing integrations and the evaluation (`scripts/evaluate.py`) rely on. A stream needs a different
response shape, and a client has to read it differently. A flag or content negotiation on `/query`
would give one route two response shapes. Both routes run the same framework-free steps of the
query pipeline (`retrieve`, then `generate` or `generate_stream`), so they can't drift apart. The
request body, the `top_k` default and validation, and the Access Token and Query Role checks are the
same.

**2. One JSON object per line (`application/x-ndjson`), read with `fetch`, not Server-Sent Events.**
The browser's `EventSource` can only send a `GET` and can't set an `Authorization` header, and the
endpoint needs both: a `POST` body like `/query`, and a bearer Access Token (ADR 0004). Once the
client reads the body with `fetch` anyway, SSE's `event:`/`data:` framing gives nothing that a
newline after each JSON object doesn't. Every message is a typed object (`articles`, `delta`,
`done`, `error`), parsed the same way.

**3. The Retrieved Articles come first, in one `articles` message.** They are known once retrieval
ends, before the chat model is called, so a client can prepare its citations while the answer is
generated. They have the same shape as `/query`'s `articles` (`ref` and `sectionParentTitre` only).
Retrieval runs before the response starts, so a retrieval failure is still a plain `500`. A
non-French question sends no Articles and the fixed refusal as a single `delta`, with no retrieval
and no chat model call, as on `/query`.

**4. A failure after the first byte is an `error` message inside the stream.** Once streaming
starts, the `200` status is already sent and can't be changed. So a generation failure ends the
stream with `{"type":"error","message":...}` instead of `done`. The message is generic and the
details only go to the server logs. A successful stream always ends with an explicit `done`, so a
client can tell a complete answer from a dropped connection. `401`, `403` and `422` still come back
as normal HTTP statuses, before any streaming.

**5. The MCP Server doesn't stream.** An MCP host only gets a tool result once it's complete, so
streaming `ask_code_civil` would make nothing appear sooner. It keeps calling `answer_query`.

## Considered options

- **Stream from `/query` itself, chosen by a request flag or the `Accept` header.** Rejected: one
  route with two response shapes is harder to document and to test, and every current consumer of
  `/query` would be one header away from a breaking change.
- **Server-Sent Events.** Rejected: `EventSource` can't `POST` or send the bearer token, and reading
  SSE over `fetch` would only add framing on top of the lines.
- **Send the Retrieved Articles last, with `done`.** Rejected: the client couldn't link citations
  until the whole answer was in, which is the wait this endpoint removes.
