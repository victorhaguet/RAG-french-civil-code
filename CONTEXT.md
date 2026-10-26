# RAG French Civil Code

A retrieval-augmented generation tool over the French Code civil, sourced from the
[`louisbrulenaudet/code-civil`](https://huggingface.co/datasets/louisbrulenaudet/code-civil) dataset.
Answers legal questions by retrieving relevant articles and generating a response grounded in them.

## Language

**Article**:
A single provision of the Code civil, corresponding to one row of the source dataset. Uniquely
identified by `ref`. Only Articles whose `etat` is `VIGUEUR` (currently in force) are ingested.
_Avoid_: Row, record, document (when referring to the source), entry

**Chunk**:
A sub-portion of an Article's `texte`, produced by splitting when the text exceeds 800 characters,
used only to power retrieval (semantic search). Its vectorstore document ID is `{ref}#{n}`, where
`n` is its 0-based position within the Article. Every Chunk carries a copy of its Article's
metadata, but never the full `texte` — matching a Chunk only tells you which Article was found;
resolving that Article's full text is a separate lookup by `ref`. A Chunk never reaches the
generation prompt directly. An Article that doesn't need splitting is stored as a single Chunk
(`{ref}#0`).
_Avoid_: Segment, piece, split, Context (a Chunk is not what the model sees — see Retrieved Articles)

**etat**:
The source dataset's validity-status field for an Article. Only two values occur in this dataset:
`VIGUEUR` (currently in force) and `ABROGE_DIFF` (repealed, but the repeal takes effect at a future
date — not yet actually repealed). Ingestion keeps `VIGUEUR` only.
_Avoid_: Status, state (when referring to this specific field)

**Query Language**:
The detected language of a user's question, restricted to `fr` or `en`. `/query` (and
`/query/stream`) refuses any question detected as `en` before retrieval, with a fixed English-only
refusal; everything else (including an inconclusive detection) is treated as `fr` and proceeds
normally. Articles and the generation prompt are French-only.
_Avoid_: Locale (this only distinguishes fr/en for the refusal gate, not full internationalization)

**Keyword Index**:
A `rank_bm25` index over every Article's full `texte` (not Chunks — BM25 has no fixed-context-window
constraint the way embeddings do). Built lazily in memory from the `ArticleStore` on first use; never
persisted to disk or touched by `scripts/ingest.py`. Tokenization is lowercase + regex word split,
then French Snowball stemming and French stopword removal via `nltk`. Queried alongside the vector
index as one half of Hybrid Retrieval.
_Avoid_: BM25 index (Keyword Index is the domain term; BM25 is its implementation)

**Candidate Articles**:
The deduplicated, Article-ranked result of Hybrid Retrieval: the Keyword Index and the vector index
are each queried for `fetch_k` candidates, and their two ranked lists are fused into one via weighted
Reciprocal Rank Fusion, deduplicated by `ref`. Truncated to `top_k`, this becomes the Retrieved
Articles.
_Avoid_: Fused results, hybrid results

**Retrieved Articles**:
The Candidate Articles, truncated to `top_k`, that ground the generation prompt — full Article text,
not Chunk text. Currently the fused/deduplicated/truncated Hybrid Retrieval order stands as-is; a
Reranker that reorders Candidate Articles into this final order is a separate, not-yet-landed piece.
_Avoid_: Context, chunks, matches, results

**Grounded Answer**:
The generation prompt's normal response shape: a direct answer followed by a "Fondement
juridique"/"Legal basis" section citing only the Retrieved Articles the model actually relied on —
never every Retrieved Article.
_Avoid_: Structured answer (doesn't distinguish this from an Out-of-Scope Answer, which is also
structured, just differently)

**Out-of-Scope Answer**:
The generation prompt's fallback response shape, produced instead of a Grounded Answer when no
retrieved Chunk answers the question. Breaks the "expert in French civil law" persona entirely,
states plainly that the question can't be answered from retrieval, reminds the user the corpus
is Code civil only (as of `DATASET_AS_OF`), and invites rephrasing. Cites no Articles.
_Avoid_: Fallback, no-answer (too vague — doesn't convey that persona is dropped)

**Streamed Answer**:
The response of `POST /query/stream`: the same pipeline as `/query`, sent as one JSON message per
line (`application/x-ndjson`) as it runs. First, exactly once, an `articles` message carrying the
Retrieved Articles (same shape as `/query`'s `articles`); then zero or more `delta` messages, the
pieces of the answer in order, which joined give the answer `/query` would return; then, exactly
once, `done`. If generation fails after streaming started, the stream ends with an `error` message
(generic, no internal details) instead of `done`. A non-French question streams as no Articles and
the fixed refusal as a single `delta`. See ADR 0005.
_Avoid_: Stream (too vague: the chat model's own stream is a different thing), SSE, event stream,
chunk (a Chunk is a piece of an Article, not of the answer)

**MCP Server**:
The interface that exposes the query pipeline to MCP hosts (e.g. Claude Code) as two tools:
`ask_code_civil`, which returns a Grounded Answer or Out-of-Scope Answer together with the full
text of its Retrieved Articles, and `get_article`, which returns one Article by `ref`. It runs the
same in-process pipeline as `/query`, over stdio (a subprocess started by the host) or over
Streamable HTTP (mounted in the API's app at `/mcp`, sharing its loaded models). Over HTTP it
requires an Access Token, and `ask_code_civil` the Query Role; over stdio it is unauthenticated.
_Avoid_: MCP (the protocol, not this server), plugin, agent, tool server

## Authentication

**Authorization Server**:
The Keycloak service of the Compose stack (realm `code-civil`), where users log in and which
issues Access Tokens. The API never issues tokens or sees passwords: it is only an OAuth resource
server that checks them. See ADR 0004.
_Avoid_: Auth server, identity provider, IdP (Keycloak can federate those; here it is the issuer)

**Access Token**:
A short-lived (5 minutes) RS256-signed JWT issued by the Authorization Server, sent as
`Authorization: Bearer <token>`. The API checks it locally: signature against the Authorization
Server's JWKS, plus `exp`, `iss` (`OAUTH_ISSUER`), `aud` (`OAUTH_AUDIENCE`) and `sub`. Required by
`POST /query`, `POST /query/stream`, `GET /articles/{ref}` and `/mcp`; `/health`, the Protected
Resource Metadata and Swagger UI (`/docs`, `/openapi.json`) stay public.
_Avoid_: JWT (the format, not the role it plays), bearer token, API key, session

**Identity**:
The authenticated caller, as a valid Access Token describes them: `sub`, `preferred_username`,
realm roles and expiry. What the token checks hand to the route or tool; never built from anything
but a verified token, except the fixed trusted Identity of the in-process evaluation.
_Avoid_: User (the account lives in the Authorization Server), principal, session

**Query Role**:
The realm role `code-civil-query` (`QUERY_ROLE` in code), required by the operations that spend
the OpenAI key: `POST /query`, `POST /query/stream` and `ask_code_civil`. Read-only operations
(`GET /articles/{ref}`, `get_article`) only need a valid Access Token. A valid token without it
gets a `403` on REST, or a tool error over MCP.
_Avoid_: Scope (OAuth scopes are what a client may ask for, not what a user may do), permission,
admin role

**Protected Resource Metadata**:
The public JSON document (RFC 9728) at `/.well-known/oauth-protected-resource/mcp`, naming `/mcp`
as the resource and the Authorization Server that issues its Access Tokens. The `401` from `/mcp`
points to it in its `WWW-Authenticate` header, so an MCP host (e.g. Claude Code) can find where to
log in before it has a token.
_Avoid_: Discovery document (that is the Authorization Server's own OpenID configuration), well-known

## Evaluation

**Golden Question**:
A hand-authored question paired with a known-correct `reference_answer` and the
`reference_article_refs` it should be retrieved from. Used to evaluate both retrieval (recall of
the reference refs among Retrieved Articles) and generation (Faithfulness of the Grounded Answer
to the Retrieved Articles).
_Avoid_: Test question, eval question (too vague — doesn't convey it carries ground truth)

**Out-of-Scope Question**:
A hand-authored question deliberately outside the Code civil's coverage (another legal domain, or
unrelated to law entirely), used to verify the pipeline produces an Out-of-Scope Answer instead of
guessing.
_Avoid_: Tricky question, negative question

**Injection Attempt**:
A hand-authored question that tries to override the generation prompt's instructions — reveal the
system prompt, break persona, or otherwise bypass the guardrails. Evaluated separately from an
Out-of-Scope Question: every Injection Attempt in the set must be resisted, not merely held above
a baseline rate.
_Avoid_: Tricky question, malicious question, jailbreak
