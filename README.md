# RAG French Civil Code

A retrieval-augmented generation API over the French Code civil. It retrieves the
relevant articles for a legal question and generates an answer grounded in their text,
citing every article it relied on.

The corpus comes from the [`louisbrulenaudet/code-civil`](https://huggingface.co/datasets/louisbrulenaudet/code-civil)
dataset on HuggingFace, filtered to articles currently in force, as a snapshot taken on
21 September 2025. The app only answers questions that the Code civil itself can
address — it won't cover other codes, case law, or repealed provisions — and when a
question falls outside what it retrieved, it says so plainly instead of guessing.

## Setup

**Prerequisites:** Python 3.12.

Install [uv](https://docs.astral.sh/uv/), the package manager this project uses:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo, then install the dependencies. `uv sync` creates a `.venv` for you and
`uv run` uses it automatically, so there's no manual virtualenv activation needed.

`torch` has no default build: pick `--extra cu124` (NVIDIA GPU, CUDA 12.4+ driver) or
`--extra cpu` (no GPU, or a GPU this project doesn't support) explicitly — see "Embedding
model" below for the tradeoff. Omitting both doesn't fail, it silently resolves to PyPI's
newest `torch` build instead, which is its own (different, larger) CUDA download:

```bash
uv sync --extra cu124    # or --extra cpu
```

Copy the example environment file and fill in your OpenAI API key:

```bash
cp .env.example .env
```

`.env` also lets you point `OPENAI_BASE_URL` at any OpenAI-compatible endpoint instead
of OpenAI itself, and change `OPENAI_MODEL` from the default (`gpt-4o-mini`).

## Embedding model

Articles and questions are embedded with
[`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
by default, one of the strongest models in the E5 family on the
[MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard for French and multilingual
retrieval. It's a ~560M-parameter model, so it wants a GPU to be fast — on an RTX 2050 (4GB
VRAM) it embeds a query in well under 50ms and re-ingests this repo's whole corpus (~2,900
articles) in under a minute. It still runs on CPU, just slowly.

**No GPU?** Install with `uv sync --extra cpu` (see "Setup" above), and set `EMBEDDING_MODEL`
in `.env` to [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
instead — a much smaller model in the same family that runs comfortably on CPU, at some cost
to retrieval quality. Only these two models are supported out of the box; using a different
embedding model family means editing `src/retrieval/embeddings.py`, since each family has its
own prefixing convention.

Switching `EMBEDDING_MODEL` requires re-running `scripts/ingest.py` — Chroma stores fixed-size
vectors per collection, so a model change makes the existing vector store incompatible.

## Running it

Build the vector store and the article database from the dataset (run once, or again
whenever you want to refresh the corpus):

```bash
uv run scripts/ingest.py
```

Then start the API:

```bash
uv run uvicorn src.api.app:app --reload
```

The server listens on `http://127.0.0.1:8000` by default.

## Running with Docker

Alternatively, run the API in a self-contained, CPU-only container — no local Python, uv or
`data/` needed, only Docker with the Compose plugin:

```bash
docker compose up --build
```

The first build runs the ingestion inside the image (downloading the dataset and the models,
then embedding every article on CPU), so it takes a few minutes; later builds reuse the cached
layers. The image uses `intfloat/multilingual-e5-small` regardless of `EMBEDDING_MODEL` in your
`.env`, and downloads nothing at runtime.

The stack has two services: the API, and [Keycloak](https://www.keycloak.org), the
authorization server that issues the Access Tokens the API requires (see "Log in from Swagger"
below). Keycloak's admin password comes from `.env`: set `KEYCLOAK_ADMIN_PASSWORD` there (see
`.env.example`), since Compose refuses to start without it.

`.env` is also loaded into the API container at runtime (it's never copied into the image), and
an `OPENAI_API_KEY` exported in your shell takes precedence over it. Without the key, the
container still starts and serves `/health` and `/articles/{ref}` — only `/query` needs it. The
API is published on `http://127.0.0.1:8000` and Keycloak on `http://localhost:8080`, both on
localhost only, and the containers restart automatically unless you stop them
(`docker compose down`).

### Log in from Swagger

`/query` and `/articles/{ref}` require an Access Token issued by Keycloak. To try them by hand,
open `http://localhost:8000/docs`, click **Authorize**, then **Authorize** again in the dialog
(the client ID `swagger-ui` is pre-filled, no secret needed). Log in on Keycloak's page, and
Swagger sends the token with every request you try. Keycloak's realm, `code-civil`, is imported
from [`keycloak/code-civil-realm.json`](keycloak/code-civil-realm.json) on every start, so no
setup is needed in the admin console.

Two demo users exist:

| User     | Password | `POST /query`                          | `GET /articles/{ref}` |
| -------- | -------- | -------------------------------------- | --------------------- |
| `demo`   | `demo`   | `200`: has the `code-civil-query` role | `200`                 |
| `reader` | `reader` | `403`: lacks the role                  | `200`                 |

**These users are for local development only.** Their passwords are public (they're in the
realm file): remove them before running Keycloak anywhere else.

Access Tokens expire after 5 minutes, and Swagger doesn't renew them: after a `401`, click
**Authorize** again. Keycloak remembers your browser session, so you won't retype the password.
To switch users, log out first, via **Logout** in the dialog and at
`http://localhost:8080/realms/code-civil/account`.

## Usage

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quand une loi entre-t-elle en vigueur ?"}'
```

```json
{
  "answer": "...",
  "articles": [
    {
      "ref": "...",
      "sectionParentTitre": "Titre préliminaire : De la publication, des effets et de l'application des lois en général"
    }
  ]
}
```

Questions must be asked in French; English questions are refused without retrieval. The reason is that the retrieval pipeline is using BM25 for hybrid retrieval and this can only work when both data language and query language are the same. Fetch the full text of a cited article by
its `ref`:

```bash
curl http://127.0.0.1:8000/articles/{ref}
```

Interactive API docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`; to call the
protected routes from there, see "Log in from Swagger".

## Use it from Claude Code

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open protocol that
lets an AI application (the *host*, here Claude Code) call tools exposed by an external
*server*. The host discovers each tool's name, description and input schema, decides on its own
when a tool is relevant, calls it, and reads the structured result. This repo ships an MCP server
that exposes the same query pipeline as `/query`, so Claude can answer a French civil-law
question from the Code civil while you work, without leaving the terminal.

The server exposes two tools:

- **`ask_code_civil(question, top_k)`**: runs the full pipeline (Query Language gate, Hybrid
  Retrieval, reranking, generation) and returns the Grounded Answer (or Out-of-Scope Answer)
  together with the Retrieved Articles it was grounded in, each with its `ref`,
  `sectionParentTitre` and **full text**, so Claude can check the answer against the sources.
  `question` must be in French: the tool description tells Claude to translate it first, and a
  question detected as English gets the same fixed refusal as `/query`. `top_k` is optional
  (default `DEFAULT_TOP_K`, must be > 0).
- **`get_article(ref)`**: returns one full Article by its `ref`, so Claude can follow a
  reference made in an answer to an Article that wasn't retrieved. An unknown `ref`
  returns a tool error ("Article not found."), like the REST 404.

It runs over two transports.

### stdio (local)

Claude Code starts the server itself, as a subprocess, with `uv run python -m src.code_civil_mcp`.
The repo commits a [`.mcp.json`](.mcp.json) declaring it, so opening the repo in Claude Code
offers the `code-civil` server automatically (approve it when prompted). It needs no login:
there's no network listener, only a subprocess of your own Claude Code. To register it by hand
instead, from the repo root:

```bash
claude mcp add code-civil -- uv run python -m src.code_civil_mcp
```

Prerequisites, since the subprocess runs on your machine with your local data:

- the dependencies are installed (`uv sync`, see "Setup") and the ingestion has run locally
  (`uv run scripts/ingest.py`, see "Running it"): the server reads `data/` and never ingests;
- `OPENAI_API_KEY` is set in `.env` at the repo root, which the server loads on startup.

### Use the MCP Server from Claude Code over HTTP

The same server is mounted inside the FastAPI app at `/mcp`, over Streamable HTTP, so the
container that serves the REST API serves MCP too. Like `/query`, it requires an Access Token
issued by Keycloak, and Claude Code gets one for you with a browser login. Start the stack, then
register the server with Claude Code:

```bash
docker compose up --build
claude mcp add --transport http --client-id claude-code --callback-port 33418 \
  code-civil-http http://localhost:8000/mcp
```

Then run `/mcp` in Claude Code, select `code-civil-http` and authenticate (or run
`claude mcp login code-civil-http` from your shell). Your browser opens Keycloak's login page:
log in as one of the demo users (see "Log in from Swagger"). As `demo`, both tools work. As
`reader`, `get_article` works, and `ask_code_civil` returns a tool error saying the caller lacks
the `code-civil-query` role. To switch users, select **Clear authentication** in `/mcp` (or run
`claude mcp logout code-civil-http`), log out at
`http://localhost:8080/realms/code-civil/account`, then authenticate again.

What happens under the hood is the standard MCP authorization flow:

1. Claude Code's first request to `/mcp` has no token and gets a `401`, whose
   `WWW-Authenticate` header points to the server's Protected Resource Metadata.
2. That metadata (`/.well-known/oauth-protected-resource/mcp`) names Keycloak as the
   authorization server, and Claude Code reads Keycloak's endpoints from there.
3. Claude Code logs in through your browser with authorization code + PKCE, as the
   pre-registered public client `claude-code` (no secret; Keycloak's dynamic client registration
   isn't used). Keycloak redirects back to Claude Code's local callback,
   `http://localhost:33418/callback`, the only redirect URI the realm allows for this client.
4. Access Tokens expire after 5 minutes. Claude Code renews them silently with the refresh
   token, so you don't log in again.

Mind the details of the command:

- `--client-id claude-code` and `--callback-port 33418` must match the client in
  [`keycloak/code-civil-realm.json`](keycloak/code-civil-realm.json). There's no
  `--client-secret`: the client is public. If port 33418 is taken on your machine, change it in
  both places: `--callback-port` and the client's `redirectUris`.
- The URL must be `http://localhost:8000/mcp`, not `127.0.0.1`: it must equal the resource the
  metadata announces, which is also the audience Keycloak puts in the tokens.
- Add `--scope user` to make the server available in every project rather than only the current
  one.

The endpoint is published on localhost only, and it accepts only localhost `Host`/`Origin`
headers, so a web page can't drive it from your browser (DNS-rebinding protection).
`uv run uvicorn src.api.app:app` serves `/mcp` the same way, without Docker (Keycloak must still
run, e.g. `docker compose up keycloak`).

### Things to know

- **The first question is slow.** Starting the server loads no model: the embedding model, the
  Reranker and the Keyword Index load on the first `ask_code_civil` call, so model loading never
  counts against Claude Code's startup timeout, and stay loaded afterwards. With stdio, that
  happens once per Claude Code session, since each session starts its own subprocess. With
  HTTP, it happens once per container lifetime, and the REST API and MCP share the same loaded
  models.
- **Every `ask_code_civil` call spends OpenAI tokens**, exactly like `/query`. `get_article` is
  a database lookup and calls no model.

## Evaluation

`scripts/evaluate.py` scores the real pipeline, driven exclusively through `/query`, against
three hand-authored question sets in `eval/`: Golden Questions (RAGAS `faithfulness` and
`context_recall`), Out-of-Scope Questions, and Injection Attempts (both scored as a binary
guardrail-pass rate). See CONTEXT.md's Evaluation section for what each set is.

Install the `eval` dependency group first — kept separate from `dev` since it pulls in RAGAS's
heavy dependency chain (`langchain`, `langgraph`, ...), which the app itself never needs.
Repeat whichever `--extra` you used in Setup — `uv sync` reconciles the venv to exactly the
extras/groups given each time, so omitting it here would swap your GPU/CPU `torch` build back
out:

```bash
uv sync --extra cu124 --group eval    # or --extra cpu
```

Then run it, either in-process (real dependency wiring, no separate server needed) or against a
running server:

```bash
uv run scripts/evaluate.py
uv run scripts/evaluate.py --base-url http://127.0.0.1:8000
```

The `eval/*.jsonl` files ship empty — every metric reports "no data" until you populate them, one
JSON object per line. Golden Questions need `question`, `reference_answer`, and
`reference_article_refs`; Out-of-Scope Questions and Injection Attempts need only `question`.

Every run also writes `eval/results.md`, a Markdown report with one section per question — its
generated answer, retrieved articles, and score or pass/fail — not just the printed aggregates.
It's regenerated on every run and never committed.

`--mode` selects what happens with the computed scores, on top of the default `report` printout:

```bash
uv run scripts/evaluate.py --mode gate              # compare against eval/baseline.json, exit non-zero on regression
uv run scripts/evaluate.py --mode gate --update-baseline    # same, and on pass overwrite eval/baseline.json with the computed scores
```

`gate` mode fails if `faithfulness`, `context_recall`, or `out_of_scope_guardrail_rate` drop below
`eval/baseline.json`'s corresponding value, or if `injection_guardrail_rate` is anything less than
`1.0` (checked unconditionally, independent of the baseline). A metric with no data — its question
set is still empty — is skipped rather than counted as a pass or a failure. `--update-baseline`
(with `--mode gate`) runs in the same evaluation, only after the gate passes, and overwrites
`eval/baseline.json` with the computed scores.

## License

MIT — see [LICENSE](LICENSE).
