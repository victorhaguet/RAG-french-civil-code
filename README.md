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

Questions can be asked in French or English. Fetch the full text of a cited article by
its `ref`:

```bash
curl http://127.0.0.1:8000/articles/{ref}
```

Interactive API docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.

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
