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
The detected language of a user's question — `fr` or `en`, defaulting to `fr` when detection is
inconclusive or the language is neither. Drives two independent choices: which instruction prefix
embeds the query for retrieval, and which language the generation prompt template is rendered in.
Articles themselves are always in French regardless of Query Language.
_Avoid_: Locale (this only distinguishes fr/en for query interpretation, not full internationalization)

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
never every Retrieved Article. Article citations stay in French even in the English template.
_Avoid_: Structured answer (doesn't distinguish this from an Out-of-Scope Answer, which is also
structured, just differently)

**Out-of-Scope Answer**:
The generation prompt's fallback response shape, produced instead of a Grounded Answer when no
retrieved Chunk answers the question. Breaks the "expert in French civil law" persona entirely,
states plainly that the question can't be answered from retrieval, reminds the user the corpus
is Code civil only (as of `DATASET_AS_OF`), and invites rephrasing. Cites no Articles.
_Avoid_: Fallback, no-answer (too vague — doesn't convey that persona is dropped)
