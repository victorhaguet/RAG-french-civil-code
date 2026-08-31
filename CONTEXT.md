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
inconclusive or the language is neither. Always drives which language the generation prompt
template is rendered in. Also drives the query's embedding instruction prefix, for embedding
models that use language-dependent instructions (the default); models with a fixed prefix
convention ignore it for embedding. Articles themselves are always in French regardless of Query
Language.
_Avoid_: Locale (this only distinguishes fr/en for query interpretation, not full internationalization)

**Retrieved Articles**:
The deduplicated set of Articles assembled for one query: one entry per unique `ref` among the
top_k retrieved Chunks, in first-seen (highest-relevance) order, even when several Chunks matched
the same Article. This — full Article text, not Chunk text — is what the generation prompt is
rendered from.
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
