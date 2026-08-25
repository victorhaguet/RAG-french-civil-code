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
A sub-portion of an Article's `texte`, produced by splitting when the text exceeds 800 characters.
Its vectorstore document ID is `{ref}#{n}`, where `n` is its 0-based position within the Article.
Every Chunk carries a copy of its Article's full metadata. An Article that doesn't need splitting
is stored as a single Chunk (`{ref}#0`).
_Avoid_: Segment, piece, split

**etat**:
The source dataset's validity-status field for an Article. Only two values occur in this dataset:
`VIGUEUR` (currently in force) and `ABROGE_DIFF` (repealed, but the repeal takes effect at a future
date — not yet actually repealed). Ingestion keeps `VIGUEUR` only.
_Avoid_: Status, state (when referring to this specific field)
