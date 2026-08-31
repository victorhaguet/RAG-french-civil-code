# Keyword Index is built over Articles, not Chunks

Status: accepted

The vector index searches Chunks — sub-portions of an Article's `texte`, sized for embedding
quality (`CHUNK_SIZE = 800`). The new BM25 Keyword Index could have mirrored that granularity, but
we decided to index each Article's full `texte` directly instead.

Chunk boundaries exist for embedding-quality reasons (a fixed context window) that don't apply to
BM25: BM25 has no context-window constraint, and its own document-length normalization (`b`
parameter of the Okapi formula) already handles the corpus's Article-size variance without needing
artificial splitting. Indexing at Article granularity also means the Keyword Index and the
`ArticleStore` share one identity (`ref`), so its ranked results plug directly into Reciprocal Rank
Fusion against the vector index's per-Article ranking (itself already deduplicated from Chunks to
`ref`s) with no extra translation step.

Benchmarked on the current corpus (2,885 Articles / 3,211 Chunks): building and querying the index
costs the same (~100ms) either way, so there was no performance case for chunking.

## Considered options

- **Index Chunks, like the vector index.** Rejected: no BM25-specific benefit, and it would force
  Chunk-to-Article deduplication on the Keyword Index side too, adding complexity that Article-level
  indexing avoids entirely.
