# Generation prompt is built from full Articles, not Chunks

Status: accepted

The generation prompt was being rendered directly from the retrieved Chunks' `page_content` —
partial Article text, sized for embedding quality (`CHUNK_SIZE = 800`), not for giving the model
full legal context. We decided the model should always see a matched Article's complete `texte`,
while Chunks stay exactly as they are today: a retrieval-only artifact that never itself reaches
the prompt.

Full Article text is not duplicated onto Chunk metadata in Chroma. Instead, ingestion also
populates a separate SQLite store (e.g. `data/articles.db`, mirroring the existing
`CHROMA_PERSIST_DIR` config pattern) mapping `ref -> Article`. At query time, the `ref`s of the
retrieved Chunks are deduplicated (one entry per unique Article, first-seen/highest-relevance
order), and each is resolved to its full Article via this store before the prompt is rendered.

`top_k` still means "Chunks searched," not "distinct Articles guaranteed" — after dedup, the
prompt may occasionally include fewer than `top_k` Articles. Only 8.7% of Articles in the current
corpus span multiple Chunks, so this collision is rare; over-fetching Chunks to guarantee an exact
count was rejected as complexity not worth it for a rare edge case.

The `/query` response's per-Article entries were also cut down to `ref` + `sectionParentTitre`
(no full text, no Chunk id), since a consumer can resolve the full Article via the new
`GET /articles/{ref}` endpoint, backed by the same SQLite store — the data is public, so exposing
it directly needs no auth.

## Considered options

- **Duplicate full `texte` onto every Chunk's metadata in Chroma.** Simplest to implement, but
  wastes space as the corpus grows (repeated once per Chunk of a multi-Chunk Article) and blurs
  Chunk's role as a retrieval-only artifact. Rejected in favor of a dedicated store, since the
  Article store is expected to outlive a single corpus (e.g. more French legal codes later).
- **Reconstruct full text by concatenating a ref's Chunks.** Rejected: `CHUNK_OVERLAP = 100` means
  naive concatenation would duplicate overlapping text.
