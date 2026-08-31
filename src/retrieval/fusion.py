"""Weighted Reciprocal Rank Fusion, combining ranked Article ref lists by rank position.

BM25 scores and cosine-similarity scores live on incomparable scales, so the
Keyword Index and vector index rankings are fused by rank position rather
than by raw score.
"""

from __future__ import annotations

from src import config


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float],
    k: int = config.RRF_K,
) -> list[str]:
    """Fuse several ranked lists of Article refs into one, via weighted RRF.

    Each list contributes `weight / (k + rank)` to a ref's fused score, where
    `rank` is the ref's 1-based position within that list (a ref missing from
    a list contributes nothing from it).

    Args:
        ranked_lists (list[list[str]]): one ranked list of Article refs per
            retrieval index, most relevant first.
        weights (list[float]): each list's weight, same length and order as
            `ranked_lists`.
        k (int): RRF's rank-damping constant.

    Returns:
        list[str]: fused Article refs, deduplicated, most relevant first.
            Ties are broken by first-seen order across `ranked_lists`.
    """
    scores: dict[str, float] = {}
    first_seen_order: list[str] = []
    for ranked_list, weight in zip(ranked_lists, weights, strict=True):
        for rank, ref in enumerate(ranked_list, start=1):
            if ref not in scores:
                scores[ref] = 0.0
                first_seen_order.append(ref)
            scores[ref] += weight / (k + rank)
    return sorted(first_seen_order, key=lambda ref: scores[ref], reverse=True)
