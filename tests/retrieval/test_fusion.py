"""Tests for weighted Reciprocal Rank Fusion, as a pure function over ranked ref lists."""

from src.retrieval.fusion import reciprocal_rank_fusion


def test_fuses_two_lists_ranking_a_ref_common_to_both_first() -> None:
    keyword_refs = ["A2", "A1", "A3"]
    vector_refs = ["A1", "A2", "A4"]

    fused = reciprocal_rank_fusion([keyword_refs, vector_refs], weights=[1.0, 1.0], k=60)

    # A1 (rank 2 keyword, rank 1 vector) and A2 (rank 1 keyword, rank 2
    # vector) both appear near the top of each list, so both should out-rank
    # A3/A4, which each appear in only one list.
    assert fused[0] in {"A1", "A2"}
    assert fused[1] in {"A1", "A2"}
    assert set(fused[:2]) == {"A1", "A2"}


def test_deduplicates_a_ref_appearing_in_both_lists() -> None:
    fused = reciprocal_rank_fusion([["A1"], ["A1"]], weights=[1.0, 1.0], k=60)

    assert fused == ["A1"]


def test_a_ref_present_in_both_lists_outranks_one_present_in_only_one() -> None:
    keyword_refs = ["A1", "A2"]
    vector_refs = ["A2"]

    fused = reciprocal_rank_fusion([keyword_refs, vector_refs], weights=[1.0, 1.0], k=60)

    # A2 is reinforced by both lists, A1 only by the keyword list.
    assert fused[0] == "A2"


def test_a_higher_weighted_list_dominates_the_fused_order() -> None:
    keyword_refs = ["A1", "A2"]
    vector_refs = ["A2", "A1"]

    fused = reciprocal_rank_fusion([keyword_refs, vector_refs], weights=[10.0, 1.0], k=60)

    # With the keyword list weighted far higher, its top rank (A1) wins
    # despite A2 ranking first in the vector list.
    assert fused[0] == "A1"


def test_ties_are_broken_by_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([["A1"], ["A2"]], weights=[1.0, 1.0], k=60)

    # Both refs get the same score (rank 1, equal weight, only one list
    # each), so the tie is broken by first-seen order across the lists.
    assert fused == ["A1", "A2"]


def test_an_empty_list_contributes_nothing() -> None:
    fused = reciprocal_rank_fusion([[], ["A1"]], weights=[1.0, 1.0], k=60)

    assert fused == ["A1"]
