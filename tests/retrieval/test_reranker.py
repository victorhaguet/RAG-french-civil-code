from src.ingestion.dataset import to_article
from src.retrieval.reranker import Reranker
from tests.factories import raw_row
from tests.fakes import FakeCrossEncoder


def test_rerank_scores_the_question_and_each_articles_texte_as_a_pair() -> None:
    model = FakeCrossEncoder()
    reranker = Reranker(model=model)
    articles = [
        to_article(raw_row(ref="A1", texte="Un texte.")),
        to_article(raw_row(ref="A2", texte="Un autre texte plus long.")),
    ]

    reranker.rerank("Quelle est la loi ?", articles)

    assert model.predict_calls == [
        [
            ("Quelle est la loi ?", "Un texte."),
            ("Quelle est la loi ?", "Un autre texte plus long."),
        ]
    ]


def test_rerank_reorders_articles_by_descending_cross_encoder_score() -> None:
    # FakeCrossEncoder scores by combined (question, texte) length, so the
    # longer article scores higher and should be moved to the front, even
    # though it comes second in the incoming (fused) order.
    model = FakeCrossEncoder()
    reranker = Reranker(model=model)
    short = to_article(raw_row(ref="SHORT", texte="Bref."))
    long = to_article(raw_row(ref="LONG", texte="Un texte nettement plus long que l'autre."))

    result = reranker.rerank("Question ?", [short, long])

    assert [article["ref"] for article in result] == ["LONG", "SHORT"]


def test_rerank_returns_an_empty_list_for_no_candidates() -> None:
    model = FakeCrossEncoder()
    reranker = Reranker(model=model)

    result = reranker.rerank("Question ?", [])

    assert result == []
    assert model.predict_calls == []
