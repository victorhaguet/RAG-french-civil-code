from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import (
    get_article_store,
    get_bm25_index,
    get_chat_model,
    get_reranker,
    get_store,
)
from src.evaluation.dataset import GoldenQuestion
from src.evaluation.golden import evaluate_golden_questions
from src.ingestion.pipeline import run_ingestion
from src.retrieval.embeddings import FIXED_PREFIX_MODEL, MultilingualE5Embeddings
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from src.storage.article_store import ArticleStore
from tests.factories import raw_row
from tests.fakes import FakeChatModel, FakeModel, PassthroughCrossEncoder


class _FakeScorableMetric:
    """Records the kwargs it was scored with; returns a fixed score."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.score_calls: list[dict[str, Any]] = []

    def score(self, **kwargs: Any) -> Any:
        self.score_calls.append(kwargs)
        return SimpleNamespace(value=self.value)


def _client(tmp_path: Path, chat_model: FakeChatModel) -> TestClient:
    model = FakeModel()
    chroma_store = run_ingestion(
        raw_rows=[raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR")],
        embeddings=MultilingualE5Embeddings(model=model, model_name=FIXED_PREFIX_MODEL),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
        sqlite_path=str(tmp_path / "articles.db"),
    )
    article_store = ArticleStore(str(tmp_path / "articles.db"))
    app.dependency_overrides[get_store] = lambda: chroma_store
    app.dependency_overrides[get_article_store] = lambda: article_store
    app.dependency_overrides[get_chat_model] = lambda: chat_model
    app.dependency_overrides[get_bm25_index] = lambda: KeywordIndex(article_store)
    app.dependency_overrides[get_reranker] = lambda: Reranker(model=PassthroughCrossEncoder())
    return TestClient(app)


def test_reports_no_data_for_an_empty_question_set(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeChatModel())

    def _fail_if_called() -> tuple[_FakeScorableMetric, _FakeScorableMetric]:
        raise AssertionError("build_metrics must not be called for an empty question set")

    report = evaluate_golden_questions(client, [], _fail_if_called)

    assert report.has_data is False
    assert report.mean_faithfulness is None
    assert report.mean_context_recall is None
    assert report.scores == []


def test_scores_a_golden_question_via_the_real_query_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeChatModel(answer="Voici la réponse."))
    faithfulness = _FakeScorableMetric(0.75)
    context_recall = _FakeScorableMetric(0.5)

    report = evaluate_golden_questions(
        client,
        [
            {
                "question": "Quand une loi entre-t-elle en vigueur ?",
                "reference_answer": "Le lendemain de sa publication.",
                "reference_article_refs": ["A1"],
            }
        ],
        lambda: (faithfulness, context_recall),
    )

    assert report.has_data is True
    [score] = report.scores
    assert score.question == "Quand une loi entre-t-elle en vigueur ?"
    assert score.answer == "Voici la réponse."
    assert score.articles == ["A1"]
    assert score.reference_answer == "Le lendemain de sa publication."
    assert score.reference_article_refs == ["A1"]
    assert score.faithfulness == 0.75
    assert score.context_recall == 0.5
    assert report.mean_faithfulness == 0.75
    assert report.mean_context_recall == 0.5

    [faithfulness_call] = faithfulness.score_calls
    assert faithfulness_call["user_input"] == "Quand une loi entre-t-elle en vigueur ?"
    assert faithfulness_call["response"] == "Voici la réponse."
    assert faithfulness_call["retrieved_contexts"] == [
        "Les lois s'appliquent dès leur entrée en vigueur."
    ]

    [context_recall_call] = context_recall.score_calls
    assert context_recall_call["reference"] == "Le lendemain de sa publication."
    assert context_recall_call["retrieved_contexts"] == [
        "Les lois s'appliquent dès leur entrée en vigueur."
    ]


def test_a_judge_error_on_one_metric_is_reported_as_none_not_raised(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeChatModel(answer="Voici la réponse."))
    faithfulness = _FakeScorableMetric(0.0)

    def _raise(**kwargs: Any) -> Any:
        raise RuntimeError("simulated judge truncation")

    faithfulness.score = _raise  # type: ignore[method-assign]
    context_recall = _FakeScorableMetric(0.5)

    report = evaluate_golden_questions(
        client,
        [
            {
                "question": "Quand une loi entre-t-elle en vigueur ?",
                "reference_answer": "Le lendemain de sa publication.",
                "reference_article_refs": ["A1"],
            }
        ],
        lambda: (faithfulness, context_recall),
    )

    [score] = report.scores
    assert score.faithfulness is None
    assert score.context_recall == 0.5
    # The /query call itself still succeeded -- only the judge verdict is missing.
    assert score.answer == "Voici la réponse."
    assert report.mean_faithfulness is None
    assert report.mean_context_recall == 0.5
    assert report.faithfulness_scored == 0
    assert report.context_recall_scored == 1


def test_mean_excludes_judge_errors_rather_than_counting_them_as_zero(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeChatModel())
    results: list[float | None] = [0.8, None]

    def _score(**kwargs: Any) -> Any:
        value = results.pop(0)
        if value is None:
            raise RuntimeError("simulated judge truncation")
        return SimpleNamespace(value=value)

    faithfulness = _FakeScorableMetric(0.0)
    faithfulness.score = _score  # type: ignore[method-assign]
    context_recall = _FakeScorableMetric(1.0)

    golden_questions: list[GoldenQuestion] = [
        {"question": "Q1", "reference_answer": "R1", "reference_article_refs": ["A1"]},
        {"question": "Q2", "reference_answer": "R2", "reference_article_refs": ["A1"]},
    ]

    report = evaluate_golden_questions(
        client, golden_questions, lambda: (faithfulness, context_recall)
    )

    assert report.mean_faithfulness == 0.8  # not (0.8 + 0) / 2
    assert report.faithfulness_scored == 1


def test_mean_averages_scores_across_multiple_golden_questions(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeChatModel())
    scores = iter([0.2, 0.6])
    faithfulness = _FakeScorableMetric(0.0)
    faithfulness.score = lambda **kwargs: SimpleNamespace(value=next(scores))  # type: ignore[method-assign]
    context_recall = _FakeScorableMetric(1.0)

    golden_questions: list[GoldenQuestion] = [
        {
            "question": "Q1",
            "reference_answer": "R1",
            "reference_article_refs": ["A1"],
        },
        {
            "question": "Q2",
            "reference_answer": "R2",
            "reference_article_refs": ["A1"],
        },
    ]

    report = evaluate_golden_questions(
        client, golden_questions, lambda: (faithfulness, context_recall)
    )

    assert report.mean_faithfulness == pytest.approx(0.4)
    assert report.mean_context_recall == 1.0
