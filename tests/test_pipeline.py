from pathlib import Path

from src.pipeline import NON_FRENCH_REFUSAL, QueryPipeline, Refusal, Retrieval, generate, retrieve
from src.retrieval.keyword_index import KeywordIndex
from src.retrieval.reranker import Reranker
from tests.app_wiring import ingest
from tests.factories import raw_row
from tests.fakes import FakeChatModel, FakeModel, PassthroughCrossEncoder


_ROWS = [
    raw_row(ref="A1", texte="Les lois s'appliquent dès leur entrée en vigueur.", etat="VIGUEUR"),
    raw_row(ref="A2", texte="Repealed provision, no longer applicable.", etat="ABROGE_DIFF"),
]


def _pipeline(tmp_path: Path, model: FakeModel, chat_model: FakeChatModel) -> QueryPipeline:
    store, article_store = ingest(tmp_path, _ROWS, model)
    reranker = Reranker(model=PassthroughCrossEncoder())
    return QueryPipeline(store, chat_model, article_store, KeywordIndex(article_store), reranker)


def test_retrieve_gives_back_the_retrieved_articles_and_the_prompt_grounded_in_them(
    tmp_path: Path,
) -> None:
    chat_model = FakeChatModel()
    pipeline = _pipeline(tmp_path, FakeModel(), chat_model)

    result = retrieve("Quand une loi entre-t-elle en vigueur ?", 1, pipeline)

    assert isinstance(result, Retrieval)
    assert [article["ref"] for article in result.articles] == ["A1"]
    assert "Quand une loi entre-t-elle en vigueur ?" in result.prompt
    assert "Les lois s'appliquent dès leur entrée en vigueur." in result.prompt
    # Retrieval alone never calls the chat model.
    assert not chat_model.invoke_calls


def test_retrieve_refuses_a_non_french_question_before_any_embedding_or_generation(
    tmp_path: Path,
) -> None:
    model = FakeModel()
    chat_model = FakeChatModel()
    pipeline = _pipeline(tmp_path, model, chat_model)
    encodes_before = len(model.encode_calls)

    result = retrieve("When does a law enter into force?", 1, pipeline)

    assert result == Refusal(answer=NON_FRENCH_REFUSAL)
    assert len(model.encode_calls) == encodes_before
    assert not chat_model.invoke_calls


def test_generate_answers_the_retrieved_prompt_with_the_chat_model(tmp_path: Path) -> None:
    chat_model = FakeChatModel(answer="Voici la réponse.")
    pipeline = _pipeline(tmp_path, FakeModel(), chat_model)
    retrieval = retrieve("Quand une loi entre-t-elle en vigueur ?", 1, pipeline)
    assert isinstance(retrieval, Retrieval)

    answer = generate(retrieval.prompt, pipeline)

    assert answer == "Voici la réponse."
    assert chat_model.invoke_calls == [retrieval.prompt]
