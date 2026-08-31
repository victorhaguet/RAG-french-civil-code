import pytest

from src.retrieval.embeddings import FIXED_PREFIX_MODEL, INSTRUCT_MODEL, MultilingualE5Embeddings
from tests.fakes import FakeModel


def test_unsupported_model_name_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_MODEL"):
        MultilingualE5Embeddings(model=FakeModel(), model_name="intfloat/some-other-model")


def test_embed_documents_prefixes_each_text_with_passage_for_the_fixed_prefix_model() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model, model_name=FIXED_PREFIX_MODEL)

    result = embeddings.embed_documents(["Les lois s'appliquent.", "Un autre texte."])

    assert model.encode_calls == [
        ["passage: Les lois s'appliquent.", "passage: Un autre texte."]
    ]
    assert result == [
        [len("passage: Les lois s'appliquent.")],
        [len("passage: Un autre texte.")],
    ]


def test_embed_query_prefixes_the_text_with_query_for_the_fixed_prefix_model() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model, model_name=FIXED_PREFIX_MODEL)

    embeddings.embed_query("Quand une loi entre-t-elle en vigueur ?")

    assert model.encode_calls == [["query: Quand une loi entre-t-elle en vigueur ?"]]


def test_embed_query_uses_the_same_prefix_regardless_of_language_for_the_fixed_prefix_model() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model, model_name=FIXED_PREFIX_MODEL)

    embeddings.embed_query("When does a law enter into force?")

    assert model.encode_calls == [["query: When does a law enter into force?"]]


def test_embed_documents_adds_no_prefix_for_the_instruct_model() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model, model_name=INSTRUCT_MODEL)

    embeddings.embed_documents(["Les lois s'appliquent.", "Un autre texte."])

    assert model.encode_calls == [["Les lois s'appliquent.", "Un autre texte."]]


def test_embed_query_uses_the_french_instruction_for_the_instruct_model() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model, model_name=INSTRUCT_MODEL)

    embeddings.embed_query("Quand une loi entre-t-elle en vigueur ?")

    [call] = model.encode_calls
    [prompt] = call
    assert prompt.startswith("Instruct: ")
    assert prompt.endswith("\nQuery: Quand une loi entre-t-elle en vigueur ?")


def test_embed_query_uses_the_english_instruction_for_the_instruct_model() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model, model_name=INSTRUCT_MODEL)

    embeddings.embed_query("When does a law enter into force?")

    [call] = model.encode_calls
    [prompt] = call
    assert prompt.endswith("\nQuery: When does a law enter into force?")
    assert "retrieve the Code civil articles" in prompt


def test_embed_query_returns_a_single_vector() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model, model_name=FIXED_PREFIX_MODEL)

    result = embeddings.embed_query("short")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)
