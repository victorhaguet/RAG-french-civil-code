from src.retrieval.embeddings import MultilingualE5Embeddings
from tests.fakes import FakeModel


def test_embed_documents_prefixes_each_text_with_passage() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    result = embeddings.embed_documents(["Les lois s'appliquent.", "Un autre texte."])

    assert model.encode_calls == [
        ["passage: Les lois s'appliquent.", "passage: Un autre texte."]
    ]
    assert result == [
        [len("passage: Les lois s'appliquent.")],
        [len("passage: Un autre texte.")],
    ]


def test_embed_query_prefixes_the_text_with_query() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    embeddings.embed_query("Quand une loi entre-t-elle en vigueur ?")

    assert model.encode_calls == [["query: Quand une loi entre-t-elle en vigueur ?"]]


def test_embed_query_returns_a_single_vector() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    result = embeddings.embed_query("short")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_embed_query_uses_the_same_prefix_regardless_of_language() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    embeddings.embed_query("When does a law enter into force?")

    assert model.encode_calls == [["query: When does a law enter into force?"]]
