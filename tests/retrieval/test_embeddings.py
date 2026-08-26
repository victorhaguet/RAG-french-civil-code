from src.retrieval.embeddings import QUERY_INSTRUCTIONS, MultilingualE5Embeddings
from tests.fakes import FakeModel


def test_embed_documents_sends_text_with_no_instruction_prefix() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    result = embeddings.embed_documents(["Les lois s'appliquent.", "Un autre texte."])

    assert model.encode_calls == [["Les lois s'appliquent.", "Un autre texte."]]
    assert result == [[len("Les lois s'appliquent.")], [len("Un autre texte.")]]


def test_embed_query_adds_an_instruction_prefix() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    embeddings.embed_query("Quand une loi entre-t-elle en vigueur ?")

    [call] = model.encode_calls
    [instructed_text] = call
    assert instructed_text.startswith("Instruct: ")
    assert instructed_text.endswith("\nQuery: Quand une loi entre-t-elle en vigueur ?")


def test_embed_query_returns_a_single_vector() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    result = embeddings.embed_query("short")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_embed_query_uses_the_french_instruction_for_a_french_query() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    embeddings.embed_query("Quand une loi entre-t-elle en vigueur ?")

    [[instructed_text]] = model.encode_calls
    assert instructed_text.startswith(f"Instruct: {QUERY_INSTRUCTIONS['fr']}")


def test_embed_query_uses_the_english_instruction_for_an_english_query() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    embeddings.embed_query("When does a law enter into force?")

    [[instructed_text]] = model.encode_calls
    assert instructed_text.startswith(f"Instruct: {QUERY_INSTRUCTIONS['en']}")


def test_embed_query_falls_back_to_french_for_an_unrecognized_language() -> None:
    model = FakeModel()
    embeddings = MultilingualE5Embeddings(model=model)

    embeddings.embed_query("いつ法律は施行されますか？")

    [[instructed_text]] = model.encode_calls
    assert instructed_text.startswith(f"Instruct: {QUERY_INSTRUCTIONS['fr']}")
