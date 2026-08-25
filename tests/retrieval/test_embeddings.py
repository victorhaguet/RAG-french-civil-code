from rag_french_civil_code.retrieval.embeddings import MultilingualE5Embeddings


class FakeModel:
    """Records what it was asked to encode; returns one fake vector per text."""

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        self.encode_calls.append(list(texts))
        return [[float(len(text))] for text in texts]


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
