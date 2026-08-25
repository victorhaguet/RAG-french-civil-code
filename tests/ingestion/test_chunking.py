from rag_french_civil_code.ingestion.chunking import build_documents, chunk_article
from rag_french_civil_code.ingestion.dataset import Article


def _article(**overrides: object) -> Article:
    article: Article = {
        "ref": "LEGIARTI000006419287",
        "texte": "Short article text.",
        "dateDebut": 1086048000000,
        "dateFin": 32472144000000,
        "etat": "VIGUEUR",
        "version_article": "2.0",
        "origine": "LEGI",
        "sectionParentTitre": "Titre préliminaire",
    }
    article.update(overrides)  # type: ignore[typeddict-item]
    return article


def test_short_article_becomes_a_single_chunk() -> None:
    article = _article(ref="A1", texte="Short article text.")

    documents = chunk_article(article)

    assert len(documents) == 1
    assert documents[0].id == "A1#0"
    assert documents[0].page_content == "Short article text."


def test_long_article_is_split_into_multiple_chunks_with_sequential_ids() -> None:
    long_text = "Une phrase juridique assez longue pour forcer un découpage. " * 30
    assert len(long_text) > 800
    article = _article(ref="A2", texte=long_text)

    documents = chunk_article(article)

    assert len(documents) > 1
    assert [doc.id for doc in documents] == [f"A2#{n}" for n in range(len(documents))]
    for doc in documents:
        assert len(doc.page_content) <= 800


def test_article_at_exactly_the_chunk_size_stays_a_single_chunk() -> None:
    article = _article(ref="A2b", texte="x" * 800)

    documents = chunk_article(article)

    assert len(documents) == 1
    assert documents[0].id == "A2b#0"


def test_article_one_character_over_the_chunk_size_is_split() -> None:
    article = _article(ref="A2c", texte="x" * 801)

    documents = chunk_article(article)

    assert len(documents) > 1
    assert [doc.id for doc in documents] == [f"A2c#{n}" for n in range(len(documents))]


def test_consecutive_chunks_overlap() -> None:
    long_text = "Une phrase juridique assez longue pour forcer un découpage. " * 30
    article = _article(ref="A3", texte=long_text)

    documents = chunk_article(article)

    first_tail = documents[0].page_content[-50:]
    assert first_tail in documents[1].page_content


def test_every_chunk_carries_the_articles_metadata() -> None:
    article = _article(
        ref="A4",
        texte="Text.",
        dateDebut=1,
        dateFin=2,
        etat="VIGUEUR",
        version_article="1.0",
        origine="LEGI",
        sectionParentTitre="Titre I",
    )

    [document] = chunk_article(article)

    assert document.metadata == {
        "ref": "A4",
        "dateDebut": 1,
        "dateFin": 2,
        "etat": "VIGUEUR",
        "version_article": "1.0",
        "origine": "LEGI",
        "sectionParentTitre": "Titre I",
    }
    assert "texte" not in document.metadata


def test_each_chunks_metadata_is_an_independent_copy() -> None:
    long_text = "Une phrase juridique assez longue pour forcer un découpage. " * 30
    article = _article(ref="A7", texte=long_text)

    documents = chunk_article(article)

    assert len(documents) > 1
    assert documents[0].metadata is not documents[1].metadata
    documents[0].metadata["etat"] = "MUTATED"
    assert documents[1].metadata["etat"] == "VIGUEUR"


def test_build_documents_chunks_every_article_in_order() -> None:
    articles = [_article(ref="A5", texte="First."), _article(ref="A6", texte="Second.")]

    documents = build_documents(articles)

    assert [doc.id for doc in documents] == ["A5#0", "A6#0"]
