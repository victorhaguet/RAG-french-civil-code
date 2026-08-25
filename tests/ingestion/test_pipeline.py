from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_french_civil_code.ingestion.pipeline import run_ingestion
from tests.factories import raw_row as _base_raw_row


class FakeEmbeddings(Embeddings):
    """Deterministic, fixed-dimension fake embeddings — no real model needed."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float((hash(text) >> (8 * i)) % 100) for i in range(4)]


def _raw_row(**overrides: object) -> dict:
    defaults = {
        "ref": "LEGIARTI1",
        "texte": "Short in-force article.",
        "dateDebut": 1,
        "version_article": "1.0",
        "sectionParentTitre": "Titre I",
    }
    defaults.update(overrides)
    return _base_raw_row(**defaults)


def _run(tmp_path: Path, raw_rows: list[dict]):
    return run_ingestion(
        raw_rows=raw_rows,
        embeddings=FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )


def test_only_vigueur_articles_are_stored(tmp_path: Path) -> None:
    rows = [
        _raw_row(ref="A1", etat="VIGUEUR"),
        _raw_row(ref="A2", etat="ABROGE_DIFF"),
    ]

    store = _run(tmp_path, rows)

    stored = store.get()
    assert stored["ids"] == ["A1#0"]


def test_short_article_stored_as_a_single_chunk_with_full_metadata(tmp_path: Path) -> None:
    rows = [
        _raw_row(
            ref="A1",
            texte="Short text.",
            dateDebut=1086048000000,
            dateFin=32472144000000,
            etat="VIGUEUR",
            version_article="2.0",
            origine="LEGI",
            sectionParentTitre="Titre préliminaire",
        )
    ]

    store = _run(tmp_path, rows)

    stored = store.get(include=["metadatas", "documents"])
    assert stored["ids"] == ["A1#0"]
    assert stored["documents"] == ["Short text."]
    assert stored["metadatas"][0] == {
        "ref": "A1",
        "dateDebut": 1086048000000,
        "dateFin": 32472144000000,
        "etat": "VIGUEUR",
        "version_article": "2.0",
        "origine": "LEGI",
        "sectionParentTitre": "Titre préliminaire",
    }


def test_long_article_split_into_multiple_sequential_chunks(tmp_path: Path) -> None:
    long_text = "Une phrase juridique assez longue pour forcer un découpage. " * 30
    rows = [_raw_row(ref="A1", texte=long_text)]

    store = _run(tmp_path, rows)

    stored = store.get()
    assert len(stored["ids"]) > 1
    assert stored["ids"] == [f"A1#{n}" for n in range(len(stored["ids"]))]


def test_dropped_fields_are_absent_from_stored_metadata(tmp_path: Path) -> None:
    rows = [_raw_row(ref="A1")]

    store = _run(tmp_path, rows)

    [metadata] = store.get(include=["metadatas"])["metadatas"]
    for dropped in ("idEliAlias", "idEli", "renvoi", "inap"):
        assert dropped not in metadata


def test_rerunning_ingestion_rebuilds_from_scratch_with_no_duplicates(tmp_path: Path) -> None:
    rows = [_raw_row(ref="A1", texte="Text one.")]
    persist_directory = str(tmp_path / "chroma")

    run_ingestion(
        raw_rows=rows,
        embeddings=FakeEmbeddings(),
        persist_directory=persist_directory,
        collection_name="test_collection",
    )
    store = run_ingestion(
        raw_rows=rows,
        embeddings=FakeEmbeddings(),
        persist_directory=persist_directory,
        collection_name="test_collection",
    )

    assert store.get()["ids"] == ["A1#0"]


def test_rerunning_ingestion_drops_articles_no_longer_present(tmp_path: Path) -> None:
    persist_directory = str(tmp_path / "chroma")

    run_ingestion(
        raw_rows=[_raw_row(ref="A1"), _raw_row(ref="A2")],
        embeddings=FakeEmbeddings(),
        persist_directory=persist_directory,
        collection_name="test_collection",
    )
    store = run_ingestion(
        raw_rows=[_raw_row(ref="A1")],
        embeddings=FakeEmbeddings(),
        persist_directory=persist_directory,
        collection_name="test_collection",
    )

    assert store.get()["ids"] == ["A1#0"]
