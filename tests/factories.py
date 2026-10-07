"""Shared test data builders."""

from typing import Any

from src.ingestion.dataset import Article, to_article


def raw_row(**overrides: object) -> dict[str, Any]:
    """A raw Code civil dataset row, shaped like a real HuggingFace row."""
    row: dict[str, Any] = {
        "ref": "LEGIARTI000006419287",
        "texte": "Les lois et actes publiés au Journal officiel entrent en vigueur.",
        "dateDebut": 1086048000000,
        "dateFin": 32472144000000,
        "num": "1",
        "id": "LEGIARTI000006419287",
        "cid": "LEGISCTA000006136318",
        "type": "AUTONOME",
        "etat": "VIGUEUR",
        "nota": "",
        "version_article": "2.0",
        "nature": "Article",
        "origine": "LEGI",
        "sectionParentTitre": "Titre préliminaire : De la publication, des effets et de l'application des lois en général",
        "idEliAlias": None,
        "idEli": None,
        "renvoi": None,
        "inap": None,
    }
    row.update(overrides)
    return row


def article(**overrides: object) -> Article:
    """An Article, as ingestion keeps it from a `raw_row` with these overrides."""
    return to_article(raw_row(**overrides))
