from src.ingestion.dataset import KEPT_FIELDS, load_articles
from tests.factories import raw_row as _raw_row


def test_keeps_only_in_force_articles() -> None:
    rows = [_raw_row(ref="A1", etat="VIGUEUR"), _raw_row(ref="A2", etat="ABROGE_DIFF")]

    articles = load_articles(raw_rows=rows)

    assert [a["ref"] for a in articles] == ["A1"]


def test_projects_down_to_kept_fields_only() -> None:
    rows = [_raw_row()]

    [article] = load_articles(raw_rows=rows)

    assert set(article.keys()) == set(KEPT_FIELDS)
    for dropped in ("idEliAlias", "idEli", "renvoi", "inap", "id", "cid", "num"):
        assert dropped not in article


def test_preserves_kept_field_values() -> None:
    rows = [_raw_row(ref="LEGIARTI000006419287", texte="Some text.", dateDebut=123, dateFin=456)]

    [article] = load_articles(raw_rows=rows)

    assert article["ref"] == "LEGIARTI000006419287"
    assert article["texte"] == "Some text."
    assert article["dateDebut"] == 123
    assert article["dateFin"] == 456
