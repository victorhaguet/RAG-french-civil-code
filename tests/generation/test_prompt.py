from src.generation.prompt import render_prompt
from src.ingestion.dataset import Article, to_article
from tests.factories import raw_row


def _article(
    ref: str = "A1",
    texte: str = "Les lois s'appliquent dès leur entrée en vigueur.",
    section: str = "Titre préliminaire",
) -> Article:
    return to_article(raw_row(ref=ref, texte=texte, sectionParentTitre=section))


def test_render_prompt_uses_the_french_template_for_a_french_question() -> None:
    prompt = render_prompt("Quand une loi entre-t-elle en vigueur ?", [_article()])

    assert "Question :" in prompt
    assert "Réponse :" in prompt
    assert "Quand une loi entre-t-elle en vigueur ?" in prompt
    assert "A1" in prompt


def test_render_prompt_uses_the_english_template_for_an_english_question() -> None:
    prompt = render_prompt("When does a law enter into force?", [_article()])

    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "When does a law enter into force?" in prompt
    assert "A1" in prompt


def test_render_prompt_falls_back_to_the_french_template_for_an_unrecognized_language() -> None:
    prompt = render_prompt("いつ法律は施行されますか？", [_article()])

    assert "Question :" in prompt
    assert "Réponse :" in prompt


def test_render_prompt_includes_every_article() -> None:
    prompt = render_prompt(
        "Quelle est la loi applicable ?",
        [_article(ref="A1", texte="Premier texte."), _article(ref="A2", texte="Second texte.")],
    )

    assert "A1" in prompt
    assert "Premier texte." in prompt
    assert "A2" in prompt
    assert "Second texte." in prompt


def test_render_prompt_includes_an_articles_full_text_without_truncation() -> None:
    long_text = "Une phrase juridique assez longue pour forcer un découpage. " * 30
    assert len(long_text) > 800

    prompt = render_prompt("Quelle est la loi applicable ?", [_article(texte=long_text)])

    assert long_text in prompt


def test_render_prompt_includes_the_section_title_for_citation() -> None:
    prompt = render_prompt(
        "Quelle est la loi applicable ?",
        [_article(section="Des contrats et des obligations conventionnelles en général")],
    )

    assert "Des contrats et des obligations conventionnelles en général" in prompt


def test_render_prompt_uses_the_default_dataset_as_of_date() -> None:
    prompt = render_prompt("Quelle est la loi applicable ?", [_article()])

    assert "21 September 2025" in prompt


def test_render_prompt_accepts_a_caller_supplied_dataset_as_of_date() -> None:
    prompt = render_prompt(
        "Quelle est la loi applicable ?", [_article()], dataset_as_of="1 January 2030"
    )

    assert "1 January 2030" in prompt
    assert "21 September 2025" not in prompt


def test_render_prompt_instructs_a_scoped_fallback_when_the_answer_is_missing() -> None:
    prompt = render_prompt("Quelle est la loi applicable ?", [_article()])

    assert "Je ne peux pas répondre à cette question" in prompt
    assert "essayez de la reformuler" in prompt
