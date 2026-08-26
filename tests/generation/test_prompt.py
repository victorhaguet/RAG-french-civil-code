from langchain_core.documents import Document

from src.generation.prompt import render_prompt


def _chunk(ref: str = "A1", text: str = "Les lois s'appliquent dès leur entrée en vigueur.") -> Document:
    return Document(page_content=text, metadata={"ref": ref})


def test_render_prompt_uses_the_french_template_for_a_french_question() -> None:
    prompt = render_prompt("Quand une loi entre-t-elle en vigueur ?", [_chunk()])

    assert "Question :" in prompt
    assert "Réponse :" in prompt
    assert "Quand une loi entre-t-elle en vigueur ?" in prompt
    assert "A1" in prompt


def test_render_prompt_uses_the_english_template_for_an_english_question() -> None:
    prompt = render_prompt("When does a law enter into force?", [_chunk()])

    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "When does a law enter into force?" in prompt
    assert "A1" in prompt


def test_render_prompt_falls_back_to_the_french_template_for_an_unrecognized_language() -> None:
    prompt = render_prompt("いつ法律は施行されますか？", [_chunk()])

    assert "Question :" in prompt
    assert "Réponse :" in prompt


def test_render_prompt_includes_every_chunk() -> None:
    prompt = render_prompt(
        "Quelle est la loi applicable ?",
        [_chunk(ref="A1", text="Premier texte."), _chunk(ref="A2", text="Second texte.")],
    )

    assert "A1" in prompt
    assert "Premier texte." in prompt
    assert "A2" in prompt
    assert "Second texte." in prompt
