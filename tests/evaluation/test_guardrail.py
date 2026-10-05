from src.evaluation.guardrail import is_out_of_scope_answer


def test_recognizes_the_french_out_of_scope_answer() -> None:
    answer = (
        "Je ne peux pas répondre à cette question à partir des informations "
        "récupérées dans le Code civil.\n\nCe chatbot répond uniquement..."
    )

    assert is_out_of_scope_answer(answer) is True


def test_recognizes_the_english_out_of_scope_answer() -> None:
    answer = (
        "I cannot answer this question based on the information retrieved from "
        "the Code civil.\n\nThis chatbot only answers..."
    )

    assert is_out_of_scope_answer(answer) is True


def test_rejects_a_grounded_answer() -> None:
    answer = (
        "Réponse :\nLa loi s'applique dès sa publication.\n\n"
        "Fondement juridique :\n- Article A1 (Titre préliminaire) : ..."
    )

    assert is_out_of_scope_answer(answer) is False
