"""Binary guardrail check: does an answer take the Out-of-Scope Answer shape?

See CONTEXT.md's Out-of-Scope Answer entry: no persona, no citations, the fixed
fallback message defined in `prompts/rag_answer_fr.jinja2`.
"""

from __future__ import annotations

# The check is a substring match, not equality: the fixed message is only the
# opening sentence of what the template renders — the rest (as-of date, rephrase
# invitation) isn't distinctive enough to assert on, and isn't guaranteed verbatim
# through the model. Keep this in sync with the prompt template.
_OUT_OF_SCOPE_MARKERS = (
    "Je ne peux pas répondre à cette question à partir des informations récupérées "
    "dans le Code civil.",
)


def is_out_of_scope_answer(answer: str) -> bool:
    """Whether `answer` matches the app's Out-of-Scope Answer shape.

    Args:
        answer (str): the `/query` response's `answer` field

    Returns:
        bool: True if `answer` contains the fixed fallback opening
    """
    return any(marker in answer for marker in _OUT_OF_SCOPE_MARKERS)
