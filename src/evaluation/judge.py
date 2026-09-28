"""Building the RAGAS judge LLM used to score Golden Questions.

Configured via `EVAL_JUDGE_MODEL` / `EVAL_JUDGE_BASE_URL`, independently of
`src/config.py`'s `OPENAI_MODEL` (the model under test) — so changing the model
being evaluated doesn't change how strictly it's judged.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from src import config
from src.evaluation.metrics import ScorableMetric

# ragas's own InstructorModelArgs default (1024) caps how many tokens the judge
# may *generate* for its structured verdict, independent of the judge model's
# actual context window. Faithfulness's NLI verdict step returns one verdict +
# reasoning per statement extracted from the answer as a single JSON blob, and
# a real, multi-statement grounded answer -- especially judged by a verbose
# open-weight model -- can exceed 1024 tokens, truncating the JSON mid-output
# and failing with instructor.IncompleteOutputException. ragas's own docs
# recommend 4096+ for exactly this failure mode.
_JUDGE_MAX_TOKENS = 4096


def build_judge_metrics() -> tuple[ScorableMetric, ScorableMetric]:
    """Build RAGAS's `Faithfulness` and `ContextRecall` metrics, bound to the judge LLM.

    Imported lazily: `ragas` is only needed once there's at least one Golden
    Question to score, and its import chain is heavy (pulls in `langchain`,
    `langgraph`, `instructor`, ...) — kept off every run against the empty seed
    files, matching this codebase's existing lazy-import convention (see
    `src/retrieval/embeddings.py`, `src/generation/chat.py`).

    Before importing `ragas`, registers a stub `langchain_community.chat_models.
    vertexai` module: `ragas` unconditionally imports `ChatVertexAI` from there at
    import time, but `langchain-community>=0.4` (required by this project's
    `langchain-core>=1.6` pin) dropped that submodule entirely. We never use
    Vertex AI here, so a stub satisfies the import without downgrading
    `langchain-community`/`langchain-core` project-wide. This is an upstream
    packaging bug in `ragas` (every release through 0.4.3 leaves
    `langchain-community` unpinned) — safe to remove once it's fixed upstream.

    Returns:
        tuple[ScorableMetric, ScorableMetric]: (faithfulness, context_recall)
    """
    _stub_langchain_community_vertexai()

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextRecall, Faithfulness

    # AsyncOpenAI, not OpenAI: ragas's collections metrics (Faithfulness,
    # ContextRecall) expose a sync `.score()` that internally does
    # `asyncio.run(self.ascore(...))`, which always calls the LLM wrapper's
    # `.agenerate()` -- that raises TypeError unless the wrapped client is
    # async (llm_factory's own docs build it from AsyncOpenAI for this reason).
    client = AsyncOpenAI(base_url=config.EVAL_JUDGE_BASE_URL, api_key=config.OPENAI_API_KEY)
    llm = llm_factory(config.EVAL_JUDGE_MODEL, client=client, max_tokens=_JUDGE_MAX_TOKENS)
    return Faithfulness(llm=llm), ContextRecall(llm=llm)


def _stub_langchain_community_vertexai() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    class _UnavailableChatVertexAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "Vertex AI isn't supported here; this is an import-time stub for ragas."
            )

    stub.ChatVertexAI = _UnavailableChatVertexAI  # type: ignore[attr-defined]
    sys.modules[module_name] = stub
