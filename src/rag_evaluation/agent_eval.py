"""Tiers 2 and 3: the judge scorers, via `mlflow.genai.evaluate()`.

Runs the real agent as the prediction function — same retriever, tools and
prompt — and scores what it answered against what it retrieved. The retrieval
scorers read that context from the trace's RETRIEVER span, which
`KnowledgeTools.search_knowledge_base` emits; without it they raise rather than
score, which is why `agent/tests/test_tools.py` asserts the span shape.

RAGAS was replaced here (2026-08-17): MLflow ships equivalents of all four of its
metrics, plus tool-call scorers for tier 3 and scheduled scorers for production
traffic, and it logs results natively — where RAGAS needed a second evaluation
stack and two version ceilings to work around its own dependency breakage.

    faithfulness       -> RetrievalGroundedness
    answer_relevancy   -> RelevanceToQuery
    context_precision  -> RetrievalRelevance
    context_recall     -> RetrievalSufficiency
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import replace

from rag_agent.settings import get_params, get_settings
from rag_config import set_param
from rag_agent.workflow import RagAgent

from rag_evaluation.golden import Entry
from rag_evaluation.mlflow_log import EXPERIMENT, active, configure

__all__ = ["DEFAULT_JUDGE_MODEL", "run_judges", "build_dataset"]

log = logging.getLogger(__name__)

# Different family from the agent *and* stronger. Different family because a
# model scoring its own output has a self-preference bias, documented at family
# level rather than only per model — so Sonnet judging Haiku is not clean.
# Stronger because the judge has to spot a claim the context does not support; a
# weaker one errs toward "looks fine" and every number comes back optimistic.
# Routed through OpenRouter, so one API key serves agent and judge.
DEFAULT_JUDGE_MODEL = "openrouter:/openai/gpt-5.4"

# Citation and refusal discipline — tier 3, expressed as rules rather than as a
# metric. Both are behaviours block 8 makes non-negotiable for a regulatory
# corpus.
GUIDELINES = [
    "The answer must only cite sources that appear in the retrieved context.",
    "If the retrieved context does not contain the answer, the response must say "
    "so plainly instead of answering from general knowledge.",
]


def build_dataset(entries: Sequence[Entry]) -> list[dict]:
    """Golden entries in MLflow's shape.

    `expectations` is filled in only where a reference exists; the scorers that
    need one are dropped when none does, rather than scoring against an empty
    string.
    """
    rows = []
    for entry in entries:
        row: dict = {"inputs": {"question": entry.question}}
        if entry.reference:
            row["expectations"] = {"expected_response": entry.reference}
        rows.append(row)
    return rows


def _predict_fn(agent: RagAgent):
    """One question through the real agent, synchronously.

    Memory is reset per question: the golden set is independent questions, not a
    conversation, and carrying context lets an earlier answer leak into a later
    one.
    """

    def predict(question: str) -> str:
        async def turn() -> str:
            agent.memory.reset()
            handler = await agent.run(question)
            response = await handler
            return str(getattr(response, "response", response))

        return asyncio.run(turn())

    return predict


def _scorers(judge: str, with_reference: bool) -> list:
    from mlflow.genai.scorers import (
        Guidelines,
        RelevanceToQuery,
        RetrievalGroundedness,
        RetrievalRelevance,
        RetrievalSufficiency,
        ToolCallEfficiency,
    )

    # `model=` on every scorer, always: MLflow's default is openai:/gpt-4.1-mini,
    # which needs an OpenAI key and is a model nobody chose.
    scorers = [
        RetrievalGroundedness(model=judge),
        RelevanceToQuery(model=judge),
        RetrievalRelevance(model=judge),
        Guidelines(name="citation_and_refusal", guidelines=GUIDELINES, model=judge),
        ToolCallEfficiency(model=judge),
    ]
    if with_reference:
        from mlflow.genai.scorers import Correctness

        scorers += [Correctness(model=judge), RetrievalSufficiency(model=judge)]
    return scorers


def _warn_about_the_judge(judge: str, agent_model: str) -> None:
    """A model grading its own homework inflates every number."""
    judge_id = judge.split(":/")[-1]
    if judge_id == agent_model:
        print(
            "WARNING: judge and agent are the same model — every metric will be "
            "inflated by self-preference bias.\n",
            file=sys.stderr,
        )
        return
    family = lambda m: m.split("/")[0] if "/" in m else m  # noqa: E731
    if family(judge_id) == family(agent_model):
        print(
            f"WARNING: judge and agent are both {family(judge_id)} models — "
            "self-preference bias is documented at family level, not only per "
            "model.\n",
            file=sys.stderr,
        )


def run_judges(
    entries: Sequence[Entry],
    judge_model: str | None = None,
    rerank: bool = False,
) -> int:
    import mlflow

    judge = judge_model or os.environ.get("RAG_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    settings = get_settings()
    params = get_params()
    if rerank:
        params = set_param(params, "agent.rerank_enabled", True)

    with_reference = sum(1 for e in entries if e.scorable_tier2)
    print(f"agent: {params.agent.llm_model}   judge: {judge}")
    print(
        f"rerank: {'on' if params.agent.rerank_enabled else 'off'}   "
        f"questions: {len(entries)}   with a reference: {with_reference}\n"
    )
    _warn_about_the_judge(judge, params.agent.llm_model)
    if not with_reference:
        print(
            "note: no entry has a `reference`, so Correctness and "
            "RetrievalSufficiency are skipped — the reference-free scorers still "
            "run.\n"
        )

    # Resolved the same way tier 1 resolves it, so one command cannot log the
    # judges and stay silent about retrieval. `settings.mlflow_tracking_uri`
    # is not used here: it falls back to the cluster address when nothing is
    # set, which would make tiers 2 and 3 talk to a server tier 1 ignores.
    if active():
        configure(EXPERIMENT)

    agent = RagAgent(settings, params)
    try:
        result = mlflow.genai.evaluate(
            data=build_dataset(entries),
            predict_fn=_predict_fn(agent),
            scorers=_scorers(judge, bool(with_reference)),
        )
    finally:
        agent.close()

    print("\n=== scores ===")
    metrics = getattr(result, "metrics", {}) or {}
    for name, value in sorted(metrics.items()):
        print(f"  {name:44} {value}")
    if with_reference < len(entries):
        print(
            f"\n  reference-dependent scorers cover {with_reference}/{len(entries)} "
            "entries — fill in `reference` in the golden set for the rest."
        )
    return 0
