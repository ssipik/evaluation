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
import faulthandler
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from rag_agent.retrieval import Retriever
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


def _predict_fn(new_agent: Callable[[], RagAgent]):
    """One question through the real agent, synchronously, on a fresh agent.

    MLflow calls this from several threads at once, so nothing per-question may
    be shared:
    - the evidence log — `run()` resets it, wiping another question's citations
    - the LLM's async client — bound to the event loop of the thread that opened it
    - the memory — the golden set is independent questions, not a conversation
    The retriever is sync and thread-safe, so `new_agent` passes one shared copy.
    """

    def predict(question: str) -> str:
        agent = new_agent()

        async def turn() -> str:
            handler = await agent.run(question)
            response = await handler
            return str(getattr(response, "response", response))

        with INFLIGHT.track("agent", question):
            return asyncio.run(turn())

    return predict


class _Inflight:
    """The agent and judge calls running right now, for finding where a run hangs.

    Each call logs its start and end, and `progress` logs the steps between them;
    `watch` turns silence into a stack dump.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[int, tuple[str, str, float]] = {}
        self._next = 0
        self._last_event = time.monotonic()

    def progress(self, what: str) -> None:
        """Log a step outside any call, and count it as progress."""
        self._last_event = time.monotonic()
        log.info("%s", what)

    @contextmanager
    def track(self, kind: str, label: str):
        label = " ".join(label.split())[:60]
        with self._lock:
            self._next += 1
            n = self._next
            self._calls[n] = (kind, label, time.monotonic())
            self._last_event = time.monotonic()
        log.info("%s #%d start: %s", kind, n, label)
        start = time.monotonic()
        ok = False
        try:
            yield
            ok = True
        finally:
            with self._lock:
                del self._calls[n]
                self._last_event = time.monotonic()
            log.info(
                "%s #%d %s after %.1f s",
                kind, n, "done" if ok else "FAILED", time.monotonic() - start,
            )

    def snapshot(self) -> list[tuple[str, str, float]]:
        """(kind, label, age in seconds) of every running call, oldest first."""
        now = time.monotonic()
        with self._lock:
            calls = [(k, lbl, now - t) for k, lbl, t in self._calls.values()]
        return sorted(calls, key=lambda c: -c[2])

    @contextmanager
    def watch(self, stacks_path: Path, every: float = 60, stuck_after: float = 300):
        """Report once a minute; dump every thread's stack, once, when a call has
        run `stuck_after` seconds or nothing has started or ended for that long.
        `kill -USR1 <pid>` dumps them on demand."""
        stop = threading.Event()
        self._last_event = time.monotonic()
        stacks_file = stacks_path.open("w")
        faulthandler.register(signal.SIGUSR1, file=stacks_file, all_threads=True)
        log.info("pid %d — `kill -USR1 %d` writes thread stacks to %s", os.getpid(), os.getpid(), stacks_path)

        def report() -> None:
            dumped = False
            while not stop.wait(every):
                calls = self.snapshot()
                idle = time.monotonic() - self._last_event
                if calls:
                    counts = {k: sum(c[0] == k for c in calls) for k in ("agent", "judge")}
                    kind, label, age = calls[0]
                    log.info(
                        "in flight: %d agent, %d judge; oldest %s %.0f s: %s",
                        counts["agent"], counts["judge"], kind, age, label,
                    )
                    stuck = age
                else:
                    log.info("in flight: nothing; no progress for %.0f s", idle)
                    stuck = idle
                if stuck > stuck_after and not dumped:
                    faulthandler.dump_traceback(file=stacks_file, all_threads=True)
                    stacks_file.flush()
                    log.warning("stuck for %.0f s — thread stacks in %s", stuck, stacks_path)
                    dumped = True

        thread = threading.Thread(target=report, name="inflight-watch", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            faulthandler.unregister(signal.SIGUSR1)
            stacks_file.close()


INFLIGHT = _Inflight()


@contextmanager
def _logging_judge_calls():
    """Route every judge call through `INFLIGHT`, labelled with the scorer's name.

    Every MLflow judge reaches its model through `BaseJudgeAdapter.invoke`,
    whichever adapter serves the model URI (`openrouter:/` is the gateway one).
    """
    from mlflow.genai.judges.adapters.base_adapter import BaseJudgeAdapter

    original = BaseJudgeAdapter.invoke

    def invoke(self, input_params):
        with INFLIGHT.track("judge", input_params.assessment_name or "?"):
            return original(self, input_params)

    BaseJudgeAdapter.invoke = invoke
    try:
        yield
    finally:
        BaseJudgeAdapter.invoke = original


def _scorers(judge: str, with_reference: bool) -> list:
    from mlflow.genai.scorers import (
        Guidelines,
        RelevanceToQuery,
        RetrievalGroundedness,
        RetrievalRelevance,
        RetrievalSufficiency,
        ToolCallCorrectness,
        ToolCallEfficiency,
    )

    # `model=` on every scorer, always: MLflow's default is openai:/gpt-4.1-mini,
    # which needs an OpenAI key and is a model nobody chose.
    scorers = [
        RetrievalGroundedness(model=judge),
        RelevanceToQuery(model=judge),
        RetrievalRelevance(model=judge),
        Guidelines(name="citation_and_refusal", guidelines=GUIDELINES, model=judge),
        # no `expected_tool_calls` in the dataset, so the judge rates whether the
        # calls fit the question — e.g. it answered without searching at all
        ToolCallCorrectness(model=judge),
        ToolCallEfficiency(model=judge),
    ]
    if with_reference:
        from mlflow.genai.scorers import Correctness

        scorers += [Correctness(model=judge), RetrievalSufficiency(model=judge)]
    return scorers


def _size_mlflow_http_pool() -> None:
    """Give MLflow's HTTP session one connection per concurrent request.

    `evaluate` logs from up to eval workers x scorer workers threads at once, plus
    the trace-export workers; with the default pool of 10 the surplus
    connections are closed after use, each with a "Connection pool is full"
    warning. MLflow reads the size once, when it creates its cached session on the
    first request — so this must run before `configure`. An explicit setting wins.
    """
    from mlflow.environment_variables import (
        MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS,
        MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS,
        MLFLOW_GENAI_EVAL_MAX_WORKERS,
    )

    size = (
        MLFLOW_GENAI_EVAL_MAX_WORKERS.get() * MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS.get()
        + MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS.get()
    )
    os.environ.setdefault("MLFLOW_HTTP_POOL_MAXSIZE", str(size))


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
    stacks = Path("mlruns") / "agent" / f"stacks-{datetime.now():%Y%m%d-%H%M%S}.txt"
    stacks.parent.mkdir(parents=True, exist_ok=True)
    with INFLIGHT.watch(stacks), _logging_judge_calls():
        if active():
            _size_mlflow_http_pool()
            INFLIGHT.progress("connecting to MLflow")
            configure(EXPERIMENT)
        INFLIGHT.progress("opening the retriever")
        retriever = Retriever(settings, params)
        try:
            INFLIGHT.progress("evaluate started")
            result = mlflow.genai.evaluate(
                data=build_dataset(entries),
                predict_fn=_predict_fn(
                    lambda: RagAgent(settings, params, retriever=retriever)
                ),
                scorers=_scorers(judge, bool(with_reference)),
            )
            INFLIGHT.progress("evaluate finished")
        finally:
            retriever.close()

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
