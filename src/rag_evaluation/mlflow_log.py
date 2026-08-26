"""Best-effort MLflow logging for tier 1.

Tier 2 goes through `mlflow.genai.evaluate()`, which logs itself; tier 1 does
not, so it logs here. Same rule as `rag_agent.tracing`: an unreachable tracking
server is an observability problem, never a reason for an eval run to fail, and
an empty tracking URI disables it outright so offline runs stay quiet.

Every knob is logged, not only the one that changed — a configuration cannot be
reconstructed from a delta, and a score without its configuration is not a data
point.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone

import urllib3

from rag_agent.settings import Settings
from rag_config import load_params
from rag_config.params import ingest_fingerprint

__all__ = [
    "log_tier1",
    "generation_run",
    "trace_session",
    "new_session_id",
    "active",
    "tracking_uri",
    "configure",
    "register_prompt",
    "EXPERIMENT",
    "DATASET_EXPERIMENT",
]

log = logging.getLogger(__name__)

# (name, template) pairs already sent this process, so a second generate() in
# one process does not repeat two HTTP roundtrips for an unchanged prompt.
_registered_prompts: set[tuple[str, str]] = set()

# Two experiments, both overridable from the environment (the local env file in
# dev, the job spec on the cluster) the way every other knob here is:
#
# - RAG-Evaluation          scoring the pipeline — tier 1 here, tiers 2 and 3 in
#                           agent_eval, so a retrieval change and a judge run
#                           sit side by side
# - RAG-Evaluation-Dataset  producing the golden set those tiers score against.
#                           Separate because it shares no metrics with them and
#                           would only clutter their comparisons.
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_EVALUATION", "RAG-Evaluation")
DATASET_EXPERIMENT = os.environ.get(
    "MLFLOW_EXPERIMENT_DATASET", "RAG-Evaluation-Dataset"
)


def tracking_uri() -> str:
    """Where to log, or "" for nowhere.

    `MLFLOW_TRACKING_URI` is MLflow's own variable, so the library picks up the
    same value without being told. Empty means tracing is off, the way it stays
    quiet in tests and on any box without the service.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", "")


def active() -> bool:
    return bool(tracking_uri())


def configure(experiment: str) -> None:
    """Point MLflow at the server, quiet it down, and select the experiment.

    The same setup `rag_agent.tracing.setup_tracing` does, duplicated here for
    the reason the stages duplicate their infra: this module owns its own MLflow
    contact. Callers do this before any other MLflow call.

    - Warnings off. The tracking server's certificate is self-signed, and this
      also silences MLflow's own complaints about calls that do nothing — so
      everything here logs its own failures rather than relying on MLflow.
    - Insecure TLS and proxy multipart upload: what that server needs to accept
      artifacts at all.
    """
    import mlflow

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)


def _params(settings: Settings) -> dict[str, object]:
    params = load_params()
    flat: dict[str, object] = {"ingest_fingerprint": ingest_fingerprint(params)}
    for section in ("parse_chunk", "embed", "agent", "plan", "finalize"):
        for key, value in asdict(getattr(params, section)).items():
            flat[f"{section}.{key}"] = value
    # The settings actually in force, which env overrides may have moved off the
    # file's values.
    flat.update(
        {
            "run.qdrant_collection": settings.qdrant_collection,
            "run.top_k": settings.top_k,
            "run.rerank_enabled": settings.rerank_enabled,
            "run.rerank_top_n": settings.rerank_top_n,
            "run.parameters_path": os.environ.get("RAG_PARAMETERS_PATH", "(packaged)"),
        }
    )
    return flat


def log_tier1(
    report,
    settings: Settings,
    golden_path: str,
    run_name: str | None = None,
    experiment: str = EXPERIMENT,
) -> None:
    """One MLflow run for one tier-1 scoring. Silent when tracing is off.

    Tier 1 scores the retrieval, so it lands in the same experiment as tiers 2
    and 3 — see EXPERIMENT above.
    """
    if not active():
        return
    try:
        import mlflow

        configure(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(_params(settings))
            mlflow.log_params(
                {"golden_set": golden_path, "golden_entries": len(report.results)}
            )
            mlflow.log_metrics(report.means)
            # Coverage is a metric, not a footnote: it says how much of the set
            # the means are actually over.
            mlflow.log_metrics(
                {
                    "scored_entries": len(report.scored),
                    "coverage": len(report.scored) / len(report.results)
                    if report.results
                    else 0.0,
                }
            )
            for attribute in ("language", "source_system"):
                for group, means in report.by(attribute).items():
                    for name, value in means.items():
                        mlflow.log_metric(f"{attribute}.{group}.{name}", value)
            mlflow.log_text(_per_entry_csv(report), "per_entry.csv")
    except Exception as exc:
        log.warning("MLflow logging failed, scores are still above: %s", exc)


def _per_entry_csv(report) -> str:
    """Per-entry rows, so a regression is traceable to the question that caused it."""
    header = "id,language,scored,recall,precision,mrr,ndcg,chunk_precision,n_chunks,n_docs"
    rows = [header]
    for r in report.results:
        if r.scores is None:
            rows.append(f"{r.entry.id},{r.entry.language or ''},0,,,,,,{len(r.chunks)},")
            continue
        s = r.scores
        rows.append(
            f"{r.entry.id},{r.entry.language or ''},1,{s.recall:.4f},{s.precision:.4f},"
            f"{s.mrr:.4f},{s.ndcg:.4f},{s.chunk_precision:.4f},{s.n_chunks},{s.k}"
        )
    return "\n".join(rows)


def register_prompt(name: str, template: str) -> None:
    """Store the prompt text in MLflow's prompt registry, best-effort.

    - new text: a new version
    - unchanged text: nothing, so versions track edits and not re-runs
    - already sent this process: skipped

    The artifact on the run says what *this* run used; the registry says how the
    prompt has changed over time, and which runs shared a version. Mirrors
    `rag_agent.tracing.register_prompt`, without that one's opt-in flag: a
    generate run is a batch job, so two roundtrips do not matter the way they do
    in a chat session.
    """
    if not active() or (name, template) in _registered_prompts:
        return
    try:
        import mlflow

        with _http_limits(retries="0", timeout="5"):
            latest = mlflow.genai.load_prompt(name, allow_missing=True)
            if latest is None or latest.template != template:
                mlflow.genai.register_prompt(
                    name=name,
                    template=template,
                    commit_message="Evaluation question writer",
                )
                log.info("registered prompt %r", name)
        _registered_prompts.add((name, template))
    except Exception as exc:
        log.warning("prompt registration skipped: %s", exc)


@contextlib.contextmanager
def _http_limits(retries: str, timeout: str) -> Iterator[None]:
    """Cap MLflow's HTTP retries and timeout; it reads both per request.

    Duplicated from `rag_agent.tracing` rather than imported — same reason the
    stages duplicate their infra: this module owns its own MLflow contact.
    Without it, an unreachable server costs minutes of exponential backoff
    before generation starts.
    """
    keys = {
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": retries,
        "MLFLOW_HTTP_REQUEST_TIMEOUT": timeout,
    }
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _best_effort(what: str) -> Iterator[None]:
    """Observability must never break the run it is watching."""
    try:
        yield
    except Exception as exc:
        log.warning("MLflow %s failed, generation is unaffected: %s", what, exc)


def new_session_id() -> str:
    """One id per execution. Timestamp first, so sessions sort by when they ran."""
    return f"generate-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


@contextlib.contextmanager
def trace_session(session_id: str) -> Iterator[None]:
    """Tag traces made in *this thread* with the session id.

    `mlflow.tracing.context` is a contextvar, and contextvars do not cross into
    ThreadPoolExecutor workers — a session entered on the main thread leaves the
    workers' traces untagged. So each worker enters it itself. (The active run
    does propagate; only the tracing context does not.)

    Setup is best-effort and the `yield` stays outside the `except`, or an
    exception from the body lands in the handler and yields a second time —
    "generator didn't stop after throw()", which hides the real error.
    """
    with contextlib.ExitStack() as stack:
        try:
            import mlflow

            stack.enter_context(mlflow.tracing.context(session_id=session_id))
        except Exception as exc:
            log.warning("session context unavailable: %s", exc)
        yield


@contextlib.contextmanager
def generation_run(
    prompt: str,
    params: dict[str, object],
    session_id: str | None = None,
    experiment: str = DATASET_EXPERIMENT,
    prompt_name: str = "evaluation_generate_prompt",
) -> Iterator[dict[str, float]]:
    """One run around a whole `generate` pass, with every model call traced.

    Yields a dict the caller fills with metrics; they are logged on exit. It is
    a real dict even when tracing is off, so the caller needs no guard.

    `mlflow.openai.autolog()` rather than the `llama_index` autolog
    `rag_agent.tracing` uses — generate drives the OpenAI SDK directly, so the
    completion call is the only span there is.

    The prompt is logged as an artifact and hashed into a param: two golden sets
    are only comparable if the instructions that produced them are on record,
    and the hash is what groups runs that share a prompt.

    Every execution gets its own session id, so the calls of one pass come back
    together under `metadata."mlflow.trace.session"` the way a chat session does
    in `rag_agent.tracing`. Timestamp first, so sessions sort by when they ran.
    """
    metrics: dict[str, float] = {}
    session_id = session_id or new_session_id()
    if not active():
        yield metrics
        return
    try:
        import mlflow

        # Setup only: a dead server must fail here in seconds, not stall the
        # whole generation pass. The run itself keeps MLflow's normal limits,
        # since by then the server has answered.
        with _http_limits(retries="0", timeout="5"):
            configure(experiment)
            mlflow.openai.autolog()
            started = mlflow.start_run()
    except Exception as exc:
        # No tracking server, no traces — but the generation still happens.
        log.warning("MLflow tracing disabled, generation continues: %s", exc)
        yield metrics
        return

    # One yield on every path below: the caller's own exceptions propagate
    # untouched, and only MLflow's are swallowed.
    with started, contextlib.ExitStack() as stack:
        with _best_effort("parameter logging"):
            mlflow.log_params(
                dict(
                    params,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()[:12],
                    session_id=session_id,
                )
            )
            mlflow.log_text(prompt, "system_prompt.txt")
        register_prompt(prompt_name, prompt)
        # Covers anything traced on this thread; the workers enter their own
        # via trace_session, since contextvars do not cross threads.
        stack.enter_context(trace_session(session_id))
        try:
            yield metrics
        finally:
            with _best_effort("metric logging"):
                if metrics:
                    mlflow.log_metrics(metrics)
