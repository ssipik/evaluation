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
import sys
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import urllib3

from rag_agent.settings import Settings
from rag_config import Params, ingest_fingerprint, load_settings, set_param
from rag_config.baseline import BaselineError, resolve_baseline

__all__ = [
    "BaselineError",
    "load_baseline",
    "log_tier1",
    "log_curve",
    "log_sweep",
    "log_search",
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


def _params(settings: Settings, params: Params) -> dict[str, object]:
    """Everything a score has to be read against: the model parameters in force,
    and the config that says which index they ran on.

    No value appears twice. Params is already the effective set — the file with
    RAG_*, --set-params and any loaded baseline applied — so there is no second
    copy of the file's own values to disagree with it.
    """
    flat: dict[str, object] = {"ingest_fingerprint": ingest_fingerprint(params)}
    # The ingest half: what built the index these numbers were measured on.
    for section in ("parse_chunk", "embed"):
        for key, value in asdict(getattr(params, section)).items():
            flat[f"{section}.{key}"] = value
    # The query half, under its own section name like the other two.
    for key, value in asdict(params.agent).items():
        flat[f"agent.{key}"] = value
    # Which index the numbers were measured on. The parameters file behind them
    # is not logged: every value in it is already above, under its own name.
    flat["settings.qdrant_collection"] = settings.qdrant_collection
    # The Qdrant field names are config, not parameters — renaming one moves a
    # vector rather than changing it — but a score still has to say which fields
    # it was measured over.
    shared = load_settings()
    flat["settings.dense_vector"] = shared.dense_vector
    flat["settings.sparse_vectors"] = shared.sparse_vectors
    return flat


def load_baseline(ref: str) -> tuple[dict[str, str], str, str | None]:
    """The query knobs of a registered baseline, for `curve --mlflow-model`.

    The registry read itself is `rag_config.baseline`, beside the parameters.yaml
    a baseline is promoted into; this is the `agent.` half of it, which is what
    a curve starts from.
    """
    baseline = resolve_baseline(ref)
    return baseline.section("agent."), baseline.summary, baseline.fingerprint


def log_tier1(
    report,
    settings: Settings,
    params: Params,
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
            _log_one_report(report, settings, params, golden_path)
    except Exception as exc:
        log.warning("MLflow logging failed, scores are still above: %s", exc)


def log_sweep(
    results,
    spec: dict,
    golden_path: str,
    run_name: str | None = None,
    experiment: str = EXPERIMENT,
) -> None:
    """One parent run for the sweep, one nested child per candidate.
    - parent: the sweep definition, plus the summary csv over every candidate
    - child:  the full report for one candidate, params carrying that knob
    """
    if not active() or not results:
        return
    try:
        import mlflow

        from rag_evaluation.sweep import sweep_csv

        configure(experiment)
        with mlflow.start_run(run_name=run_name) as parent:
            mlflow.log_params(
                {
                    "swept_params": ",".join(spec.get("sweep") or {}),
                    **{
                        f"sweep.{knob}": ",".join(str(v) for v in values)
                        for knob, values in (spec.get("sweep") or {}).items()
                    },
                    **{
                        f"baseline.{knob}": value
                        for knob, value in (spec.get("baseline") or {}).items()
                    },
                    # A `candidates:` entry moves several knobs at once, so it
                    # is logged whole rather than as one value list per knob.
                    **{
                        f"candidate.{entry.get('label', i)}": ",".join(
                            f"{k}={v}" for k, v in (entry.get("set") or {}).items()
                        )
                        for i, entry in enumerate(spec.get("candidates") or [])
                    },
                }
            )
            for scored in results:
                if scored.report is None:
                    continue
                label = scored.candidate.label
                child = f"{run_name}-{label}" if run_name else label
                with mlflow.start_run(run_name=child, nested=True):
                    _log_one_report(
                        scored.report, scored.settings, scored.params, golden_path
                    )
            csv = sweep_csv(results)
            mlflow.log_text(csv, "summary.csv")
            _save_sweep_locally(csv, run_name, parent.info.run_id)
    except Exception as exc:
        log.warning("MLflow logging failed, scores are still above: %s", exc)


def log_search(
    results,
    study,
    objective,
    spec: dict,
    settings: Settings,
    golden_path: str,
    run_name: str | None = None,
    experiment: str = EXPERIMENT,
) -> None:
    """One parent run for the study, one nested child per trial.

    - parent: the search space, the winner, and the study's plots and csv
    - child:  the full report for one trial, params carrying that trial's knobs

    `log_curve`'s shape. The parent carries no step-indexed metrics, because a
    joint search has no single x axis to index them by — its equivalent is
    Optuna's own history and contour plots.
    """
    if not active() or not results:
        return
    try:
        import mlflow

        from rag_evaluation.search import best, make_figures, search_csv

        configure(experiment)
        with mlflow.start_run(run_name=run_name) as parent:
            mlflow.log_params(
                {
                    # Resolved, not what the file asked for: `floor: baseline`
                    # is a number only once run_search has scored the baseline.
                    **objective.params,
                    "trials": len(results),
                    **{
                        f"search.{knob}": str(space)
                        for knob, space in (spec.get("search") or {}).items()
                    },
                    **{
                        f"baseline.{knob}": value
                        for knob, value in (spec.get("baseline") or {}).items()
                    },
                }
            )
            for trial in results:
                if trial.report is None:
                    continue
                child = (
                    f"{run_name}-{trial.label}" if run_name else trial.label
                )
                with mlflow.start_run(run_name=child, nested=True):
                    mlflow.log_metric("objective", trial.value)
                    # Cost beside the score: with reranking on, trials differ by
                    # orders of magnitude and the comparison is meaningless
                    # without it.
                    mlflow.log_metric("seconds", trial.seconds)
                    _log_one_report(trial.report, settings, trial.params, golden_path)
            winner = best(results, objective)
            if winner:
                with _best_effort("winner"):
                    mlflow.log_params(
                        {f"best.{k}": v for k, v in winner.knobs.items()}
                        | {"best.trial": winner.number}
                    )
                    mlflow.log_metrics(winner.report.means)
            wall = float(study.user_attrs.get("wall_seconds", 0.0))
            if wall:
                mlflow.log_metric("wall_seconds", wall)
                mlflow.log_metric("scoring_seconds", sum(r.seconds for r in results))
            csv = search_csv(results)
            mlflow.log_text(csv, "trials.csv")
            figures = make_figures(study)
            for path, figure in figures.items():
                mlflow.log_figure(figure, path)
            _save_search_locally(csv, run_name, parent.info.run_id, figures)
            _close_figures(figures.values())
    except Exception as exc:
        # The type, not only the message: a NameError here is a bug in this
        # module and reads nothing like an unreachable tracking server.
        log.warning(
            "MLflow logging failed (%s), scores are still above: %s",
            type(exc).__name__,
            exc,
        )


def _save_search_locally(
    csv: str, run_name: str | None, run_id: str, figures
) -> None:
    """`mlruns/search/<run name>/<parent run id>/`, the layout `_save_locally`
    writes a curve into and for the same reason: names repeat."""
    out = Path("mlruns") / "search" / (run_name or "unnamed") / run_id
    try:
        (out / "plots").mkdir(parents=True, exist_ok=True)
        (out / "trials.csv").write_text(csv, encoding="utf-8")
        for path, figure in figures.items():
            figure.savefig(out / path, dpi=110)
        print(f"  saved to {out}", file=sys.stderr)
    except OSError as exc:
        log.warning("could not write %s: %s", out, exc)


def _save_sweep_locally(csv: str, run_name: str | None, run_id: str) -> None:
    """`mlruns/sweep/<run name>/<parent run id>/summary.csv`, the same layout
    `_save_locally` writes a curve into and for the same reason: names repeat."""
    out = Path("mlruns") / "sweep" / (run_name or "unnamed") / run_id
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.csv").write_text(csv, encoding="utf-8")
        print(f"  saved to {out}", file=sys.stderr)
    except OSError as exc:
        log.warning("could not write %s: %s", out, exc)


def _log_golden_dataset(report, golden_path: str) -> None:
    """Record which golden set produced these numbers, as a hashed dataset input.

    The digest is what makes two runs comparable — same digest, same questions
    and labels — so `mlflow.search_runs(filter_string="datasets.digest='…'")`
    answers whether a delta is attributable to the config or to the set having
    moved under it (EVALUATION.md's frozen-set rule).

    The input also carries the file it came from (`source`) and how many entries
    it holds (`profile.num_rows`), so neither is logged as a parameter.

    - sorted by id, so reordering the file is not a new set
    - `tests` left out: it is a note about an entry, not something scored
    """
    import mlflow
    import pandas as pd

    rows = [
        {
            "id": entry.id,
            "question": entry.question,
            "language": entry.language or "",
            "reference": entry.reference,
            "expect_refusal": entry.expect_refusal,
            "relevant_documents": "; ".join(
                sorted(f"{d.source_system}|{d.url}" for d in entry.relevant_documents)
            ),
        }
        for entry in sorted((r.entry for r in report.results), key=lambda e: e.id)
    ]
    dataset = mlflow.data.from_pandas(
        pd.DataFrame(rows), name="golden_set", source=golden_path
    )
    mlflow.log_input(dataset, context="eval")


def _log_parameter_carrier() -> None:
    """A placeholder model on the run, so the UI offers "Register Model".

    The registry versions models, not runs, and the UI only shows the button on
    a run that has one — this is that hook and nothing more. The parameters it
    stands for are the run's own `agent.*` params, which is what
    `promote` and `curve --mlflow-model` read back; the model is never loaded.

    `pip_requirements=[]` skips MLflow's environment capture, which would
    otherwise export the whole uv.lock into every child run of every sweep.
    """
    import mlflow

    class ParameterSet(mlflow.pyfunc.PythonModel):
        # Type hints only to keep MLflow from warning about signature
        # inference; nothing calls this.
        def predict(self, context, model_input: list[str], params=None) -> list[str]:
            return model_input

    mlflow.pyfunc.log_model(
        name="rag-retrieval-params", python_model=ParameterSet(), pip_requirements=[]
    )


def _log_one_report(
    report, settings: Settings, params: Params, golden_path: str
) -> None:
    """Params, means, the per-attribute split and the per-entry csv of one
    report, into whichever run is active."""
    import mlflow

    _log_golden_dataset(report, golden_path)
    mlflow.log_params(_params(settings, params))
    mlflow.log_metrics(report.means)
    # Coverage is a metric, not a footnote: it says how much of the set the
    # means are actually over.
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
    _log_parameter_carrier()


def log_curve(
    rows,
    param: str,
    settings: Settings,
    params: Params,
    golden_path: str,
    run_name: str | None = None,
    experiment: str = EXPERIMENT,
    collections: dict | None = None,
) -> None:
    """One parent run for the sweep, one nested child per value.
    - parent: the swept values as step-indexed metrics, plus the plots and csv
    - child:  the full report for one value, params carrying that value, and
      the collection it scored when `collections` says each value had its own
    Steps are the values themselves when they are whole numbers, so MLflow's
    own chart reads as metric vs parameter; otherwise the index, and the csv
    artifact carries the real values.
    """
    if not active() or not rows:
        return
    try:
        import mlflow

        from dataclasses import replace

        from rag_evaluation.curve import curve_csv, make_figures

        configure(experiment)
        values = [value for value, _ in rows]
        stepped = all(float(v) == int(v) for v in values if isinstance(v, (int, float)))
        deepest = rows[-1][1]
        with mlflow.start_run(run_name=run_name) as parent:
            _log_golden_dataset(deepest, golden_path)
            mlflow.log_params(
                {
                    **_params(settings, params),
                    "swept_param": param,
                    "swept_values": ",".join(str(v) for v in values),
                    **(
                        {"swept_collections": ",".join(collections[v] for v in values)}
                        if collections
                        else {}
                    ),
                }
            )
            for i, (value, report) in enumerate(rows):
                step = int(value) if stepped and not isinstance(value, bool) else i
                for name, mean in report.means.items():
                    mlflow.log_metric(name, mean, step=step)
                child = f"{run_name}-{param}={value}" if run_name else f"{param}={value}"
                # _log_one_report logs the value under the same `section.field`
                # name, the one promote and load_baseline read.
                scored = (
                    replace(settings, qdrant_collection=collections[value])
                    if collections
                    else settings
                )
                with mlflow.start_run(run_name=child, nested=True):
                    _log_one_report(
                        report,
                        scored,
                        set_param(params, param, value),
                        golden_path,
                    )
            mlflow.log_text(curve_csv(rows, param), f"{param}_curve.csv")
            figures = make_figures(rows, param)
            for path, figure in figures.items():
                mlflow.log_figure(figure, path)
            _save_locally(rows, param, run_name, parent.info.run_id, figures)
            _close_figures(figures.values())
    except Exception as exc:
        log.warning("MLflow logging failed, scores are still above: %s", exc)


def _save_locally(rows, param: str, run_name: str | None, run_id: str, figures) -> None:
    """A copy of the table and the charts on disk, beside what MLflow stores.

    `mlruns/curve/<run name>/<parent run id>/` — the run id because names repeat
    and only the id says which run a directory belongs to.
    """
    from rag_evaluation.curve import curve_csv, format_table

    out = Path("mlruns") / "curve" / (run_name or "unnamed") / run_id
    try:
        (out / "plots").mkdir(parents=True, exist_ok=True)
        (out / f"{param}_curve.txt").write_text(
            format_table(rows, param) + "\n", encoding="utf-8"
        )
        (out / f"{param}_curve.csv").write_text(curve_csv(rows, param), encoding="utf-8")
        for path, figure in figures.items():
            figure.savefig(out / path, dpi=110)
        print(f"  saved to {out}", file=sys.stderr)
    except OSError as exc:
        log.warning("could not write %s: %s", out, exc)


def _close_figures(figures) -> None:
    """Matplotlib keeps every figure alive until it is closed."""
    import matplotlib.pyplot as plt

    for figure in figures:
        plt.close(figure)


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
