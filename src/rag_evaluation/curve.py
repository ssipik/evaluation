"""Score tier 1 across a range of one parameter, and plot every metric.

One knob, many values, one MLflow parent run with a child per value. Same shape
as sweep.py's inner loop, but driven from the command line and it draws the
curves — the question it answers is "where does this parameter stop paying",
which a single number cannot show.

`run_curve` is `sweep.run_sweep` over one knob — same candidates, same
fingerprint routing, same snapshot, ledger and stages when `--reingest` builds.
Read the two side by side; the docstring of `run_curve` lists every difference.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import fields
from typing import Any

from rag_agent.settings import get_params, get_settings
from rag_config import (
    AgentParams,
    EmbedParams,
    Params,
    ParseChunkParams,
    ingest_fingerprint,
    load_params,
)

from rag_evaluation.golden import Entry
from rag_evaluation.metrics import METRIC_NAMES
from rag_evaluation.retrieval_eval import Report, run_tier1
from rag_evaluation.sweep import (
    Candidate,
    Scored,
    _set,
    build,
    collection_exists,
    frozen_snapshot,
    write_candidate,
)

__all__ = [
    "INGEST_SWEEPABLE",
    "SWEEPABLE",
    "candidates",
    "cast_params",
    "curve_csv",
    "format_table",
    "make_figures",
    "parse_overrides",
    "parse_values",
    "run_curve",
]

log = logging.getLogger(__name__)

# Every model parameter the agent has, minus the ones tier 1 cannot move.
# Derived rather than listed, so a knob added to rag-config becomes sweepable
# without anyone remembering this file — the same "include the section, name the
# exceptions" rule INGEST_FINGERPRINT_EXCLUDE uses, and for the same reason: a
# knob wrongly included is an obvious wasted run, one wrongly left out is never
# noticed.
NOT_SWEEPABLE = frozenset(
    {
        # run_tier1 passes `limit`, which overrides rerank_top_n inside
        # Retriever.search, so sweeping it would plot a flat line.
        "rerank_top_n",
        # A container serves one reranker, chosen at start (../tei_rerank), so a
        # value here would be scored against whatever is actually running.
        "rerank_model",
        # Tier 1 is judge-free retrieval scoring and never calls the model.
        "llm_model",
        "llm_temperature",
        "llm_max_tokens",
    }
)
# Spelled `section.field` like everywhere else — sweep files, set_param, the
# MLflow params — so one name means one knob in every command.
SWEEPABLE = tuple(
    f"agent.{f.name}" for f in fields(AgentParams) if f.name not in NOT_SWEEPABLE
)

# The index-shaping knobs a curve can move: the numeric fields of the two
# fingerprinted sections. Model names and the sparse-model map
# are left out — no value list on a command line spells them.
INGEST_SWEEPABLE = tuple(
    f"{section}.{f.name}"
    for section, cls in (("parse_chunk", ParseChunkParams), ("embed", EmbedParams))
    for f in fields(cls)
    if isinstance(f.default, (int, float)) and not isinstance(f.default, bool)
)

_TYPES = {
    f"{section}.{f.name}": type(f.default).__name__
    for section, cls in (
        ("agent", AgentParams),
        ("parse_chunk", ParseChunkParams),
        ("embed", EmbedParams),
    )
    for f in fields(cls)
}


class CurveError(ValueError):
    """The sweep was asked for something it cannot do."""


def _cast(raw: str, kind: str) -> Any:
    raw = raw.strip()
    if kind == "bool":
        if raw.lower() not in {"true", "false"}:
            raise CurveError(f"{raw!r} is not true or false")
        return raw.lower() == "true"
    try:
        return float(raw) if kind == "float" else int(raw)
    except ValueError as exc:
        raise CurveError(f"{raw!r} is not a {kind}") from exc


def parse_values(spec: str, param: str) -> list[Any]:
    """A value list from `1,5,20` or an inclusive interval from `1:21[:step]`.

    A negative step counts down, so `21:1:-1` is `1:21` reversed.
    """
    kind = _TYPES.get(param, "int")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise CurveError(f"{spec!r}: an interval is start:stop or start:stop:step")
        start, stop = (_cast(parts[0], kind), _cast(parts[1], kind))
        step = _cast(parts[2], kind) if len(parts) == 3 else (1 if kind == "int" else 1.0)
        if step == 0:
            raise CurveError("step cannot be zero")
        if (stop - start) * step < 0:
            raise CurveError(
                f"{spec!r}: step {step} runs away from {stop} — "
                f"use {stop}:{start}:{step} to count down"
            )
        slack = 1e-9 if kind == "float" else 0
        values, v = [], start
        while (v <= stop + slack) if step > 0 else (v >= stop - slack):
            values.append(round(v, 6) if kind == "float" else v)
            v += step
        return values
    return [_cast(part, kind) for part in spec.split(",") if part.strip()]


def parse_overrides(specs: Sequence[str]) -> dict[str, Any]:
    """`name=value` pairs: the knobs held at a chosen value while another is swept.

    The runtime half of the precedence in CLAUDE.md — parameters.yaml is the
    source, this moves one value for one run without editing the shared file.
    """
    out: dict[str, Any] = {}
    for spec in specs:
        name, sep, raw = spec.partition("=")
        name = name.strip()
        if not sep:
            raise CurveError(f"{spec!r}: expected name=value")
        if name not in SWEEPABLE and name not in INGEST_SWEEPABLE:
            raise CurveError(
                f"{name!r} cannot be set — one of "
                f"{', '.join(sorted(SWEEPABLE + INGEST_SWEEPABLE))}"
            )
        out[name] = _cast(raw, _TYPES.get(name, "int"))
    return out


def cast_params(raw: Mapping[str, str]) -> dict[str, Any]:
    """A baseline's `agent.` section, typed, under the names --param uses.

    MLflow params come back as strings, and `Baseline.section` strips the
    prefix. Anything outside SWEEPABLE is dropped — a baseline records the
    collection and the parameters file it ran against too, and neither is a knob.
    """
    return {
        f"agent.{key}": _cast(value, _TYPES[f"agent.{key}"])
        for key, value in raw.items()
        if f"agent.{key}" in SWEEPABLE
    }


def candidates(
    param: str, values: Sequence[Any], baseline: Params | None = None
) -> Iterator[Candidate]:
    """One candidate per value, each changing `param` from the baseline — a
    sweep file's `sweep:` list for one knob, without the baseline candidate."""
    if param not in SWEEPABLE and param not in INGEST_SWEEPABLE:
        raise CurveError(
            f"{param!r} is not sweepable — one of "
            f"{', '.join(sorted(SWEEPABLE + INGEST_SWEEPABLE))}"
        )
    params = baseline or load_params()
    for value in values:
        yield Candidate(
            label=f"{param}={value}",
            knob=param,
            value=value,
            params=_set(params, param, value),
            changes=((param, value),),
        )


def run_curve(
    param: str,
    values: Sequence[Any],
    entries: list[Entry],
    reingest: bool = False,
    dry_run: bool = False,
    overwrite_frozen: bool = False,
    baseline: Params | None = None,
    ingest_spec: dict | None = None,
) -> list[Scored]:
    """Every value, scored on tier 1 — `sweep.run_sweep` over one knob.

    The body is run_sweep's, line for line. The differences, all of them:
    - candidates come from `candidates(param, values)`, not a sweep file, and
      there is no baseline candidate: the other values are the comparison
    - the ingest settings come in as `ingest_spec` (--ingest), not a file's
      `ingest:` block
    - no per-candidate report is printed; the caller prints the table
    """
    results: list[Scored] = []
    baseline_fp = ingest_fingerprint(load_params())

    current_collection = get_settings().qdrant_collection
    qdrant_url = get_settings().qdrant_url

    ingest_spec = ingest_spec or {}
    manifest_dsn = ""
    if reingest and not dry_run:
        manifest_dsn = frozen_snapshot(ingest_spec, overwrite_frozen)

    for candidate in candidates(param, values, baseline):
        if reingest:
            # See run_sweep: every candidate builds its own collection, the one
            # at the shipped parameters included.
            needs_build, collection = True, candidate.collection
        else:
            needs_build = candidate.fingerprint != baseline_fp
            collection = (
                candidate.collection
                if needs_build or collection_exists(candidate.collection, qdrant_url)
                else current_collection
            )
        path = write_candidate(candidate, collection)
        print(f"\n=== {candidate.label}  [{candidate.fingerprint}]  {collection}")

        if needs_build and not reingest:
            print(
                f"  skipped: {candidate.knob} changes the index, which needs "
                "--reingest and a full re-ingest"
            )
            results.append(Scored(candidate, None))
            continue
        if not needs_build and not collection_exists(collection, qdrant_url):
            print(
                f"  skipped: no collection {collection} to score — "
                "--reingest builds one"
            )
            results.append(Scored(candidate, None))
            continue
        if dry_run:
            results.append(Scored(candidate, None))
            continue

        os.environ["RAG_PARAMETERS_PATH"] = str(path)
        os.environ["RAG_QDRANT_COLLECTION"] = collection
        env = {"RAG_PARAMETERS_PATH": str(path), "RAG_QDRANT_COLLECTION": collection}
        settings = get_settings()
        if needs_build:
            build(candidate, collection, env, ingest_spec, manifest_dsn)

        effective = get_params()
        report = run_tier1(entries, settings=settings, params=effective)
        results.append(Scored(candidate, report, settings, effective))

    return results


def format_table(rows: Sequence[tuple[Any, Report]], param: str) -> str:
    """The metric means at every value, one row per value."""
    if not rows:
        return "\n  nothing to score — the sweep produced no reports."
    last = rows[-1][1]
    if not last.scored:
        return "\n  nothing to score — no entry has relevant_documents yet."
    lines = [
        f"\n=== {param}: {len(last.scored)} of {len(last.results)} entries scored, "
        f"rerank={'on' if last.rerank else 'off'} ===",
        f"  {param:>16}  " + "  ".join(f"{n:>10}" for n in METRIC_NAMES),
    ]
    for value, report in rows:
        means = report.means
        lines.append(
            f"  {str(value):>16}  "
            + "  ".join(f"{means[n]:>10.4f}" for n in METRIC_NAMES)
        )
    return "\n".join(lines)


def make_figures(rows: Sequence[tuple[Any, Report]], param: str) -> dict[str, Any]:
    """One figure per metric plus a combined overview, keyed by artifact path."""
    import matplotlib

    matplotlib.use("Agg")  # no display on this box, and none needed
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    sns.set_theme(style="whitegrid", context="notebook")

    # Every metric is computed over the top-k ranking, so the label carries the
    # depth. Sweeping top_k makes the depth the x axis, and it stays symbolic.
    depth = "k" if param == "agent.top_k" else str(rows[0][1].k)

    xs = [value for value, _ in rows]
    # A bool or string sweep has no numeric axis, so plot against position and
    # label the ticks with the values.
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs)
    positions = xs if numeric else list(range(len(xs)))

    def draw(axis, name: str) -> None:
        sns.lineplot(
            x=positions,
            y=[report.means.get(name, 0.0) for _, report in rows],
            marker="o",
            ax=axis,
        )
        label = f"{name}@{depth}"
        # y autoscaled, not pinned to 0-1: a curve whose metrics move within a
        # narrow band is flat and unreadable on the full range.
        axis.set(xlabel=param, ylabel=label, title=f"{label} vs {param}")
        if not numeric:
            axis.set_xticks(positions, [str(x) for x in xs])
        elif all(isinstance(x, int) for x in xs):
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    figures: dict[str, Any] = {}
    for name in METRIC_NAMES:
        fig, axis = plt.subplots(figsize=(6, 4))
        draw(axis, name)
        fig.tight_layout()
        figures[f"plots/{name}_vs_{param}.png"] = fig

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, name in zip(axes.ravel(), METRIC_NAMES, strict=False):
        draw(axis, name)
    title = f"tier 1 metrics vs {param}"
    fig.suptitle(title if param == "agent.top_k" else f"{title}  (top_k={depth})")
    fig.tight_layout()
    figures[f"plots/all_metrics_vs_{param}.png"] = fig
    return figures


def curve_csv(rows: Sequence[tuple[Any, Report]], param: str) -> str:
    """The numbers behind the plots, so a curve can be re-read without them."""
    header = f"{param}," + ",".join(METRIC_NAMES) + ",scored_entries,coverage"
    lines = [header]
    for value, report in rows:
        means = report.means
        coverage = len(report.scored) / len(report.results) if report.results else 0.0
        lines.append(
            f"{value},"
            + ",".join(f"{means.get(n, 0.0):.6f}" for n in METRIC_NAMES)
            + f",{len(report.scored)},{coverage:.4f}"
        )
    return "\n".join(lines) + "\n"
