"""Score tier 1 across a range of one query parameter, and plot every metric.

One knob, many values, one MLflow parent run with a child per value. Same shape
as sweep.py's inner loop, but driven from the command line and it draws the
curves — the question it answers is "where does this parameter stop paying",
which a single number cannot show.

Query-time parameters only: everything here changes without a re-ingest, so a
whole curve costs seconds per value against the standing collection.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

from rag_agent.settings import Settings, get_settings

from rag_evaluation.golden import Entry
from rag_evaluation.metrics import METRIC_NAMES
from rag_evaluation.retrieval_eval import Report, run_tier1

__all__ = [
    "SWEEPABLE",
    "curve_csv",
    "format_table",
    "make_figures",
    "parse_overrides",
    "parse_values",
    "run_curve",
]

log = logging.getLogger(__name__)

# The retrieval shape (block 7) plus the excerpt knobs. Deliberately not every
# Settings field: urls, model ids and the LLM settings are not tuning knobs, and
# a typo should say so rather than sweep something meaningless. rerank_top_n is
# absent on purpose: run_tier1 passes `limit`, which overrides it inside
# Retriever.search, so sweeping it here would measure nothing.
SWEEPABLE = (
    "prefetch_limit",
    "top_k",
    "rrf_k",
    "rrf_dense_weight",
    "rrf_sparse_weight",
    "rerank_enabled",
    "truncate_chunks",
    "max_excerpt_chars",
)

_TYPES = {f.name: f.type for f in fields(Settings)}


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
        if name not in SWEEPABLE:
            raise CurveError(
                f"{name!r} cannot be set — one of {', '.join(sorted(SWEEPABLE))}"
            )
        out[name] = _cast(raw, _TYPES.get(name, "int"))
    return out


def run_curve(
    entries: Sequence[Entry],
    param: str,
    values: Sequence[Any],
    settings: Settings | None = None,
) -> list[tuple[Any, Report]]:
    """One tier-1 report per value: set the field on a copy of Settings, and
    score with that."""
    if param not in SWEEPABLE:
        raise CurveError(
            f"{param!r} is not sweepable — one of {', '.join(sorted(SWEEPABLE))}"
        )
    if not values:
        raise CurveError("no values to sweep")
    base = settings or get_settings()

    rows: list[tuple[Any, Report]] = []
    for value in values:
        log.info("%s=%s", param, value)
        rows.append((value, run_tier1(entries, settings=replace(base, **{param: value}))))
    return rows


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
        axis.set(xlabel=param, ylabel=name, title=f"{name} vs {param}", ylim=(0, 1.02))
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
    fig.suptitle(f"tier 1 metrics vs {param}")
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
