"""The tuning loop: change one parameter, measure, log, iterate.

Two nested loops, and the nesting is the point:

    for each ingest configuration:          expensive - builds a collection
        for each query configuration:       cheap - seconds, reuses it
            score tier 1, log one MLflow run

Reversed, every `top_k` value would pay for a re-ingest. Which loop a candidate
lands in is decided by `rag_config.ingest_fingerprint`, not by hand: it hashes
exactly those parameters whose change invalidates the index, so a query-side
sweep reuses `rag_chunks_<fingerprint>` automatically and a chunk-size change
gets a fresh one.

One knob at a time from a baseline, rather than a full grid: 3+2+2 candidates
instead of 12, each interpretable on its own. Take the winner, make it the new
baseline, sweep the next knob.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rag_agent.settings import get_params, get_settings
from rag_config.params import (
    Params,
    Settings,
    ingest_fingerprint,
    load_params,
    load_settings,
    set_param,
)

from rag_evaluation.golden import Entry
from rag_evaluation.retrieval_eval import Report, format_report, run_tier1

__all__ = [
    "Candidate",
    "SweepError",
    "load_sweep",
    "candidates",
    "run_sweep",
    "Scored",
    "sweep_csv",
    "format_summary",
]

log = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
COLLECTION_PREFIX = "rag_chunks_"

# What the guard refuses to touch. A sweep that writes into these has scored
# nothing and destroyed the working index.
PRODUCTION_COLLECTION = "rag_chunks"
PRODUCTION_DB = "rag_ingestion"


class SweepError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    """One configuration: a full parameter set plus what changed and why."""

    label: str
    knob: str
    value: Any
    params: Params

    @property
    def fingerprint(self) -> str:
        return ingest_fingerprint(self.params)

    @property
    def collection(self) -> str:
        return f"{COLLECTION_PREFIX}{self.fingerprint}"

    @property
    def directory(self) -> Path:
        """One directory per *collection*, named by the ingest fingerprint."""
        return RUNS_DIR / self.fingerprint

    @property
    def slug(self) -> str:
        """File name inside it: several query-side candidates share a collection,
        and each still needs its own parameter file."""
        if not self.knob:
            return "baseline"
        return f"{self.knob}={self.value}".replace("/", "_").replace(" ", "")


def load_sweep(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "sweep" not in raw:
        raise SweepError(f"{path}: no `sweep:` section")
    return raw


def _set(params: Params, dotted: str, value: Any) -> Params:
    """rag_config's setter, speaking this module's error type.

    A bad name in a sweep file is a sweep problem, and the CLI catches
    SweepError — so the ValueError does not escape as a traceback.
    """
    try:
        return set_param(params, dotted, value)
    except ValueError as exc:
        raise SweepError(str(exc)) from None


def candidates(spec: dict, baseline: Params | None = None) -> Iterator[Candidate]:
    """The baseline, then one candidate per (knob, value) — coordinate descent."""
    params = baseline or load_params()
    for dotted, value in (spec.get("baseline") or {}).items():
        params = _set(params, dotted, value)
    yield Candidate(label="baseline", knob="", value=None, params=params)
    for dotted, values in spec["sweep"].items():
        for value in values:
            yield Candidate(
                label=f"{dotted}={value}",
                knob=dotted,
                value=value,
                params=_set(params, dotted, value),
            )


@dataclass(frozen=True)
class Scored:
    """One candidate's outcome, with the configuration it actually ran under.

    `settings` and `params` are captured per candidate rather than read back
    later: a rebuilt candidate scores against its own collection, and the
    process env holds only the last one by the time the loop ends.
    """

    candidate: Candidate
    report: Report | None
    settings: Settings | None = None
    params: Params | None = None


def format_summary(results: list[Scored]) -> str:
    """The metric means of every candidate — `curve.format_table`, with the
    candidate label where a curve has the swept value."""
    from rag_evaluation.metrics import METRIC_NAMES

    scored = [s for s in results if s.report is not None and s.report.scored]
    if not results:
        return "\n  nothing to score — the sweep produced no candidates."
    if not scored:
        return "\n  nothing to score — no entry has relevant_documents yet."
    last = scored[-1].report
    # Labels are longer than a curve's values, so the column grows to fit rather
    # than truncating the one thing that says which candidate a row is.
    width = max(16, *(len(s.candidate.label) for s in results))
    lines = [
        f"\n=== sweep: {len(last.scored)} of {len(last.results)} entries scored, "
        f"rerank={'on' if last.rerank else 'off'} ===",
        f"  {'candidate':>{width}}  " + "  ".join(f"{n:>10}" for n in METRIC_NAMES),
    ]
    for s in results:
        if s.report is None or not s.report.scored:
            lines.append(f"  {s.candidate.label:>{width}}  {'(not scored)':>10}")
            continue
        means = s.report.means
        lines.append(
            f"  {s.candidate.label:>{width}}  "
            + "  ".join(f"{means[n]:>10.4f}" for n in METRIC_NAMES)
        )
    return "\n".join(lines)


def sweep_csv(results: list[Scored]) -> str:
    """One row per candidate — the curve's csv, for a sweep's star of candidates.

    Candidates that were skipped or not scored keep their row with empty metrics:
    which ones did not run is part of the record.
    """
    from rag_evaluation.metrics import METRIC_NAMES

    header = (
        "label,knob,value,fingerprint,"
        + ",".join(METRIC_NAMES)
        + ",scored_entries,coverage"
    )
    lines = [header]
    for candidate, report in ((s.candidate, s.report) for s in results):
        # The baseline changed no knob, so it carries no value — an empty
        # cell rather than the literal "None".
        value = candidate.value if candidate.knob else ""
        head = f"{candidate.label},{candidate.knob},{value},{candidate.fingerprint}"
        if report is None or not report.scored:
            lines.append(head + "," * (len(METRIC_NAMES) + 2))
            continue
        means = report.means
        coverage = len(report.scored) / len(report.results) if report.results else 0.0
        lines.append(
            head
            + ","
            + ",".join(f"{means.get(n, 0.0):.6f}" for n in METRIC_NAMES)
            + f",{len(report.scored)},{coverage:.4f}"
        )
    return "\n".join(lines) + "\n"


def write_candidate(candidate: Candidate, collection: str | None = None) -> Path:
    """The candidate's parameter file, under the directory naming its collection.

    `runs/<fingerprint>/<candidate>.yaml`. The directory *is* the registry of
    which parameters built which collection — no lookup table to keep in sync —
    and the winning file is already the artifact that gets blessed into
    `rag-config` under a new version.
    """
    directory = candidate.directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{candidate.slug}.yaml"
    body = {
        "version": f"sweep-{candidate.fingerprint}",
        **{
            section: _asdict(getattr(candidate.params, section))
            for section in ("parse_chunk", "embed", "agent")
        },
    }
    path.write_text(
        "# Generated by rag_evaluation.sweep — the reasoning for each parameter\n"
        "# lives in rag_config.params, which regenerating cannot destroy.\n"
        f"# written {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    # The config half, beside it. Only the collection actually varies per
    # candidate — its own fingerprint-named one, or, for a query-side candidate
    # reusing an index built before the sweep existed, whatever that index is
    # called — but the whole set is written so the pair is a complete
    # configuration rather than a delta.
    settings = replace(
        load_settings(), qdrant_collection=collection or candidate.collection
    )
    (directory / "settings.yaml").write_text(
        "# Generated by rag_evaluation.sweep — the config half of the candidate\n"
        "# beside it. Reasoning lives in rag_config.params.\n"
        + yaml.safe_dump(_asdict(settings), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _asdict(section) -> dict:
    from dataclasses import asdict

    return asdict(section)


def guard(collection: str, pg_dsn: str | None) -> None:
    """Refuse to run against production. This is what the `--eval` flag would
    have bought, at one check in one repo rather than a parallel schema."""
    if collection == PRODUCTION_COLLECTION:
        raise SweepError(
            f"refusing to sweep into the production collection {collection!r} — "
            "a sweep re-chunks and would mix two chunkings into one index"
        )
    if pg_dsn and PRODUCTION_DB in pg_dsn and "eval" not in pg_dsn:
        raise SweepError(
            f"refusing to sweep against the production ledger ({pg_dsn!r}) — "
            "give the run its own database, so delta sees an empty ledger"
        )


def collection_exists(collection: str, qdrant_url: str) -> bool:
    """Whether this ingest configuration is already built — the cache that makes
    an ingest sweep affordable."""
    import httpx

    try:
        r = httpx.get(f"{qdrant_url}/collections/{collection}", timeout=10)
        return r.status_code == 200
    except Exception as exc:
        log.warning("could not check %s: %s", collection, exc)
        return False


def run_sweep(
    spec: dict,
    entries: list[Entry],
    reingest: bool = False,
    dry_run: bool = False,
) -> list[Scored]:
    """Every candidate, scored on tier 1.

    `reingest=False` (the default) scores query-side candidates against the
    collection that is already there and refuses anything needing a rebuild —
    which keeps a mistyped chunk size from silently scoring the previous index.

    Scores only: the caller logs, the way `run_curve` and `log_curve` divide it.
    """
    from rag_evaluation.stages import ingest

    results: list[Scored] = []
    baseline_fp = ingest_fingerprint(load_params())

    current_collection = get_settings().qdrant_collection
    qdrant_url = get_settings().qdrant_url

    for candidate in candidates(spec):
        needs_build = candidate.fingerprint != baseline_fp
        # Reuse rather than rename: a query-side candidate scores against the
        # index that is already there, whatever it happens to be called. Only a
        # candidate that changes the index gets a fingerprint-named collection.
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
                "--reingest and hours of TEI"
            )
            results.append(Scored(candidate, None))
            continue
        if dry_run:
            results.append(Scored(candidate, None))
            continue

        # Set in this process, not only handed to subprocesses: `get_settings`
        # resolves the file through RAG_PARAMETERS_PATH, and tier 1 scores
        # in-process.
        os.environ["RAG_PARAMETERS_PATH"] = str(path)
        os.environ["RAG_QDRANT_COLLECTION"] = collection
        env = {"RAG_PARAMETERS_PATH": str(path), "RAG_QDRANT_COLLECTION": collection}
        settings = get_settings()
        if needs_build:
            pg_dsn = os.environ.get("RAG_PG_DSN")
            guard(collection, pg_dsn)
            if collection_exists(collection, settings.qdrant_url):
                print(f"  collection {collection} exists — reusing it")
            else:
                ingest(env, run_id=f"sweep-{candidate.fingerprint}")

        effective = get_params()
        report = run_tier1(entries, settings=settings, params=effective)
        print(format_report(report))
        results.append(Scored(candidate, report, settings, effective))

    return results
