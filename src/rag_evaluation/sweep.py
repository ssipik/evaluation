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
baseline, sweep the next knob. Knobs that cannot move independently — chunk size
and its overlap — go in the file's `candidates:` section instead, one entry
setting all of them at once.

An ingest candidate builds against a frozen copy of the manifest in its own
database (`ledger.py`), so every candidate re-chunks the same corpus.
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
from sqlalchemy.engine import make_url

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

# Where the frozen manifest is copied from when neither the sweep file nor
# RAG_PG_DSN says. Same default as generate.py, for the same reason: the stage
# packages are not installed here to import it from.
DEFAULT_PG_DSN = "postgresql+psycopg://rag:rag@localhost:5433/rag_ingestion"


class SweepError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    """One configuration: a full parameter set plus what changed and why."""

    label: str
    knob: str
    value: Any
    params: Params
    # Every (knob, value) this candidate changed. One pair for a `sweep:` entry;
    # several for a `candidates:` entry, where `knob` and `value` are the joined
    # forms the summary and the csv print.
    changes: tuple[tuple[str, Any], ...] = ()

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
        stem = self.label if len(self.changes) > 1 else f"{self.knob}={self.value}"
        return stem.replace("/", "_").replace(" ", "")


def load_sweep(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not raw.get("sweep") and not raw.get("candidates"):
        raise SweepError(f"{path}: no `sweep:` or `candidates:` section")
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
    """The baseline, then the candidates of both kinds.

    - `sweep:` — one candidate per (knob, value), each changing one knob from
      the baseline. Coordinate descent: the winner is not carried into the next
      knob's candidates, you put it in the next file's `baseline:` yourself.
    - `candidates:` — one candidate per named entry, changing as many knobs as
      it lists. For knobs that cannot be moved independently: chunk size and
      overlap sweep as pairs, or a 350-word chunk with the baseline's 100-word
      overlap would change the overlap fraction too.
    """
    params = baseline or load_params()
    for dotted, value in (spec.get("baseline") or {}).items():
        params = _set(params, dotted, value)
    yield Candidate(label="baseline", knob="", value=None, params=params)
    for dotted, values in (spec.get("sweep") or {}).items():
        for value in values:
            yield Candidate(
                label=f"{dotted}={value}",
                knob=dotted,
                value=value,
                params=_set(params, dotted, value),
                changes=((dotted, value),),
            )
    for entry in spec.get("candidates") or []:
        changes = entry.get("set") or {}
        if not changes:
            raise SweepError(f"candidate {entry.get('label', '?')!r}: empty `set:`")
        candidate_params = params
        for dotted, value in changes.items():
            candidate_params = _set(candidate_params, dotted, value)
        # Joined with "+" and "|", not ",": both land in a csv column.
        yield Candidate(
            label=entry.get("label") or "+".join(f"{k}={v}" for k, v in changes.items()),
            knob="+".join(changes),
            value="|".join(str(v) for v in changes.values()),
            params=candidate_params,
            changes=tuple(changes.items()),
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
    # The database name, never the dsn: it carries a password.
    if pg_dsn and make_url(pg_dsn).database == PRODUCTION_DB:
        raise SweepError(
            f"refusing to sweep against the production ledger ({PRODUCTION_DB}) — "
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


def frozen_snapshot(ingest_spec: dict, overwrite: bool) -> str:
    """The campaign's snapshot of the manifest, taken once, before any candidate.

    Every candidate ledger is copied from it rather than from the live manifest,
    so one built tomorrow replays what the first one ingested today.
    """
    from rag_evaluation.ledger import DEFAULT_RUN_ID, DEFAULT_SNAPSHOT, LedgerError
    from rag_evaluation.ledger import ensure_snapshot

    source_dsn = ingest_spec.get("source_dsn") or os.environ.get(
        "RAG_PG_DSN", DEFAULT_PG_DSN
    )
    name = ingest_spec.get("snapshot", DEFAULT_SNAPSHOT)
    try:
        dsn, documents, run_id = ensure_snapshot(
            source_dsn,
            name,
            ingest_spec.get("run_id", DEFAULT_RUN_ID),
            overwrite=overwrite,
        )
    except LedgerError as exc:
        raise SweepError(str(exc)) from None
    print(f"\nsnapshot {name}: {documents} documents, run {run_id!r}")
    return dsn


def build(
    candidate: Candidate,
    collection: str,
    env: dict[str, str],
    ingest_spec: dict,
    manifest_dsn: str,
) -> None:
    """The expensive half: the candidate's own ledger, then stages 2-6 into its
    own collection.

    Resumable rather than all-or-nothing. A crashed campaign leaves a part-built
    collection and a ledger that records exactly which documents reached it, so
    running again ingests the remainder — the reason "the collection is there"
    alone is not taken as "it is built".
    """
    from rag_evaluation.ledger import (
        DEFAULT_PREFIX,
        DEFAULT_RUN_ID,
        LedgerError,
        database_exists,
        eval_dsn,
        ledger_state,
        prepare_ledger,
    )
    from rag_evaluation.stages import (
        DEFAULT_BATCH_MAX_DOCS,
        DEFAULT_PARSE_WORKERS,
        ingest,
        ingest_env,
    )

    target_dsn = eval_dsn(
        manifest_dsn, candidate.fingerprint, ingest_spec.get("prefix", DEFAULT_PREFIX)
    )
    guard(collection, target_dsn)

    built = collection_exists(collection, get_settings().qdrant_url)
    if built and not database_exists(target_dsn):
        # Built by an earlier campaign whose ledger has since been dropped.
        # Nothing left to compare against, so take the collection as it is.
        print(f"  collection {collection} exists — reusing it")
        return
    try:
        run_id = prepare_ledger(
            manifest_dsn, target_dsn, ingest_spec.get("run_id", DEFAULT_RUN_ID)
        )
        manifest, ingested, _ = ledger_state(target_dsn)
    except LedgerError as exc:
        raise SweepError(str(exc)) from None
    if built and ingested >= manifest:
        print(f"  collection {collection} exists, {ingested} documents — reusing it")
        return
    if ingested:
        print(f"  resuming: {ingested} of {manifest} documents already ingested")

    # The stages read the candidate's ledger, never the source one, and its
    # chunking from the environment as well as from its parameters file.
    stage_env = {**env, **ingest_env(candidate.params), "RAG_PG_DSN": target_dsn}
    # Texts per TEI request. Throughput only, so outside the fingerprint; set
    # here because TEI's cost per text jumps past ~16 long ones, and the
    # requests queued behind a slow one then pass tei_timeout.
    if "embed_batch_size" in ingest_spec:
        stage_env["RAG_EMBED_BATCH_SIZE"] = str(ingest_spec["embed_batch_size"])
    ingest(
        stage_env,
        run_id=run_id,
        parse_workers=ingest_spec.get("parse_workers", DEFAULT_PARSE_WORKERS),
        batch_max_docs=ingest_spec.get("batch_max_docs", DEFAULT_BATCH_MAX_DOCS),
    )


def run_sweep(
    spec: dict,
    entries: list[Entry],
    reingest: bool = False,
    dry_run: bool = False,
    overwrite_frozen: bool = False,
) -> list[Scored]:
    """Every candidate, scored on tier 1.

    `reingest=False` (the default) scores query-side candidates against the
    collection that is already there and refuses anything needing a rebuild —
    which keeps a mistyped chunk size from silently scoring the previous index.

    Scores only: the caller logs, the way `run_curve` and `log_curve` divide it.
    """
    results: list[Scored] = []
    baseline_fp = ingest_fingerprint(load_params())

    current_collection = get_settings().qdrant_collection
    qdrant_url = get_settings().qdrant_url

    ingest_spec = spec.get("ingest") or {}
    manifest_dsn = ""
    if reingest and not dry_run:
        manifest_dsn = frozen_snapshot(ingest_spec, overwrite_frozen)

    for candidate in candidates(spec):
        if reingest:
            # Every candidate builds its own collection, the one at the shipped
            # parameters included. Scoring that one against the standing index
            # instead would compare a build made tonight with one embedded on
            # another day, over a corpus that has grown since — a difference in
            # the corpus would read as a difference in the knob.
            needs_build, collection = True, candidate.collection
        else:
            needs_build = candidate.fingerprint != baseline_fp
            # Reuse rather than rename: a query-side candidate scores the index
            # that is already there, whatever it happens to be called. An
            # index-changing one is named by its fingerprint and skipped below,
            # for want of --reingest.
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

        # Set in this process, not only handed to subprocesses: `get_settings`
        # resolves the file through RAG_PARAMETERS_PATH, and tier 1 scores
        # in-process.
        os.environ["RAG_PARAMETERS_PATH"] = str(path)
        os.environ["RAG_QDRANT_COLLECTION"] = collection
        env = {"RAG_PARAMETERS_PATH": str(path), "RAG_QDRANT_COLLECTION": collection}
        settings = get_settings()
        if needs_build:
            build(candidate, collection, env, ingest_spec, manifest_dsn)

        effective = get_params()
        report = run_tier1(entries, settings=settings, params=effective)
        print(format_report(report))
        results.append(Scored(candidate, report, settings, effective))

    return results
