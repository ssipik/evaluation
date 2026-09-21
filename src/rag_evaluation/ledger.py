"""The frozen ledger an ingest sweep replays.

One database per ingest configuration, `rag_eval_<fingerprint>`, holding a copy
of the real `ingest_manifest` and nothing else. An empty `ingest_docs` is what
makes delta report the whole corpus as new, which is the point: the candidate
re-chunks and re-embeds all of it.

What is frozen is the *list* of documents, not their bytes — stage 4 fetches
every URL live (`parse_chunk/CLAUDE.md`). Candidates run back to back so the
window in which a source can change stays small.

Discover never runs here. It reads a moving world, so two candidates that each
listed it would score different corpora.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from rag_evaluation.stages import ROOT

__all__ = [
    "DEFAULT_PREFIX",
    "DEFAULT_RUN_ID",
    "DEFAULT_SNAPSHOT",
    "LedgerError",
    "database_exists",
    "eval_dsn",
    "prepare_ledger",
    "ledger_state",
    "snapshot_dsn",
    "ensure_snapshot",
]

log = logging.getLogger(__name__)

SCHEMA_DIR = ROOT / "postgres" / "sql"
DEFAULT_PREFIX = "rag_eval_"
# The campaign's snapshot of the manifest. Candidate ledgers are copied from
# here, not from the live one, so a candidate built later — a resumed campaign,
# one more chunk size — replays the same corpus as the first. Recreate it with
# --overwrite-frozen when the campaign should follow the live manifest again.
DEFAULT_SNAPSHOT = "rag_eval_frozen"
# Never dropped, never written into. Same name as sweep.PRODUCTION_DB, kept
# here so this module guards itself without importing its caller.
PRODUCTION_DB = "rag_ingestion"
# Every candidate replays the same frozen listing, so the id names the listing
# rather than the candidate. Stages 2-6 all filter on it.
DEFAULT_RUN_ID = "frozen"

# The manifest columns a copy carries. manifest_id is left out on purpose: it is
# GENERATED ALWAYS and nothing in a fresh database references the old ids.
MANIFEST_COLUMNS = (
    "run_id",
    "source_system",
    "doc_url",
    "source_id",
    "title",
    "published_at",
    "doc_version",
    "size_bytes",
    "extra",
    "execution_ts",
)


class LedgerError(RuntimeError):
    pass


def eval_dsn(source_dsn: str, fingerprint: str, prefix: str = DEFAULT_PREFIX) -> str:
    """The candidate's own database on the same server as the source ledger.

    `render_as_string(hide_password=False)`, not `str()`: str masks the password
    as `***`, and this DSN is handed to psycopg and to the stage subprocesses.
    """
    url = make_url(source_dsn).set(database=f"{prefix}{fingerprint}")
    return url.render_as_string(hide_password=False)


def snapshot_dsn(source_dsn: str, name: str = DEFAULT_SNAPSHOT) -> str:
    """The campaign snapshot's database, beside the source ledger."""
    url = make_url(source_dsn).set(database=name)
    return url.render_as_string(hide_password=False)


def _maintenance_engine(dsn: str) -> Engine:
    """CREATE DATABASE cannot run inside a transaction, nor against the database
    being created — so it runs on `postgres`, autocommit."""
    url = make_url(dsn).set(database="postgres")
    return create_engine(url, isolation_level="AUTOCOMMIT")


def database_exists(dsn: str) -> bool:
    """Whether that database is on the server — asked before connecting to it,
    so a missing one is a fact rather than an OperationalError."""
    name = make_url(dsn).database
    with _maintenance_engine(dsn).connect() as conn:
        return (
            conn.execute(
                text("select 1 from pg_database where datname = :n"), {"n": name}
            ).first()
            is not None
        )


def _create_database(target_dsn: str) -> bool:
    """Returns whether it had to be created."""
    name = make_url(target_dsn).database
    with _maintenance_engine(target_dsn).connect() as conn:
        if database_exists(target_dsn):
            return False
        # An identifier cannot be a bind parameter, and the name is ours:
        # prefix + a hex fingerprint.
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    log.info("created database %s", name)
    return True


def _drop_database(dsn: str) -> None:
    """Drop it, disconnecting whoever is still attached — psql left open in
    another window is the usual one, and it would block the drop."""
    name = make_url(dsn).database
    if name in (PRODUCTION_DB, "postgres"):
        raise LedgerError(f"refusing to drop {name!r}")
    with _maintenance_engine(dsn).connect() as conn:
        conn.execute(
            text(
                "select pg_terminate_backend(pid) from pg_stat_activity "
                "where datname = :n and pid <> pg_backend_pid()"
            ),
            {"n": name},
        )
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
    log.info("dropped database %s", name)


def _apply_schema(engine: Engine) -> None:
    """`postgres/sql/*.sql` in order — authoritative for the schema, and all of
    it `IF NOT EXISTS`, so applying it twice is a no-op."""
    files = sorted(SCHEMA_DIR.glob("*.sql"))
    if not files:
        raise LedgerError(f"no schema files under {SCHEMA_DIR}")
    with engine.begin() as conn:
        for path in files:
            conn.exec_driver_sql(path.read_text(encoding="utf-8"))
    log.info("applied %s schema files", len(files))


def _copy_manifest(source: Engine, target: Engine, run_id: str) -> int:
    """The newest sighting of each document, stamped with one run_id.

    Newest-per-document first, because the table's natural key is
    `(run_id, source_system, doc_url)`: relabeling a document that two discover
    runs both listed would collide. And because delta filtered to one run_id
    would otherwise see only what that single run happened to list.
    """
    columns = ", ".join(MANIFEST_COLUMNS)
    select_newest = text(f"""
        select distinct on (source_system, doc_url) {columns}
          from ingest_manifest
         order by source_system, doc_url, execution_ts desc
    """)
    with source.connect() as conn:
        rows = [dict(r) for r in conn.execute(select_newest).mappings()]
    if not rows:
        raise LedgerError(
            "the source ledger's ingest_manifest is empty — run discover first, "
            "or point RAG_PG_DSN at the database that holds the corpus"
        )
    for row in rows:
        row["run_id"] = run_id
        # `extra` arrives as the dict psycopg decoded the jsonb into, and goes
        # back as text the insert casts — an untyped bind cannot adapt a dict.
        row["extra"] = json.dumps(row["extra"] or {})
    placeholders = ", ".join(
        f"cast(:{c} as jsonb)" if c == "extra" else f":{c}" for c in MANIFEST_COLUMNS
    )
    insert = text(f"insert into ingest_manifest ({columns}) values ({placeholders})")
    with target.begin() as conn:
        conn.execute(insert, rows)
    log.info("copied %s manifest rows as run %r", len(rows), run_id)
    return len(rows)


def ledger_state(dsn: str) -> tuple[int, int, str | None]:
    """`(manifest rows, ledger rows, run_id)` — how much of the corpus this
    database has already ingested, and under which run."""
    engine = create_engine(dsn)
    with engine.connect() as conn:
        manifest = conn.execute(text("select count(*) from ingest_manifest")).scalar_one()
        docs = conn.execute(text("select count(*) from ingest_docs")).scalar_one()
        runs: Sequence[str] = conn.execute(
            text("select distinct run_id from ingest_manifest")
        ).scalars().all()
    if len(runs) > 1:
        raise LedgerError(
            f"{make_url(dsn).database} holds {len(runs)} manifest runs "
            f"({', '.join(sorted(runs))}) — a frozen ledger holds exactly one"
        )
    return manifest, docs, runs[0] if runs else None


def ensure_snapshot(
    source_dsn: str,
    snapshot: str,
    run_id: str = DEFAULT_RUN_ID,
    overwrite: bool = False,
) -> tuple[str, int, str]:
    """The campaign's snapshot of the manifest: `(dsn, documents, run_id)`.

    Taken once and copied from thereafter, so every candidate replays the same
    corpus however far apart they are built. `overwrite=True` drops it and takes
    a new one from the live manifest — the only way its contents ever change.
    """
    dsn = snapshot_dsn(source_dsn, snapshot)
    if make_url(dsn).database == make_url(source_dsn).database:
        raise LedgerError(
            f"the snapshot and the source ledger are the same database "
            f"({snapshot!r}) — the snapshot is a copy, it needs its own"
        )
    if overwrite and database_exists(dsn):
        _drop_database(dsn)
    fresh = _create_database(dsn)
    engine = create_engine(dsn)
    _apply_schema(engine)
    documents, _, existing = ledger_state(dsn)
    if documents and not fresh:
        log.info("snapshot %s: %s documents, run %r", snapshot, documents, existing)
        return dsn, documents, existing or run_id
    documents = _copy_manifest(create_engine(source_dsn), engine, run_id)
    return dsn, documents, run_id


def prepare_ledger(
    source_dsn: str, target_dsn: str, run_id: str = DEFAULT_RUN_ID
) -> str:
    """Create the candidate's ledger if it is not there, and return its run_id.

    `source_dsn` is the campaign snapshot, not the live ledger — see
    `ensure_snapshot`.

    Idempotent, and that is what makes a crashed campaign resumable: a database
    that already holds the frozen manifest is left exactly as it is, so a second
    ingest picks up the documents the first one did not finish.
    """
    if make_url(target_dsn).database == make_url(source_dsn).database:
        raise LedgerError(
            f"the sweep's ledger and the source ledger are the same database "
            f"({make_url(target_dsn).database!r}) — a sweep needs its own, so "
            "delta sees an empty ingest_docs"
        )
    _create_database(target_dsn)
    target = create_engine(target_dsn)
    _apply_schema(target)
    manifest, docs, existing = ledger_state(target_dsn)
    if manifest:
        log.info(
            "reusing ledger %s: %s manifest rows, %s already ingested",
            make_url(target_dsn).database,
            manifest,
            docs,
        )
        return existing or run_id
    _copy_manifest(create_engine(source_dsn), target, run_id)
    return run_id
