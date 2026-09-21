"""Driving the six ingestion stages from the evaluation module.

Shells out to each stage's `python -m` entrypoint rather than importing it, so
this module depends on `rag-agent` and `rag-config` only — not on six stage
packages, each with its own lockfile.

Host venvs, not containers: the stage images predate the `rag-config` split, and
`scripts/run_pipeline.sh` is compose-only and re-discovers on every cycle, which a sweep
must not do. A container mode, or handing off to Airflow, comes once the images
are rebuilt.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

__all__ = ["ROOT", "StageError", "run_stage", "ingest"]

log = logging.getLogger(__name__)

# The working tree holding the six stage repos as siblings of evaluation/.
ROOT = Path(__file__).resolve().parents[3]

# parse_chunk processes at a time. Parsing is CPU-bound (PyMuPDF), so this is
# bounded by cores, not by any service.
DEFAULT_PARSE_WORKERS = 4
# Plan's cap on documents per batch, passed as --batch-max-docs. The shipped
# 200 turns this corpus into 3 batches, too few to keep 4 workers busy; smaller
# batches also finish more evenly, so no worker sits idle waiting on the last.
DEFAULT_BATCH_MAX_DOCS = 25

STAGES = {
    "discover": "rag_discover",
    "delta": "rag_delta",
    "plan": "rag_plan",
    "parse_chunk": "rag_parse_chunk",
    "embed": "rag_embed",
    "finalize": "rag_finalize",
}


class StageError(RuntimeError):
    """A stage exited non-zero. Fatal: a partial ingest scores a partial index."""


# The index-shaping parameters a stage also reads from the environment, where an
# env var beats the parameters file (rag_config precedence). Passed explicitly
# per candidate, because a RAG_CHUNK_SIZE left in a shell or in a stage's own
# dotfile would otherwise ingest every candidate at the same size while the run
# records them as different.
INGEST_ENV = {
    "RAG_CHUNK_SIZE": lambda p: p.parse_chunk.chunk_size,
    "RAG_CHUNK_OVERLAP": lambda p: p.parse_chunk.chunk_overlap,
    "RAG_SPARSE_AVG_LEN": lambda p: p.embed.sparse_avg_len,
}


def ingest_env(params) -> dict[str, str]:
    """The candidate's index-shaping parameters as stage environment variables."""
    return {name: str(read(params)) for name, read in INGEST_ENV.items()}


def run_stage(name: str, *args: str, env: dict[str, str] | None = None) -> str:
    """One stage, returning its stdout — which is the report the next stage reads."""
    python = ROOT / name / ".venv" / "bin" / "python"
    if not python.exists():
        raise StageError(f"{name}: no venv at {python} — run `uv sync` in that stage")
    log.info("stage %s %s", name, " ".join(args))
    proc = subprocess.run(
        [str(python), "-m", STAGES[name], *args],
        cwd=ROOT / name,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise StageError(f"{name} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return proc.stdout


def _parse_batches(batch_ids: Sequence[str], env: dict[str, str], workers: int) -> None:
    """parse_chunk over every batch, `workers` at a time.

    Batches are independent by design — in production each is its own Nomad
    dispatch (block 11) — so running several here is the concurrency the
    cluster already has. Every batch runs to the end even when one fails, the
    way sibling dispatches would; the failures are raised together afterwards.
    """
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(run_stage, "parse_chunk", "--batch-id", b, env=env): b
            for b in batch_ids
        }
        for future in as_completed(futures):
            if (exc := future.exception()) is not None:
                failures.append(f"batch {futures[future]}: {exc}")
    if failures:
        raise StageError("\n\n".join(failures))


def ingest(
    env: dict[str, str],
    run_id: str,
    parse_workers: int = DEFAULT_PARSE_WORKERS,
    batch_max_docs: int | None = DEFAULT_BATCH_MAX_DOCS,
) -> None:
    """Stages 2-6 over an already-populated manifest, into the env's collection.

    Stage 1 is deliberately absent: a sweep scores every configuration against
    the *same* corpus, and discover reads a moving world — two configurations
    listed on different days are not comparable. So `ledger.prepare_ledger` puts
    a frozen manifest in the target ledger first, and every configuration
    replays that one.

    `run_id` is the one the manifest rows carry, not a name of the caller's
    choosing: delta filters the manifest on it, plan filters the worklist rows
    delta stamped with it, and embed filters the batches. A value that is on no
    manifest row fails at delta with "no manifest rows for run".
    """
    run_stage("delta", "--run-id", run_id, env=env)
    plan_args = ["--run-id", run_id]
    if batch_max_docs:
        plan_args += ["--batch-max-docs", str(batch_max_docs)]
    plan_out = run_stage("plan", *plan_args, env=env)
    batch_ids = re.findall(r"^\d+$", plan_out, flags=re.M)
    if not batch_ids:
        raise StageError(
            "plan produced no batches — the manifest is empty, or the ledger "
            "already has every document (a sweep needs its own empty ledger)"
        )
    log.info("parse_chunk: %s batches, %s at a time", len(batch_ids), parse_workers)
    _parse_batches(batch_ids, env, parse_workers)
    # embed takes the whole run and parallelizes over TEI replicas itself
    run_stage("embed", "--run-id", run_id, env=env)
    run_stage("finalize", "--run-id", run_id, env=env)
