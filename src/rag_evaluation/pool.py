"""Pooling: turn what several configurations retrieved into labeling work.

Judging every (question, document) pair does not scale — 50 questions against a
100-document corpus is already 5,000 judgements, and the corpus grows. Pooling
judges only what some configuration actually returned, which costs
*questions x (configurations x k)* and is therefore independent of corpus size.
It is how TREC builds its relevance judgments (EVALUATION.md cites the overview
paper).

Two properties this module has to preserve:

- **Run diversity beats pool depth.** A deeper pool from one configuration mostly
  re-surfaces what that configuration already ranked highly. Pooling from
  configurations that disagree is what keeps unjudged-relevant documents rare.
- **Labels accumulate.** A judgement is per (question, document) and never
  expires, so re-pooling must present only what is new. Anything already labeled
  is listed apart, under `relevant_documents_generated`, as context rather than
  work.

A document here is `source_system` + `url` together (`DocRef`), the same key the
golden set labels and tier 1 scores.

Questions come from one `generate` session directory — `entries.yaml` and, when
the session was aggregated, `aggregate.yaml`. That is the point in the flow
where pooling is worth doing: those questions each carry only the document they
were written from, and pooling is what finds the others that answer them. The
output lands in the same directory, so one session directory holds the whole
review pile.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rag_agent.retrieval import Retriever
from rag_config import set_param
from rag_agent.settings import Settings, get_params, get_settings

from rag_evaluation.golden import DEFAULT_PATH as GOLDEN_PATH
from rag_evaluation.golden import DocRef, Entry, GoldenSetError, load_golden
from rag_evaluation.retrieval_eval import chunk_ref

__all__ = [
    "PoolRun",
    "DEFAULT_RUNS",
    "pool_entries",
    "format_pool",
    "load_session_entries",
    "run_pool",
]

# Same base directory as generate and aggregate: one subdirectory per session,
# next to golden_set.yaml. Nothing here is authoritative — this is a staging
# area for review, the golden set is the result.
DATA_DIR = GOLDEN_PATH.parent

# The files a session can hold questions in. entries.yaml is always there;
# aggregate.yaml only when the session was aggregated.
QUESTION_FILES = ("entries.yaml", "aggregate.yaml")


@dataclass(frozen=True)
class PoolRun:
    """One contributing configuration."""

    name: str
    k: int = 20


# The retriever always queries all three legs, so leg-level diversity
# (dense-only vs sparse-only) is not available without a retrieval change —
# EVALUATION.md notes it as what to add when the corpus grows enough for pool
# bias to matter.
DEFAULT_RUNS = (PoolRun("hybrid", k=20),)

# There is deliberately no reranked run. Reranking does not widen the candidate
# set: it rescores the same `top_k` fused chunks and returns its best `k` of
# them (../agent retrieval.py), so a reranked run at k can only return what a
# plain run at `top_k` already returns. Raising --k to top_k pools a superset
# for free, where reranking costs ~222 s per query on this box (Candle
# fallback — ../tei_rerank). Whether the reranker helps the *agent* is a
# separate question, and tier 1 answers it.


@dataclass
class Candidate:
    doc: DocRef
    title: str | None
    best_rank: int
    runs: set[str]
    already_labeled: bool


def pool_entries(
    entries: Sequence[Entry],
    runs: Sequence[PoolRun] = DEFAULT_RUNS,
    settings: Settings | None = None,
) -> dict[str, list[Candidate]]:
    """Per entry, the union of what every run returned, minus what is labeled."""
    base = settings or get_settings()
    # Rerank off regardless of the file or the environment; see DEFAULT_RUNS.
    params = set_param(get_params(), "agent.rerank_enabled", False)
    pooled: dict[str, dict[DocRef, Candidate]] = {e.id: {} for e in entries}

    for run in runs:
        retriever = Retriever(base, params)
        try:
            for entry in entries:
                found = retriever.search(entry.question, limit=run.k)
                seen: set[DocRef] = set()
                for rank, chunk in enumerate(found, 1):
                    doc = chunk_ref(chunk)
                    if doc is None or doc in seen:
                        continue
                    seen.add(doc)
                    existing = pooled[entry.id].get(doc)
                    if existing:
                        existing.runs.add(run.name)
                        existing.best_rank = min(existing.best_rank, rank)
                    else:
                        pooled[entry.id][doc] = Candidate(
                            doc=doc,
                            title=chunk.title,
                            best_rank=rank,
                            runs={run.name},
                            already_labeled=doc in entry.relevant_documents,
                        )
        finally:
            close = getattr(retriever, "close", None)
            if close:
                close()

    return {
        entry_id: sorted(cands.values(), key=lambda c: c.best_rank)
        for entry_id, cands in pooled.items()
    }


def format_pool(entries: Sequence[Entry], pooled: dict[str, list[Candidate]]) -> str:
    """Paste-ready YAML: delete the wrong lines rather than typing documents.

    Two lists per entry, so what came from `generate` and what pooling added are
    never confused:

    - `relevant_documents_generated` — the documents the entry already carries,
      each annotated with the rank retrieval gave it, or `not retrieved`.
    - `relevant_documents_pooled` — everything else the runs returned. This is
      the list to judge.

    Documents are written in golden_set.yaml's own block style, so a kept
    candidate is copied across as-is. Rejecting one means deleting its two
    lines.
    """
    by_id = {e.id: e for e in entries}
    out: list[str] = [
        "# Pooled candidates. For each entry, delete the NOT relevant lines under",
        "# relevant_documents_pooled, and add what remains to that entry's",
        "# relevant_documents in golden_set.yaml.",
        "# relevant_documents_generated is what the entry already carries — shown for",
        "# context, already labeled, nothing to do. `not retrieved` there means the",
        "# run missed a known-relevant document: an entry worth keeping, not fixing.",
    ]
    for entry_id, candidates in pooled.items():
        entry = by_id[entry_id]
        by_doc = {c.doc: c for c in candidates}
        fresh = [c for c in candidates if not c.already_labeled]
        out.append(f"\n# --- {entry_id}: {entry.question}")
        out.append(f"  - id: {entry_id}")

        out.append(_list_header("relevant_documents_generated", entry.relevant_documents))
        for doc in entry.relevant_documents:
            hit = by_doc.get(doc)
            note = (
                f"rank {hit.best_rank} [{'+'.join(sorted(hit.runs))}] {(hit.title or '')[:60]}"
                if hit
                else "not retrieved"
            )
            out.extend(_doc_lines(doc, note))

        out.append(_list_header("relevant_documents_pooled", fresh))
        for c in fresh:
            runs = "+".join(sorted(c.runs))
            title = (c.title or "")[:60]
            out.extend(_doc_lines(c.doc, f"rank {c.best_rank} [{runs}] {title}"))
    return "\n".join(out)


def _list_header(key: str, items: Sequence[object]) -> str:
    """The key line, closed with `[]` when nothing follows it."""
    return f"    {key}:" if items else f"    {key}: []"


def _doc_lines(doc: DocRef, note: str) -> list[str]:
    """One document as golden_set.yaml writes it, the note on the first line."""
    return [
        f"      - source_system: {doc.source_system}    # {note}",
        f"        url: {json.dumps(doc.url)}",
    ]


def load_session_entries(
    run_dir: str | Path, ids: Iterable[str] | None = None
) -> list[Entry]:
    """Every question a session wrote, from entries.yaml and aggregate.yaml.

    Both files use the golden set's own format, so they load with `load_golden`.
    Ids are checked across the two together — `aggregate` is given the ids
    already taken, so a collision here means one of the files was hand-edited.
    """
    directory = Path(run_dir)
    entries: list[Entry] = []
    for name in QUESTION_FILES:
        path = directory / name
        if path.exists():
            entries.extend(load_golden(path))
    if not entries:
        raise GoldenSetError(
            f"no {' or '.join(QUESTION_FILES)} in {directory}"
        )

    seen: set[str] = set()
    duplicates = {e.id for e in entries if e.id in seen or seen.add(e.id)}
    if duplicates:
        raise GoldenSetError(
            f"{directory}: duplicate id(s) {', '.join(sorted(duplicates))} across "
            f"{' and '.join(QUESTION_FILES)}"
        )

    if ids is not None:
        wanted = set(ids)
        entries = [e for e in entries if e.id in wanted]
        missing = wanted - {e.id for e in entries}
        if missing:
            raise GoldenSetError(
                f"{directory}: unknown id(s) {', '.join(sorted(missing))}"
            )
    return entries


def run_pool(
    session_id: str | Path,
    ids: Iterable[str] | None = None,
    k: int = 20,
    data_dir: str | None = None,
    settings: Settings | None = None,
) -> int:
    """Pool one session's questions and write pool.yaml beside them."""
    run_dir = (Path(data_dir) if data_dir else DATA_DIR) / session_id
    if not run_dir.is_dir():
        print(f"no such session directory: {run_dir}", file=sys.stderr)
        return 2
    try:
        entries = load_session_entries(run_dir, ids)
    except GoldenSetError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    runs = tuple(replace(r, k=k) for r in DEFAULT_RUNS)
    print(
        f"{len(entries)} questions, {len(runs)} run(s) at k={k}",
        file=sys.stderr,
    )
    try:
        pooled = pool_entries(entries, runs, settings=settings)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot retrieve (are ../qdrant and ../tei up?): {exc}", file=sys.stderr)
        return 2

    out_path = run_dir / "pool.yaml"
    out_path.write_text(format_pool(entries, pooled) + "\n", encoding="utf-8")

    labeled = {d for e in entries for d in e.relevant_documents}
    fresh = sum(
        1 for cands in pooled.values() for c in cands if not c.already_labeled
    )
    print(f"  pool.yaml  {fresh} candidates to judge", file=sys.stderr)
    print(f"             {len(labeled)} already labeled, listed apart", file=sys.stderr)
    return 0 if fresh else 1
