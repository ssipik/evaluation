"""Seed golden-set entries from the corpus: sample documents, ask an LLM for a
summary and one question answered by it.

The labeling problem this solves is the reverse of `pool`. Pooling starts from a
question and asks which documents answer it, so every label is a judgement a
human has to make. Here the document comes first: the question is *written from*
one document, so that document is relevant by construction and
`relevant_documents` needs no pooling at all. What still needs a human is whether
the question is a fair one — see `format_entries` on why the answer does not go
straight into `reference`.

**A third model family.** The agent answers (`anthropic/claude-haiku-4.5`) and
the judge scores (`openai/gpt-5.4`), so the question writer is Google's. Same
reason the judge is not the agent's family: a model tends to favour text shaped
like its own, and a question written by the family that later answers or grades
it flatters both. Routed through OpenRouter, so one API key still serves all
three.

**Where the text comes from.** Postgres is a ledger, not a library — it carries
identifiers and hashes, and `staged_nodes` is transient (stage 6 clears it). So
the ledger says *which* documents exist and Qdrant supplies *what they say*,
reassembled from the indexed chunks in `node_seq` order. That has a useful side
effect: a question can only be written from text that is actually in the index,
so a generated entry is answerable by construction rather than in principle.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys

import yaml
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from rag_agent.qdrant import QdrantSearcher
from rag_agent.settings import Settings, get_provider, get_settings
from sqlalchemy import BigInteger, Column, MetaData, Table, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from rag_evaluation.golden import DEFAULT_PATH as GOLDEN_PATH
from rag_evaluation.golden import DocRef, Entry
from rag_evaluation.mlflow_log import generation_run, new_session_id, trace_session

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PG_DSN",
    "Sampled",
    "Generated",
    "Failure",
    "sample_documents",
    "document_text",
    "generate",
    "format_entries",
    "format_review",
    "run_generate",
    "_slug",
    "DATA_DIR",
]

log = logging.getLogger(__name__)

# Duplicated from the stages' settings.py rather than imported: this module
# depends on rag-agent and rag-config only, and the stage packages are not
# installed here (stages.py shells out to them for the same reason).
DEFAULT_PG_DSN = "postgresql+psycopg://rag:rag@localhost:5433/rag_ingestion"

# One subdirectory per session, next to golden_set.yaml because the whole point
# is reading a run and copying the entries you keep across into it. Nothing here
# is authoritative — the golden set is; this is a staging area for review.
DATA_DIR = GOLDEN_PATH.parent

# Overridable with RAG_EVAL_GENERATE_MODEL, the way the judge reads
# RAG_EVAL_JUDGE_MODEL.
# Newest stable Gemini on OpenRouter — no `-preview` suffix, so the id does not
# expire under the script, and a 1M context takes any document in this corpus
# whole. `--model google/gemini-3.1-pro-preview` buys better questions for
# roughly five times the price; at 50 documents both are small change.
DEFAULT_MODEL = "google/gemini-3.7-flash"

# The provider table is rag-agent's, so this and the agent read one API key and
# one base URL. The model is the part that differs.
PROVIDER = "openrouter"

_metadata = MetaData()

# Mirrors ../postgres/sql/ (authoritative for the schema), the way each stage's
# db.py mirrors the tables it touches. Only the columns this one select reads:
# it never writes, so the rest of each table would be dead weight to keep in
# step. Add a column here when this module starts reading it.
ingest_manifest = Table(
    "ingest_manifest",
    _metadata,
    Column("manifest_id", BigInteger, primary_key=True),
    Column("source_system", Text, nullable=False),
    Column("doc_url", Text, nullable=False),
    Column("source_id", Text, nullable=False),
    Column("title", Text),
)

ingest_docs = Table(
    "ingest_docs",
    _metadata,
    Column("manifest_id", BigInteger, primary_key=True),
    Column("status", Text, nullable=False),
)


@dataclass(frozen=True)
class Sampled:
    """One document drawn from the ledger."""

    doc: DocRef
    title: str | None
    source_id: str | None
    language: str | None = None  # from the chunk payload, filled in on fetch


@dataclass(frozen=True)
class Generated:
    """What the model wrote for one document."""

    sampled: Sampled
    summary: tuple[str, ...]
    question: str
    answer: str
    point_index: int
    language: str

    @property
    def point(self) -> str:
        """The summary point the answer came from."""
        return self.summary[self.point_index]


@dataclass(frozen=True)
class Failure:
    doc: DocRef
    reason: str


def sample_documents(
    n: int = 50,
    seed: int | None = None,
    dsn: str | None = None,
    engine: Engine | None = None,
) -> list[Sampled]:
    """`n` documents from the ledger, drawn with `seed`.

    The join is what makes the result one row per document. `ingest_manifest`
    holds a row per (run_id, source_system, doc_url), so a document listed by
    ten runs is ten sightings — but `ingest_docs` holds one row per document,
    its manifest_id moved to the live sighting on each re-ingest (embed's
    `_write_ledger`). Joining on it therefore picks exactly that sighting.

    Only `active`: `deleted` is a tombstone and `empty` parsed to no text, so
    neither has vectors, and a question written from either could never be
    answered by retrieval.

    The draw is Python's, not `order by random()`: seeding Postgres means a
    `setseed` that lasts one session and interacts with the plan, while
    `random.Random(seed).sample` over a sorted candidate list reproduces exactly
    as long as the corpus has not changed underneath it. It changing is the real
    limit on reproducibility here, and no seeding scheme fixes that.
    """
    owned = engine is None
    engine = engine or create_engine(dsn or os.environ.get("RAG_PG_DSN", DEFAULT_PG_DSN))
    query = (
        select(
            ingest_manifest.c.source_system,
            ingest_manifest.c.doc_url,
            ingest_manifest.c.title,
            ingest_manifest.c.source_id,
        )
        .select_from(
            ingest_docs.join(
                ingest_manifest,
                ingest_docs.c.manifest_id == ingest_manifest.c.manifest_id,
            )
        )
        .where(ingest_docs.c.status == "active")
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(query).all()
    finally:
        if owned:
            engine.dispose()

    candidates = [
        Sampled(
            doc=DocRef(source_system=row.source_system, url=row.doc_url),
            title=row.title,
            source_id=row.source_id,
        )
        for row in rows
    ]
    candidates.sort(key=lambda s: (s.doc.source_system, s.doc.url))
    if n >= len(candidates):
        log.info("corpus has %d active documents — taking all of them", len(candidates))
        return candidates
    return random.Random(seed).sample(candidates, n)


def document_text(
    searcher: QdrantSearcher, sampled: Sampled, max_chars: int
) -> tuple[str, str | None]:
    """One document's indexed text in `node_seq` order, and its language.

    A document is source_system + url together, so both halves go into the
    scroll — two sources listing one url are two documents.
    """
    records = searcher.scroll_document(sampled.doc.url, sampled.doc.source_system)
    parts: list[str] = []
    language: str | None = None
    size = 0
    for record in records:
        payload = record.payload or {}
        language = language or payload.get("language")
        chunk = payload.get("text") or ""
        if size + len(chunk) > max_chars:
            parts.append(chunk[: max_chars - size])
            break
        parts.append(chunk)
        size += len(chunk)
    return "\n\n".join(parts), language


SYSTEM_PROMPT = """\
You write evaluation questions for a retrieval system over a corpus of financial \
and regulatory documents (ECB and BaFin publications, fund factsheets, internal \
wiki pages), in German and English.

You are given the full text of ONE document. Do two things:

1. Summarise it as a list of 5 to 10 points. Each point is one self-contained \
factual statement drawn from the document — a figure, a date, a decision, a \
definition, a rule. Write the points in the document's own language.

2. Write ONE question whose answer is contained in exactly one of those points, \
and give that answer.

Rules for the question, all of them about how it will be used. It will be asked \
against the WHOLE corpus, with the document withheld, and the system is scored \
on whether it finds this document again:

- Self-contained. Never refer to "the document", "this text", "the attached \
report" or "the above" — at scoring time there is no document in view, only the \
question.
- Specific enough to identify this document among many similar ones. Name the \
institution, instrument, fund, ISIN, date or figure the answer turns on. "What \
were the interest rates?" matches fifty documents; "What were euro area bank \
interest rates in June 2026?" matches one.
- Not a restatement of the title. If the title alone answers it, it tests \
nothing.
- Answerable from one point, not assembled from several.
- In the same language as the document.

The answer is short — a figure, a name, a date, one or two sentences — and says \
only what the point says.\
"""

_SCHEMA = {
    "name": "golden_entry",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "question", "answer", "point_index", "language"],
        "properties": {
            "summary": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 10,
            },
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "point_index": {
                "type": "integer",
                "description": "0-based index of the summary point holding the answer",
            },
            "language": {"type": "string", "enum": ["de", "en"]},
        },
    },
}


class MissingApiKey(RuntimeError):
    pass


def _build_client(settings: Settings):
    from openai import OpenAI

    provider = get_provider(PROVIDER)
    key = os.environ.get(provider.api_key_var, "")
    # The agent's key only counts when the agent is on this provider too —
    # otherwise it is a Mistral or OpenAI key being sent to OpenRouter, the
    # mix-up rag-agent's provider table exists to prevent.
    if not key and settings.llm_provider == PROVIDER:
        key = settings.llm_api_key
    if not key:
        raise MissingApiKey(
            f"{provider.api_key_var} is unset — export it (the same key the judge "
            "uses; see README)."
        )
    return OpenAI(base_url=provider.base_url, api_key=key, timeout=180.0)


class GenerationError(RuntimeError):
    """The model's answer was unusable. Per-document, never fatal — one bad
    response must not cost the other 49 their API calls."""


def _generate_one(client, model: str, sampled: Sampled, body: str) -> Generated:
    header = f"Title: {sampled.title or '(untitled)'}\nSource: {sampled.doc.url}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{header}\n\n---\n\n{body}"},
        ],
        # The one knob that matters here: near-zero would rewrite the same
        # question shape for every document, and a golden set wants variety.
        temperature=0.7,
        response_format={"type": "json_schema", "json_schema": _SCHEMA},
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"not JSON: {exc}") from None

    summary = tuple(str(p) for p in data.get("summary") or ())
    index = data.get("point_index")
    # The whole contract of this script: the answer has to live in a point. A
    # model that indexes past the end has not honoured it, whatever it wrote.
    if not isinstance(index, int) or not 0 <= index < len(summary):
        raise GenerationError(
            f"point_index {index!r} outside the {len(summary)} summary points"
        )
    if not data.get("question") or not data.get("answer"):
        raise GenerationError("empty question or answer")
    return Generated(
        sampled=sampled,
        summary=summary,
        question=str(data["question"]).strip(),
        answer=str(data["answer"]).strip(),
        point_index=index,
        language=str(data.get("language") or sampled.language or "en"),
    )


def generate(
    sampled: Sequence[Sampled],
    model: str = DEFAULT_MODEL,
    settings: Settings | None = None,
    max_chars: int = 1_000_000,
    workers: int = 4,
    session_id: str | None = None,
) -> tuple[list[Generated], list[Failure]]:
    """A summary, question and answer per document, plus what failed.

    Failures are collected rather than raised: 50 documents are 50 independent
    calls, and the run is worth keeping when three of them come back malformed.
    """
    settings = settings or get_settings()
    client = _build_client(settings)
    searcher = QdrantSearcher(settings.qdrant_url, settings.qdrant_collection)

    done: list[Generated] = []
    failed: list[Failure] = []

    session_id = session_id or new_session_id()

    def one(item: Sampled) -> None:
        # Entered per worker: the session is a contextvar and does not reach
        # pool threads from the main one (see trace_session).
        with trace_session(session_id):
            try:
                body, language = document_text(searcher, item, max_chars)
                if not body.strip():
                    # Ledger says active, index has nothing. Normal after a
                    # sweep rebuilt the collection under a different name.
                    failed.append(Failure(item.doc, "no chunks in the collection"))
                    return
                item = replace(item, language=language or item.language)
                done.append(_generate_one(client, model, item, body))
            except Exception as exc:  # noqa: BLE001 — per-document isolation
                failed.append(Failure(item.doc, f"{type(exc).__name__}: {exc}"))

    tracked = {
        "model": model,
        "workers": workers,
        "max_chars": max_chars,
        "documents": len(sampled),
        "qdrant_collection": settings.qdrant_collection,
    }
    try:
        with generation_run(SYSTEM_PROMPT, tracked, session_id) as metrics:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one, sampled))
            metrics["generated"] = len(done)
            metrics["failed"] = len(failed)
    finally:
        searcher.close()

    done.sort(key=lambda g: (g.sampled.doc.source_system, g.sampled.doc.url))
    return done, failed


_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(question: str, taken: set[str]) -> str:
    """A stable, readable id — the question's own words, since that is what a
    reader of the golden set is looking for."""
    kept: list[str] = []
    size = 0
    for word in _SLUG.sub("-", question.lower()).strip("-").split("-"):
        if len(word) <= 2:  # articles and prepositions carry nothing
            continue
        if size + len(word) > 44:
            break  # on a word boundary — a slug cut mid-word reads as a typo
        kept.append(word)
        size += len(word) + 1
    base = "-".join(kept) or "generated"
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        candidate = f"{base}-{n}"
    taken.add(candidate)
    return candidate


def _yaml_text(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _comment(value: str) -> str:
    """One line, always. A newline inside a `#` comment ends the comment, and
    the rest of the answer would be parsed as YAML."""
    return " ".join(value.split())


def format_entries(
    generated: Sequence[Generated],
    existing_ids: Sequence[str] = (),
    with_reference: bool = False,
) -> str:
    """Golden-set entries, ready to paste under `questions:`.

    `reference` is left empty by default even though the model wrote an answer.
    EVALUATION.md's rule is that references are written by hand — a reference the
    machine produced turns tier 2 into a test of agreement between two models
    rather than of correctness. The proposed answer is right there in a comment
    (and in the review file) for a human to accept, edit or throw out.
    `--reference` writes it into the field instead, for anyone who would rather
    edit than retype; it is still unreviewed text either way.
    """
    taken = set(existing_ids)
    out: list[str] = [
        "# Generated entries — review before merging into golden_set.yaml.",
        "# relevant_documents is exact: each question was written from that document.",
        "# The question is what needs a human. Check that it is self-contained, that it",
        "# names enough to pick this document out of the corpus, and that it is not just",
        "# the title in question form. Delete the ones that are not.",
    ]
    if not with_reference:
        out.append("# reference is left empty on purpose (EVALUATION.md: written by")
        out.append("# hand). Each entry carries the proposed answer as a comment.")
    for item in generated:
        entry_id = _slug(item.question, taken)
        out.append(f"\n  - id: {entry_id}")
        out.append(f"    question: {_yaml_text(item.question)}")
        out.append(f"    language: {item.language}")
        reference = item.answer if with_reference else ""
        out.append(f"    reference: {_yaml_text(reference)}")
        out.append("    relevant_documents:")
        out.append(f"      - source_system: {item.sampled.doc.source_system}")
        # Quoted: a url is opaque text, and one containing ` #` would otherwise
        # turn the rest of the line into a comment.
        out.append(f"        url: {_yaml_text(item.sampled.doc.url)}")
        note = f"generated from {item.sampled.title or item.sampled.doc.url}"
        out.append(f"    tests: {_yaml_text(note)}")
        if not with_reference:
            # Only the answer. The summary point behind it is the same text
            # review.yaml already carries, and `point_index` there says which
            # one it was — no reason to repeat it in the file you paste from.
            out.append(f"    # proposed answer: {_comment(item.answer)}")
    return "\n".join(out)


class _ReviewDumper(yaml.SafeDumper):
    """Long prose as a literal block, so a summary point reads as a sentence
    rather than wrapping mid-word across indented lines."""

    def increase_indent(self, flow=False, indentless=False):
        # Indent list items under their key; PyYAML's default puts them at the
        # same column, which is valid YAML and harder to scan.
        return super().increase_indent(flow, False)


def _represent_str(dumper: yaml.SafeDumper, data: str):
    style = "|" if ("\n" in data or len(data) > 80) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_ReviewDumper.add_representer(str, _represent_str)


def format_review(generated: Sequence[Generated], failed: Sequence[Failure]) -> str:
    """The whole summary per document, not only the point the question used, so
    a reviewer can see what else was on offer.
    """
    report = {
        "generated": [
            {
                "source_system": item.sampled.doc.source_system,
                "url": item.sampled.doc.url,
                "title": item.sampled.title,
                "language": item.language,
                "question": item.question,
                "answer": item.answer,
                "point_index": item.point_index,
                "summary": list(item.summary),
            }
            for item in generated
        ],
        "failed": [
            {
                "source_system": failure.doc.source_system,
                "url": failure.doc.url,
                "error": failure.reason,
            }
            for failure in failed
        ],
    }
    return yaml.dump(
        report,
        Dumper=_ReviewDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    ).rstrip("\n")


def run_generate(
    entries: Sequence[Entry],
    n: int = 50,
    seed: int | None = None,
    dsn: str | None = None,
    model: str | None = None,
    max_chars: int = 1_000_000,
    workers: int = 4,
    reference: bool = False,
    data_dir: str | None = None,
    dry_run: bool = False,
) -> int:
    """Sample, generate, write files; returns the exit code.

    `entries` is the existing golden set, read only for the ids already taken.
    """
    try:
        sampled = sample_documents(n=n, seed=seed, dsn=dsn)
    except OperationalError as exc:
        # The usual first-run failure on this box: Docker Desktop is not
        # started, so ../postgres is not listening.
        print(f"cannot read the ledger: {exc.orig}", file=sys.stderr)
        return 2
    if not sampled:
        print("no active documents in the ledger", file=sys.stderr)
        return 2
    print(f"sampled {len(sampled)} documents", file=sys.stderr)
    if dry_run:
        for item in sampled:
            print(f"  {item.doc.source_system:14} {item.doc.url}")
        return 0

    chosen_model = model or os.environ.get("RAG_EVAL_GENERATE_MODEL", DEFAULT_MODEL)
    session_id = new_session_id()
    try:
        done, failed = generate(
            sampled,
            model=chosen_model,
            max_chars=max_chars,
            workers=workers,
            session_id=session_id,
        )
    except MissingApiKey as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # One directory per run, both files inside it:
    # entries.yaml - question - answer data, same format as the golden set
    # review.yaml  - the full summaries behind them, plus what failed
    run_dir = (Path(data_dir) if data_dir else DATA_DIR) / session_id
    run_dir.mkdir(parents=True, exist_ok=True)
    entries_path = run_dir / "entries.yaml"
    review_path = run_dir / "review.yaml"

    entries_path.write_text(
        "questions:\n" + format_entries(done, [e.id for e in entries], reference) + "\n",
        encoding="utf-8",
    )
    review_path.write_text(
        format_review(done, failed) + "\n", encoding="utf-8"
    )
    print(f"session {session_id} -> {run_dir}", file=sys.stderr)
    print(f"  entries.yaml  {len(done)} entries, review before merging", file=sys.stderr)
    print(f"  review.yaml   the summaries behind them", file=sys.stderr)
    for failure in failed:
        print(f"  failed {failure.doc.url}: {failure.reason}", file=sys.stderr)
    print(
        f"{len(done)} generated, {len(failed)} failed, model {chosen_model}",
        file=sys.stderr,
    )
    return 0 if done else 1
