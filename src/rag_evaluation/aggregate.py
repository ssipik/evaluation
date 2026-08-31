"""Multi-document entries, built from the summary points `generate` already wrote.

`generate` samples one document per call, so every entry it writes has exactly
one `relevant_documents` row. That leaves two gaps the golden set needs filled
(EVALUATION.md): a question written from document A is often also answered by
document B, and nothing in the set requires *several* documents at once.

This module fills the second. It reads a run's `review.yaml`, embeds every
summary point, and looks for the signature of a periodic series — the same
statement repeated across documents with different figures:

    "annual growth rate of M3 increased to 3.2% in March 2026 from 3.0%..."
    "annual growth rate of M3 decreased to 2.7% in April 2026 from 3.2%..."

Those describe one measure at different times, so one question can be written
that needs all of them, and recall becomes graded rather than binary.

Three filters stand between "similar text" and "usable cluster", each for a
failure seen on the real corpus:

- **an upper bound on documents per cluster.** Boilerplate is what similarity
  search finds first — BaFin's cookie banner scored 0.990 across documents,
  higher than anything with content in it. A statement appearing in many
  documents is a template, not a fact.
- **figures must differ.** Identical text in two documents is a duplicated fact,
  answerable from either one alone. That is the *other* gap, and labelling both
  as required would score correct retrieval as 50% recall — see
  `metrics.score_query`, where recall is hits over all wanted documents.
- **wording must match.** Two statements about unrelated things can drift over
  the threshold; sharing a numeric shape as well is what says "same measure".

What survives goes to the model as a group, with one instruction: write a
question that cannot be answered without all of them.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from rag_agent.settings import Settings, get_settings
from rag_agent.tei import TeiClient

from rag_evaluation.generate import (
    DEFAULT_MODEL,
    MissingApiKey,
    _build_client,
    _comment,
    _slug,
    _yaml_text,
)
from rag_evaluation.golden import DocRef
from rag_evaluation.golden import DEFAULT_PATH as GOLDEN_PATH
from rag_evaluation.mlflow_log import generation_run, new_session_id, trace_session

__all__ = [
    "Point",
    "Cluster",
    "Aggregated",
    "load_points",
    "find_clusters",
    "figures_differ",
    "aggregate",
    "format_entries",
    "run_aggregate",
    "SYSTEM_PROMPT",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MAX_DOCUMENTS",
]

log = logging.getLogger(__name__)

# Tuned on the 50-document run of 2026-08-21, where 311 points gave 38 pairs
# above 0.90 and the monetary-developments series sat at 0.978. Lower than this
# starts pairing unrelated statements that share a sentence shape.
DEFAULT_THRESHOLD = 0.90

# Above this a cluster is a template, not a fact. The cookie banner spans 7
# documents; the ECB monetary series spans 4.
DEFAULT_MAX_DOCUMENTS = 5

# One subdirectory per session, next to golden_set.yaml because the whole point
# is reading a run and copying the entries you keep across into it. Nothing here
# is authoritative — the golden set is; this is a staging area for review.
DATA_DIR = GOLDEN_PATH.parent

# Any run of digits, so "3.2%", "2026" and "23 July" all count as figures.
_FIGURE = re.compile(r"\d[\d.,]*")

SYSTEM_PROMPT = """\
You write evaluation questions for a retrieval system over financial and \
regulatory documents.

You are given several statements. Each comes from a DIFFERENT document, and they \
describe the same measure or the same kind of fact at different times or for \
different subjects.

Write ONE question that cannot be answered from any single one of them — it must \
require all of the statements together. Then give the answer, drawing on every \
statement.

Rules, all of them about how the question will be used. It is asked against the \
whole corpus with the documents withheld, and the system is scored on whether it \
finds every one of them again:

- Self-contained. Never refer to "the statements", "these documents", "the \
above", "the title", "the page" or "the section" — at scoring time there is no \
document in view. "die im Titel angegebene Änderung" is unusable however well it \
reads.
- It must genuinely need all of them. "What was M3 growth in June?" needs one \
document; "How did M3 growth develop between March and June?" needs four. Write \
the second kind.
- Name the range explicitly — the months, the periods, the subjects covered — so \
that a reader knows what is being asked without seeing the sources.
- Identify the documents by what they say, not by where they are stored. \
Identifiers a reader would use are good: an ISIN, a circular number like 05/2018 \
(WA), a fund name, a date. Storage handles are not: Confluence page ids, wiki \
space keys, URLs, file names. A question quoting one is answered by string \
matching and tests nothing about retrieval.
- Do not ask for a specific answer length or format.
- Write in the language the statements are written in.

Return JSON only:
{"question": "...", "answer": "...", "language": "de" or "en"}
"""

_SCHEMA = {
    "name": "aggregated_question",
    "schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "language": {"type": "string", "enum": ["de", "en"]},
        },
        "required": ["question", "answer", "language"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class Point:
    """One summary point, and the document it came from."""

    doc: DocRef
    title: str | None
    text: str


@dataclass(frozen=True)
class Cluster:
    """Points from different documents describing the same measure."""

    points: tuple[Point, ...]
    similarity: float

    @property
    def documents(self) -> tuple[DocRef, ...]:
        seen: dict[DocRef, None] = {}
        for p in self.points:
            seen.setdefault(p.doc, None)
        return tuple(seen)


@dataclass(frozen=True)
class Aggregated:
    """A question needing every document in its cluster."""

    cluster: Cluster
    question: str
    answer: str
    language: str


def load_points(review_path: str | Path) -> list[Point]:
    """Every summary point in a run's `review.yaml`, with its document."""
    data = yaml.safe_load(Path(review_path).read_text(encoding="utf-8")) or {}
    points: list[Point] = []
    for record in data.get("generated") or []:
        doc = DocRef(record["source_system"], record["url"])
        for text in record.get("summary") or []:
            points.append(Point(doc=doc, title=record.get("title"), text=str(text)))
    return points


def figures_differ(points: Sequence[Point]) -> bool:
    """True when the points carry different figures — the series signature.

    `Sequence`, not `list`: `find_clusters` passes `Cluster.points`, which is a
    tuple because the dataclass is frozen.

    Same wording *and* same numbers is one fact duplicated across documents,
    which needs a different kind of entry than this module writes (see the
    module docstring).
    """
    seen = {tuple(_FIGURE.findall(p.text)) for p in points}
    return len(seen) > 1 and all(_FIGURE.search(p.text) for p in points)


def _similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    """Every pairwise cosine at once.

    Normalise the rows, then one matrix product — with unit rows the dot product
    *is* the cosine.
    """
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    # A zero vector has no direction, so call it similar to nothing.
    m = np.divide(m, norms, out=np.zeros_like(m), where=norms != 0)
    return m @ m.T


def find_clusters(
    points: list[Point],
    threshold: float = DEFAULT_THRESHOLD,
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    settings: Settings | None = None,
    vectors: list[list[float]] | None = None,
) -> list[Cluster]:
    """Groups of points from different documents describing the same measure.

    Grouping is greedy from the strongest pair outwards rather than by connected
    components: components chain, so one boilerplate line linking two real
    clusters would merge them. `vectors` is for tests, which have no TEI.
    """
    if len(points) < 2:
        return []
    if vectors is None:
        base = settings or get_settings()
        vectors = TeiClient(base.tei_url, timeout=120.0).embed([p.text for p in points])

    sim = _similarity_matrix(vectors)
    pairs = [
        (float(sim[i, j]), i, j)
        for i in range(len(points))
        for j in range(i + 1, len(points))
        if points[i].doc != points[j].doc and sim[i, j] >= threshold
    ]
    pairs.sort(reverse=True)


    # A group represents a cluster of similar points
    # Each pair is added to an existing group if either point is already in it, or to a newly created group
    groups: list[list[int]] = []  
    where: dict[int, int] = {}
    for score, i, j in pairs:
        gi, gj = where.get(i), where.get(j)
        if gi is None and gj is None:
            where[i] = where[j] = len(groups)
            groups.append([i, j])
        elif gi is None:
            groups[gj].append(i)
            where[i] = gj
        elif gj is None:
            groups[gi].append(j)
            where[j] = gi

    out: list[Cluster] = []
    for members in groups:
        chosen = tuple(points[i] for i in sorted(members))
        docs = {p.doc for p in chosen}
        if not 2 <= len(docs) <= max_documents:
            continue
        if not figures_differ(chosen):
            continue
        ordered = sorted(members)
        scores = [
            float(sim[i, j])
            for a, i in enumerate(ordered)
            for j in ordered[a + 1 :]
        ]
        out.append(
            Cluster(points=chosen, similarity=sum(scores) / len(scores) if scores else 0.0)
        )
    out.sort(key=lambda c: c.similarity, reverse=True)
    return out


def _ask(client, model: str, cluster: Cluster) -> Aggregated:
    lines = [
        f"{i}. [{p.title or p.doc.url}] {p.text}"
        for i, p in enumerate(cluster.points, 1)
    ]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        # Lower than generate's 0.7: there is one right shape of question here,
        # and variety is not worth a question that needs only one document.
        temperature=0.3,
        response_format={"type": "json_schema", "json_schema": _SCHEMA},
    )
    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)
    if not data.get("question") or not data.get("answer"):
        raise ValueError("empty question or answer")
    return Aggregated(
        cluster=cluster,
        question=str(data["question"]).strip(),
        answer=str(data["answer"]).strip(),
        language=str(data.get("language") or "en"),
    )


def aggregate(
    clusters: list[Cluster],
    model: str = DEFAULT_MODEL,
    settings: Settings | None = None,
    session_id: str | None = None,
) -> tuple[list[Aggregated], list[str]]:
    """One question per cluster, plus a reason per cluster that produced none.

    Sequential, not pooled like `generate`: a handful of clusters is a handful
    of calls, and the cost of a thread pool is not worth it here.
    """
    base = settings or get_settings()
    client = _build_client(base)
    session_id = session_id or new_session_id()

    done: list[Aggregated] = []
    failed: list[str] = []
    tracked = {"model": model, "clusters": len(clusters), "mode": "aggregate"}
    # Its own prompt name: this prompt and generate's are edited independently,
    # and one registry entry receiving both would read as one prompt flipping
    # between two texts.
    with generation_run(
        SYSTEM_PROMPT,
        tracked,
        session_id,
        prompt_name="evaluation_aggregate_prompt",
    ) as metrics:
        with trace_session(session_id):
            for cluster in clusters:
                try:
                    done.append(_ask(client, model, cluster))
                except Exception as exc:  # noqa: BLE001 — per-cluster isolation
                    failed.append(f"{type(exc).__name__}: {exc}")
        metrics["generated"] = len(done)
        metrics["failed"] = len(failed)
    return done, failed


def format_entries(
    aggregated: list[Aggregated], existing_ids: Sequence[str] = ()
) -> str:
    """Golden-set entries, ready to paste under `questions:`.

    Same shape `generate.format_entries` writes, with every document of the
    cluster in `relevant_documents` — which is the whole point, so `tests` says
    so and names the count a reviewer should check against.
    """
    taken = set(existing_ids)
    out: list[str] = [
        "# Multi-document entries — review before merging into golden_set.yaml.",
        "# Each question is meant to need EVERY document listed under it. Check that:",
        "#   - it really cannot be answered from one of them alone",
        "#   - the range it names matches the documents actually listed",
        "#   - no document in the list is redundant",
        "# reference is left empty on purpose (EVALUATION.md: written by hand).",
    ]
    for item in aggregated:
        entry_id = _slug(item.question, taken)
        out.append(f"\n  - id: {entry_id}")
        out.append(f"    question: {_yaml_text(item.question)}")
        out.append(f"    language: {item.language}")
        out.append('    reference: ""')
        out.append("    relevant_documents:")
        for doc in item.cluster.documents:
            out.append(f"      - source_system: {doc.source_system}")
            out.append(f"        url: {_yaml_text(doc.url)}")
        note = (
            f"aggregation across {len(item.cluster.documents)} documents — "
            "recall fails if any is missing"
        )
        out.append(f"    tests: {_yaml_text(note)}")
        out.append(f"    # proposed answer: {_comment(item.answer)}")
    return "\n".join(out)


def format_review(aggregated: list[Aggregated], failed: list[str]) -> str:
    """The clusters behind the entries, so a reviewer can see what was joined."""
    report = {
        "aggregated": [
            {
                "question": item.question,
                "answer": item.answer,
                "language": item.language,
                "similarity": round(item.cluster.similarity, 4),
                "sources": [
                    {"url": p.doc.url, "title": p.title, "point": p.text}
                    for p in item.cluster.points
                ],
            }
            for item in aggregated
        ],
        "failed": list(failed),
    }
    from rag_evaluation.generate import _ReviewDumper

    return yaml.dump(
        report,
        Dumper=_ReviewDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    ).rstrip("\n")


def run_aggregate(
    session_id: str | Path,
    existing_ids: Sequence[str] = (),
    threshold: float = DEFAULT_THRESHOLD,
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    model: str | None = None,
    data_dir: str | None = None,
    dry_run: bool = False,
) -> int:
    """Cluster one run's summary points and write the entries beside them."""
    import sys

    run_dir = (Path(data_dir) if data_dir else DATA_DIR) / session_id
    review = run_dir / "review.yaml"
    if not review.exists():
        print(f"no review.yaml in {run_dir}", file=sys.stderr)
        return 2

    points = load_points(review)
    print(f"{len(points)} summary points", file=sys.stderr)
    try:
        clusters = find_clusters(points, threshold, max_documents)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot embed (is ../tei up?): {exc}", file=sys.stderr)
        return 2
    print(f"{len(clusters)} clusters worth asking about", file=sys.stderr)
    if not clusters:
        return 1
    if dry_run:
        for c in clusters:
            print(f"\n  {len(c.documents)} documents, similarity {c.similarity:.3f}")
            for p in c.points:
                print(f"    [{(p.title or p.doc.url)[:44]}] {p.text[:90]}")
        return 0

    chosen = model or os.environ.get("RAG_EVAL_GENERATE_MODEL", DEFAULT_MODEL)
    try:
        done, failed = aggregate(clusters, model=chosen)
    except MissingApiKey as exc:
        print(str(exc), file=sys.stderr)
        return 2

    (run_dir / "aggregate.yaml").write_text(
        "questions:\n" + format_entries(done, existing_ids) + "\n", encoding="utf-8"
    )
    (run_dir / "aggregate-review.yaml").write_text(
        format_review(done, failed) + "\n", encoding="utf-8"
    )
    print(f"  aggregate.yaml         {len(done)} entries", file=sys.stderr)
    print("  aggregate-review.yaml  the clusters behind them", file=sys.stderr)
    for reason in failed:
        print(f"  failed: {reason}", file=sys.stderr)
    return 0 if done else 1
