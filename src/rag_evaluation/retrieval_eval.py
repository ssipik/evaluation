"""Tier 1: score retrieval against the golden set. No LLM, no judge.

Deterministic and free, which is what makes it the tier a sweep runs on every
candidate — the judge tiers only see the two or three it shortlists. It imports
`rag_agent.retrieval.Retriever` directly rather than going through `RagAgent`,
because nothing here needs the model.

The output is a document ranking, not a chunk ranking: see metrics.py for why
labels are documents and what that costs. A document is `source_system` + `url`
together, so a chunk is turned into a `DocRef` before anything is compared.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from rag_agent.retrieval import Chunk, Retriever
from rag_config import Params, set_param
from rag_agent.settings import Settings, get_params, get_settings

from rag_evaluation.golden import DocRef, Entry
from rag_evaluation.metrics import METRIC_NAMES, Scores, collapse, mean_scores, score_query

__all__ = ["QueryResult", "Report", "chunk_ref", "run_tier1"]

log = logging.getLogger(__name__)


def chunk_ref(chunk: Chunk) -> DocRef | None:
    """The document a chunk belongs to, or None when it carries no url and so
    can never match a label."""
    if not chunk.source_url:
        return None
    return DocRef(source_system=chunk.source_system or "", url=chunk.source_url)


@dataclass(frozen=True)
class QueryResult:
    entry: Entry
    chunks: list[Chunk]
    scores: Scores | None  # None when the entry has no labels yet

    @property
    def retrieved_documents(self) -> list[DocRef]:
        return collapse(chunk_ref(c) for c in self.chunks)


@dataclass(frozen=True)
class Report:
    results: list[QueryResult]
    k: int
    rerank: bool

    @property
    def scored(self) -> list[QueryResult]:
        return [r for r in self.results if r.scores is not None]

    @property
    def means(self) -> dict[str, float]:
        return mean_scores([r.scores for r in self.scored])

    def by(self, attribute: str) -> dict[str, dict[str, float]]:
        """Means split by an entry attribute — `language` answers whether German
        recall lags English, which is block 5's open question."""
        groups: dict[str, list[Scores]] = {}
        for r in self.scored:
            groups.setdefault(str(getattr(r.entry, attribute, None)), []).append(r.scores)
        return {name: mean_scores(s) for name, s in sorted(groups.items())}


def run_tier1(
    entries: Sequence[Entry],
    settings: Settings | None = None,
    params: Params | None = None,
    rerank: bool | None = None,
    retriever: Retriever | None = None,
) -> Report:
    """Search once per entry and score the entries that carry labels.

    Unlabeled entries are kept in the report but not scored: counting them as
    misses would report a labeling gap as a retrieval failure.
    """
    settings = settings or get_settings()
    params = params or get_params()
    if rerank is not None:
        params = set_param(params, "agent.rerank_enabled", rerank)
    # No limit is passed: search reads top_k and rerank_top_n from params itself,
    # so anything given here could only cap what it already decided. Tier 1 used
    # to pass top_k, which won over rerank_top_n and made it score top_k
    # reordered chunks — a configuration the agent never serves.
    #
    # `k` is the depth the report labels itself with: what comes back is
    # rerank_top_n when reranking, top_k otherwise.
    rerank_on = params.agent.rerank_enabled
    k = params.agent.rerank_top_n if rerank_on else params.agent.top_k
    owned = retriever is None
    retriever = retriever or Retriever(settings, params)

    results = []
    try:
        for entry in entries:
            chunks = retriever.search(entry.question)
            refs = [chunk_ref(c) for c in chunks]
            ranking = collapse(refs)
            scores = (
                score_query(ranking, entry.relevant_documents, chunk_docs=refs)
                if entry.scorable_tier1
                else None
            )
            results.append(QueryResult(entry=entry, chunks=chunks, scores=scores))
    finally:
        if owned:
            close = getattr(retriever, "close", None)
            if close:
                close()
    return Report(results=results, k=k, rerank=params.agent.rerank_enabled)


def format_report(report: Report) -> str:
    """The text summary. Coverage first, because a mean over three of fifteen
    entries is not a score of the system."""
    lines = [
        f"\nscored {len(report.scored)} of {len(report.results)} entries "
        f"(k={report.k}, rerank={'on' if report.rerank else 'off'})",
    ]
    if not report.scored:
        lines.append(
            "\n  nothing to score — no entry has relevant_documents yet.\n"
            "  Run `pool` and label a few, then come back."
        )
        return "\n".join(lines)

    lines.append("\n=== means ===")
    for name, value in report.means.items():
        lines.append(f"  {name:18} {value:.4f}")

    for attribute in ("language", "source_system"):
        groups = report.by(attribute)
        if len(groups) > 1:
            lines.append(f"\n=== by {attribute} ===")
            header = "  ".join(f"{n:>10}" for n in METRIC_NAMES)
            lines.append(f"  {'':14}{header}")
            for group, means in groups.items():
                row = "  ".join(f"{means[n]:>10.4f}" for n in METRIC_NAMES)
                lines.append(f"  {group:14}{row}")

    lines.append("\n=== per entry ===")
    for r in report.results:
        if r.scores is None:
            lines.append(f"  {r.entry.id:28} (unlabeled)")
        else:
            s = r.scores
            lines.append(
                f"  {r.entry.id:28} recall {s.recall:.2f}  ndcg {s.ndcg:.2f}  "
                f"mrr {s.mrr:.2f}  ({len(r.chunks)} chunks, {s.k} docs)"
            )
    return "\n".join(lines)
