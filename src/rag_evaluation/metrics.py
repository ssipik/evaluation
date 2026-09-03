"""Tier-1 retrieval metrics: document-level, deterministic, no LLM.

The golden set labels *documents* (`relevant_documents`) while retrieval returns
*chunks*, so every scored ranking is the retrieved chunk list collapsed to
distinct documents, first occurrence winning. Without that collapse three chunks
of one document count as three hits and recall goes above 1.

A document is `source_system` + `url` together, never the url alone (`DocRef` in
golden.py). Nothing here needs that shape though: the functions take any hashable
document key, so the tests can score plain strings.

Labels are documents rather than chunk ids because chunk ids regenerate on every
re-chunk, and chunk size is the parameter the sweep most wants to vary — see
EVALUATION.md. The cost is granularity: a relevant document whose least useful
chunk was retrieved still scores as a hit. Tier 2's groundedness scorer is what
catches that, because the judge sees the chunk text.

Definitions and a worked example are in EVALUATION.md; the tests here assert
that example.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, fields
from typing import TypeVar

__all__ = ["Scores", "collapse", "score_ranking", "score_query", "mean_scores"]

Doc = TypeVar("Doc", bound=Hashable)


@dataclass(frozen=True)
class Scores:
    """One question's scores. `k` is documents in the collapsed ranking."""

    hit_rate: float
    recall: float
    precision: float
    mrr: float
    ndcg: float
    # Prompt waste, and the one chunk-level number: what share of the retrieved
    # chunks came from a relevant document. The collapse hides it, because a
    # document is one entry however many of its chunks came back.
    chunk_precision: float
    k: int
    n_relevant: int
    n_chunks: int


def collapse(docs: Iterable[Doc | None]) -> list[Doc]:
    """Chunk documents in rank order -> distinct ones, first occurrence winning.

    Chunks that carry no document key are dropped: nothing can be labeled
    against them.
    """
    seen: dict[Doc, None] = {}
    for doc in docs:
        if doc and doc not in seen:
            seen[doc] = None
    return list(seen)


def score_ranking(ranking: Sequence[Doc], relevant: Iterable[Doc]) -> Scores:
    """Score an already-collapsed document ranking."""
    return score_query(ranking, relevant, chunk_docs=ranking)


def score_query(
    ranking: Sequence[Doc],
    relevant: Iterable[Doc],
    chunk_docs: Sequence[Doc | None],
) -> Scores:
    """Metrics for one question.

    `ranking` is the collapsed document list, `chunk_docs` the uncollapsed one
    it came from — only `chunk_precision` uses the second.
    """
    wanted = {d for d in relevant if d}
    k = len(ranking)
    hits = [i for i, doc in enumerate(ranking) if doc in wanted]

    recall = len(hits) / len(wanted) if wanted else 0.0
    precision = len(hits) / k if k else 0.0
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0

    # Rank i (0-based) is worth 1/log2(i+2); the ideal packs every relevant
    # document into the top positions.
    dcg = sum(1.0 / math.log2(i + 2) for i in hits)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(wanted), k)))
    ndcg = dcg / idcg if idcg else 0.0

    chunk_hits = sum(1 for doc in chunk_docs if doc in wanted)
    return Scores(
        hit_rate=1.0 if hits else 0.0,
        recall=recall,
        precision=precision,
        mrr=mrr,
        ndcg=ndcg,
        chunk_precision=chunk_hits / len(chunk_docs) if chunk_docs else 0.0,
        k=k,
        n_relevant=len(wanted),
        n_chunks=len(chunk_docs),
    )


METRIC_NAMES = ("hit_rate", "recall", "precision", "mrr", "ndcg", "chunk_precision")


def mean_scores(scores: Sequence[Scores]) -> dict[str, float]:
    """Per-metric mean over the questions. Empty in, empty out — not zeros."""
    if not scores:
        return {}
    return {
        name: sum(getattr(s, name) for s in scores) / len(scores)
        for name in METRIC_NAMES
    }


assert set(METRIC_NAMES) <= {f.name for f in fields(Scores)}
