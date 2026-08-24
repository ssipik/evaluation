"""Evaluation module (ARCHITECTURE.md block 12).

Three steps — build the golden set, tune against it, evaluate online — and three
scoring tiers inside the tuning loop. EVALUATION.md is the procedure; this
package is the machinery.

Its stated output is an optimized `parameters.yaml` for `rag-config`, which is
why the module lives outside every stage: a sweep drives stages 1-6 to build one
collection per ingest configuration, then scores it through the query path.
"""

from rag_evaluation.golden import DocRef, Entry, load_golden
from rag_evaluation.metrics import Scores, collapse, mean_scores, score_query

__all__ = [
    "DocRef",
    "Entry",
    "load_golden",
    "Scores",
    "collapse",
    "mean_scores",
    "score_query",
]
