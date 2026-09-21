"""The held-out score: tier 1 on the test golden set, for one chosen configuration.

The configuration comes from exactly one of:
- a yaml with a `baseline:` block, as in a sweep file — individual substitutions
  on top of the shipped parameters (`params_from_file`)
- a registered parameter set, all three sections (`params_from_mlflow`)

Scoring only: the index is looked up by the configuration's fingerprint, never
built — `sweep` or `curve` with `--reingest` build it (`collection_for`).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from rag_agent.settings import Settings
from rag_config import Params, ingest_fingerprint, load_params, set_param

from rag_evaluation.golden import DEFAULT_PATH as GOLDEN_PATH
from rag_evaluation.sweep import COLLECTION_PREFIX, _set, collection_exists

__all__ = [
    "DEFAULT_TEST_PATH",
    "HoldoutError",
    "collection_for",
    "params_from_file",
    "params_from_mlflow",
]

DEFAULT_TEST_PATH = GOLDEN_PATH.parent / "golden_set_test.yaml"


class HoldoutError(RuntimeError):
    pass


def params_from_file(path: str | Path) -> Params:
    """The shipped parameters with the file's `baseline:` substitutions applied."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not spec.get("baseline"):
        raise HoldoutError(f"{path}: no `baseline:` block to take the parameters from")
    params = load_params()
    for dotted, value in spec["baseline"].items():
        params = _set(params, dotted, value)
    return params


def params_from_mlflow(ref: str) -> tuple[Params, str]:
    """A registered parameter set, `name@alias` or `name/version`, and its summary.

    Every section, not only `agent`: the chunking and embedding half decides
    which index the configuration is scored on.
    """
    from rag_config.baseline import resolve_baseline
    from rag_config.promote import section_values

    baseline = resolve_baseline(ref)
    params = load_params()
    for section, values in section_values(baseline).items():
        for name, value in values.items():
            params = set_param(params, f"{section}.{name}", value)
    return params, baseline.summary


def collection_for(params: Params, settings: Settings) -> str:
    """The index built for this configuration's fingerprint.

    - `rag_chunks_<fingerprint>`, when a sweep or curve built it
    - the standing collection, when the fingerprint is the shipped one and
      nothing else was built — the sweep's rule for a query-side candidate
    - otherwise an error: any other index holds a different chunking
    """
    fingerprint = ingest_fingerprint(params)
    built = f"{COLLECTION_PREFIX}{fingerprint}"
    if collection_exists(built, settings.qdrant_url):
        return built
    if fingerprint == ingest_fingerprint(load_params()):
        return settings.qdrant_collection
    raise HoldoutError(
        f"no collection {built} for this configuration — build it with "
        "`sweep` or `curve` and --reingest"
    )
