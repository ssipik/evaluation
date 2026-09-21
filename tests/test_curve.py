"""Curve: the knob vocabulary and the reingest path. Offline — the build and
the scorer are replaced, no stage or service runs."""

from __future__ import annotations

import pytest

import rag_evaluation.curve as curve
import rag_evaluation.sweep as sweep
from rag_agent.settings import get_settings
from rag_config import load_params
from rag_evaluation.curve import (
    INGEST_SWEEPABLE,
    SWEEPABLE,
    CurveError,
    cast_params,
    candidates,
    parse_overrides,
    parse_values,
    run_curve,
)


class TestVocabulary:
    def test_ingest_knobs_are_the_numeric_fingerprinted_fields(self):
        assert "parse_chunk.chunk_size" in INGEST_SWEEPABLE
        assert "embed.sparse_avg_len" in INGEST_SWEEPABLE
        # names, not numbers: no command-line value list spells them
        assert "embed.dense_model" not in INGEST_SWEEPABLE
        assert "embed.sparse_models" not in INGEST_SWEEPABLE

    def test_every_knob_is_spelled_section_dot_field(self):
        assert "agent.top_k" in SWEEPABLE
        assert "top_k" not in SWEEPABLE
        assert all("." in name for name in SWEEPABLE + INGEST_SWEEPABLE)

    def test_a_baseline_comes_back_under_the_same_names(self):
        # Baseline.section strips the prefix; cast_params puts it back
        assert cast_params({"top_k": "7", "collection": "x"}) == {"agent.top_k": 7}

    def test_an_ingest_interval_parses_as_ints(self):
        assert parse_values("300:500:100", "parse_chunk.chunk_size") == [300, 400, 500]

    def test_an_ingest_knob_can_be_held_with_set_params(self):
        assert parse_overrides(["parse_chunk.chunk_overlap=50"]) == {
            "parse_chunk.chunk_overlap": 50
        }


class TestCandidates:
    def test_one_per_value_each_moving_only_the_knob(self):
        produced = list(candidates("parse_chunk.chunk_size", [300, 400]))
        assert [c.value for c in produced] == [300, 400]
        assert [c.params.parse_chunk.chunk_size for c in produced] == [300, 400]
        assert produced[0].params.agent == load_params().agent

    def test_an_unknown_knob_raises(self):
        with pytest.raises(CurveError, match="not sweepable"):
            list(candidates("nonsense", [1]))


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """run_curve with the services and the stages replaced. Records what was
    built and what was scored, and restores the env run_curve writes."""
    seen = {"built": [], "scored": []}
    monkeypatch.delenv("RAG_PARAMETERS_PATH", raising=False)
    monkeypatch.delenv("RAG_QDRANT_COLLECTION", raising=False)
    monkeypatch.setattr(sweep, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(curve, "frozen_snapshot", lambda spec, overwrite: "dsn")
    # only the standing index exists, as on a Qdrant before any build
    standing = get_settings().qdrant_collection
    monkeypatch.setattr(curve, "collection_exists", lambda name, url: name == standing)
    monkeypatch.setattr(
        curve,
        "build",
        lambda candidate, collection, env, spec, dsn: seen["built"].append(
            (candidate.value, collection, env["RAG_QDRANT_COLLECTION"], spec)
        ),
    )

    def score(entries, settings, params):
        seen["scored"].append((params.parse_chunk.chunk_size, settings.qdrant_collection))
        return object()

    monkeypatch.setattr(curve, "run_tier1", score)
    return seen


class TestRouting:
    """Which values change the index is the fingerprint's call, as in a sweep."""

    def test_an_ingest_knob_without_reingest_is_skipped(self, offline):
        results = run_curve("parse_chunk.chunk_size", [300], [])
        assert results[0].report is None
        assert offline["built"] == [] and offline["scored"] == []

    def test_a_query_knob_scores_the_standing_collection(self, offline):
        standing = get_settings().qdrant_collection
        results = run_curve("agent.top_k", [5, 8], [])
        assert {s.settings.qdrant_collection for s in results} == {standing}
        assert offline["built"] == []

    def test_dry_run_builds_and_scores_nothing(self, offline):
        results = run_curve("parse_chunk.chunk_size", [300], [], reingest=True, dry_run=True)
        assert results[0].report is None
        assert offline["built"] == [] and offline["scored"] == []


class TestReingest:
    def test_each_value_is_built_and_scored_in_its_own_collection(self, offline):
        results = run_curve(
            "parse_chunk.chunk_size", [300, 400], [],
            reingest=True, ingest_spec={"parse_workers": 2},
        )
        collections = {s.candidate.value: s.settings.qdrant_collection for s in results}
        assert collections[300] != collections[400]
        assert all(c.startswith("rag_chunks_") for c in collections.values())
        # scored against the collection built for that value, with its chunking
        assert offline["scored"] == [(300, collections[300]), (400, collections[400])]
        # the stages are pointed at it, and get the ingest settings
        assert [b[1] for b in offline["built"]] == [collections[300], collections[400]]
        assert all(b[1] == b[2] for b in offline["built"])
        assert all(b[3] == {"parse_workers": 2} for b in offline["built"])

    def test_a_query_knob_shares_one_freshly_built_collection(self, offline):
        results = run_curve("agent.top_k", [5, 8], [], reingest=True)
        assert len({s.settings.qdrant_collection for s in results}) == 1
