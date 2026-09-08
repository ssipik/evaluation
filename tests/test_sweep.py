"""Candidate generation, the fingerprint routing, and the production guard.

All offline: no Qdrant, no TEI, no stages.
"""

from __future__ import annotations

import pytest
import yaml

from rag_config.params import load_params
from rag_evaluation.sweep import (
    PRODUCTION_COLLECTION,
    Candidate,
    SweepError,
    candidates,
    guard,
    write_candidate,
)


SPEC = {
    "baseline": {"agent.top_k": 10},
    "sweep": {"agent.top_k": [5, 20], "parse_chunk.chunk_size": [384]},
}


class TestCandidates:
    def test_baseline_comes_first_and_carries_the_overrides(self):
        first = next(iter(candidates(SPEC)))
        assert first.label == "baseline"
        assert first.params.agent.top_k == 10

    def test_one_candidate_per_knob_value_not_a_grid(self):
        """Coordinate descent: 1 baseline + 2 + 1, not 1 + 2x1 combinations."""
        assert len(list(candidates(SPEC))) == 4

    def test_each_candidate_changes_exactly_one_knob_from_the_baseline(self):
        produced = list(candidates(SPEC))
        baseline = produced[0].params
        for candidate in produced[1:]:
            differing = [
                section
                for section in ("parse_chunk", "embed", "agent")
                if getattr(candidate.params, section) != getattr(baseline, section)
            ]
            assert len(differing) == 1

    def test_unknown_parameter_raises(self):
        with pytest.raises(SweepError, match="unknown parameter"):
            list(candidates({"sweep": {"agent.nonsense": [1]}}))

    def test_parameter_without_a_section_raises(self):
        with pytest.raises(SweepError, match="unknown parameter"):
            list(candidates({"sweep": {"top_k": [1]}}))


class TestFingerprintRouting:
    """Which loop a candidate lands in is decided by the hash, not by hand."""

    def test_query_side_candidates_share_the_baseline_collection(self):
        produced = {c.label: c for c in candidates(SPEC)}
        assert produced["agent.top_k=5"].fingerprint == produced["baseline"].fingerprint
        assert produced["agent.top_k=20"].collection == produced["baseline"].collection

    def test_a_chunk_size_candidate_gets_its_own_collection(self):
        produced = {c.label: c for c in candidates(SPEC)}
        assert (
            produced["parse_chunk.chunk_size=384"].collection
            != produced["baseline"].collection
        )

    def test_candidates_sharing_a_collection_still_get_their_own_file(self, tmp_path, monkeypatch):
        import rag_evaluation.sweep as sweep

        monkeypatch.setattr(sweep, "RUNS_DIR", tmp_path)
        produced = {c.label: c for c in candidates(SPEC)}
        a = write_candidate(produced["agent.top_k=5"])
        b = write_candidate(produced["agent.top_k=20"])
        assert a.parent == b.parent and a != b
        assert yaml.safe_load(a.read_text())["agent"]["top_k"] == 5
        assert yaml.safe_load(b.read_text())["agent"]["top_k"] == 20


class TestWrittenFile:
    def test_it_is_a_complete_parameter_set_not_a_delta(self, tmp_path, monkeypatch):
        """A score without its whole configuration is not a data point."""
        import rag_evaluation.sweep as sweep

        monkeypatch.setattr(sweep, "RUNS_DIR", tmp_path)
        path = write_candidate(next(iter(candidates(SPEC))))
        body = yaml.safe_load(path.read_text())
        assert set(body) >= {"version", "parse_chunk", "embed", "agent"}
        assert "settings" not in body  # config lives in its own file beside it
        assert body["parse_chunk"]["chunk_size"] == load_params().parse_chunk.chunk_size

    def test_the_collection_can_be_pinned_to_an_existing_index(self, tmp_path, monkeypatch):
        import rag_evaluation.sweep as sweep

        monkeypatch.setattr(sweep, "RUNS_DIR", tmp_path)
        path = write_candidate(next(iter(candidates(SPEC))), collection="rag_chunks")
        beside = yaml.safe_load((path.parent / "settings.yaml").read_text())
        assert beside["qdrant_collection"] == "rag_chunks"


class TestGuard:
    """What the rejected `--eval` flag would have bought, in one check."""

    def test_refuses_the_production_collection(self):
        with pytest.raises(SweepError, match="production collection"):
            guard(PRODUCTION_COLLECTION, "postgresql://rag@x/rag_eval_a3f19c")

    def test_refuses_the_production_ledger(self):
        with pytest.raises(SweepError, match="production ledger"):
            guard("rag_chunks_a3f19c", "postgresql://rag@localhost/rag_ingestion")

    def test_allows_a_dedicated_pair(self):
        guard("rag_chunks_a3f19c", "postgresql://rag@localhost/rag_eval_a3f19c")
