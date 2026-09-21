"""The test command's parameter sources and index lookup. Offline: Qdrant and
MLflow are replaced."""

from __future__ import annotations

import pytest

import rag_evaluation.holdout as holdout
from rag_agent.settings import get_settings
from rag_config import ingest_fingerprint, load_params, set_param
from rag_evaluation.holdout import HoldoutError, collection_for, params_from_file
from rag_evaluation.sweep import SweepError


class TestParamsFromFile:
    def test_the_baseline_substitutions_land_on_the_shipped_parameters(self, tmp_path):
        f = tmp_path / "s.yaml"
        f.write_text("baseline:\n  agent.top_k: 3\n  parse_chunk.chunk_size: 500\n")
        params = params_from_file(f)
        assert params.agent.top_k == 3
        assert params.parse_chunk.chunk_size == 500
        # everything not named keeps the shipped value
        assert params.agent.rrf_k == load_params().agent.rrf_k

    def test_a_file_without_a_baseline_block_raises(self, tmp_path):
        f = tmp_path / "s.yaml"
        f.write_text("sweep:\n  agent.top_k: [3]\n")
        with pytest.raises(HoldoutError, match="baseline"):
            params_from_file(f)

    def test_an_unknown_name_raises(self, tmp_path):
        f = tmp_path / "s.yaml"
        f.write_text("baseline:\n  agent.nonsense: 1\n")
        with pytest.raises(SweepError):
            params_from_file(f)


class TestCollectionFor:
    @pytest.fixture
    def existing(self, monkeypatch):
        names: set[str] = set()
        monkeypatch.setattr(holdout, "collection_exists", lambda name, url: name in names)
        return names

    def test_the_built_collection_when_there_is_one(self, existing):
        params = set_param(load_params(), "parse_chunk.chunk_size", 500)
        built = f"rag_chunks_{ingest_fingerprint(params)}"
        existing.add(built)
        assert collection_for(params, get_settings()) == built

    def test_the_standing_one_for_the_shipped_fingerprint(self, existing):
        settings = get_settings()
        assert collection_for(load_params(), settings) == settings.qdrant_collection

    def test_another_chunking_with_nothing_built_raises(self, existing):
        params = set_param(load_params(), "parse_chunk.chunk_size", 500)
        with pytest.raises(HoldoutError, match="--reingest"):
            collection_for(params, get_settings())
