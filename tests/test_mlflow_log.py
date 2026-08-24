"""generation_run's contract, offline — no tracking server involved."""

from __future__ import annotations

import pytest

from rag_evaluation.mlflow_log import (
    active,
    DATASET_EXPERIMENT,
    EXPERIMENT,
    generation_run,
    tracking_uri,
    new_session_id,
    trace_session,
)


class TestGenerationRun:
    def test_the_experiment_names_are_the_agreed_ones(self):
        # the two experiments: scoring the pipeline, and building the set it is
        # scored against
        assert EXPERIMENT == "RAG-Evaluation"
        assert DATASET_EXPERIMENT == "RAG-Evaluation-Dataset"

    def test_without_a_tracking_uri_it_is_a_no_op(self, monkeypatch):
        # the offline default: generation still runs, nothing is logged
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with generation_run("prompt", {"model": "m"}) as metrics:
            metrics["generated"] = 1
        assert metrics == {"generated": 1}

    def test_the_caller_gets_a_dict_it_can_fill_unconditionally(self, monkeypatch):
        # the reason it yields a dict rather than None: generate() needs no guard
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with generation_run("prompt", {}) as metrics:
            assert isinstance(metrics, dict)

    def test_a_caller_exception_propagates_rather_than_breaking_the_generator(
        self, monkeypatch
    ):
        # the bug this guards: a second yield on the failure path raises
        # RuntimeError and hides what actually went wrong
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with pytest.raises(ValueError, match="boom"):
            with generation_run("prompt", {}):
                raise ValueError("boom")

    def test_an_unreachable_server_does_not_stop_generation(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
        with generation_run("prompt", {"model": "m"}) as metrics:
            metrics["generated"] = 2
        assert metrics == {"generated": 2}


class TestSession:
    def test_every_execution_gets_its_own_id(self):
        assert new_session_id() != new_session_id()

    def test_the_id_sorts_by_when_it_ran(self):
        # timestamp first, so the UI's session list is chronological
        first, second = sorted([new_session_id(), new_session_id()])
        assert first.startswith("generate-") and second.startswith("generate-")

    def test_the_run_records_which_session_it_was(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with generation_run("prompt", {}, "generate-fixed") as metrics:
            metrics["generated"] = 1
        assert metrics == {"generated": 1}

    def test_trace_session_is_transparent_when_tracing_is_off(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with trace_session("generate-x"):
            pass

    def test_trace_session_lets_the_workers_exception_through(self):
        # each worker enters this; swallowing here would lose the per-document
        # failure that generate() records
        with pytest.raises(ValueError, match="doc failed"):
            with trace_session("generate-x"):
                raise ValueError("doc failed")


class TestTrackingUri:
    """MLflow's own variable is the switch, so the library agrees with us."""

    def test_it_is_read_from_mlflows_own_variable(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://somewhere:5000")
        assert tracking_uri() == "http://somewhere:5000"
        assert active()

    def test_unset_means_off(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert tracking_uri() == ""
        assert not active()

    def test_empty_means_off_too(self, monkeypatch):
        # how the offline suite and any box without the service stay quiet
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
        assert not active()
