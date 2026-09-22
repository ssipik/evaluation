"""Tiers 2 and 3 wiring. Offline: no agent, no judge calls — the scorers are only
constructed, never run."""

from __future__ import annotations

import os
from types import SimpleNamespace

from rag_evaluation.agent_eval import (
    _predict_fn,
    _scorers,
    _size_mlflow_http_pool,
    _warn_about_the_judge,
    build_dataset,
)
from rag_evaluation.golden import Entry

JUDGE = "openrouter:/openai/gpt-5.4"


class TestBuildDataset:
    def test_a_reference_becomes_the_expected_response(self):
        rows = build_dataset([Entry(id="a", question="q?", reference="r.")])
        assert rows == [
            {"inputs": {"question": "q?"}, "expectations": {"expected_response": "r."}}
        ]

    def test_no_reference_means_no_expectations_key(self):
        rows = build_dataset([Entry(id="a", question="q?")])
        assert rows == [{"inputs": {"question": "q?"}}]


class TestScorers:
    def names(self, scorers):
        return {s.name for s in scorers}

    def test_the_reference_free_set(self):
        names = self.names(_scorers(JUDGE, with_reference=False))
        assert names == {
            "retrieval_groundedness",
            "relevance_to_query",
            "retrieval_relevance",
            "citation_and_refusal",
            "tool_call_correctness",
            "tool_call_efficiency",
        }

    def test_a_reference_adds_correctness_and_sufficiency(self):
        extra = self.names(_scorers(JUDGE, with_reference=True)) - self.names(
            _scorers(JUDGE, with_reference=False)
        )
        assert extra == {"correctness", "retrieval_sufficiency"}

    def test_every_scorer_uses_the_given_judge(self):
        # MLflow's own default is a model nobody chose
        assert {s.model for s in _scorers(JUDGE, with_reference=True)} == {JUDGE}


class TestWarnAboutTheJudge:
    def test_same_model_warns(self, capsys):
        _warn_about_the_judge("openrouter:/anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5")
        assert "same model" in capsys.readouterr().err

    def test_same_family_warns(self, capsys):
        _warn_about_the_judge("openrouter:/anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5")
        assert "both anthropic" in capsys.readouterr().err

    def test_other_family_is_silent(self, capsys):
        _warn_about_the_judge(JUDGE, "anthropic/claude-haiku-4.5")
        assert capsys.readouterr().err == ""


class FakeAgent:
    """Answers with the question and how many turns this agent has run."""

    def __init__(self):
        self.turns = 0

    async def run(self, question):
        self.turns += 1

        async def handler():
            return SimpleNamespace(response=f"{question}#{self.turns}")

        return handler()


class TestPredictFn:
    def test_returns_the_response_text(self):
        assert _predict_fn(FakeAgent)("q?") == "q?#1"

    def test_every_question_gets_a_fresh_agent(self):
        made = []

        def new_agent():
            made.append(FakeAgent())
            return made[-1]

        predict = _predict_fn(new_agent)
        predict("a?")
        # a shared agent would be on its second turn here
        assert predict("b?") == "b?#1"
        assert len(made) == 2


class TestSizeMlflowHttpPool:
    def test_covers_every_concurrent_logger(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_HTTP_POOL_MAXSIZE", raising=False)
        monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "4")
        monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "8")
        monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS", "10")
        _size_mlflow_http_pool()
        assert os.environ["MLFLOW_HTTP_POOL_MAXSIZE"] == "42"

    def test_an_explicit_setting_wins(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_HTTP_POOL_MAXSIZE", "7")
        _size_mlflow_http_pool()
        assert os.environ["MLFLOW_HTTP_POOL_MAXSIZE"] == "7"
