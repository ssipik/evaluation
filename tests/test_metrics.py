"""The worked example in EVALUATION.md, asserted."""

from __future__ import annotations

import pytest

from rag_evaluation.metrics import Scores, collapse, mean_scores, score_query, score_ranking


class TestCollapse:
    def test_keeps_rank_order_and_first_occurrence(self):
        assert collapse(["X", "A", "Y", "A", "B"]) == ["X", "A", "Y", "B"]

    def test_drops_chunks_without_a_document(self):
        assert collapse(["A", None, "B", ""]) == ["A", "B"]

    def test_several_chunks_of_one_document_are_one_hit(self):
        """The whole reason the collapse exists: recall must stay <= 1."""
        chunks = ["A", "A", "A", "Z", "Z"]
        s = score_query(collapse(chunks), {"A"}, chunk_docs=chunks)
        assert s.recall == 1.0
        assert s.k == 2  # A and Z, not five chunks
        # ...while chunk-level precision still reports the prompt waste
        assert s.chunk_precision == pytest.approx(3 / 5)


class TestWorkedExample:
    """R={A,B}, retrieved [X, A, Y, B, Z] — EVALUATION.md's Q1."""

    @pytest.fixture
    def q1(self):
        return score_ranking(["X", "A", "Y", "B", "Z"], {"A", "B"})

    def test_hit_rate(self, q1):
        assert q1.hit_rate == 1.0

    def test_recall(self, q1):
        assert q1.recall == pytest.approx(1.0)

    def test_precision(self, q1):
        assert q1.precision == pytest.approx(0.4)

    def test_mrr(self, q1):
        assert q1.mrr == pytest.approx(0.5)

    def test_ndcg(self, q1):
        # hits at positions 2 and 4 -> 0.631 + 0.431 = 1.062
        # ideal at 1 and 2         -> 1.000 + 0.631 = 1.631
        assert q1.ndcg == pytest.approx(0.651, abs=1e-3)

    def test_q2_perfect_ranking_still_has_low_precision(self):
        """One relevant document, five asked for: precision is capped at |R|/k."""
        q2 = score_ranking(["C", "W", "V", "U", "T"], {"C"})
        assert (q2.hit_rate, q2.recall, q2.mrr, q2.ndcg) == (1.0, 1.0, 1.0, 1.0)
        assert q2.precision == pytest.approx(0.2)

    def test_q3_nothing_relevant_scores_zero_everywhere(self):
        q3 = score_ranking(["P", "Q", "R", "S", "T"], {"D", "E"})
        assert (q3.hit_rate, q3.recall, q3.precision, q3.mrr, q3.ndcg) == (0, 0, 0, 0, 0)

    def test_means_over_the_three_questions(self):
        scores = [
            score_ranking(["X", "A", "Y", "B", "Z"], {"A", "B"}),
            score_ranking(["C", "W", "V", "U", "T"], {"C"}),
            score_ranking(["P", "Q", "R", "S", "T"], {"D", "E"}),
        ]
        m = mean_scores(scores)
        assert m["hit_rate"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["precision"] == pytest.approx(0.2)
        assert m["mrr"] == pytest.approx(0.5)
        assert m["ndcg"] == pytest.approx(0.55, abs=1e-2)


class TestEdges:
    def test_empty_ranking(self):
        s = score_ranking([], {"A"})
        assert (s.hit_rate, s.recall, s.precision, s.mrr, s.ndcg) == (0, 0, 0, 0, 0)

    def test_ndcg_is_one_when_every_hit_is_at_the_top(self):
        assert score_ranking(["A", "B", "X"], {"A", "B"}).ndcg == pytest.approx(1.0)

    def test_no_scores_means_no_numbers_rather_than_zeros(self):
        """An unscored set must not look like a system that scored 0."""
        assert mean_scores([]) == {}
