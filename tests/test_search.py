"""The search space, the two objectives, and the winner selection.

All offline: no Qdrant, no TEI, no Optuna study.
"""

from __future__ import annotations

import pytest

from rag_config.params import load_params
from rag_evaluation.golden import DocRef, Entry
from rag_evaluation.metrics import Scores
from rag_evaluation.search import (
    F_SCORE,
    LADDER,
    Evaluated,
    Objective,
    SearchError,
    best,
    in_space,
    epsilon_weights,
    format_search,
    load_search,
    search_csv,
)

METRICS = ("recall", "precision", "ndcg")


def ladder(
    metrics=METRICS, weights=(1.0, 0.005, 0.0001), floor=None, tolerance=0.0
) -> Objective:
    return Objective(
        LADDER,
        tuple(metrics),
        weights=tuple(weights),
        floor=floor,
        tolerance=tolerance,
    )


def entry(entry_id: str, n_relevant: int = 2) -> Entry:
    return Entry(
        id=entry_id,
        question=f"q{entry_id}",
        relevant_documents=[
            DocRef(source_system="ecb", url=f"http://x/{entry_id}/{i}")
            for i in range(n_relevant)
        ],
    )


class FakeReport:
    """Only what the selection and the tables read off a Report."""

    def __init__(self, **means):
        self.means = {name: means.get(name, 0.0) for name in Scores.__annotations__ if name in
                      ("hit_rate", "recall", "precision", "mrr", "ndcg", "chunk_precision")}
        self.results = [1, 2, 3]
        self.scored = [1, 2, 3]
        self.rerank = False
        self.k = 10


def trial(number: int, **means) -> Evaluated:
    return Evaluated(
        number=number,
        knobs={"agent.top_k": 10 + number},
        params=load_params(),
        report=FakeReport(**means),
        value=0.0,
    )


class TestLoadSearch:
    def test_it_refuses_a_file_without_a_search_section(self, tmp_path):
        path = tmp_path / "s.yaml"
        path.write_text("baseline:\n  agent.top_k: 10\n")
        with pytest.raises(SearchError, match="no `search:` section"):
            load_search(path)

    def test_it_refuses_an_ingest_side_knob(self, tmp_path):
        """A trial scores the standing collection, so a re-chunk would score the
        wrong index rather than a new one."""
        path = tmp_path / "s.yaml"
        path.write_text("search:\n  parse_chunk.chunk_size: {low: 200, high: 900}\n")
        with pytest.raises(SearchError, match="not a query-side knob"):
            load_search(path)

    def test_it_refuses_an_objective_that_is_not_a_metric(self, tmp_path):
        path = tmp_path / "s.yaml"
        path.write_text("search:\n  agent.top_k: {low: 3, high: 9}\nobjective: [f1]\n")
        with pytest.raises(SearchError, match="unknown objective metric"):
            load_search(path)

    @pytest.mark.parametrize(
        "name, objective",
        [
            (
                "optuna_ladder_no_floor",
                {
                    "type": LADDER,
                    "metrics": ["recall", "precision"],
                    "floor": None,
                    "tolerance": 0.0,
                },
            ),
            (
                "optuna_ladder_floor",
                {"type": LADDER, "metrics": ["recall", "precision"], "floor": 0.975},
            ),
            (
                "optuna_f_score",
                {"type": F_SCORE, "metrics": ["recall", "precision"], "beta": 5.0},
            ),
        ],
    )
    def test_the_shipped_examples_load(self, name, objective):
        """Each of the three objectives has a file, and each file is the one
        thing that proves its own form still parses."""
        block = load_search(f"sweeps/{name}.yaml")["objective"]
        assert {key: block.get(key) for key in objective} == objective


class TestEpsilonWeights:
    def test_the_tiebreaks_cannot_outvote_the_metric_above_them(self):
        """The whole construction: every lower term summed is still smaller than
        one step of the metric above it."""
        entries = [entry(str(i), n_relevant=3) for i in range(15)]
        weights = epsilon_weights(entries, METRICS, k_max=40)

        recall_step = 1 / (15 * 3)
        assert weights[1] + weights[2] < recall_step

        precision_step = 1 / (15 * 40)
        assert weights[2] < weights[1] * precision_step

    def test_it_shrinks_as_the_golden_set_grows(self):
        """More questions means finer steps, so the ladder has to tighten — the
        hardcoded 1e-3 that would have been wrong at N=50."""
        small = epsilon_weights([entry(str(i)) for i in range(15)], METRICS, 30)
        large = epsilon_weights([entry(str(i)) for i in range(50)], METRICS, 30)
        assert large[1] < small[1]

    def test_the_leading_weight_is_one(self):
        weights = epsilon_weights([entry("a")], METRICS, 10)
        assert weights[0] == 1.0
        assert len(weights) == len(METRICS)


class TestBest:
    def test_recall_decides_before_anything_else(self):
        winner = best(
            [
                trial(0, recall=0.9, precision=0.01, ndcg=0.01),
                trial(1, recall=0.8, precision=0.99, ndcg=0.99),
            ],
            ladder(),
        )
        assert winner.number == 0

    def test_precision_breaks_a_recall_tie(self):
        winner = best(
            [
                trial(0, recall=0.8, precision=0.2, ndcg=0.9),
                trial(1, recall=0.8, precision=0.3, ndcg=0.1),
            ],
            ladder(),
        )
        assert winner.number == 1

    def test_ndcg_breaks_a_precision_tie(self):
        winner = best(
            [
                trial(0, recall=0.8, precision=0.2, ndcg=0.5),
                trial(1, recall=0.8, precision=0.2, ndcg=0.7),
            ],
            ladder(),
        )
        assert winner.number == 1

    def test_tolerance_treats_a_near_tie_on_recall_as_a_tie(self):
        """0.02 of recall is inside the noise of a small golden set, so the
        choice moves down to precision."""
        trials = [
            trial(0, recall=0.82, precision=0.10, ndcg=0.5),
            trial(1, recall=0.80, precision=0.40, ndcg=0.5),
        ]
        assert best(trials, ladder()).number == 0
        assert best(trials, ladder(tolerance=0.05)).number == 1

    def test_no_scored_trial_has_no_winner(self):
        assert best([], ladder()) is None


class TestTables:
    def test_the_csv_has_one_row_per_trial_plus_the_header(self):
        csv = search_csv([trial(0, recall=0.8), trial(1, recall=0.9)])
        lines = csv.strip().split("\n")
        assert lines[0].startswith("trial,objective,agent.top_k,")
        assert len(lines) == 3

    def test_the_table_names_the_winner(self):
        table = format_search([trial(0, recall=0.8), trial(1, recall=0.9)], ladder())
        assert "best: trial 1" in table

    def test_an_empty_study_says_so_rather_than_printing_a_header(self):
        assert "nothing to score" in format_search([], ladder())


class TestFloor:
    """The knee rule: recall only has to clear the floor, then precision decides.

    Numbers are the tune_top_k curve — recall saturates at 0.9841 by top_k 9,
    and every step past 7 costs precision for a fraction of one question.
    """

    K7 = {"recall": 0.9788, "precision": 0.3389, "ndcg": 0.9072}
    K9 = {"recall": 0.9841, "precision": 0.2917, "ndcg": 0.9104}
    K20 = {"recall": 0.9841, "precision": 0.1576, "ndcg": 0.9104}
    K5 = {"recall": 0.9312, "precision": 0.3984, "ndcg": 0.8867}

    def value(self, means, floor=None):
        return ladder(floor=floor).value(means)

    def test_without_a_floor_any_recall_gain_beats_any_precision_loss(self):
        assert self.value(self.K9) > self.value(self.K7)

    def test_above_the_floor_the_shallower_depth_wins(self):
        """Both clear 0.975, so recall stops mattering and precision — the one
        metric that falls as depth grows — picks top_k 7."""
        assert self.value(self.K7, 0.975) > self.value(self.K9, 0.975)

    def test_a_ladder_led_by_ndcg_cannot_settle_depth(self):
        """Why precision ranks above ndcg: ndcg only ever grows with top_k, so
        led by it the deepest trial wins — which is what the knee-a study did."""
        # The ladder `epsilon_weights` actually produces for 63 questions at
        # k_max 20 — the third rung is ~4e-4 of the second, far too small for
        # precision's 0.18 to answer ndcg's 0.0032.
        led_by_ndcg = ladder(
            metrics=("recall", "ndcg", "precision"),
            weights=(1.0, 0.00066, 2.6e-7),
            floor=0.975,
        )
        assert led_by_ndcg.value(self.K20) > led_by_ndcg.value(self.K7)

    def test_a_trial_below_the_floor_cannot_outrank_one_above_it(self):
        """K5 has the best precision of all and must still lose: no amount of
        the second metric pulls a trial back across the line."""
        assert self.value(self.K5, 0.975) < self.value(self.K20, 0.975)

    def test_below_the_floor_the_lead_metric_still_orders_them(self):
        """So the sampler has a gradient to climb back with."""
        worse = {"recall": 0.90, "precision": 0.9, "ndcg": 0.9}
        assert self.value(worse, 0.975) < self.value(self.K5, 0.975)

    def test_best_is_the_objective_argmax_when_a_floor_is_set(self):
        """The whole point of a fixed reference: no separate selection logic, so
        the sampler optimises exactly what gets picked."""
        trials = [
            Evaluated(0, {}, load_params(), FakeReport(**self.K9), 1.2),
            Evaluated(1, {}, load_params(), FakeReport(**self.K7), 1.9),
        ]
        assert best(trials, ladder(floor=0.975)).number == 1

    def test_a_floor_outside_what_a_metric_can_be_is_refused(self, tmp_path):
        path = tmp_path / "s.yaml"
        path.write_text("search:\n  agent.top_k: {low: 3, high: 9}\nfloor: 1.5\n")
        with pytest.raises(SearchError, match="neither a value the metric can take"):
            load_search(path)

    def test_baseline_is_accepted_as_a_floor(self, tmp_path):
        """The floor for when the saturation point is not known in advance."""
        path = tmp_path / "s.yaml"
        path.write_text(
            "search:\n  agent.top_k: {low: 3, high: 9}\nfloor: baseline\n"
        )
        assert load_search(path)["objective"]["floor"] == "baseline"


class TestFScore:
    """The other objective: recall and precision trade continuously, and beta is
    the exchange rate. Numbers are the tune_top_k curve, as in TestFloor.
    """

    K5 = {"recall": 0.9312, "precision": 0.3984}
    K7 = {"recall": 0.9788, "precision": 0.3389}
    K20 = {"recall": 0.9841, "precision": 0.1576}

    def f(self, beta, means):
        return Objective(F_SCORE, ("recall", "precision"), beta=beta).value(means)

    def test_beta_one_is_the_plain_harmonic_mean(self):
        assert self.f(1.0, {"recall": 0.5, "precision": 0.5}) == pytest.approx(0.5)
        assert self.f(1.0, {"recall": 1.0, "precision": 0.2}) == pytest.approx(1 / 3)

    def test_a_low_beta_buys_precision_with_recall(self):
        """F-1 at this operating point prefers top_k 5 over 7 — which is the
        reason a retrieval study needs a large beta."""
        assert self.f(1.0, self.K5) > self.f(1.0, self.K7)

    def test_a_high_beta_reaches_the_knee_the_floor_picks(self):
        """beta 5 lands on the same top_k 7 as `floor: 0.975` — over 5, whose
        recall is short, and over 20, whose precision is spent."""
        scores = [self.f(5.0, m) for m in (self.K5, self.K7, self.K20)]
        assert max(scores) == scores[1]

    def test_beta_is_monotone_in_how_much_recall_is_worth(self):
        """The knob does what it says: raising it moves the winner from the
        shallow trial to the deep one."""
        assert self.f(0.5, self.K5) > self.f(0.5, self.K20)
        assert self.f(20.0, self.K20) > self.f(20.0, self.K5)

    def test_a_trial_that_retrieved_nothing_scores_zero(self):
        assert self.f(3.0, {"recall": 0.0, "precision": 0.0}) == 0.0

    def test_best_is_the_argmax(self):
        """No lead metric means no rung to fall through to, so the objective
        alone decides — the same rule a floored ladder follows."""
        objective = Objective(F_SCORE, ("recall", "precision"), beta=5.0)
        trials = [
            Evaluated(0, {}, load_params(), FakeReport(**self.K20), 0.80),
            Evaluated(1, {}, load_params(), FakeReport(**self.K7), 0.85),
        ]
        assert best(trials, objective).number == 1


class TestObjectiveSpec:
    """The `objective:` block, both forms."""

    SEARCH = "search:\n  agent.top_k: {low: 3, high: 9}\n"

    def load(self, tmp_path, body):
        path = tmp_path / "s.yaml"
        path.write_text(self.SEARCH + body)
        return load_search(path)

    def test_a_bare_list_is_a_ladder_and_takes_the_top_level_floor(self, tmp_path):
        """The shorthand every sweeps/ file is written in still means what it
        did before the block existed."""
        spec = self.load(
            tmp_path, "floor: 0.975\nobjective: [recall, precision]\n"
        )
        assert spec["objective"]["type"] == LADDER
        assert spec["objective"]["metrics"] == ["recall", "precision"]
        assert spec["objective"]["floor"] == 0.975
        # Folded in and dropped, so there is one place to read it from.
        assert "floor" not in spec

    def test_f_score_defaults_to_recall_against_precision(self, tmp_path):
        spec = self.load(tmp_path, "objective:\n  type: f_score\n  beta: 5\n")
        assert spec["objective"]["metrics"] == ["recall", "precision"]
        assert spec["objective"]["beta"] == 5.0

    def test_a_floor_inside_the_ladder_block_is_read(self, tmp_path):
        spec = self.load(
            tmp_path,
            "objective:\n  type: ladder\n  metrics: [recall, precision]\n"
            "  floor: 0.975\n",
        )
        assert spec["objective"]["floor"] == 0.975

    def test_an_f_score_refuses_a_floor(self, tmp_path):
        """Two answers to the same question — the floor draws a hard line where
        the f_score draws a curve, and taking both leaves neither meaning it."""
        with pytest.raises(SearchError, match="no lead metric for a floor"):
            self.load(
                tmp_path, "objective:\n  type: f_score\n  beta: 5\n  floor: 0.9\n"
            )

    def test_an_f_score_refuses_anything_but_two_metrics(self, tmp_path):
        with pytest.raises(SearchError, match="exactly two metrics"):
            self.load(
                tmp_path,
                "objective:\n  type: f_score\n  metrics: [recall, precision, ndcg]\n",
            )

    def test_a_non_positive_beta_is_refused(self, tmp_path):
        with pytest.raises(SearchError, match="not a positive number"):
            self.load(tmp_path, "objective:\n  type: f_score\n  beta: 0\n")

    def test_a_ladder_refuses_a_beta(self, tmp_path):
        with pytest.raises(SearchError, match="beta belongs to an f_score"):
            self.load(tmp_path, "objective:\n  type: ladder\n  beta: 2\n")

    def test_a_tolerance_defaults_to_exact_ties(self, tmp_path):
        spec = self.load(tmp_path, "objective:\n  type: ladder\n")
        assert spec["objective"]["tolerance"] == 0.0

    def test_a_tolerance_beside_a_floor_is_refused(self, tmp_path):
        """Both say when the leading metric stops deciding, one against a fixed
        number and one against the best trial — taking both means neither."""
        with pytest.raises(SearchError, match="use whichever the study means"):
            self.load(
                tmp_path,
                "objective:\n  type: ladder\n  floor: 0.975\n  tolerance: 0.01\n",
            )

    def test_an_f_score_refuses_a_tolerance(self, tmp_path):
        with pytest.raises(SearchError, match="no lead metric for a tolerance"):
            self.load(
                tmp_path, "objective:\n  type: f_score\n  tolerance: 0.01\n"
            )

    def test_a_tolerance_outside_what_a_metric_can_span_is_refused(self, tmp_path):
        with pytest.raises(SearchError, match="not a distance the metric can span"):
            self.load(tmp_path, "objective:\n  type: ladder\n  tolerance: 2\n")

    def test_an_unknown_type_is_refused(self, tmp_path):
        with pytest.raises(SearchError, match="unknown objective type"):
            self.load(tmp_path, "objective:\n  type: weighted_metrics\n")

    def test_a_floor_in_both_places_is_refused(self, tmp_path):
        """Nothing downstream should have to pick which one wins."""
        with pytest.raises(SearchError, match="ambiguous"):
            self.load(
                tmp_path, "floor: 0.9\nobjective:\n  type: ladder\n"
            )


class TestSearchSpace:
    """A study must not answer with a configuration its own file excluded."""

    SPACE = {"agent.top_k": {"low": 4, "high": 20}, "agent.rrf_k": [1, 2, 3]}

    def test_a_value_inside_every_range_is_in_space(self):
        assert in_space({"agent.top_k": 9, "agent.rrf_k": 2}, self.SPACE)

    def test_the_seeded_baseline_can_fall_outside_a_range(self):
        """Optuna warns and uses an enqueued value out of range, so top_k 30
        from parameters.yaml lands in a study that asked for 4-20."""
        assert not in_space({"agent.top_k": 30, "agent.rrf_k": 2}, self.SPACE)

    def test_a_categorical_outside_the_choices_is_out_of_space(self):
        assert not in_space({"agent.top_k": 9, "agent.rrf_k": 7}, self.SPACE)

    def test_out_of_space_is_a_label_not_a_veto(self):
        """The baseline competes on the objective like any other trial — a point
        that wins is the objective's problem, not something to filter away."""
        winner = best(
            [
                Evaluated(0, {}, load_params(), FakeReport(recall=1.0), 9.9,
                          in_space=False),
                Evaluated(1, {}, load_params(), FakeReport(recall=0.98), 1.5),
            ],
            ladder(floor=0.9),
        )
        assert winner.number == 0
