"""Tune several query knobs jointly, with Optuna choosing where to look next.

The third driver over `run_tier1`, and the one that gives up interpretability
for coverage:

    curve   one knob, every value, plotted     — where does this knob stop paying
    sweep   n knobs, one at a time             — what is each knob worth alone
    search  n knobs, jointly, TPE-guided       — what combination wins

A sweep cannot see interactions by construction: it only ever scores candidates
one step from the baseline, so the best `rrf_k` at `top_k=30` is the only one it
ever measures. This varies them together, and `plot_contour` shows the surface a
star of candidates cannot.

Query-side knobs only, like a curve: everything here changes without a re-ingest,
so a whole study runs against the standing collection. An ingest-side knob is
refused rather than silently scored against the wrong index — a joint search over
`chunk_size` needs the two-level design in EVALUATION.md, not this.

The study lives in SQLite under `runs/optuna/`, so an interrupted study resumes
by name instead of starting over.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from rag_agent.settings import Settings, get_params, get_settings
from rag_config import AgentParams, Params, set_param

from rag_evaluation.golden import Entry
from rag_evaluation.metrics import METRIC_NAMES
from rag_evaluation.retrieval_eval import Report, run_tier1

__all__ = [
    "Evaluated",
    "Objective",
    "SearchError",
    "BASELINE_FLOOR",
    "DEFAULT_OBJECTIVE",
    "F_SCORE",
    "LADDER",
    "best",
    "in_space",
    "epsilon_weights",
    "format_search",
    "load_search",
    "make_figures",
    "resolve_objective",
    "run_search",
    "search_csv",
]

log = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
STORAGE_DIR = RUNS_DIR / "optuna"

# The two `objective.type:` values a search file can ask for. See `Objective`.
LADDER = "ladder"
F_SCORE = "f_score"
OBJECTIVE_KINDS = (LADDER, F_SCORE)

# The default ladder: recall first, precision on a recall tie, ndcg on a
# precision tie. Any prefix of METRIC_NAMES works; the file can name its own —
# `sweeps/knee.yaml` drops ndcg, because a rung that only breaks exact ties
# decides nothing when the metric varies in its fifth decimal.
#
# precision has to sit above ndcg, however much better a retrieval-quality
# metric ndcg is. recall, ndcg, mrr and hit_rate are all non-decreasing in top_k
# — they only gain as more documents come back — so precision and
# chunk_precision are the only two here that ever prefer a shallower retrieval,
# and a ladder not led by one of them picks the deepest trial in the study.
DEFAULT_OBJECTIVE = ("recall", "precision", "ndcg")

# The same two metrics for an f_score, and the same reason: one that grows with
# top_k against one that falls, or the score cannot choose a depth either.
DEFAULT_F_SCORE_METRICS = ("recall", "precision")

# `floor: baseline` — the floor nobody has to know in advance. See `_floor`.
BASELINE_FLOOR = "baseline"

_TYPES = {f.name: type(f.default).__name__ for f in fields(AgentParams)}


class SearchError(ValueError):
    """The study was asked for something it cannot do."""


@dataclass(frozen=True)
class Evaluated:
    """One trial: what was tried, what it scored, and the single number the
    sampler saw."""

    number: int
    knobs: dict[str, Any]
    params: Params
    report: Report | None
    value: float
    # False for the seeded baseline when parameters.yaml holds a value the file
    # did not ask for. Marked in the table so a row with an out-of-range value is
    # explained; it competes on the objective like any other trial.
    in_space: bool = True

    @property
    def label(self) -> str:
        return f"trial-{self.number:04d}"


@dataclass(frozen=True)
class Objective:
    """How a trial's metric means become the one number the sampler ranks on.

    Two kinds, differing in what the leading metric is allowed to do:

    - `ladder`  the first metric decides and the rest break its exact ties, with
                `tolerance` widening "ties" to what the golden set cannot tell
                apart. A `floor` replaces both with the knee rule — clear it,
                then maximise the next metric — and above the floor the leading
                metric stops mattering entirely.
    - `f_score` the two metrics trade against each other continuously, `beta`
                setting the exchange rate. Nothing decides alone, so there is no
                lead metric for a floor to hold up or a tolerance to loosen.

    Built by `resolve_objective` rather than read off the file: both the weights
    and a `floor: baseline` are measured, not declared.
    """

    kind: str
    metrics: tuple[str, ...]
    # ladder only: one weight per metric, from `epsilon_weights`.
    weights: tuple[float, ...] = ()
    # ladder only: the resolved number, never the string `baseline`.
    floor: float | None = None
    # ladder without a floor only: how close to the best the leading metric has
    # to be to count as tied, so the choice falls to the next one down.
    tolerance: float = 0.0
    # f_score only: how much more the first metric weighs than the second.
    beta: float = 1.0

    @property
    def scalar(self) -> bool:
        """Whether the value alone orders the trials, so `best` is its argmax
        and a tolerance has no rung left to fall through to."""
        return self.kind == F_SCORE or self.floor is not None

    def value(self, means: dict[str, float]) -> float:
        """The one number the sampler ranks on."""
        if self.kind == F_SCORE:
            return _f_score(
                means.get(self.metrics[0], 0.0),
                means.get(self.metrics[1], 0.0),
                self.beta,
            )
        return _ladder_value(means, self.metrics, self.weights, self.floor)

    def describe(self) -> str:
        """One line for a table header."""
        if self.kind == F_SCORE:
            return f"F-{self.beta:g} of {self.metrics[0]} and {self.metrics[1]}"
        rungs = " > ".join(self.metrics)
        if self.floor is not None:
            return f"{rungs}, {self.metrics[0]} floor {self.floor:g}"
        if self.tolerance:
            return f"{rungs}, tolerance {self.tolerance:g}"
        return rungs

    @property
    def params(self) -> dict[str, Any]:
        """What MLflow records, so two studies can be told apart by their
        parameters alone."""
        return {
            "objective": self.kind,
            "objective.metrics": ",".join(self.metrics),
            "objective.beta": f"{self.beta:g}" if self.kind == F_SCORE else "none",
            "floor": "none" if self.floor is None else round(self.floor, 6),
            "tolerance": self.tolerance,
        }


def _f_score(favoured: float, other: float, beta: float) -> float:
    """The weighted harmonic mean of two metrics, `beta**2` times as much weight
    on the first as on the second.

    Computed from the means rather than per question, so the metric columns of
    the results table explain the objective column exactly. A per-question F
    would also score 0 on every question that retrieved nothing relevant, which
    on a golden set this size is mostly noise.

    Note how large beta has to be at a retrieval operating point: at recall 0.98
    and precision 0.20, F-1 is 0.33 and its gradient points at precision hard
    enough to drive top_k to 1. Reproducing the knee those numbers come from
    takes a beta around 5.
    """
    b2 = beta * beta
    denominator = b2 * other + favoured
    if denominator <= 0.0:
        return 0.0
    return (1.0 + b2) * favoured * other / denominator


def _objective_spec(raw: dict) -> dict:
    """The `objective:` block as a validated mapping, whichever form was used.

    - `objective: [recall, precision]` is the ladder, and takes the file's
      top-level `floor`/`floor_slack` — the shorthand `sweeps/` was written in.
    - `objective: {type: ..., ...}` carries its own, floor included.
    """
    spec = raw.get("objective")
    if spec is None or isinstance(spec, list):
        spec = {
            "type": LADDER,
            "metrics": list(spec or DEFAULT_OBJECTIVE),
            "floor": raw.get("floor"),
            "floor_slack": raw.get("floor_slack"),
        }
    elif isinstance(spec, dict):
        spec = dict(spec)
        if raw.get("floor") is not None or raw.get("floor_slack") is not None:
            raise SearchError(
                "a top-level `floor:` alongside an `objective:` block is "
                "ambiguous — put the floor inside the block"
            )
    else:
        raise SearchError(
            "objective: expected a list of metrics or a block with a `type:`"
        )

    kind = spec.get("type") or LADDER
    if kind not in OBJECTIVE_KINDS:
        raise SearchError(
            f"unknown objective type {kind!r} — one of {', '.join(OBJECTIVE_KINDS)}"
        )
    default = DEFAULT_F_SCORE_METRICS if kind == F_SCORE else DEFAULT_OBJECTIVE
    metrics = list(spec.get("metrics") or default)
    unknown = [n for n in metrics if n not in METRIC_NAMES]
    if unknown:
        raise SearchError(
            f"unknown objective metric {unknown[0]!r} — one of "
            f"{', '.join(METRIC_NAMES)}"
        )
    spec["type"], spec["metrics"] = kind, metrics
    return _check_f_score(spec) if kind == F_SCORE else _check_ladder(spec)


def _check_f_score(spec: dict) -> dict:
    if len(spec["metrics"]) != 2:
        raise SearchError(
            "an f_score trades exactly two metrics against each other, not "
            f"{len(spec['metrics'])}"
        )
    try:
        beta = float(spec.get("beta", 1.0))
    except (TypeError, ValueError):
        beta = -1.0
    if not beta > 0.0:
        raise SearchError(f"beta {spec.get('beta')!r} is not a positive number")
    spec["beta"] = beta
    if spec.get("floor") is not None or spec.get("floor_slack") is not None:
        raise SearchError(
            "an f_score has no lead metric for a floor to hold up — it answers "
            "the same question as the floor with a continuous trade-off, so "
            "use one or the other"
        )
    if spec.get("tolerance") is not None:
        raise SearchError(
            "an f_score has no lead metric for a tolerance to loosen — beta "
            "already says what a bit of the first metric is worth"
        )
    return spec


def _check_ladder(spec: dict) -> dict:
    if spec.get("beta") is not None:
        raise SearchError("beta belongs to an f_score, not a ladder")
    floor = spec.get("floor")
    if floor is not None and floor != BASELINE_FLOOR:
        try:
            in_range = 0.0 <= float(floor) <= 1.0
        except (TypeError, ValueError):
            in_range = False
        if not in_range:
            raise SearchError(
                f"floor {floor!r} is neither a value the metric can take nor "
                f"{BASELINE_FLOOR!r}"
            )
    tolerance = spec.get("tolerance")
    if tolerance is None:
        spec["tolerance"] = 0.0
        return spec
    if floor is not None:
        raise SearchError(
            "a floor and a tolerance both decide when the leading metric stops "
            "deciding, one against a fixed number and one against the best "
            "trial — use whichever the study means, not both"
        )
    try:
        in_range = 0.0 <= float(tolerance) <= 1.0
    except (TypeError, ValueError):
        in_range = False
    if not in_range:
        raise SearchError(
            f"tolerance {tolerance!r} is not a distance the metric can span"
        )
    spec["tolerance"] = float(tolerance)
    return spec


def load_search(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "search" not in raw:
        raise SearchError(f"{path}: no `search:` section")
    for dotted in raw["search"]:
        if not dotted.startswith("agent."):
            raise SearchError(
                f"{dotted!r} is not a query-side knob — a search scores against "
                "the standing collection, and anything else needs a re-ingest"
            )
    raw["objective"] = _objective_spec(raw)
    # Folded into the objective above, and dropped so nothing downstream can
    # read a floor from two places and disagree.
    raw.pop("floor", None)
    raw.pop("floor_slack", None)
    return raw


def _suggest(trial, dotted: str, spec: Any) -> Any:
    """One knob's value for this trial, in the type its dataclass field has.

    - a list is a categorical choice
    - a mapping is a range: `low`, `high`, and optionally `step` or `log`
    """
    name = dotted.partition(".")[2]
    kind = _TYPES.get(name, "int")
    if isinstance(spec, list):
        if not spec:
            raise SearchError(f"{dotted}: empty list of choices")
        return trial.suggest_categorical(dotted, spec)
    if not isinstance(spec, dict) or "low" not in spec or "high" not in spec:
        raise SearchError(
            f"{dotted}: expected a list of choices or a range with low and high"
        )
    low, high, step, log_scale = (
        spec["low"],
        spec["high"],
        spec.get("step"),
        bool(spec.get("log", False)),
    )
    if kind == "bool":
        raise SearchError(f"{dotted} is a flag — give it a list, not a range")
    if kind == "float":
        return trial.suggest_float(dotted, float(low), float(high), step=step, log=log_scale)
    return trial.suggest_int(dotted, int(low), int(high), step=int(step or 1), log=log_scale)


def in_space(knobs: dict[str, Any], space: dict) -> bool:
    """Whether every knob is inside the range the file declared.

    The seeded baseline carries whatever `parameters.yaml` holds, and Optuna
    accepts an enqueued value outside the distribution (it warns and uses it).
    So a study can contain a point its own search space excluded.
    """
    for dotted, spec in space.items():
        value = knobs.get(dotted)
        if isinstance(spec, list):
            if value not in spec:
                return False
        elif isinstance(spec, dict) and not (spec["low"] <= value <= spec["high"]):
            return False
    return True


def epsilon_weights(
    entries: Sequence[Entry], names: Sequence[str], k_max: int
) -> list[float]:
    """Weights that turn the metric ladder into one number the sampler can rank.

    `recall + w1*precision + w2*ndcg` orders trials lexicographically as long as
    each weight is smaller than the smallest real difference in the metric above
    it. Those differences are not arbitrary: over N labelled questions recall
    moves in steps of `1/(N*lcm(n_relevant))` and precision in steps of at least
    `1/(N*k_max)`, so the ladder is computed from the golden set rather than
    guessed. Half the gap, at every level, leaves a factor-of-two margin.

    The argument dies if the metrics ever stop being quantized — bootstrapped
    means, weighted questions — and then `best()`'s tolerance is what to trust.
    """
    labelled = [e for e in entries if getattr(e, "scorable_tier1", False)]
    n = len(labelled) or 1
    counts = [len(e.relevant_documents) or 1 for e in labelled] or [1]
    lcm = math.lcm(*counts)
    gaps = {
        "recall": 1.0 / (n * lcm),
        "precision": 1.0 / (n * max(k_max, 1)),
        "hit_rate": 1.0 / n,
    }
    weights = [1.0]
    for name in names[:-1]:
        # Anything not quantized in a way worth trusting gets the coarsest gap
        # of the ones that are, which only ever makes the ladder more cautious.
        weights.append(weights[-1] * gaps.get(name, 1.0 / (n * max(k_max, 1))) / 2)
    return weights


def _ladder_value(
    means: dict[str, float],
    names: Sequence[str],
    weights: Sequence[float],
    floor: float | None = None,
) -> float:
    """The ladder as one number.

    Without a floor this is the epsilon ladder: the first metric decides, the
    rest break its exact ties.

    With a floor it is the knee instead — "the first metric only has to be good
    enough, then maximise the next one". Above the floor every trial scores
    `1 + <the rest>`, so no amount of the second metric can pull a trial back
    across the line; below it, the score is the first metric itself, so the
    sampler climbs back toward the floor. Unlike a tolerance the reference is
    fixed, so this is a function of the parameters alone and `best()` is its
    argmax — the sampler optimises exactly what gets picked.
    """
    lead = means.get(names[0], 0.0)
    tail = zip(names[1:], weights[1:], strict=True)
    if len(names) == 1:
        return lead
    if floor is None:
        return lead + sum(w * means.get(n, 0.0) for n, w in tail)
    if lead < floor:
        return min(lead + sum(w * means.get(n, 0.0) for n, w in tail), floor)
    
    # Above the floor the first metric contributes exactly 1.0 to the total score,
    # its value becomes irrelevant so the next metric becomes relevant now.
    # Rescaling is only such that the total score become more readable and interpretable.
    scale = weights[1]
    return 1.0 + sum((w / scale) * means.get(n, 0.0) for n, w in tail)

def _floor(
    spec: dict,
    metric: str,
    entries: Sequence[Entry],
    settings: Settings,
    base: Params,
) -> float | None:
    """The `objective:` block's floor as a number, scoring the baseline first if
    the file asked for it.

    `floor: baseline` is for when the saturation point is not known in advance:
    the floor becomes what the configuration in force already achieves, so the
    study means "buy me precision without giving up the recall we have today".
    `floor_slack` is how much of it may be given up after all.

    Measured before the study rather than from trial 0, so the floor is fixed for
    every trial including the baseline's own — a floor discovered mid-study would
    score the trials before it on a different scale than the ones after.
    """
    floor = spec.get("floor")
    if floor is None:
        return None
    if floor != BASELINE_FLOOR:
        return float(floor)
    report = run_tier1(entries, settings=settings, params=base)
    achieved = report.means.get(metric, 0.0)
    floor = achieved - float(spec.get("floor_slack") or 0.0)
    print(
        f"  floor from the baseline: {metric} {achieved:.4f}"
        + (f" less {spec['floor_slack']} slack -> {floor:.4f}" if spec.get("floor_slack") else "")
    )
    return floor


def resolve_objective(
    spec: dict,
    entries: Sequence[Entry],
    k_max: int,
    settings: Settings,
    base: Params,
) -> Objective:
    """The objective the file asked for, with its floor and weights as numbers.

    Both are measured rather than read: `floor: baseline` is whatever the
    configuration in force scores, and the ladder's weights come from the golden
    set. Resolved once, before the study, so every trial is scored on the same
    scale — a floor discovered mid-study would rank the trials before it
    differently from the ones after.
    """
    block = spec["objective"]
    metrics = tuple(block["metrics"])
    if block["type"] == F_SCORE:
        return Objective(F_SCORE, metrics, beta=float(block["beta"]))
    return Objective(
        LADDER,
        metrics,
        weights=tuple(epsilon_weights(entries, metrics, k_max)),
        floor=_floor(block, metrics[0], entries, settings, base),
        tolerance=float(block["tolerance"]),
    )


def run_search(
    entries: Sequence[Entry],
    spec: dict,
    n_trials: int,
    settings: Settings | None = None,
    params: Params | None = None,
    study_name: str = "search",
    seed: int | None = None,
) -> tuple[list[Evaluated], Any, Objective]:
    """`n_trials` trials against the standing collection, newest knowledge first.

    Returns the trials in the order they ran, the Optuna study — which holds the
    sampler's own view and draws the figures — and the resolved objective, which
    the table and MLflow both need and neither can rebuild from the file alone.
    """
    import optuna

    if n_trials < 1:
        raise SearchError("a study needs at least one trial")
    wiring = settings or get_settings()
    base = params or get_params()
    for dotted, value in (spec.get("baseline") or {}).items():
        base = _set(base, dotted, value)

    space = spec["search"]
    # The deepest a trial can retrieve, which is what precision is quantized by.
    depth = space.get("agent.top_k")
    k_max = (
        max(depth) if isinstance(depth, list) else
        int(depth["high"]) if isinstance(depth, dict) else base.agent.top_k
    )
    goal = resolve_objective(spec, entries, k_max, wiring, base)
    log.info(
        "objective %s, weights %s",
        goal.describe(),
        [f"{w:.3g}" for w in goal.weights],
    )

    results: list[Evaluated] = []

    def score(trial) -> float:
        tuned = base
        knobs = {}
        for dotted, choices in space.items():
            value = _suggest(trial, dotted, choices)
            knobs[dotted] = value
            tuned = _set(tuned, dotted, value)
        report = run_tier1(entries, settings=wiring, params=tuned)
        means = report.means
        for name in METRIC_NAMES:
            trial.set_user_attr(name, means.get(name, 0.0))
        value = goal.value(means)
        results.append(
            Evaluated(
                trial.number,
                knobs,
                tuned,
                report,
                value,
                in_space=in_space(knobs, space),
            )
        )
        print(f"  trial {trial.number:>4}  {value:.6f}  " + "  ".join(
            f"{n}={means.get(n, 0.0):.4f}" for n in goal.metrics
        ))
        return value

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{STORAGE_DIR / f'{study_name}.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    # The configuration in force is trial 0, so a study can never report a
    # winner without having measured what it is being compared against.
    if not study.trials:
        study.enqueue_trial(
            {d: _current(base, d) for d in space}, skip_if_exists=True
        )
    # On the study as well as in the returned objective, so a resumed study
    # still carries the number a `baseline` floor resolved to.
    study.set_user_attr("floor", goal.floor)
    study.optimize(score, n_trials=n_trials)
    return results, study, goal


def _current(params: Params, dotted: str) -> Any:
    section, _, name = dotted.partition(".")
    return getattr(getattr(params, section), name)


def _set(params: Params, dotted: str, value: Any) -> Params:
    """rag_config's setter, speaking this module's error type."""
    try:
        return set_param(params, dotted, value)
    except ValueError as exc:
        raise SearchError(str(exc)) from None


def best(results: Sequence[Evaluated], objective: Objective) -> Evaluated | None:
    """The winner: the objective's argmax, unless the ladder leaves a choice open.

    A floored ladder and an f_score both encode the whole rule in the value, so
    the winner is simply the highest-scoring trial — the sampler and the pick
    agree by construction.

    For a plain ladder the objective's `tolerance` is what "saturated" means:
    every trial within it of the best `recall` is treated as tied on recall, and
    the choice moves to `precision`, and so on. Zero reproduces the sampler's own
    ordering; set it from the noise of the golden set (a bootstrap CI), not from
    the quantization.
    """
    pool = [r for r in results if r.report is not None and r.report.scored]
    if not pool:
        return None
    if objective.scalar:
        return max(pool, key=lambda r: r.value)
    for name in objective.metrics:
        top = max(r.report.means.get(name, 0.0) for r in pool)
        pool = [
            r
            for r in pool
            if r.report.means.get(name, 0.0) >= top - objective.tolerance
        ]
        if len(pool) == 1:
            break
    return pool[0]


def _cell(value: Any) -> str:
    """A knob value at a width a table can hold — a suggested float carries
    sixteen digits and none of them are meaningful here."""
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _table(
    results: Sequence[Evaluated], title: str, knobs: Sequence[str], width: int
) -> list[str]:
    """One table's lines: the header, then a row per trial in the order given."""
    lines = [
        title,
        f"  {'trial':>10}  {'objective':>10}  "
        + "  ".join(f"{k.partition('.')[2]:>{min(width, 16)}}" for k in knobs)
        + "  "
        + "  ".join(f"{n:>10}" for n in METRIC_NAMES),
    ]
    for r in results:
        # The number the sampler ranked on, beside the metrics it was built
        # from — a row that wins on a tiebreak is unreadable without it.
        mark = "" if r.in_space else "*"
        row = f"  {str(r.number) + mark:>10}  {r.value:>10.6f}  " + "  ".join(
            f"{_cell(r.knobs[k]):>{min(width, 16)}}" for k in knobs
        )
        if r.report is None or not r.report.scored:
            lines.append(row + f"  {'(not scored)':>10}")
            continue
        means = r.report.means
        lines.append(row + "  " + "  ".join(f"{means[n]:>10.4f}" for n in METRIC_NAMES))
    return lines


def format_search(results: Sequence[Evaluated], objective: Objective) -> str:
    """Every trial's metric means, twice — `curve.format_table` with the trial
    where a curve has the swept value.

    - in the order they ran, which is where the sampler's progress is visible
    - best first, which is where the answer is

    Both, because they say different things: a good row late in the first table
    means the search was still improving, and the same row at the top of the
    second is the configuration to promote.
    """
    scored = [r for r in results if r.report is not None and r.report.scored]
    if not results:
        return "\n  nothing to score — the study ran no trials."
    if not scored:
        return "\n  nothing to score — no entry has relevant_documents yet."
    last = scored[-1].report
    knobs = list(results[0].knobs)
    width = max(12, *(len(k) for k in knobs))
    header = (
        f"{len(last.scored)} of {len(last.results)} entries scored, "
        f"rerank={'on' if last.rerank else 'off'}, {objective.describe()}"
    )
    lines = _table(
        results, f"\n=== search, in the order they ran: {header} ===", knobs, width
    )
    lines += _table(
        sorted(results, key=lambda r: r.value, reverse=True),
        f"\n=== search, best first: {header} ===",
        knobs,
        width,
    )
    if any(not r.in_space for r in results):
        lines.append(
            "\n  * outside the declared search space (the seeded baseline), "
            "competing on the objective like any other trial"
        )
    winner = best(results, objective)
    if winner:
        lines.append(
            f"\n  best: trial {winner.number} — "
            + ", ".join(f"{k}={_cell(v)}" for k, v in winner.knobs.items())
        )
        lead, floor = objective.metrics[0], objective.floor
        if floor is not None:
            missed = [r for r in scored if r.report.means.get(lead, 0.0) < floor]
            lines.append(
                f"  ({len(scored) - len(missed)} of {len(scored)} trials cleared "
                f"the floor; the rest are ranked by {lead} alone)"
            )
        elif objective.tolerance and winner is not max(scored, key=lambda r: r.value):
            # Worth saying out loud: the sampler ranked on the objective, and a
            # tolerance can pick a different row than the one it put on top.
            lines.append(
                f"  (tolerance {objective.tolerance:g} chose it over the "
                "highest objective)"
            )
    return "\n".join(lines)


def search_csv(results: Sequence[Evaluated]) -> str:
    """The numbers behind the study, so it can be re-read without Optuna."""
    if not results:
        return "trial,objective\n"
    knobs = list(results[0].knobs)
    header = (
        "trial,objective,"
        + ",".join(knobs)
        + ","
        + ",".join(METRIC_NAMES)
        + ",scored_entries,coverage"
    )
    lines = [header]
    for r in results:
        head = f"{r.number},{r.value:.8f}," + ",".join(str(r.knobs[k]) for k in knobs)
        if r.report is None or not r.report.scored:
            lines.append(head + "," * (len(METRIC_NAMES) + 2))
            continue
        means = r.report.means
        coverage = (
            len(r.report.scored) / len(r.report.results) if r.report.results else 0.0
        )
        lines.append(
            head
            + ","
            + ",".join(f"{means.get(n, 0.0):.6f}" for n in METRIC_NAMES)
            + f",{len(r.report.scored)},{coverage:.4f}"
        )
    return "\n".join(lines) + "\n"


def make_figures(study) -> dict[str, Any]:
    """Optuna's own views, keyed by artifact path — the study's answer to a
    curve's per-metric plots.

    Each is best-effort: a plot that needs more trials than the study has, or an
    importance evaluator that needs scikit-learn, is skipped rather than losing
    the whole study's charts.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on this box, and none needed
    import optuna.visualization.matplotlib as viz

    def figure(drawn):
        """A grid plot hands back an array of Axes rather than one — every Axes
        in it belongs to the same Figure, so any of them names it."""
        return (drawn.flat[0] if hasattr(drawn, "flat") else drawn).figure

    # Matplotlib's default canvas holds one plot. The grid plots put an n-by-n
    # panel on that same canvas, so they are sized by how many knobs the study
    # has rather than left to shrink into illegibility.
    n = max(1, len(study.trials[0].params) if study.trials else 1)
    figures: dict[str, Any] = {}
    plots = [
        ("plots/history.png", viz.plot_optimization_history, (11, 6)),
        ("plots/slice.png", viz.plot_slice, (max(11, 5.0 * n), 5.5)),
        ("plots/importances.png", viz.plot_param_importances, (11, 6)),
    ]
    if len(study.trials) > 1:
        side = max(14.0, 4.5 * n)
        plots.append(("plots/contour.png", viz.plot_contour, (side, side)))
    for path, draw, size in plots:
        try:
            fig = figure(draw(study))
            fig.set_size_inches(*size)
            # The enlarged canvas needs its layout recomputed or the panels
            # crowd their own axis labels — except where a colorbar is one of
            # the axes, which tight_layout packs into the middle of the grid.
            has_colorbar = any(ax.get_label() == "<colorbar>" for ax in fig.axes)
            if not has_colorbar:
                with contextlib.suppress(Exception):
                    fig.tight_layout()
            figures[path] = fig
        except Exception as exc:
            log.info("skipped %s: %s", path, exc)
    return figures
