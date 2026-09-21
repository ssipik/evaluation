"""Argparse only — the logic lives in the library modules.

    python -m rag_evaluation retrieval               tier 1, no LLM
    python -m rag_evaluation pool --session-id …     labeling helper
    python -m rag_evaluation agent --ids isin-lookup tiers 2 and 3 (needs a key)
    python -m rag_evaluation generate --n 50         seed entries from the corpus
    python -m rag_evaluation aggregate --session …  multi-document entries from a run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

from rag_agent.settings import get_params, get_settings

from rag_evaluation.generate import DEFAULT_MODEL as GENERATE_MODEL
from rag_config import set_param

from rag_evaluation.curve import INGEST_SWEEPABLE, SWEEPABLE
from rag_evaluation.golden import DEFAULT_PATH, GoldenSetError, load_golden
from rag_evaluation.holdout import DEFAULT_TEST_PATH

log = logging.getLogger(__name__)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--golden", default=str(DEFAULT_PATH), help="the golden set")
    parser.add_argument("--ids", nargs="*", help="only these entry ids")


def _ingest_settings(specs: list[str]) -> dict:
    """`--ingest name=value` pairs as a sweep file's `ingest:` block. Whole numbers
    become ints; anything else (a snapshot name, a dsn) stays a string."""
    out: dict = {}
    for spec in specs:
        name, sep, raw = spec.partition("=")
        if not sep:
            raise SystemExit(f"--ingest {spec!r}: expected name=value")
        out[name.strip()] = int(raw) if raw.strip().lstrip("-").isdigit() else raw.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_evaluation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tier1 = sub.add_parser("retrieval", help="tier 1: score retrieval, no LLM")
    _common(tier1)
    tier1.add_argument("--mlflow-run", help="name of the MLflow run")
    tier1.add_argument(
        "--rerank", action="store_true", help="score with the cross-encoder on"
    )
    tier1.add_argument(
        "--no-mlflow", action="store_true", help="print only, log nothing"
    )

    test = sub.add_parser(
        "test", help="tier 1 on the test golden set, for one chosen configuration"
    )
    test.add_argument(
        "--golden", default=str(DEFAULT_TEST_PATH), help="the test golden set"
    )
    test.add_argument("--ids", nargs="*", help="only these entry ids")
    source = test.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file",
        help="a yaml with a `baseline:` block, as in a sweep file — its "
        "substitutions on top of the shipped parameters",
    )
    source.add_argument(
        "--mlflow-model",
        metavar="NAME@ALIAS|NAME/VERSION",
        help="a registered parameter set, as MLflow addresses one",
    )
    test.add_argument(
        "--mlflow-run", help="name of the MLflow run (default: test-<file or model>)"
    )
    test.add_argument(
        "--no-mlflow", action="store_true", help="print only, log nothing"
    )

    agent = sub.add_parser("agent", help="tiers 2 and 3: judge scorers")
    _common(agent)
    agent.add_argument("--judge-model", help="scoring model; must not be the agent's")
    agent.add_argument("--rerank", action="store_true")

    # generate command and its arguments
    gen = sub.add_parser(
        "generate", help="generates data points (questions, answers) for the golden set from sampled documents"
    )
    _common(gen)
    gen.add_argument("--n", type=int, default=50, help="number of documents to sample")
    gen.add_argument(
        "--seed", type=int, help="random generator seed used to sample documents"
    )
    gen.add_argument(
        "--model",
        help=f"model that analyses the samples documents and generates the (answer, question) data points (default {GENERATE_MODEL})",
    )
    gen.add_argument("--dsn", help="Postgres DSN (default RAG_PG_DSN)")
    gen.add_argument(
        "--max-chars", type=int, default=1_000_000, help="maximum document size per LLM call"
    )
    gen.add_argument("--workers", type=int, default=4, help="number of concurrent LLM calls")
    gen.add_argument(
        "--data-dir", help="base directory for the run (default data/); a session subdirectory is made under it"
    )
    gen.add_argument(
        "--reference",
        action="store_true",
        help="put the generated answer in `reference` — still unreviewed text",
    )
    gen.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sampled documents and stop, before any LLM call",
    )

    # aggregate command and its arguments
    agg = sub.add_parser(
        "aggregate",
        help="generates (question, answer) data points for the golden set based on information retrieved from multiple documents",
    )
    _common(agg)
    agg.add_argument(
        "--session-id",
        required=True,
        help="session id of a previouls `generate` run; data is usually found at data/generate-<session>",
    )
    agg.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="lower threshold for the similarity (cosine) of a pair to be considered a match (default 0.90)",
    )
    agg.add_argument(
        "--max-documents",
        type=int,
        default=5,
        help="clusters wider than this are boilerplate, not facts",
    )
    agg.add_argument("--model", help=f"writer model (default {GENERATE_MODEL})")
    agg.add_argument(
        "--data-dir", help="base directory for the run (default data/); a session subdirectory should exist under it"
    )
    agg.add_argument(
        "--dry-run", action="store_true", help="print the clusters, call nothing"
    )
    
    # pool command and its arguments
    pool = sub.add_parser(
        "pool",
        help="retrieves candidate documents for a session's questions",
    )
    pool.add_argument(
        "--session-id",
        required=True,
        help="session id of a previouls `generate` run; data is usually found at data/generate-<session>",
    )
    pool.add_argument(
        "--ids",
        nargs="*",
        help="only these entry ids from the session files (default: every question in them)",
    )
    pool.add_argument(
        "--k",
        type=int,
        default=20,
        help="maximum chunks retrieved per question; they are deduplicated to documents, so the pool holds at most k per question and usually fewer. Capped by the retriever's top_k (30), above which nothing more comes back (default 20)",
    )
    pool.add_argument(
        "--data-dir",
        help="base directory holding the session directories (default data/); the --session-id subdirectory must exist under it",
    )

    # curve command and its arguments
    curve = sub.add_parser(
        "curve",
        help="score tier 1 across a range of one parameter and plot every metric",
    )
    _common(curve)
    curve.add_argument(
        "--param",
        required=True,
        help=f"the parameter to vary, as section.field — a query one "
        f"({', '.join(sorted(SWEEPABLE))}) or, with --reingest, an ingest one "
        f"({', '.join(INGEST_SWEEPABLE)})",
    )
    curve.add_argument(
        "--values",
        required=True,
        help="an inclusive interval `1:21` or `1:21:2`, or a list `1,5,20` / `true,false`",
    )
    curve.add_argument(
        "--mlflow-model",
        metavar="NAME@ALIAS|NAME/VERSION",
        help="start from a registered baseline, as MLflow addresses one: name@alias or name/version; --set-params wins over it, and it wins over parameters.yaml",
    )
    curve.add_argument(
        "--set-params",
        action="append",
        default=[],
        dest="overrides",
        metavar="NAME=VALUE",
        help="hold another parameter at a value for this sweep, e.g. --set-params agent.top_k=7; repeatable",
    )
    curve.add_argument(
        "--reingest",
        action="store_true",
        help="build a collection for each value first (stages 2-6 over the frozen "
        "snapshot) — required for an ingest parameter",
    )
    curve.add_argument(
        "--dry-run", action="store_true", help="write the candidates, score nothing"
    )
    curve.add_argument(
        "--overwrite-frozen",
        action="store_true",
        help="with --reingest: retake the manifest snapshot, dropping the one there",
    )
    curve.add_argument(
        "--ingest",
        action="append",
        default=[],
        dest="ingest",
        metavar="NAME=VALUE",
        help="with --reingest: a sweep file's `ingest:` setting, e.g. "
        "--ingest parse_workers=4 --ingest embed_batch_size=8; repeatable",
    )
    curve.add_argument(
        "--mlflow-run",
        help="name of the MLflow parent run; children are <name>-<param>=<value>",
    )
    curve.add_argument(
        "--no-mlflow", action="store_true", help="print only, log nothing"
    )

    search = sub.add_parser(
        "search",
        help="tune several query knobs jointly, Optuna choosing where to look",
    )
    _common(search)
    search.add_argument("--file", required=True, help="the search space")
    search.add_argument(
        "--trials", type=int, default=30, help="how many trials to run (default 30)"
    )
    search.add_argument(
        "--study",
        help="study name; resumes an existing one under runs/optuna "
        "(default: the search file's name)",
    )
    search.add_argument(
        "--seed", type=int, help="sampler seed, for a reproducible study"
    )
    search.add_argument(
        "--mlflow-run",
        help="name of the parent MLflow run (default: the search file's name); "
        "children are <name>-trial-0007",
    )
    search.add_argument(
        "--no-mlflow", action="store_true", help="print only, log nothing"
    )

    sweep = sub.add_parser("sweep", help="the tuning loop over a sweep file")
    _common(sweep)
    sweep.add_argument("--file", required=True, help="the sweep definition")
    sweep.add_argument(
        "--reingest",
        action="store_true",
        help="allow candidates that need a new collection (hours of TEI)",
    )
    sweep.add_argument(
        "--dry-run", action="store_true", help="write the candidates, score nothing"
    )
    sweep.add_argument(
        "--overwrite-frozen",
        action="store_true",
        help="retake the manifest snapshot every candidate is built from, "
        "dropping the one that is there",
    )
    sweep.add_argument(
        "--mlflow-run",
        help="name of the parent MLflow run (default: the sweep file's name); "
        "children are <name>-<knob>=<value>",
    )
    sweep.add_argument(
        "--no-mlflow", action="store_true", help="print only, log nothing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    
    if args.command == "pool":
        from rag_evaluation.pool import run_pool

        return run_pool(
            args.session_id,
            ids=args.ids,
            k=args.k,
            data_dir=args.data_dir,
        )

    try:
        entries = load_golden(args.golden, args.ids)
    except GoldenSetError as exc:
        print(f"golden set: {exc}", file=sys.stderr)
        return 2

    if args.command == "generate":
        from rag_evaluation.generate import run_generate

        return run_generate(
            entries,
            n=args.n,
            seed=args.seed,
            dsn=args.dsn,
            model=args.model,
            max_chars=args.max_chars,
            workers=args.workers,
            reference=args.reference,
            data_dir=args.data_dir,
            dry_run=args.dry_run,
        )

    if args.command == "aggregate":
        from rag_evaluation.aggregate import run_aggregate

        return run_aggregate(
            args.session_id,
            existing_ids=[e.id for e in entries],
            threshold=args.threshold,
            max_documents=args.max_documents,
            model=args.model,
            data_dir=args.data_dir,
            dry_run=args.dry_run,
        )

    if args.command == "test":
        from dataclasses import replace

        from rag_config.baseline import BaselineError
        from rag_config.promote import PromoteError

        from rag_evaluation.holdout import (
            HoldoutError,
            collection_for,
            params_from_file,
            params_from_mlflow,
        )
        from rag_evaluation.mlflow_log import log_tier1
        from rag_evaluation.retrieval_eval import format_report, run_tier1
        from rag_evaluation.sweep import SweepError

        try:
            if args.file:
                params = params_from_file(args.file)
                name = Path(args.file).stem
            else:
                params, summary = params_from_mlflow(args.mlflow_model)
                print(summary, file=sys.stderr)
                name = args.mlflow_model.replace("@", "-").replace("/", "-")
            settings = get_settings()
            settings = replace(settings, qdrant_collection=collection_for(params, settings))
        except (HoldoutError, SweepError, BaselineError, PromoteError) as exc:
            print(f"test: {exc}", file=sys.stderr)
            return 2
        print(f"scoring {settings.qdrant_collection} on {args.golden}", file=sys.stderr)
        report = run_tier1(entries, settings=settings, params=params)
        print(format_report(report))
        if not args.no_mlflow:
            log_tier1(
                report,
                settings,
                params,
                golden_path=args.golden,
                run_name=args.mlflow_run or f"test-{name}",
                context="test",
            )
        return 0

    if args.command == "retrieval":
        from rag_evaluation.mlflow_log import log_tier1
        from rag_evaluation.retrieval_eval import format_report, run_tier1

        settings = get_settings()
        params = get_params()
        if args.rerank:
            params = set_param(params, "agent.rerank_enabled", True)
        report = run_tier1(entries, settings=settings, params=params)
        print(format_report(report))
        if not args.no_mlflow:
            log_tier1(
                report,
                settings,
                params,
                golden_path=args.golden,
                run_name=args.mlflow_run,
            )
        return 0

    if args.command == "curve":
        from rag_config.params import ingest_fingerprint

        from rag_evaluation.curve import (
            CurveError,
            cast_params,
            format_table,
            parse_overrides,
            parse_values,
            run_curve,
        )
        from rag_evaluation.mlflow_log import BaselineError, load_baseline, log_curve
        from rag_evaluation.sweep import SweepError

        try:
            overrides = parse_overrides(args.overrides)
            if args.param in overrides:
                raise CurveError(
                    f"--set-params {args.param} conflicts with --param {args.param}"
                )
            # Two objects, kept apart: settings is wiring, params is what is
            # being tuned. Only params moves, and its precedence lowest first is
            # parameters.yaml, then RAG_* (both inside get_params), then a
            # registered baseline, then --set-params. The same object is
            # measured and logged, so the two cannot disagree.
            settings = get_settings()
            params = get_params()
            if args.mlflow_model:
                raw, summary, fingerprint = load_baseline(args.mlflow_model)
                for name, value in cast_params(raw).items():
                    params = set_param(params, name, value)
                print(summary, file=sys.stderr)
                current = ingest_fingerprint()
                if fingerprint and fingerprint != current:
                    print(
                        f"  warning: tuned against index {fingerprint}, "
                        f"this one is {current}",
                        file=sys.stderr,
                    )
            for name, value in overrides.items():
                params = set_param(params, name, value)
            values = parse_values(args.values, args.param)
            results = run_curve(
                args.param,
                values,
                entries,
                reingest=args.reingest,
                dry_run=args.dry_run,
                overwrite_frozen=args.overwrite_frozen,
                baseline=params,
                ingest_spec=_ingest_settings(args.ingest),
            )
        except (CurveError, BaselineError, SweepError) as exc:
            print(f"curve: {exc}", file=sys.stderr)
            return 2
        # A curve plots values, not candidates: the scored ones, in order.
        scored = [s for s in results if s.report is not None]
        rows = [(s.candidate.value, s.report) for s in scored]
        collections = {s.candidate.value: s.settings.qdrant_collection for s in scored}
        print(format_table(rows, args.param))
        if not args.no_mlflow:
            log_curve(
                rows,
                args.param,
                settings,
                params,
                golden_path=args.golden,
                run_name=args.mlflow_run,
                # only when each value had its own; otherwise all are the
                # standing collection, already in the parent's params
                collections=collections if args.reingest else None,
            )
        return 0

    if args.command == "agent":
        from rag_evaluation.agent_eval import run_judges

        return run_judges(
            entries, judge_model=args.judge_model, rerank=args.rerank
        )

    if args.command == "search":
        from rag_evaluation.mlflow_log import log_search
        from rag_evaluation.search import (
            SearchError,
            format_search,
            load_search,
            run_search,
        )

        name = args.study or Path(args.file).stem
        try:
            spec = load_search(args.file)
            results, study, objective = run_search(
                entries,
                spec,
                n_trials=args.trials,
                study_name=name,
                seed=args.seed,
            )
        except SearchError as exc:
            print(f"search: {exc}", file=sys.stderr)
            return 2
        print(format_search(results, objective))
        if not args.no_mlflow:
            log_search(
                results,
                study,
                objective,
                spec,
                get_settings(),
                golden_path=args.golden,
                run_name=args.mlflow_run or name,
            )
        return 0

    if args.command == "sweep":
        from rag_evaluation.mlflow_log import log_sweep
        from rag_evaluation.sweep import (
            SweepError,
            format_summary,
            load_sweep,
            run_sweep,
        )

        try:
            spec = load_sweep(args.file)
            results = run_sweep(
                spec,
                entries,
                reingest=args.reingest,
                dry_run=args.dry_run,
                overwrite_frozen=args.overwrite_frozen,
            )
        except SweepError as exc:
            print(f"sweep: {exc}", file=sys.stderr)
            return 2
        print(format_summary(results))
        if not args.no_mlflow:
            log_sweep(
                results,
                spec,
                golden_path=args.golden,
                # The definition file names the sweep: `sweeps/sweep1.yaml` is
                # the run `sweep1`, and its candidates `sweep1-agent.top_k=5`.
                run_name=args.mlflow_run or Path(args.file).stem,
            )
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())
