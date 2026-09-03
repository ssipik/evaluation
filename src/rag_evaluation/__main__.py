"""Argparse only — the logic lives in the library modules.

    python -m rag_evaluation retrieval --k 10        tier 1, no LLM
    python -m rag_evaluation pool --session-id …     labeling helper
    python -m rag_evaluation agent --ids isin-lookup tiers 2 and 3 (needs a key)
    python -m rag_evaluation generate --n 50         seed entries from the corpus
    python -m rag_evaluation aggregate --session …  multi-document entries from a run
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from dotenv import load_dotenv

from rag_agent.settings import get_settings

from rag_evaluation.generate import DEFAULT_MODEL as GENERATE_MODEL
from rag_evaluation.curve import SWEEPABLE
from rag_evaluation.golden import DEFAULT_PATH, GoldenSetError, load_golden

log = logging.getLogger(__name__)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--golden", default=str(DEFAULT_PATH), help="the golden set")
    parser.add_argument("--ids", nargs="*", help="only these entry ids")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_evaluation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tier1 = sub.add_parser("retrieval", help="tier 1: score retrieval, no LLM")
    _common(tier1)
    tier1.add_argument("--k", type=int, help="results per query (default: top_k)")
    tier1.add_argument("--run-name", help="name of the MLflow run")
    tier1.add_argument(
        "--rerank", action="store_true", help="score with the cross-encoder on"
    )
    tier1.add_argument(
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
        help="score tier 1 across a range of one query parameter and plot every metric",
    )
    _common(curve)
    curve.add_argument(
        "--param",
        required=True,
        help=f"the query parameter to vary — one of {', '.join(sorted(SWEEPABLE))}",
    )
    curve.add_argument(
        "--values",
        required=True,
        help="an inclusive interval `1:21` or `1:21:2`, or a list `1,5,20` / `true,false`",
    )
    curve.add_argument(
        "--run-name",
        help="name of the MLflow parent run; children are <name>-<param>=<value>",
    )
    curve.add_argument(
        "--no-mlflow", action="store_true", help="print only, log nothing"
    )

    sweep = sub.add_parser("sweep", help="the tuning loop over a sweep file")
    _common(sweep)
    sweep.add_argument("--file", required=True, help="the sweep definition")
    sweep.add_argument("--k", type=int, help="results per query")
    sweep.add_argument(
        "--reingest",
        action="store_true",
        help="allow candidates that need a new collection (hours of TEI)",
    )
    sweep.add_argument(
        "--dry-run", action="store_true", help="write the candidates, score nothing"
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

    if args.command == "retrieval":
        from rag_evaluation.mlflow_log import log_tier1
        from rag_evaluation.retrieval_eval import format_report, run_tier1

        settings = get_settings()
        if args.rerank:
            settings = replace(settings, rerank_enabled=True)
        report = run_tier1(entries, settings=settings, k=args.k)
        print(format_report(report))
        if not args.no_mlflow:
            log_tier1(
                report, settings, golden_path=args.golden, run_name=args.run_name
            )
        return 0

    if args.command == "curve":
        from rag_evaluation.curve import (
            CurveError,
            format_table,
            parse_values,
            run_curve,
        )
        from rag_evaluation.mlflow_log import log_curve

        settings = get_settings()
        try:
            values = parse_values(args.values, args.param)
            rows = run_curve(entries, args.param, values, settings=settings)
        except CurveError as exc:
            print(f"curve: {exc}", file=sys.stderr)
            return 2
        print(format_table(rows, args.param))
        if not args.no_mlflow:
            log_curve(
                rows,
                args.param,
                settings,
                golden_path=args.golden,
                run_name=args.run_name,
            )
        return 0

    if args.command == "agent":
        from rag_evaluation.agent_eval import run_judges

        return run_judges(
            entries, judge_model=args.judge_model, rerank=args.rerank
        )

    if args.command == "sweep":
        from rag_evaluation.sweep import SweepError, load_sweep, run_sweep

        try:
            results = run_sweep(
                load_sweep(args.file),
                entries,
                k=args.k,
                reingest=args.reingest,
                dry_run=args.dry_run,
            )
        except SweepError as exc:
            print(f"sweep: {exc}", file=sys.stderr)
            return 2
        print("\n=== sweep summary ===")
        for candidate, report in results:
            if report is None or not report.scored:
                print(f"  {candidate.label:28} (not scored)")
                continue
            means = report.means
            print(
                f"  {candidate.label:28} recall {means['recall']:.4f}  "
                f"ndcg {means['ndcg']:.4f}  mrr {means['mrr']:.4f}"
            )
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())
