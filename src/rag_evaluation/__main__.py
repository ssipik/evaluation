"""Argparse only — the logic lives in the library modules.

    python -m rag_evaluation retrieval --k 10        tier 1, no LLM
    python -m rag_evaluation pool                    labeling helper
    python -m rag_evaluation agent --ids isin-lookup tiers 2 and 3 (needs a key)
    python -m rag_evaluation generate --n 50         seed entries from the corpus
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from dotenv import load_dotenv

from rag_agent.settings import get_settings

from rag_evaluation.generate import DEFAULT_MODEL as GENERATE_MODEL
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
    tier1.add_argument(
        "--rerank", action="store_true", help="score with the cross-encoder on"
    )
    tier1.add_argument(
        "--no-mlflow", action="store_true", help="print only, log nothing"
    )

    pool = sub.add_parser("pool", help="pooled candidates to label by hand")
    _common(pool)
    pool.add_argument("--k", type=int, default=20, help="depth per contributing run")
    pool.add_argument(
        "--with-rerank",
        action="store_true",
        help="add a reranked run to the pool — more diverse, ~222 s per query here",
    )

    agent = sub.add_parser("agent", help="tiers 2 and 3: judge scorers")
    _common(agent)
    agent.add_argument("--judge-model", help="scoring model; must not be the agent's")
    agent.add_argument("--rerank", action="store_true")

    gen = sub.add_parser(
        "generate", help="seed golden entries from sampled documents (needs a key)"
    )
    _common(gen)
    gen.add_argument("--n", type=int, default=50, help="documents to sample")
    gen.add_argument(
        "--seed", type=int, help="draw seed; same seed + same corpus = same sample"
    )
    gen.add_argument("--model", help=f"writer model (default {GENERATE_MODEL})")
    gen.add_argument("--dsn", help="Postgres DSN (default RAG_PG_DSN)")
    gen.add_argument(
        "--max-chars", type=int, default=1_000_000, help="document text sent per call"
    )
    gen.add_argument("--workers", type=int, default=4, help="concurrent calls")
    gen.add_argument(
        "--out", help="base directory for the run (default data/); a session subdirectory is made under it"
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

    try:
        entries = load_golden(args.golden, args.ids)
    except GoldenSetError as exc:
        print(f"golden set: {exc}", file=sys.stderr)
        return 2

    if args.command == "retrieval":
        from rag_evaluation.mlflow_log import log_tier1
        from rag_evaluation.retrieval_eval import format_report, run_tier1

        settings = get_settings()
        if args.rerank:
            settings = replace(settings, rerank_enabled=True)
        report = run_tier1(entries, settings=settings, k=args.k)
        print(format_report(report))
        if not args.no_mlflow:
            log_tier1(report, settings, golden_path=args.golden)
        return 0

    if args.command == "pool":
        from rag_evaluation.pool import (
            DEFAULT_RUNS,
            RERANKED_RUN,
            format_pool,
            pool_entries,
        )

        configured = DEFAULT_RUNS + ((RERANKED_RUN,) if args.with_rerank else ())
        runs = tuple(replace(r, k=args.k) for r in configured)
        print(format_pool(entries, pool_entries(entries, runs)))
        return 0

    if args.command == "agent":
        from rag_evaluation.agent_eval import run_judges

        return run_judges(
            entries, judge_model=args.judge_model, rerank=args.rerank
        )

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
            out=args.out,
            dry_run=args.dry_run,
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
