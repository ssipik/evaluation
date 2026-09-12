"""Compare rerank latency between two TEI reranker instances.

Not a pytest test — it needs the network and takes a few minutes, where the rest
of tests/ is offline. Named bench_* so pytest does not collect it.

    .venv/bin/python tests/bench_tei_rerank.py
    .venv/bin/python tests/bench_tei_rerank.py --passages 10 --repeat 2

The sibling of bench_tei_embed.py, measuring the other half of the query path.
Two things differ, both because reranking is a different kind of work:

- **Latency, not throughput.** Embedding runs over thousands of chunks during an
  ingest, so chunks/s is the number. Reranking runs once per user question, in
  the request path, so seconds-per-query is.
- **Capped at 32 passages.** TEI refuses a larger client batch with a 422, and
  rag_agent.rerank sends the whole candidate set in one POST, so this is the
  agent's ceiling too. The default is 30 — the shipped `top_k`, and what the
  agent actually reranks.

This model has no ONNX export, so TEI falls back to Candle (see
../tei_rerank/README.md), which on a laptop costs seconds per passage. A default
run against the local reranker therefore takes many minutes; `--skip-local`
measures the cluster alone.

**Passage length dominates, superlinearly** — a cross-encoder's attention is
quadratic in sequence length, so `--chunk-chars` moves the result far more than
`--passages` does. Measured on the cluster, 8 passages:

    chars  tokens  per passage  vs 1000ch
     1000     239      0.59 s       1.0x
     2000     476      1.49 s       2.5x
     4266    1013      4.34 s       7.4x

The default is 4000, near the corpus p95 of 4266, because that is what actually
governs whether a rerank finishes inside a timeout. An earlier 1000-char default
reported 0.62 s per passage and made reranking look ~7x cheaper than a real
candidate set turned out to be.

The corpora are imported from bench_tei_embed so both benchmarks score the same
text. Running this file puts tests/ on sys.path, so the plain import works from
either the package root or from inside tests/.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import httpx
from dotenv import find_dotenv, load_dotenv

from bench_tei_embed import CORPORA, count_tokens, make_chunk

DEFAULT_ENDPOINTS = {
    "local": "http://localhost:8081",
    "cluster": "https://tei-rerank.service.get",
}

# TEI refuses a client batch above --max-client-batch-size with a 422, and this
# is not a batching problem the caller can work around: rag_agent.rerank sends
# the whole candidate set in one POST, so 32 is the ceiling for the agent too.
# top_k = 30 sits just under it.
MAX_PASSAGES = 32
# The shipped agent.top_k — what the agent actually hands the reranker.
TOP_K = 30

# One realistic question per language. Length barely matters: a cross-encoder
# scores (query, passage) pairs and the passage dominates the token count, so the
# passage size is the knob that moves the number.
QUERIES = {
    "en": "What liquidity requirements apply to institutions under supervisory review?",
    "de": "Welche Liquiditätsanforderungen gelten für Institute nach dem aufsichtlichen "
          "Überprüfungsverfahren?",
}


def rerank_once(client: httpx.Client, url: str, query: str, texts: list[str]) -> float:
    """One /rerank call, returning its wall time.

    Model errors stay fatal — a benchmark that skips failures reports the broken
    instance as the fast one. Gateway errors are retried once, because they come
    from the ingress rather than the service: a 502 with no task restart behind
    it was seen during a long run, and losing five minutes of measurement to a
    Traefik blip is not a result worth keeping. A retry that is used is printed,
    so it cannot silently hide a service that is genuinely flapping.
    """
    for attempt in (1, 2):
        start = time.perf_counter()
        try:
            resp = client.post(
                f"{url}/rerank",
                json={"query": query, "texts": texts, "raw_scores": False},
                timeout=1800,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (502, 503, 504) and attempt == 1:
                print(f"\n  {url}: {exc.response.status_code} from the ingress, retrying once")
                time.sleep(5)
                continue
            raise
        got = len(resp.json())
        if got != len(texts):
            raise RuntimeError(f"{url}: scored {got} passages, expected {len(texts)}")
        return time.perf_counter() - start
    raise AssertionError("unreachable")


def run(name: str, url: str, query: str, passages: list[str], rounds: int) -> dict:
    with httpx.Client() as client:
        # Warm-up on a single passage, excluded: the first call after idle pays
        # model-load and allocator costs no later call repeats. One passage
        # rather than the full set because on the local side that is minutes.
        rerank_once(client, url, query, passages[:1])
        times = []
        for i in range(1, rounds + 1):
            times.append(rerank_once(client, url, query, passages))
            print(f"\r  {name}: round {i}/{rounds}", end="", flush=True)
    print("\r" + " " * 44, end="\r")
    # Mean rather than best-of-N: this is a latency measurement, and what a user
    # waits is the typical call, not the luckiest one. min/max carry the spread,
    # which on a shared cluster is the part worth seeing.
    return {
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "per_passage": statistics.mean(times) / len(passages),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", default=os.environ.get("BENCH_RERANK_LOCAL", DEFAULT_ENDPOINTS["local"]))
    p.add_argument("--cluster", default=os.environ.get("BENCH_RERANK_CLUSTER", DEFAULT_ENDPOINTS["cluster"]))
    p.add_argument("--passages", type=int, default=TOP_K,
                   help=f"candidates per query (default {TOP_K} = top_k; max {MAX_PASSAGES})")
    p.add_argument("--chunk-chars", type=int, default=4000,
                   help="characters per passage (default 4000, near the corpus p95 of 4266)")
    p.add_argument("--language", choices=sorted(CORPORA), default="mixed",
                   help="corpus language (default mixed: the golden set is ~57%% en, 43%% de)")
    p.add_argument("--repeat", type=int, default=1, help="measured rounds per endpoint")
    p.add_argument("--skip-local", action="store_true",
                   help="cluster only — the local reranker costs tens of seconds per passage")
    args = p.parse_args()

    if args.passages > MAX_PASSAGES:
        print(f"--passages {args.passages} exceeds TEI's client batch limit of "
              f"{MAX_PASSAGES}; it would return 422 rather than a slow answer.",
              file=sys.stderr)
        return 2

    endpoints = [("cluster", args.cluster)]
    if not args.skip_local:
        endpoints.insert(0, ("local", args.local))

    for name, url in endpoints:
        try:
            httpx.get(f"{url}/health", timeout=10).raise_for_status()
        except Exception as exc:
            print(f"{name} ({url}) not reachable: {exc}", file=sys.stderr)
            if "CERTIFICATE_VERIFY" in str(exc):
                print("  hint: SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt", file=sys.stderr)
            return 2

    corpus = CORPORA[args.language]
    passages = [make_chunk(args.chunk_chars, i, corpus) for i in range(args.passages)]
    # The query language follows the corpus; a mixed run asks in German, which is
    # the harder direction for retrieval and the one worth timing.
    query = QUERIES["de" if args.language in ("de", "mixed") else "en"]

    tokens = count_tokens(args.cluster, query) + sum(
        count_tokens(args.cluster, t) for t in passages[: len(corpus)]
    ) // len(corpus) * len(passages)
    print(f"query: {query[:60]}...")
    print(f"payload: {args.passages} passages x {args.chunk_chars} chars ({args.language}), "
          f"~{tokens} tokens per call")
    if not args.skip_local:
        print("  note: the local reranker has no ONNX export and falls back to Candle —\n"
              "        expect tens of seconds per passage there.")
    print()

    results = {name: run(name, url, query, passages, args.repeat) for name, url in endpoints}

    spread = f"{'min-max':>16}" if args.repeat > 1 else ""
    print(f"{'endpoint':>10}  {'per query':>10}  {'per passage':>12}  {'tok/s':>8}  {spread}")
    for name, _ in endpoints:
        r = results[name]
        rng = f"{r['min']:.2f}-{r['max']:.2f}s".rjust(16) if args.repeat > 1 else ""
        print(f"{name:>10}  {r['mean']:>9.2f}s  {r['per_passage']:>11.2f}s  "
              f"{tokens / r['mean']:>8.0f}  {rng}")

    if len(endpoints) == 2:
        ratio = results["local"]["mean"] / results["cluster"]["mean"]
        print(f"\n  cluster is {ratio:.1f}x faster per query")
    # What the agent would actually pay, which is the number that decides whether
    # reranking is affordable in the request path rather than only in an eval.
    if args.passages != TOP_K:
        at_top_k = results["cluster"]["per_passage"] * TOP_K
        print(f"  at top_k={TOP_K} the cluster would cost ~{at_top_k:.0f}s per query")
    return 0


if __name__ == "__main__":
    # Same as rag_evaluation.__main__: the module's environment file carries
    # SSL_CERT_FILE, which the cluster's traefik certificate needs because it is
    # not in certifi's bundle.
    load_dotenv(find_dotenv(usecwd=True))
    raise SystemExit(main())
