"""Measure how TEI embedding throughput changes with the number of concurrent clients.

Not a pytest test — it needs the network and puts load on the shared tei-embed
service, which the chat also uses. Named bench_* so pytest does not collect it.

    .venv/bin/python tests/bench_tei_embed_concurrency.py
    .venv/bin/python tests/bench_tei_embed_concurrency.py --clients 1,2 --seconds 10

Each client sends 32-chunk requests one after another, the way one embed batch job
calls TEI (rag_embed.TeiClient), so N clients stand for N embed batches running at
once. The knee — the last client count before throughput stops rising — is how many
embed batches one TEI replica can serve at the same time.

- Run it with tei-embed at count = 1. With more replicas behind traefik the result
  is for the whole service, not for one replica.
- Timeouts and HTTP errors are counted, not raised: overload is what this measures.
- English only and every chunk the same length, so absolute numbers are a best
  case. The ratio between client counts is what this script is for.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass

import httpx
from dotenv import find_dotenv, load_dotenv

from bench_tei_embed import BATCH, ENGLISH, count_tokens, make_chunk

DEFAULT_URL = "https://tei-embed.service.get"
# The next client count must add at least this much throughput, or the current one is the knee.
KNEE_GAIN = 0.10
# Distinct request payloads the clients cycle through.
POOL_BATCHES = 64
INFO_KEYS = ("model_id", "max_concurrent_requests", "max_batch_tokens",
             "max_client_batch_size", "tokenization_workers")


@dataclass(frozen=True)
class Request:
    seconds: float
    finished: float
    chunks: int
    error: str | None


def embed_request(client: httpx.Client, url: str, texts: list[str], timeout: float) -> Request:
    """One /embed call; an error is returned, not raised."""
    start = time.perf_counter()
    error = None
    try:
        resp = client.post(f"{url}/embed", json={"inputs": texts, "truncate": True}, timeout=timeout)
        resp.raise_for_status()
        got = len(resp.json())
        if got != len(texts):
            error = f"got {got}/{len(texts)} vectors"
    except httpx.TimeoutException:
        error = "timeout"
    except httpx.HTTPStatusError as exc:
        error = f"http {exc.response.status_code}"
    except httpx.HTTPError as exc:
        error = type(exc).__name__
    end = time.perf_counter()
    return Request(end - start, end, 0 if error else len(texts), error)


def client_loop(url: str, batches: list[list[str]], offset: int, seconds: float,
                timeout: float, barrier: threading.Barrier, out: list[Request]) -> None:
    """Send requests one after another until `seconds` have passed since the common start."""
    with httpx.Client() as client:
        barrier.wait()
        deadline = time.perf_counter() + seconds
        i = offset
        while time.perf_counter() < deadline:
            out.append(embed_request(client, url, batches[i % len(batches)], timeout))
            i += 1


def run_level(url: str, clients: int, batches: list[list[str]], seconds: float,
              timeout: float, tokens_per_chunk: int) -> dict:
    barrier = threading.Barrier(clients + 1)
    per_client: list[list[Request]] = [[] for _ in range(clients)]
    threads = [
        threading.Thread(
            target=client_loop,
            # different offsets, so the clients don't send the same payloads
            args=(url, batches, i * len(batches) // clients, seconds, timeout, barrier, per_client[i]),
        )
        for i in range(clients)
    ]
    for t in threads:
        t.start()
    barrier.wait()
    start = time.perf_counter()
    for t in threads:
        t.join()

    requests = [r for rs in per_client for r in rs]
    ok = [r for r in requests if r.error is None]
    # Until the last request finished, not until the deadline: requests still running at the deadline count too.
    elapsed = max((r.finished for r in requests), default=start) - start
    chunks = sum(r.chunks for r in requests)
    times = sorted(r.seconds for r in ok)
    return {
        "clients": clients,
        "requests": len(requests),
        "chunks_per_s": chunks / elapsed if elapsed > 0 else 0.0,
        "tokens_per_s": chunks * tokens_per_chunk / elapsed if elapsed > 0 else 0.0,
        "mean": statistics.mean(times) if times else math.nan,
        # ceil, not int: int() picks a percentile below the mean for small samples
        "p95": times[max(0, math.ceil(0.95 * len(times)) - 1)] if times else math.nan,
        "errors": Counter(r.error for r in requests if r.error),
    }


def find_knee(levels: list[dict]) -> int | None:
    """The last client count before adding clients gains less than KNEE_GAIN throughput."""
    for prev, nxt in zip(levels, levels[1:]):
        if prev["chunks_per_s"] == 0 or nxt["chunks_per_s"] / prev["chunks_per_s"] - 1 < KNEE_GAIN:
            return prev["clients"]
    return None


def parse_clients(value: str) -> list[int]:
    try:
        counts = sorted({int(v) for v in value.split(",")})
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}")
    if not counts or counts[0] < 1:
        raise argparse.ArgumentTypeError("client counts must be >= 1")
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("BENCH_TEI_URL", DEFAULT_URL))
    p.add_argument("--clients", type=parse_clients, default=[1, 2, 4, 8, 16],
                   help="client counts to measure, comma-separated (default 1,2,4,8,16)")
    p.add_argument("--seconds", type=float, default=120.0, help="measuring time per client count")
    p.add_argument("--chunk-chars", type=int, default=1000,
                   help="characters per chunk (default 1000, near the corpus mean of 957)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="per-request timeout; default matches rag-config's tei_timeout")
    p.add_argument("--pause", type=float, default=10.0,
                   help="seconds between client counts, so TEI's queue is empty again")
    args = p.parse_args()

    try:
        httpx.get(f"{args.url}/health", timeout=10).raise_for_status()
        info = httpx.get(f"{args.url}/info", timeout=10).json()
    except Exception as exc:
        print(f"{args.url} not reachable: {exc}", file=sys.stderr)
        if "CERTIFICATE_VERIFY" in str(exc):
            print("  hint: SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt", file=sys.stderr)
        return 2
    print(f"TEI {args.url}")
    for key in INFO_KEYS:
        print(f"  {key}: {info.get(key)}")

    texts = [make_chunk(args.chunk_chars, i, [ENGLISH]) for i in range(POOL_BATCHES * BATCH)]
    batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
    tokens_per_chunk = count_tokens(args.url, texts[0])
    runtime = len(args.clients) * args.seconds + (len(args.clients) - 1) * args.pause
    print(f"payload: {BATCH} chunks x {args.chunk_chars} chars (en, ~{tokens_per_chunk} tokens each) "
          f"per request; ~{runtime / 60:.1f} min total\n")

    # Warm-up, not counted: the first call after idle pays one-off tokenizer and ONNX costs.
    with httpx.Client() as client:
        warm = embed_request(client, args.url, batches[0][:4], args.timeout)
    if warm.error:
        print(f"warm-up request failed: {warm.error}", file=sys.stderr)
        return 2

    levels = []
    for i, clients in enumerate(args.clients):
        if i:
            time.sleep(args.pause)
        print(f"  {clients} client(s): measuring {args.seconds:.0f}s ...", flush=True)
        levels.append(run_level(args.url, clients, batches, args.seconds, args.timeout, tokens_per_chunk))

    base = levels[0]["chunks_per_s"]
    print(f"\n{'clients':>7}  {'requests':>8}  {'chunks/s':>8}  {'tok/s':>7}  "
          f"{'speedup':>7}  {'mean ms':>8}  {'p95 ms':>8}  {'errors':>6}")
    for lv in levels:
        speedup = lv["chunks_per_s"] / base if base else math.nan
        print(f"{lv['clients']:>7}  {lv['requests']:>8}  {lv['chunks_per_s']:>8.2f}  {lv['tokens_per_s']:>7.0f}  "
              f"{speedup:>6.2f}x  {lv['mean'] * 1000:>8.0f}  {lv['p95'] * 1000:>8.0f}  "
              f"{sum(lv['errors'].values()):>6}")
    print(f"  (speedup is relative to {levels[0]['clients']} client(s))")

    for lv in levels:
        if lv["errors"]:
            detail = ", ".join(f"{err}: {n}" for err, n in lv["errors"].most_common())
            print(f"  warning: {lv['clients']} client(s) had errors — {detail}")
        if lv["p95"] > args.timeout / 2:
            print(f"  warning: {lv['clients']} client(s) p95 {lv['p95']:.0f}s is over half the "
                  f"{args.timeout:.0f}s timeout")

    knee = find_knee(levels)
    if knee is None:
        print(f"\n  throughput still rising at {levels[-1]['clients']} client(s) — "
              "measure higher --clients to find the knee")
    else:
        print(f"\n  knee: {knee} client(s) — more concurrent embed batches than this gain "
              f"< {KNEE_GAIN:.0%} throughput on this TEI (per replica if tei-embed runs at count = 1)")
    return 0


if __name__ == "__main__":
    # Same as bench_tei_embed.py: the environment file carries SSL_CERT_FILE for the cluster's
    # traefik certificate. find_dotenv walks upwards, so this works from tests/ and from the package root.
    load_dotenv(find_dotenv(usecwd=True))
    raise SystemExit(main())
