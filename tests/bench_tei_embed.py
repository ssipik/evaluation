"""Compare embedding throughput between two TEI instances.

Not a pytest test — it needs the network and takes a minute, where the rest of
tests/ is offline. Named bench_* so pytest does not collect it.

    .venv/bin/python tests/bench_tei.py
    .venv/bin/python tests/bench_tei.py --seconds 30 --repeat 2

Sends the *same* payload to each endpoint, in serial sub-batches of
``embed_batch_size`` — which is how rag_embed.TeiClient actually calls TEI (one
request in flight at a time), so the number reflects a real ingest rather than a
parallel best case.

The corpus is hardcoded English prose, repeated to the requested chunk size: it
needs no services and is identical on every run, so both endpoints get
byte-identical work. English rather than lorem ipsum because tokenization density
differs — measured against bge-m3 per 1000 characters: lorem 165 tokens, English
200, German 221. A lorem benchmark understates the work a real chunk costs by
about a fifth.

Every chunk is the same length, which is a uniform-length best case; real chunks
vary and pad, so absolute throughput here is optimistic. The *ratio* between the
two endpoints is the number this script is for.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import httpx
from dotenv import find_dotenv, load_dotenv

DEFAULT_ENDPOINTS = {
    "local": "http://localhost:8080",
    "cluster": "https://tei-embed.service.get",
}
# TEI rejects a client batch larger than its --max-client-batch-size (32 here),
# which is also what rag_config ships as embed_batch_size.
BATCH = 32
# Floor on batches per run, so mean and p95 describe more than one sample.
MIN_BATCHES = 4

ENGLISH = """
Preamble
Whereas recognition of the inherent dignity and of the equal and inalienable rights of all members
of the human family is the foundation of freedom, justice and peace in the world,

Whereas disregard and contempt for human rights have resulted in barbarous acts which have outraged
the conscience of mankind, and the advent of a world in which human beings shall enjoy freedom of
speech and belief and freedom from fear and want has been proclaimed as the highest aspiration of
the common people,

Whereas it is essential, if man is not to be compelled to have recourse, as a last resort, to
rebellion against tyranny and oppression, that human rights should be protected by the rule of law,

Whereas it is essential to promote the development of friendly relations between nations,

Whereas the peoples of the United Nations have in the Charter reaffirmed their faith in fundamental
human rights, in the dignity and worth of the human person and in the equal rights of men and women
and have determined to promote social progress and better standards of life in larger freedom,

Whereas Member States have pledged themselves to achieve, in co-operation with the United Nations,
the promotion of universal respect for and observance of human rights and fundamental freedoms,

Whereas a common understanding of these rights and freedoms is of the greatest importance for the
full realization of this pledge,

Now, therefore,

The General Assembly,

Proclaims this Universal Declaration of Human Rights as a common standard of achievement for all
peoples and all nations, to the end that every individual and every organ of society, keeping this
Declaration constantly in mind, shall strive by teaching and education to promote respect for these
rights and freedoms and by progressive measures, national and international, to secure their
universal and effective recognition and observance, both among the peoples of Member States
themselves and among the peoples of territories under their jurisdiction.

Article 1
All human beings are born free and equal in dignity and rights. They are endowed with reason and
conscience and should act towards one another in a spirit of brotherhood.

Article 2

Everyone is entitled to all the rights and freedoms set forth in this Declaration, without
distinction of any kind, such as race, colour, sex, language, religion, political or other
opinion, national or social origin, property, birth or other status. Furthermore, no distinction
shall be made on the basis of the political, jurisdictional or international status of the country
or territory to which a person belongs, whether it be independent, trust, non-self-governing or
under any other limitation of sovereignty.

Article 3
Everyone has the right to life, liberty and security of person.

Article 4
No one shall be held in slavery or servitude; slavery and the slave trade shall be prohibited in
all their forms.

Article 5

No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment.

Article 6
Everyone has the right to recognition everywhere as a person before the law.

Article 7
All are equal before the law and are entitled without any discrimination to equal protection of the
law. All are entitled to equal protection against any discrimination in violation of this
Declaration and against any incitement to such discrimination.

Article 8
Everyone has the right to an effective remedy by the competent national tribunals for acts
violating the fundamental rights granted him by the constitution or by law.

Article 9
No one shall be subjected to arbitrary arrest, detention or exile.

Article 10
Everyone is entitled in full equality to a fair and public hearing by an independent and impartial
tribunal, in the determination of his rights and obligations and of any criminal charge against
him.
"""

# The same Declaration in German. Comparable content and register, so the two
# differ by language rather than by subject — which is what makes a mixed run
# meaningful. Exact wording does not matter here: only the token stream does, and
# German is ~10% denser than English (221 vs 200 tokens per 1000 characters
# against bge-m3), which is why a German share makes the number more honest for
# this corpus.
GERMAN = """
Präambel
Da die Anerkennung der angeborenen Würde und der gleichen und unveräußerlichen Rechte aller
Mitglieder der Gemeinschaft der Menschen die Grundlage von Freiheit, Gerechtigkeit und Frieden in
der Welt bildet,

da die Nichtanerkennung und Verachtung der Menschenrechte zu Akten der Barbarei geführt haben, die
das Gewissen der Menschheit mit Empörung erfüllen, und da verkündet worden ist, dass einer Welt, in
der die Menschen Rede- und Glaubensfreiheit und Freiheit von Furcht und Not genießen, das höchste
Streben des Menschen gilt,

da es notwendig ist, die Menschenrechte durch die Herrschaft des Rechtes zu schützen, damit der
Mensch nicht gezwungen wird, als letztes Mittel zum Aufstand gegen Tyrannei und Unterdrückung zu
greifen,

da es notwendig ist, die Entwicklung freundschaftlicher Beziehungen zwischen den Nationen zu
fördern,

da die Völker der Vereinten Nationen in der Charta ihren Glauben an die grundlegenden
Menschenrechte, an die Würde und den Wert der menschlichen Person und an die Gleichberechtigung von
Mann und Frau erneut bekräftigt und beschlossen haben, den sozialen Fortschritt und bessere
Lebensbedingungen in größerer Freiheit zu fördern,

Artikel 1
Alle Menschen sind frei und gleich an Würde und Rechten geboren. Sie sind mit Vernunft und
Gewissen begabt und sollen einander im Geiste der Brüderlichkeit begegnen.

Artikel 2
Jeder hat Anspruch auf die in dieser Erklärung verkündeten Rechte und Freiheiten ohne irgendeinen
Unterschied, etwa nach Rasse, Hautfarbe, Geschlecht, Sprache, Religion, politischer oder sonstiger
Überzeugung, nationaler oder sozialer Herkunft, Vermögen, Geburt oder sonstigem Stand.

Artikel 3
Jeder hat das Recht auf Leben, Freiheit und Sicherheit der Person.

Artikel 5
Niemand darf der Folter oder grausamer, unmenschlicher oder erniedrigender Behandlung oder Strafe
unterworfen werden.

Artikel 7
Alle Menschen sind vor dem Gesetz gleich und haben ohne Unterschied Anspruch auf gleichen Schutz
durch das Gesetz. Alle haben Anspruch auf gleichen Schutz gegen jede Diskriminierung, die gegen
diese Erklärung verstößt, und gegen jede Aufhetzung zu einer derartigen Diskriminierung.

Artikel 8
Jeder hat Anspruch auf einen wirksamen Rechtsbehelf bei den zuständigen innerstaatlichen Gerichten
gegen Handlungen, durch die seine ihm nach der Verfassung oder nach dem Gesetz zustehenden
Grundrechte verletzt werden.
"""

# The golden set runs roughly 57% English and 43% German, so "mixed" is the
# closest match to what an ingest actually embeds.
CORPORA = {"en": [ENGLISH], "de": [GERMAN], "mixed": [ENGLISH, GERMAN]}


def make_chunk(chars: int, seed: int, corpus: list[str]) -> str:
    """One chunk of `chars`, repeating the corpus text as needed.

    The index is prefixed so no two chunks are byte-identical — same length and
    same work, but nothing a server could collapse as a duplicate. With a
    multi-language corpus the seed also picks the language, so a mixed run
    alternates rather than embedding one language and then the other.
    """
    text = corpus[seed % len(corpus)]
    body = text * (chars // len(text) + 1)
    return f"[{seed}] {body[:chars]}"


def embed_once(client: httpx.Client, url: str, texts: list[str]) -> float:
    """One /embed call, returning its wall time. Errors are fatal: a benchmark
    that silently skips failures reports the broken instance as the fast one."""
    start = time.perf_counter()
    resp = client.post(
        f"{url}/embed", json={"inputs": texts, "truncate": True}, timeout=600
    )
    resp.raise_for_status()
    got = len(resp.json())
    if got != len(texts):
        raise RuntimeError(f"{url}: asked for {len(texts)} vectors, got {got}")
    return time.perf_counter() - start


def count_tokens(url: str, text: str) -> int:
    """Real token count from TEI, so throughput is measured rather than guessed
    from a chars/4 rule that is wrong by 20-35% depending on the language."""
    resp = httpx.post(f"{url}/tokenize", json={"inputs": text}, timeout=30)
    resp.raise_for_status()
    return len(resp.json()[0])


def run(name: str, url: str, batches: list[list[str]], tokens_per_chunk: int) -> dict:
    with httpx.Client() as client:
        # Warm-up, excluded: the first call after idle pays tokenizer and ONNX
        # arena costs no later call repeats.
        embed_once(client, url, batches[0][:4])
        times, start = [], time.perf_counter()
        for i, batch in enumerate(batches, 1):
            times.append(embed_once(client, url, batch))
            print(f"\r  {name}: batch {i}/{len(batches)}", end="", flush=True)
        total = time.perf_counter() - start
    print("\r" + " " * 44, end="\r")
    chunks = sum(len(b) for b in batches)
    return {
        "total": total,
        "chunks": chunks,
        "chunks_per_s": chunks / total,
        "tokens_per_s": (chunks * tokens_per_chunk) / total,
        "batch_mean": statistics.mean(times),
        # ceil, not int: with 2 samples int(0.95*2)-1 indexes 0, reporting the
        # *fastest* batch as p95 — a percentile below the mean.
        "batch_p95": sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)],
    }


def calibrate(url: str, chunk: str, seconds: float) -> int:
    """How many chunks this endpoint needs to run for roughly `seconds`.

    Calibrating on the *first* endpoint bounds the total runtime: if it turns out
    to be the slower of the two, the faster one simply finishes early. Sizing off
    the faster one would make the slow side run for a multiple of the target.
    """
    with httpx.Client() as client:
        sample = [chunk] * BATCH
        embed_once(client, url, sample[:4])
        elapsed = embed_once(client, url, sample)
    per_chunk = elapsed / BATCH
    # Round *up*, and never below MIN_BATCHES: a single batch makes mean and p95
    # the same number as the total, which measures one sample and calls it a
    # distribution. round() also banker's-rounds 0.5 to 0, which produced a
    # one-batch run on the first version of this script.
    wanted = math.ceil(seconds / per_chunk / BATCH)
    n = max(MIN_BATCHES, wanted) * BATCH
    print(f"  calibrated on {url}: {per_chunk * 1000:.0f} ms/chunk -> {n} chunks for ~{n * per_chunk:.0f}s")
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", default=os.environ.get("BENCH_TEI_LOCAL", DEFAULT_ENDPOINTS["local"]))
    p.add_argument("--cluster", default=os.environ.get("BENCH_TEI_CLUSTER", DEFAULT_ENDPOINTS["cluster"]))
    p.add_argument("--seconds", type=float, default=60.0, help="target duration per endpoint")
    p.add_argument("--chunks", type=int, help="fixed chunk count, skipping calibration")
    p.add_argument("--chunk-chars", type=int, default=1000,
                   help="characters per chunk (default 1000, near the corpus mean of 957)")
    p.add_argument("--repeat", type=int, default=1, help="alternate the endpoints this many rounds")
    p.add_argument("--language", choices=sorted(CORPORA), default="mixed",
                   help="corpus language (default mixed: the golden set is ~57%% en, 43%% de)")
    args = p.parse_args()

    endpoints = [
        ("local", args.local),
        ("cluster", args.cluster)
    ]
    for name, url in endpoints:
        try:
            httpx.get(f"{url}/health", timeout=10).raise_for_status()
        except Exception as exc:
            print(f"{name} ({url}) not reachable: {exc}", file=sys.stderr)
            if "CERTIFICATE_VERIFY" in str(exc):
                print("  hint: SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt", file=sys.stderr)
            return 2

    corpus = CORPORA[args.language]
    n = args.chunks or calibrate(
        endpoints[0][1], make_chunk(args.chunk_chars, 0, corpus), args.seconds
    )
    texts = [make_chunk(args.chunk_chars, i, corpus) for i in range(n)]
    batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
    # One tokenize call per language, averaged, so tok/s is the model's own count
    # rather than a chars/4 guess — and a mixed run is not scored at the English
    # rate when almost half its chunks are denser German.
    per_chunk = round(
        sum(count_tokens(endpoints[1][1], t) for t in texts[: len(corpus)]) / len(corpus)
    )
    print(f"payload: {n} chunks x {args.chunk_chars} chars ({args.language}, "
          f"~{per_chunk} tokens each) in {len(batches)} batches of {BATCH}\n")

    results: dict[str, list[dict]] = {name: [] for name, _ in endpoints}
    for r in range(args.repeat):
        # Alternate the order each round so a warm cache or a busy moment does not
        # systematically favour whichever endpoint goes first.
        order = endpoints if r % 2 == 0 else list(reversed(endpoints))
        for name, url in order:
            results[name].append(run(name, url, batches, per_chunk))

    print(f"{'endpoint':>10}  {'total s':>8}  {'chunks/s':>9}  {'tok/s':>8}  {'batch mean':>11}  {'batch p95':>10}")
    for name, _ in endpoints:
        best = min(results[name], key=lambda x: x["total"])
        print(f"{name:>10}  {best['total']:>8.1f}  {best['chunks_per_s']:>9.2f}  "
              f"{best['tokens_per_s']:>8.0f}  {best['batch_mean'] * 1000:>10.0f}ms  "
              f"{best['batch_p95'] * 1000:>9.0f}ms")

    a = min(results["local"], key=lambda x: x["total"])["total"]
    b = min(results["cluster"], key=lambda x: x["total"])["total"]
    faster, ratio = ("cluster", a / b) if b < a else ("local", b / a)
    rounds = f"best of {args.repeat} rounds" if args.repeat > 1 else "single round"
    print(f"\n  {faster} is {ratio:.2f}x faster on identical payloads ({rounds})")
    return 0


if __name__ == "__main__":
    # Same as rag_evaluation.__main__: the module's environment file carries
    # SSL_CERT_FILE, which the cluster's traefik certificate needs because it is
    # not in certifi's bundle. find_dotenv walks upwards, so this works when the
    # script is run from tests/ as well as from the package root.
    load_dotenv(find_dotenv(usecwd=True))
    raise SystemExit(main())
