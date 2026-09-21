# rag-evaluation

The evaluation module (ARCHITECTURE.md block 12). `EVALUATION.md` is the
procedure — what the metrics mean, why labels are documents, how the tuning
loop works. This file is how to run it.

Its stated output is an optimized `parameters.yaml` for `rag-config`: a sweep
writes one candidate file per configuration, and the winner is blessed into that
package under a new `version:`.

```bash
uv sync --extra dev
```

## Building the golden set

Three commands, run in this order, and each one needs a human pass afterwards.
They fill different gaps, so none of them replaces another:

| command | writes | what a human does |
|---|---|---|
| `generate` | a question per sampled document, one labeled document each | judge whether the question is fair |
| `aggregate` | a question per group of documents saying the same thing at different times | judge whether the question really needs all of them |
| `pool` | candidate documents for questions that already exist | judge each candidate relevant or not |

`generate` and `aggregate` write questions and label only the documents they were
written from. `pool` writes no questions — it labels the *other* documents that
answer them. Without it, a document no sample happened to cover stays unlabeled,
and `metrics.score_query` counts it as a miss when retrieval finds it, which
scores the better configurations worse.

Nothing any of them writes is authoritative. Each run leaves its files in
`data/<session-id>/`; you review them there and paste what you keep into
`golden_set.yaml`.

### Seeding questions — `generate`

Needs `../qdrant` up, `../postgres` up, and `OPENROUTER_API_KEY`. Samples
documents from the ledger and asks Gemini for a summary plus one question the
summary answers.

```bash
.venv/bin/python -m rag_evaluation generate --n 50 --seed 7 --dry-run   # the sample only
.venv/bin/python -m rag_evaluation generate --n 50 --seed 7
```

The question is written *from* one document, so that document is relevant by
construction and needs no labeling — `pool` below is what finds the others. What needs a human is the
question — that it stands on its own, and that it names enough to pick this
document out of a corpus of similar ones. Each run writes
`data/<session_id>/entries.yaml` (the entries) and `review.yaml` (the full
summaries and the failures) — the same session id its MLflow traces carry.
Review `entries.yaml`, delete the weak entries, paste the rest into
`golden_set.yaml`. `--data-dir` moves the base directory; the session subdirectory is
always made under it.

`reference` comes out empty on purpose; the proposed answer sits in a comment
under each entry (`--reference` puts it in the field instead). EVALUATION.md's
rule is that references are written by hand — a machine-written one turns tier 2
into a test of two models agreeing.

The writer model is a third family, `google/gemini-3.7-flash`, so it is neither
the agent's (Claude) nor the judge's (GPT). `--model` or
`RAG_EVAL_GENERATE_MODEL` override it.

### Multi-document questions — `aggregate`

Needs `../tei` up and `OPENROUTER_API_KEY`. Reads a `generate` run's summary
points, finds the same measure stated across several documents, and writes one
question that needs all of them.

```bash
.venv/bin/python -m rag_evaluation aggregate \
    --session-id generate-20260821-192458-e4bdc8 --dry-run   # the clusters only
.venv/bin/python -m rag_evaluation aggregate \
    --session-id generate-20260821-192458-e4bdc8
```

Writes `aggregate.yaml` (entries) and `aggregate-review.yaml` (the clusters
behind them) into the same session directory.

`generate` labels exactly one document per question, so nothing it writes tests
whether retrieval returns a *set*. This closes that: four monthly ECB releases
stating M3 growth become one question about the whole period, and recall becomes
graded instead of binary.

Three filters decide what is worth asking about, each for a failure seen on the
real corpus:

| filter | default | what it rejects |
|---|---|---|
| cosine floor | `--threshold 0.90` | statements that merely share a sentence shape |
| documents per cluster | `--max-documents 5` | templates repeated across the corpus |
| figures must differ | always | one fact duplicated verbatim — answerable from any single document, so requiring all of them would score correct retrieval as partial recall |

The last one is what rejects BaFin's cookie banner, which pairs at cosine 0.990 —
higher than anything with content in it — but carries no figures at all.

### Labeling documents — `pool`

Needs `../qdrant` and `../tei` up. No API key, no spend — it runs the retriever
and hands you the results; the judging is yours.

```bash
.venv/bin/python -m rag_evaluation pool --session-id generate-20260825-184221-e18b2b --k 20
.venv/bin/python -m rag_evaluation pool --session-id … --k 30          # the retriever's top_k, the deepest pool available
```

Questions come from that session's `entries.yaml` and `aggregate.yaml`; the
output is written to `pool.yaml` in the same directory, so one session directory
holds the whole review pile. `--ids` narrows it to named entries.

Every candidate is one line — delete the wrong ones and paste the rest into
`relevant_documents`. A candidate is a `source_system` + `url` pair, because the
url alone does not identify a document. Already-labeled documents are omitted —
labels accumulate, so re-pooling shows new work only.

## Tier 1 — retrieval, no LLM

Needs `../qdrant` and `../tei` up. No API key, no spend, seconds per run.

```bash
.venv/bin/python -m rag_evaluation retrieval
.venv/bin/python -m rag_evaluation retrieval --rerank     # the ablation
.venv/bin/python -m rag_evaluation retrieval --ids isin-lookup
```

Only entries with `relevant_documents` are scored; the summary says how many, because
a mean over three of fifteen entries is not a score of the system.

## Test set — the held-out score

Tier 1 on `data/golden_set_test.yaml` for one configuration, given as exactly
one of:

```bash
.venv/bin/python -m rag_evaluation test --file sweeps/chunking.yaml          # its baseline: block
.venv/bin/python -m rag_evaluation test --mlflow-model rag-retrieval-params@best
```

`--file` takes a sweep-style yaml and applies its `baseline:` substitutions to
the shipped parameters; `--mlflow-model` takes a registered set, all three
sections. The index is the one built for that configuration's fingerprint
(`rag_chunks_<fingerprint>`), or the standing collection for the shipped
fingerprint; `test` never builds one — `sweep` or `curve` with `--reingest` do.

## Tiers 2 and 3 — the judge scorers

Needs `OPENROUTER_API_KEY`. Runs the real agent, so it costs time and money.

```bash
.venv/bin/python -m rag_evaluation agent --ids isin-lookup
.venv/bin/python -m rag_evaluation agent --judge-model openrouter:/openai/gpt-5.4
```

The judge is a different family from the agent *and* stronger; the harness warns
if you point both at the same model or the same family.

## Sweeps

```bash
.venv/bin/python -m rag_evaluation sweep --file sweeps/topk.yaml --dry-run
.venv/bin/python -m rag_evaluation sweep --file sweeps/query_side.yaml
.venv/bin/python -m rag_evaluation sweep --file sweeps/chunking.yaml --reingest
```

A sweep file is a baseline plus candidates of two kinds:

```yaml
baseline:                       # the starting point every candidate departs from
  agent.top_k: 7
sweep:                          # one candidate per value, one knob each
  agent.rrf_k: [2, 10, 60]
candidates:                     # one candidate per entry, as many knobs as it lists
  - label: chunk350
    set:
      parse_chunk.chunk_size: 350
      parse_chunk.chunk_overlap: 50
```

`sweep:` is coordinate descent — the winner of one knob is *not* carried into
the next knob's candidates; put it in the next file's `baseline:` yourself. Use
`candidates:` for knobs that cannot move independently, chunk size and overlap
being the case that matters: 350 words against the baseline's 100-word overlap
changes the overlap fraction too, so the score could not be attributed.

Query-side candidates reuse the collection that is already built and score in
seconds. A candidate that changes the index gets its own collection named by
`rag_config.ingest_fingerprint` and needs `--reingest`, which for each candidate:

1. derives `rag_eval_<fingerprint>` on the same server as `RAG_PG_DSN`, creates
   it, applies `postgres/sql/*.sql`, and copies in the newest sighting of every
   document from the source `ingest_manifest` under one run_id;
2. runs stages 2-6 (never discover — see `ledger.py`) into
   `rag_chunks_<fingerprint>`;
3. scores tier 1 against it.

So `RAG_PG_DSN` points at the database holding the real manifest, and the sweep
writes only to the copies. Overrides, all optional:

```yaml
ingest:
  source_dsn: postgresql+psycopg://rag@host/rag_ingestion   # default: $RAG_PG_DSN
  run_id: frozen                                            # stamped on the copy
  prefix: rag_eval_
```

The runner refuses to write into the production collection or ledger. It needs
`CREATE DATABASE` on that server, and `RAG_TEI_REPLICAS` should match the
running `tei-embed` count — embed sizes its worker threads from it.

**Resumable.** Step 1 is idempotent and step 2 re-runs only what the ledger does
not record as ingested, so a campaign killed halfway through picks up where it
stopped. A candidate whose collection exists and whose ledger is complete is
skipped.

**What is frozen is the list of documents, not their bytes** — stage 4 fetches
every URL live. Run a campaign's candidates back to back so the window in which
a source can change stays small.

## Promoting a tuned set

This module's part is the upstream half: `curve` and `sweep` produce the runs,
and every run logs a `rag-retrieval-params` carrier model, so the winner can be
registered from the MLflow UI's "Register Model" button under a name and an
alias. That is what gives a configuration a version.

Writing that version into the `parameters.yaml` `rag-config` ships is
`rag-config`'s own command — the target is that package's file and the edit has
to preserve its layout. See `config/README.md` for it.

## Environment

| Variable | What it does |
|---|---|
| `RAG_PARAMETERS_PATH` | the candidate `parameters.yaml` every stage and the agent read |
| `RAG_PG_DSN` | the ledger; give a sweep its own database |
| `RAG_QDRANT_COLLECTION` | the index being written or queried |
| `MLFLOW_TRACKING_URI` | where to log; unset disables logging entirely, for every tier |
| `RAG_EVAL_JUDGE_MODEL` | judge for tiers 2 and 3 |
| `MLFLOW_EXPERIMENT_EVALUATION` | experiment for the scoring tiers (default `RAG-Evaluation`) |
| `MLFLOW_EXPERIMENT_DATASET` | experiment for `generate` (default `RAG-Evaluation-Dataset`) |
| `RAG_EVAL_GENERATE_MODEL` | writer model for `generate` |

## Tests

Offline — no services, no network, no key.

```bash
.venv/bin/python -m pytest
```

## Dependencies

`rag-config` and `rag-agent` resolve from the sibling directories, editable —
where the six stages pin git revisions. A stage pins because its Docker build
context cannot reach a sibling; this module is never built into an image, and it
already reaches its siblings by path to run them. Editable also means a sweep
measures the working tree rather than the last push.
