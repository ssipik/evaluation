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

## Tier 1 — retrieval, no LLM

Needs `../qdrant` and `../tei` up. No API key, no spend, seconds per run.

```bash
.venv/bin/python -m rag_evaluation retrieval --k 10
.venv/bin/python -m rag_evaluation retrieval --k 10 --rerank     # the ablation
.venv/bin/python -m rag_evaluation retrieval --ids isin-lookup
```

Only entries with `relevant_documents` are scored; the summary says how many, because
a mean over three of fifteen entries is not a score of the system.

## Seeding entries — `generate`

Needs `../qdrant` up, `../postgres` up, and `OPENROUTER_API_KEY`. Samples
documents from the ledger and asks Gemini for a summary plus one question the
summary answers.

```bash
.venv/bin/python -m rag_evaluation generate --n 50 --seed 7 --dry-run   # the sample only
.venv/bin/python -m rag_evaluation generate --n 50 --seed 7
```

The reverse of `pool`: the question is written *from* one document, so
`relevant_documents` is exact and needs no labeling. What needs a human is the
question — that it stands on its own, and that it names enough to pick this
document out of a corpus of similar ones. Each run writes
`data/<session_id>/entries.yaml` (the entries) and `review.jsonl` (the full
summaries and the failures) — the same session id its MLflow traces carry.
Review `entries.yaml`, delete the weak entries, paste the rest into
`golden_set.yaml`. `--out` moves the base directory; the session subdirectory is
always made under it.

`reference` comes out empty on purpose; the proposed answer sits in a comment
under each entry (`--reference` puts it in the field instead). EVALUATION.md's
rule is that references are written by hand — a machine-written one turns tier 2
into a test of two models agreeing.

The writer model is a third family, `google/gemini-3.7-flash`, so it is neither
the agent's (Claude) nor the judge's (GPT). `--model` or
`RAG_EVAL_GENERATE_MODEL` override it. `generated.jsonl` holds each document's
full summary and every failure.

## Labeling — `pool`

```bash
.venv/bin/python -m rag_evaluation pool --k 20            # paste-ready YAML
.venv/bin/python -m rag_evaluation pool --with-rerank     # +1 run, ~222 s/query
```

Prints what several configurations retrieved, one candidate per line; delete the
wrong lines and paste the rest into `relevant_documents`. A candidate is a
`source_system` + `url` pair, because the url alone does not identify a document.
Already-labeled documents are omitted — labels accumulate, so re-pooling shows
new work only.

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

A sweep file is a baseline plus one list of values per knob; candidates are one
knob at a time from that baseline. Query-side candidates reuse the collection
that is already built. A candidate that changes the index gets its own
collection named by `rag_config.ingest_fingerprint`, and needs `--reingest` plus
its own ledger:

```bash
export RAG_PG_DSN=postgresql://rag@localhost/rag_eval_<fingerprint>
```

The runner refuses to write into the production collection or ledger.

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
