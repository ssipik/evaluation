# Evaluation procedure — plan

> **Status (2026-09-21): implemented, except B8; tiers 2 and 3 not yet run.** This is the design record — what was
> decided and why. The living procedure is `EVALUATION.md` (Part A below became that file);
> `README.md` is how to run it.
>
> | Part | State |
> |---|---|
> | B0 `RETRIEVER` span | done, `../agent/src/rag_agent/{tracing,tools}.py`, tested |
> | B1 `evaluation/` package | done |
> | B2 config propagation | done — no stage changes were needed |
> | B3 `ingest_fingerprint` | done, `../config/src/rag_config/params.py`, tested |
> | B4 sweep runner | query-side verified live; the ingest path is written but never run end to end, and preparing a ledger is still by hand (see `stages.py`) |
> | B5 golden set | done — 63 tuning entries (`golden_set.yaml`) + 29 held-out (`golden_set_test.yaml`), all with `relevant_documents` and `reference`; no `expect_refusal` negative controls |
> | B6 tier 1 | done, verified against the live corpus and MLflow |
> | B7 tiers 2 and 3 | written and unit-tested (`tests/test_agent_eval.py`), not yet run live — needs `OPENROUTER_API_KEY` and real spend; judge calibration not done |
> | B8 CI | **not started**, including the Qdrant snapshot it depends on |
> | B9 documentation | done |
>
> Two things the implementation corrected: `rag-config` is already a git source, so the
> Docker-builds-blocked note in B4 is stale; and a `--pool` run with reranking costs ~222 s
> per query here, so the reranked run is opt-in rather than a default contributor.

## Context

Block 12 of `masterplan/ARCHITECTURE.md` names MLflow + RAGAS and a golden set of ~50–100
Q&A pairs, but only one piece of it is built: `agent/eval/ragas_eval.py` runs the real agent
and scores answers with an LLM judge. Everything that makes tuning *measurable* is missing —
there is no judge-free retrieval scoring, no relevance labels, no reference answers, and eval
scores print to stdout and vanish, so no configuration can be compared against another.

**RAGAS is being dropped (decided 2026-08-17).** MLflow 3.15 — already installed, already
running as the tracing backend — ships `mlflow.genai.evaluate()` with built-in judge scorers
that cover every RAGAS metric this harness uses, plus tool-call scorers that cover what this
plan originally described as manual trace reading, plus scheduled scorers for production
traffic. Keeping RAGAS would mean maintaining a second evaluation stack, two version ceilings
that exist only to work around its own dependency breakage (`ragas<0.5`,
`langchain-community<0.4`), and a results path that logs nowhere.

Verified before deciding:

- MLflow's judge URI parser is provider-generic — `openrouter:/openai/gpt-5.4` becomes
  litellm's `openrouter/openai/gpt-5.4` (`mlflow/metrics/genai/model_utils.py:98`), and
  litellm 1.97 resolves that to provider `openrouter`, reading `OPENROUTER_API_KEY`.
- A live judge call works: agent on Haiku 4.5, judge on GPT 5.4, one OpenRouter key for both.
- The judge model is **not** inherited from autologging.
  `get_default_model()` (`mlflow/genai/judges/utils/__init__.py:35`) resolves
  `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` → Databricks (if the tracking URI is Databricks) →
  hardcoded `openai:/gpt-4.1-mini`. That last fallback needs an OpenAI key, so **both
  harnesses must pass `model=` explicitly** rather than rely on the default.

**Deliverable 1** — a new document `evaluation/EVALUATION.md` holding the procedure below
(steps, tiers, metrics with formulas, the tuning loop). It is a reference document, not a
design decision record; block 12 stays the authority on *why* MLflow was chosen, and gets a
pointer to this file plus the RAGAS reversal.

**Deliverable 2** — a new `evaluation/` module: the sweep runner, the tier-1 retrieval harness,
and tier 2 rewritten on `mlflow.genai.evaluate()`.

---

# Part A — the document (`evaluation/EVALUATION.md`)

**This became `EVALUATION.md`.** Kept here as the record of what was specified; read the
living file instead, which is maintained.

## The three steps

| Step | What it is | Uses the golden set? |
|---|---|---|
| 1. Build the golden set | authoring + validating the evaluation data | it *is* the set |
| 2. Tune — the run/compare loop | change a knob, re-score, diff. Scoring happens **inside** this loop | yes, and it must stay frozen |
| 3. Online evaluation | real traffic, thumbs, sampled auto-eval | no — it *writes* new entries into it |

Scoring is not a step of its own: it is what step 2 does. What *is* worth separating, inside
step 2, is **which scorer** runs — the judge-free retrieval scorers (tier 1) or the judge
scorers (tiers 2 and 3) — because they differ by three orders of magnitude in cost and answer
different questions.

## Step 1 — the golden set

Data only. No system runs at this step; it produces a YAML file, not a score.

**Three kinds of entry:**

- **Answerable** — the answer exists in the corpus today. Spread deliberately across
  `source_system`, both languages, and the query shapes that stress different retrieval legs
  (exact identifier → BM25; paraphrase → dense; German compound → the block-5 gap).
- **Negative controls** — answerable nowhere in the corpus; correct behaviour is refusal.
  Target ~10% of the set. Without these the worst failure mode here — answering a regulatory
  question from model weights — is invisible.
- **Adversarial / near-miss** — two similar documents where only one is right (the
  March-vs-June interest-rate pair), date-qualified questions, question and answer in
  different languages.

**Fields per entry:**

| Field | Used by | Notes |
|---|---|---|
| `id` | all | stable slug, so a score tracks across runs |
| `question` | all | |
| `language` | all | of the question; the answer may come from the other language |
| `tests` | humans | free-text note on what the entry stresses |
| `reference` | tier 2 | the correct answer, in your words. **Written by hand** |
| `relevant_urls` | tier 1 | **new** — the `source_url`s that must be retrieved |
| `expect_refusal` | tier 3 | **new** — true for negative controls |

`reference` is written by hand on purpose: a reference generated from the same retrieval the
system uses measures self-consistency, not correctness. Synthetic generation is fine for the
*questions*, never for the references.

**Getting `relevant_urls` — pooling.** Do not read the corpus. Run the system at a high `k`
(20) under a few configurations, take the union of what came back, and label those by hand.
The method is called **pooling**, and it is how TREC builds its relevance judgments: the
top-ranked documents from each run are unioned into a pool, and only the pool is judged
(Voorhees, *Overview of TREC 2002*, <https://trec.nist.gov/pubs/trec11/papers/OVERVIEW.11.pdf>
— top 100 per run, average pool 769 documents).

Its bias is known and acceptable: anything no configuration ever retrieved stays unlabeled and
counts as irrelevant. The same paper reports that Harman and Zobel independently found early
TREC collections did contain unjudged documents that would have been judged relevant, and that
the approximately-complete judgments pooling produces are still sufficient to **fairly compare
runs**. That is precisely what this loop needs — it compares configurations rather than quoting
absolute numbers.

**Why pooling rather than labeling exhaustively**, even though the corpus is small today:
exhaustive labeling costs *questions x documents*, so 50 questions against 100 documents is
already 5,000 judgements. Pooling costs *questions x (configurations x k)* — bounded by how
many runs contribute, **independent of corpus size**. Three configurations at `k=20` is at most
60 documents per question whether the corpus holds 700 chunks or 700,000. That is the property
that makes this the method to build on rather than a demo-scale shortcut.

**As the corpus grows, the lever is run diversity, not pool depth.** TREC's reliability comes
from many genuinely different systems contributing. The local equivalent is pooling from
dense-only, sparse-only, hybrid and reranked runs rather than three near-identical hybrid
settings — a deeper pool from one configuration mostly re-surfaces what that configuration
already ranked highly, so it costs judging effort without reducing bias.

**Labels accumulate.** Judging is per (question, document) and does not expire: when a new
configuration surfaces documents nobody has judged, only those are new work, and the coverage
of the set improves monotonically. So bias shrinks over time as a side effect of running the
loop, provided newly-pooled documents are labeled when they appear rather than silently
counted as irrelevant.

**Validating the set** (a separate pass, before any score counts):

- every answerable question really is answerable from the corpus today
- each `reference` is checkable against its source document, not written from memory
- no ambiguous questions — if two documents answer differently, the question must say which
- no near-duplicates, or one topic silently gets double weight in the mean
- the spread is counted, not assumed (entries per `source_system`, per language, per shape)
- a schema check: ids unique, required fields present, negative controls flagged

**The one feedback loop:** entries where *every* configuration retrieves nothing are usually
broken questions, not retrieval failures. Running step 2 exposes them; fix them back in step 1.

**Size**: 50–100. Below ~30 the per-metric noise swamps the effect of any config change.

## Step 2 — tune: the run/compare loop

Everything that produces a number happens here. Two subsections: **the scorers** (which
metric can be run, at what cost) and **the loop** (the procedure that runs them).

Steps and tiers are different axes: a **step** is a phase of the work, a **tier** is which
scorer runs.

### The scorers — three tiers

Three tiers because they answer different questions at very different costs.

#### Tier 1 — retrieval only, no LLM

Compares the retrieved chunks against `relevant_urls` by **exact identifier match** — set
membership, nothing semantic. That is what makes it deterministic and free.

Matching is on `source_url`, not chunk id: stage 5 rewrites `manifest_id` in place and chunk
ids regenerate on every re-chunk, while `source_url` is stable — the same reason
`fetch_document` keys on it. The cost is granularity (you score document retrieval, not
paragraph retrieval), and the gain is that labels survive the chunk-size experiment you most
want to run.

**The principle: label the unit that is fixed, not the unit that is a parameter.** TREC's
classic qrels are document-level; MS MARCO's are passage-level, but only because those
passages are a fixed part of the collection rather than something the system re-cuts. Chunks
here are a tuning knob, so chunk-level labels would be a function of the thing under test.
Practitioner guidance converges on the same shape — relevance defined at knowledge-base
article level with a hit counted when any of its chunks is retrieved, golden sets described as
query-*document* pairs, and chunk-size sweeps plotted as recall@k against size, which only
works if labels survive re-chunking. This is common practice rather than a formal standard;
there is no standards body for RAG evaluation.

**Chunks in, documents out — the one conversion that has to happen first.** Retrieval returns
*chunks*; the golden set labels *documents* (`relevant_urls`). Comparing the two directly
would let three chunks of the same document count as three hits and push recall above 1. So
before any formula runs, the retrieved chunk list is **collapsed to a ranked list of distinct
`source_url`s, first occurrence winning**:

```
retrieved chunks:  [c1 docA, c2 docB, c3 docA, c4 docC, c5 docB]
collapsed ranking: [docA, docB, docC]          <- what the metrics see
```

Everything below is then document-level and `|hits| ≤ |R|` by construction. Two consequences
to keep in mind:

- `k` in `precision@k` is the number of **distinct documents** in the collapsed list, not the
  number of chunks requested. It varies per question, which is one more reason precision is
  read with care.
- Chunk-level precision — what fraction of the *chunks* in the prompt came from a relevant
  document — is a genuinely different and useful number, since it measures prompt waste
  directly. Report it separately as `chunk_precision@k` rather than folding it in.

**Notation**: `R` = the labeled relevant documents for one question, `hits` = collapsed
ranking ∩ R, `|hits|` = the number of hits, `k` = how many documents the collapsed list holds.

**hit_rate@k** — Did at least one relevant document appear in the top k? Per question this is
1 or 0; the "rate" is the average over the set.
```
hit_rate = 1 if |hits| > 0 else 0
```
Classic IR calls this `success@k`. LlamaIndex's `HitRate` has a `use_granular_hit_rate` flag
(default `False`) whose granular form is `|hits| / |R|` — mathematically identical to recall,
so do not report both.

**recall@k** — Of all the relevant documents, what fraction did retrieval find?
```
recall = |hits| / |R|
```

**precision@k** — Of the results returned, what fraction were relevant?
```
precision = |hits| / k
```
Capped at `|R| / k`, so it is **not comparable across questions with different `|R|`**.

**MRR (Mean Reciprocal Rank)** — How far down the list was the *first* relevant result?
Anything after the first hit is ignored.
```
MRR = 1 / rank(first hit)          0 if there is no hit
```

**nDCG@k (normalized Discounted Cumulative Gain)** — How close is this ranking to the best
possible one? Each hit is worth `1 / log₂(position + 1)`, so later positions are discounted.
```
DCG  = Σ  1 / log₂(pos + 1)        over the positions of the hits
IDCG = the same sum with all |R| hits packed into positions 1..|R|
nDCG = DCG / IDCG
```
Position weights: 1 → 1.000, 2 → 0.631, 3 → 0.500, 4 → 0.431, 5 → 0.387.
The normalization is why nDCG *is* comparable across questions where precision is not.

Every metric is computed per question, then averaged over the set.

**Worked example**, three questions. Letters are documents — i.e. the collapsed ranking, five
distinct documents each, so `k = 5`:

| | Q1 `R={A,B}`, got `[X,A,Y,B,Z]` | Q2 `R={C}`, got `[C,W,V,U,T]` | Q3 `R={D,E}`, got `[P,Q,R,S,T]` | Mean |
|---|---|---|---|---|
| hit_rate | 1 | 1 | 0 | **0.67** |
| recall | 2/2 = 1.0 | 1/1 = 1.0 | 0/2 = 0 | **0.67** |
| precision | 2/5 = 0.4 | 1/5 = 0.2 | 0/5 = 0 | **0.20** |
| MRR | 1/2 = 0.5 | 1/1 = 1.0 | 0 | **0.50** |
| nDCG | 0.65 | 1.0 | 0 | **0.55** |

Q1's nDCG in full: hits at positions 2 and 4 → `DCG = 0.631 + 0.431 = 1.062`; ideal is both
at positions 1 and 2 → `IDCG = 1.000 + 0.631 = 1.631`; `1.062 / 1.631 = 0.65`.

Q2 shows why precision is read with care: retrieval was perfect, but only one document was
relevant and five were requested, so precision is 0.2.

#### Tier 2 — generation, LLM as judge

`mlflow.genai.evaluate()` with built-in scorers. It runs the real agent as the `predict_fn` —
same retriever, tools and prompt — and the retrieval scorers judge the answer **against the
chunks that were actually retrieved**: `RetrievalGroundedness` calls
`judges.is_grounded(request=…, response=…, context=…)`, so these are RAG scorers, not
prompt-only answer scorers. Nothing is re-retrieved for the eval's benefit.

The context comes from spans typed `RETRIEVER` in the trace. The agent emitted none when this
was written, which is what B0 fixed.

| Scorer | Was (RAGAS) | Needs `reference` | Fails when |
|---|---|---|---|
| `RetrievalGroundedness` | `faithfulness` | no | the answer states things the retrieved chunks do not support → hallucination |
| `RelevanceToQuery` | `answer_relevancy` | no | the answer is supported but does not address the question |
| `RetrievalRelevance` | `context_precision` | no | retrieval returned junk alongside the good chunks |
| `RetrievalSufficiency` | `context_recall` | **yes** | the retrieved context cannot support the expected answer |
| `Correctness` | — | **yes** | the answer contradicts the reference |
| `Guidelines` | — | no | free-text rules — used here for the citation and refusal rules |

Read them as a pair: **low groundedness = generation problem; low sufficiency = retrieval
problem.**

**The judge model.** Different family from the agent *and* stronger:

```
agent:  openrouter/anthropic/claude-haiku-4.5     RAG_LLM_MODEL
judge:  openrouter:/openai/gpt-5.4                RAG_EVAL_JUDGE_MODEL
```

One OpenRouter key covers both. Two independent reasons, and both matter:

- **Different family** — a model scoring its own output has a self-preference bias that
  inflates every number, and the bias is documented at family level, not just model level. So
  Sonnet judging Haiku is not clean.
- **Stronger** — the judge has to spot a claim the context does not support. A judge weaker
  than the agent misses subtle unfaithfulness and errs toward "looks fine", so scores come
  back optimistic.

A cheaper judge (`gpt-5.4-mini`) is acceptable for **relative** comparisons in the tuning
loop, where a constant bias cancels between config A and config B. It is not acceptable for
quoting an absolute number to anyone. If cost forces the downgrade, say so alongside the
results.

Always pass `model=` explicitly. The MLflow default is `openai:/gpt-4.1-mini`, which needs an
OpenAI key and is a model nobody chose.

**Calibrate the judge once.** Hand-label 20–30 judgements and measure agreement with the
judge on the metric that matters most (groundedness). Above ~80% agreement its numbers mean
something; below that, no amount of model strength fixes it. This is the step most people
skip, and it is what separates a score from a number.

#### Tier 3 — agent behaviour

Also scorers, not manual trace reading:

| Check | How |
|---|---|
| did it call `search` at all, or answer from its own weights | `ToolCallCorrectness` |
| did it call `fetch_document` unnecessarily (the truncation-marker problem in `CLAUDE.md`) | `ToolCallEfficiency` |
| did it refuse the negative controls instead of inventing an answer | `Guidelines`, driven by `expect_refusal` |
| did it cite only chunks that were actually retrieved | `Guidelines` |
| did it honour caller-pinned `SearchFilters` | trace inspection — no scorer covers this |

#### Cost, and why the tiers are separate

| Tier | Per configuration, ~50 questions | Deterministic |
|---|---|---|
| 1 | seconds, free | yes |
| 2 | ~an hour + judge cost (far worse with rerank on: ~222 s/query on this box) | no — judge is stochastic |
| 3 | runs with tier 2, same traces | no |

Three reasons the separation earns its keep:

1. **Cost.** A 10-configuration sweep is 10 × tier 2 (~10 h) without it, versus 10 × tier 1
   (seconds) + tier 2 on the 2–3 finalists.
2. **Determinism.** Tier 2's stochastic noise hides small deltas that tier 1 measures exactly.
3. **Diagnosis.** A bad tier-2 score does not say whether retrieval missed the chunk or the
   model mishandled it. Tier 1 isolates the retrieval half.

### The loop

The procedure that runs the scorers above. Tier 1, tier 2 or both, depending on the knob.

```
1. Baseline    — run, record scores AND the full config. A score without its config is not a data point.
2. One knob    — change exactly one thing. Two at once and the delta cannot be attributed.
3. Re-run      — same set, same k.
4. Diff        — per-entry deltas, not only the mean.
5. Keep or revert. If kept, this becomes the new baseline.
```

**The tiers are independent, not a fixed pipeline.** Which tier runs depends on what changed:

| Knob changed | Tier 1 | Tier 2 |
|---|---|---|
| query-side retrieval (`top_k`, prefetch, RRF weights, rerank) | sweep all candidates | finalists only |
| ingest-side (chunk size, embedding model, stemmer) | one re-ingest per candidate, then score | finalists only |
| agent model, system prompt, tool descriptions | not applicable | directly, no pre-filter |

**Knobs, split by cost — exhaust the cheap column first:**

| Query-side (a setting change + re-run) | Ingest-side (needs a full six-stage re-ingest) |
|---|---|
| `top_k` (retrieved nodes) | chunk size / overlap / chunking strategy |
| prefetch limits per leg | embedding model |
| RRF weights (dense vs the two sparse legs) | sparse analyzer / stemmer (the block-5 German splitter) |
| rerank on/off, `rerank_top_n` | |
| metadata filters | |
| agent model, system prompt, tool descriptions | |
| `RAG_TRUNCATE_CHUNKS` | |

**Ingest-side knobs are tunable too — the loop just gets longer.** A chunk-size candidate
cannot be swept against a fixed index; it has to be *built* first. So the loop grows a step:
re-ingest into a separate collection, point the retriever at it, score, repeat.

- **The golden set does not change**, because `relevant_urls` labels documents, not chunks.
  That is the payoff of the labeling choice in step 1: one set scores every chunk
  configuration. Chunk-id labels would have to be re-made for each candidate, which is what
  makes chunk size untunable in most setups.
- **One collection per candidate** (`rag_chunks_cs512`, …), not a re-ingest over the working
  one. `RAG_QDRANT_COLLECTION` already selects it, so nothing in the query path changes and
  the current index stays intact for comparison.
- **Cost is the constraint, not correctness.** A full six-stage cycle at this box's ~100–160
  tokens/s is a long job — so treat an ingest sweep as an occasional campaign (two or three
  candidates, run overnight), not part of the per-knob rhythm. Tier 1 still scores each
  candidate in seconds once its collection exists.
- Ingest parameters go into the MLflow run's params alongside the query-side ones, or the two
  halves of the configuration cannot be compared later.

The RRF **k constant** (the 60 in `1/(k+rank)`) is *not* exposed by Qdrant — only the weights
are tunable.

**Reading the results — the pattern diagnoses, not the number:**

| Pattern | Diagnosis | Fix |
|---|---|---|
| recall low | the chunk never came back | chunking, embedding, prefetch limit, query legs |
| recall high, MRR/nDCG low | found it, ranked it badly | reranker, RRF weights |
| recall@20 high, recall@5 low | ranking problem — a reranker will pay | turn rerank on, re-measure |
| precision low, recall high | prompt full of junk | lower `top_k` / `rerank_top_n` |

**Discipline:**

- The golden set stays **frozen** while tuning. Growing it moves the mean, and then a delta
  cannot be attributed to the config. Grow between campaigns, and re-run the baseline
  afterwards to establish a new reference point.
- Measure run-to-run variance once (run the same config twice on tier 2) and treat any
  smaller delta as no delta.
- Judge temperature at 0.
- Read per-question scores. An entry scoring 0 is either a real failure worth fixing or a bad
  golden-set entry worth deleting, and only the entry tells you which.

**The three questions this settles**, currently carried on assumption in ARCHITECTURE.md:
whether the reranker earns its second container (tier 1, rerank on/off, compare nDCG);
whether German compound recall measurably lags English (split the tier-1 mean by `language`);
whether Haiku 4.5 is strong enough (tier 2 + tier 3).

## Step 3 — online evaluation

Offline sets go stale — they do not contain the questions users actually ask.

- traces of real traffic (MLflow, already running): latency, token cost, tool-call counts
- thumbs up/down in the UI (Chainlit supports it natively)
- sampled auto-eval: `RetrievalGroundedness` over a sample of production traces —
  reference-free, so it works on live traffic with no labels. MLflow's
  `mlflow.genai.scorers.online` / `scheduled_scorers` run exactly this against a trace
  sampling rate, which is the whole reason step 3 is cheap once tier 2 exists.
- **feed failures back**: a down-voted question with no good answer becomes a new golden-set
  entry. That is the loop that keeps step 1 from going stale.

## Sources

Kept for future reference; none of them is a standard, and where they disagree with this
document the reasoning here is what was actually decided.

**Pooling and IR evaluation methodology**

- Voorhees, *Overview of TREC 2002* — <https://trec.nist.gov/pubs/trec11/papers/OVERVIEW.11.pdf>
  How pools are built (top 100 per run, average pool 769 documents) and the finding that
  pooled judgments are incomplete but still sufficient to fairly compare runs. The source for
  the pooling method in step 1.
- Text Retrieval Conference — <https://en.wikipedia.org/wiki/Text_Retrieval_Conference>
  One-paragraph definition of pooling, if the paper is more than you need.

**RAG evaluation practice**

- *Retrieval Evaluation for RAG* (Data AI Hub) —
  <https://www.dataaihub.co/learn/retrieval-evaluation>
  Relevance defined at knowledge-base-article level with a hit counted when any of its chunks
  is retrieved — the same collapse rule used in tier 1. Also golden sets as 50-200
  query-document pairs.
- *RAG evaluation metrics* (Braintrust) —
  <https://www.braintrust.dev/articles/rag-evaluation-metrics>
  Gating retrieval changes on recall@k before touching prompts or the LLM, which is the
  tier-1-before-tier-2 ordering in step 2.
- *RAG Evaluation Metrics: Retrieval, Reranking, Generation* —
  <https://slavadubrov.github.io/blog/2026/05/10/rag-evaluation-metrics/>
  Metric-by-metric walkthrough across all three stages, and the chunk-size sweep as recall@k
  plotted against token size.
- *Long-Context Isn't All You Need: How Retrieval & Chunking Impact Finance RAG* (Snowflake) —
  <https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/>
  A measured chunking/retrieval ablation on a regulated-domain corpus — the closest published
  analogue to the ingest-side sweep described in step 2.
- *Result Evaluation for RAG: Metrics & Best Practices* (IBM) —
  <https://www.ibm.com/think/architectures/rag-cookbook/result-evaluation>
  Vendor-neutral overview of the metric landscape, useful as an orientation piece.


---

# Part B — implementation

## B0. Emit a `RETRIEVER` span — the prerequisite for every judge scorer

Stays in the **agent package**: the span has to come from the code being traced.

`RetrievalGroundedness`, `RetrievalRelevance` and `RetrievalSufficiency` take their context
**only** from spans of type `RETRIEVER`
(`extract_retrieval_context_from_trace`, `mlflow/genai/utils/trace_utils.py:747`), and raise
`"No retrieval context found in the trace"` when there is none. There is none today:
`Retriever` (`src/rag_agent/retrieval.py:77`) is a plain class over qdrant-client, not a
LlamaIndex `BaseRetriever`, so `mlflow.llama_index.autolog()` never instruments it.

- Decorate `KnowledgeTools.search_knowledge_base` (`src/rag_agent/tools.py:141`) with
  `@mlflow.trace(span_type=SpanType.RETRIEVER)` — not `Retriever.search`, so the span brackets
  what the agent asked for and what it got back.
- The span's **output shape is a contract**: a list of dicts, each with one of
  `("page_content", "content", "text")` plus optional `"metadata"` (`_parse_chunk`,
  `trace_utils.py:838` / `:818`). Anything else is dropped with a log warning and the scorer
  sees an empty context — a silent zero, not an error. Test the shape.
- **Multi-call is fine.** The scorer docstrings say "if multiple spans are found, MLflow will
  use the last one", but the implementation iterates every top-level retrieval span and returns
  one `Feedback` each. Do not collapse searches into one span to work around a limitation that
  does not exist.
- **Keep tracing best-effort** (`agent/CLAUDE.md`): verify with `MLFLOW_TRACKING_URI` empty.

## B1. The `evaluation/` package

A standalone module by the same convention as the six stages: own `pyproject.toml`,
`uv.lock`, `.venv`, thin `python -m rag_evaluation` entrypoint with the logic in library
functions. **This module is already the plan of record** — `CLAUDE.md:23` and
`ARCHITECTURE.md:722` both name it as what generates `parameters.yaml`.

```
evaluation/
├── golden_set.yaml              moved from agent/eval
├── runs/<fingerprint>/          one directory per ingest configuration
│   └── parameters.yaml          the candidate config; the directory IS the registry
├── src/rag_evaluation/
│   ├── golden.py                load + validate the set
│   ├── pool.py                  the --pool labeling helper
│   ├── metrics.py               tier-1 metrics, pure functions
│   ├── retrieval_eval.py        tier 1
│   ├── agent_eval.py            tiers 2 and 3
│   ├── sweep.py                 the ingest/query loop
│   └── __main__.py              argparse only
└── tests/                       offline, CI
```

Depends on `rag-agent` (the query path) and `rag-config`. It does **not** depend on the six
stage packages — the sweep shells out to their entrypoints.

Why not `agent/eval/`: an ingest sweep has to drive stages 1-6, which the agent package has no
business doing. Why not one eval script per stage: the outcome metrics (recall, nDCG,
groundedness) are only observable after a document has been chunked, embedded, indexed *and*
queried, so no single stage can compute them; per-stage checks are a separate, intrinsic layer,
recorded as a TODO in `ARCHITECTURE.md` block 12.

## B2. Configuration propagation — no stage changes needed

One candidate file plus one env var reaches all six stages and the agent:

```bash
RAG_PARAMETERS_PATH=evaluation/runs/a3f19c/parameters.yaml   # every tunable knob
RAG_PG_DSN=postgresql://rag@localhost/rag_eval_a3f19c        # deployment value, env-only
```

`rag_config.params_path()` (`config/src/rag_config/params.py:177`) checks
`RAG_PARAMETERS_PATH` before the copy shipped in the package, so changing `chunk_size` is
editing one YAML. No `--config` flag, no `eval_*` tables, no re-lock, no rebuild.
`qdrant_collection` lives *in* the candidate file, since every stage falls back to
`params.qdrant_collection`.

**Rejected: parallel `eval_ingest_*` tables.** Same isolation is available from a separate
database with the identical DDL, at zero code cost, and it scales to N ledgers rather than two.
Duplicated tables would thread a table-name parameter through the SQL of six repos and drift
against `postgres/sql/*.sql`.

**The safety guard** replaces what an `--eval` flag would have given: the runner refuses to
start unless `RAG_PARAMETERS_PATH` is set, and unless the candidate's `qdrant_collection` and
`RAG_PG_DSN` both differ from the production defaults. One check, one repo.

## B3. `ingest_fingerprint()` in `rag-config`

The collection name is a hash of exactly those parameters whose change invalidates the index.
The classification already exists as the dataclass docstrings:

| Dataclass | Docstring | In the fingerprint |
|---|---|---|
| `PlanParams` | *"re-plans; no re-ingest"* | no |
| `ParseChunkParams` | *"needs stages 4-5 re-run"* | **all of it** |
| `EmbedParams` | *"invalidates the vectors already in Qdrant"* | **all but two fields** |
| `FinalizeParams` | stage 6 | no |
| `AgentParams` | *"All query-time: changing these needs no re-ingest"* | no |

```python
INGEST_FINGERPRINT_EXCLUDE = {"embed.batch_size", "embed.qdrant_collection"}

def ingest_fingerprint(params) -> str:
    """sha256 over parse_chunk + embed minus the exclusions, sorted keys, first 6 hex."""
```

**Include by default, exclude explicitly** — the failure modes are asymmetric. A field wrongly
included costs an unnecessary re-ingest and is obvious; a field wrongly excluded silently
scores the wrong index. So a new parameter is covered without anyone remembering.

Lives in `rag-config`, beside the field definitions, so adding a field and updating the
fingerprint is one commit in one repo.

## B4. The sweep runner

```
once:          discover -> snapshot the manifest        (frozen corpus)
per ingest configuration:
    fingerprint = ingest_fingerprint(candidate)
    if collection rag_chunks_<fp> exists:  skip the ingest       <- the cache
    else: fresh ledger DB + replay manifest snapshot
          -> delta -> plan -> parse_chunk -> embed -> finalize
    per query configuration:
        Retriever.search over the golden questions -> tier 1 -> one MLflow run
finalists only: full RagAgent + mlflow.genai scorers (tiers 2 and 3)
```

- **Freeze the corpus.** `discover` reads live feeds and its own contract says a re-run unions
  what it listed rather than reproducing a snapshot. Two configurations discovered on different
  days scored different corpora, and the comparison is void. Run discover once, snapshot
  `ingest_manifest`, replay it into each configuration's fresh ledger. Stages 2-6 are
  idempotent off a fixed manifest, which is exactly what this needs.
- **Nesting matters**: ingest outer, query inner. Reversed, every `top_k` value costs a
  re-ingest.
- **One knob at a time from a baseline** (coordinate descent), not a full grid: 3+2+2 = 7 runs
  instead of 12, each interpretable alone. Take the winner, make it the new baseline. Full grid
  only where two knobs are suspected to interact.
- **Shell out to the stage venvs**, not to containers: the images are weeks stale and cannot be
  rebuilt while `rag-config` is a `path = "../config"` source. `scripts/run_pipeline.sh` is compose-only
  and needs a frozen manifest and per-configuration collections that it does not have, so the
  runner is its own thin orchestrator rather than a wrapper around it. It grows a `--containers`
  mode, or hands off to Airflow, once the repos have remotes.
- **MLflow structure**: one parent run per sweep, one child run per configuration. Params =
  *every* knob plus the fingerprint and collection name (a config cannot be reconstructed from a
  delta). Metrics = the tier-1 means plus per-language and per-source breakdowns. Tags =
  `baseline=true`, `swept_knob=<name>`. Artifact = the per-entry CSV.
- **Output**: the winning `parameters.yaml` is blessed into `rag-config` under a new `version:`
  — no conversion step, because it is already that file.

## B5. Golden set

Moves to `evaluation/golden_set.yaml`, with two new optional fields documented in the header:

- `relevant_urls: []` — the `source_url`s that must be retrieved (tier 1)
- `expect_refusal: false` — true for negative controls (tier 3)

Both empty by default, exactly as `reference` already is, so entries are labeled gradually. The
YAML stays the source of truth and is *mapped* into MLflow's dataset shape at load time
(`inputs` / `expectations`) — tier 1 has nothing to do with MLflow's schema.

**The `--pool` helper**: runs the retriever at `k=20` and prints each entry's returned
`source_url`s in a paste-ready YAML shape.

- **Unions across configurations** (dense-only, sparse-only, hybrid, reranked) rather than
  reporting one run — run diversity is what keeps pool bias down as the corpus grows.
- **Marks URLs already labeled** and prints only the new ones. Labels accumulate; re-pooling
  must not re-present judgements already made.

## B6. Tier 1 — `retrieval_eval.py`

- Imports `rag_agent.retrieval` (`Retriever`, `build_filter`) directly. Never `RagAgent`: tier 1
  involves no LLM. Library import, not the `rag_agent search` CLI, so it gets `Chunk` objects
  rather than formatted text.
- **Collapse retrieved chunks to distinct `source_url`s, first occurrence winning, before
  scoring.** One function shared by the metrics and `--pool`, so labeling and scoring cannot
  disagree about what a result is. A test must cover several chunks from one document: recall
  stays <= 1.
- Hand-write the five metrics — no new dependency, and the matching rule is `source_url`
  membership, which LlamaIndex's id-based `RetrieverEvaluator` would have to be bent into.
- Report `chunk_precision@k` separately: the fraction of retrieved *chunks* from a relevant
  document, the prompt-waste number the collapse hides.
- Skip entries with an empty `relevant_urls` and print coverage — `x of y entries scored`.
- Breakdowns by `language` and `source_system` — the German-compound question needs the
  language split to be readable.
- Flags: `--k`, `--rerank`, `--ids`, `--golden`, `--pool`.
- **MLflow logging is manual here** (tier 1 does not go through `genai.evaluate()`), best-effort
  exactly like `rag_agent/tracing.py`: an unreachable MLflow never fails an eval run, and an
  empty `MLFLOW_TRACKING_URI` disables it so offline runs stay quiet.

## B7. Tiers 2 and 3 — `agent_eval.py`, replacing RAGAS

- `predict_fn` wraps `RagAgent.run`, resetting `agent.memory` between questions — the golden set
  is independent questions, not a conversation. This exists in the current harness and must
  survive the rewrite.
- Dataset from the golden set: `inputs={"question": ...}`, `expectations` carrying
  `expected_response` (from `reference`) and the refusal expectation.
- Scorers: `RetrievalGroundedness`, `RelevanceToQuery`, `RetrievalRelevance`,
  `RetrievalSufficiency`, `Correctness`, `Guidelines` (citation + refusal rules),
  `ToolCallCorrectness` and `ToolCallEfficiency` (tier 3).
- **Every scorer gets `model=` explicitly** — MLflow's default is `openai:/gpt-4.1-mini`, which
  needs an OpenAI key and is a model nobody chose. From `--judge-model` /
  `RAG_EVAL_JUDGE_MODEL`, default `openrouter:/openai/gpt-5.4`: different family *and* stronger.
  Keep the existing same-model warning and extend it to same-family.
- Reference-dependent scorers drop per entry when `reference` is empty; the summary keeps
  reporting coverage.
- Results land in MLflow natively — no logging code on this side.

**Dependencies**: drop `ragas` and `langchain-community` with both version-ceiling comments;
add `mlflow>=3.15` and `litellm` (the judge transport, not a default MLflow dependency).
`tei_embedding.py` is no longer needed unless a scorer wants embeddings. The production package
is unaffected: `mlflow-skinny` stays the runtime dependency, full `mlflow` the `dev` extra.

## B8. CI

Three tiers, because most of this cannot run per-commit.

1. **Per-commit, no services** — metric math (the worked example; the collapse with duplicate
   `source_url`s; recall <= 1), golden-set schema (unique ids, required fields, negative
   controls flagged, well-formed `relevant_urls`), sweep-file parsing. Ordinary offline pytest.
2. **Nightly, services up, no re-ingest** — the actual regression guard, made possible by a
   **frozen Qdrant snapshot**: build the eval collection once, export it via the snapshot API,
   store it as a CI artifact. CI then restores it, runs tier 1, and diffs against a baseline
   committed as JSON in `evaluation/` so the gate is reviewable in a diff. Gate on a threshold
   (`recall@10 must not drop more than 0.02`). Minutes, no LLM, no API key.
   **Build the snapshot first — without it every CI option collapses back into a six-stage
   ingest.**
3. **Manual or scheduled campaign** — the full sweep and tiers 2/3. Hours plus real spend, never
   on push. Its output is a `parameters.yaml` candidate to bless, not a pass/fail.

**Cross-repo triggering**: the commit that breaks retrieval lands in `agent/` or `embed/`, not
in `evaluation/`. Start with a **nightly checkout** — `evaluation`'s CI checks out all six
siblings at `main` and runs tier 1 — rather than `repository_dispatch` from six repos. One
workflow file in one repo; attribution costs one bisect, which is cheaper than six coupled
pipelines until several people are committing.

## B9. Documentation

- `evaluation/EVALUATION.md` — Part A above.
- `evaluation/README.md` — how to run: the sweep, tier 1, `--pool`, tiers 2/3.
- `masterplan/ARCHITECTURE.md` block 12 — record the decisions with their reasons: **RAGAS
  dropped for MLflow's built-in scorers** (one evaluation stack, no version ceilings, native
  logging, tool-call and online scorers included); the judge routes through OpenRouter via
  litellm so one key serves agent and judge; the judge is different family *and* stronger; tier 1
  is a separate judge-free harness (cost, determinism, retrieval-vs-generation diagnosis);
  relevance labels key on `source_url` rather than chunk id (labels survive re-chunking);
  collections are named by an ingest fingerprint; parallel `eval_*` tables rejected in favour of
  a separate database. Update the stack summary row from "MLflow + RAGAS" to MLflow alone.
- `CLAUDE.md` (root) — add `evaluation/` to the layout list.
- `agent/CLAUDE.md` — the `RETRIEVER` span and its output-shape contract, since B0 lives there.

## Verification

```bash
cd evaluation && uv sync

# tier 1 — needs qdrant + tei, no LLM, no API key
.venv/bin/python -m rag_evaluation retrieval --k 5
.venv/bin/python -m rag_evaluation pool
.venv/bin/python -m rag_evaluation retrieval --k 5 --rerank

# tiers 2 + 3 — needs OPENROUTER_API_KEY
.venv/bin/python -m rag_evaluation agent --ids isin-lookup

# a two-point ingest sweep
.venv/bin/python -m rag_evaluation sweep --file sweeps/chunk_size.yaml
```

1. **The `RETRIEVER` span (B0) — check first, everything in tier 2 depends on it.** Run one
   question, then `mlflow.get_trace(...)`: a span of type `RETRIEVER` must exist and
   `extract_retrieval_context_from_trace(trace)` must return a non-empty dict. Then
   `RetrievalGroundedness(model=...)(trace=trace)` must return a value rather than raise. Also
   run a question that triggers **two** searches and confirm two feedbacks.
2. **Metric correctness** — offline test with the worked example (`R={A,B}`, `[X,A,Y,B,Z]`)
   asserting 1, 1.0, 0.4, 0.5, 0.65; plus a case where three chunks share one `source_url`,
   asserting the collapse and recall <= 1.
3. **Coverage honesty** — with no `relevant_urls` anywhere, tier 1 reports `0 of 15 scored` and
   exits cleanly rather than printing a meaningless 0.0.
4. **The safety guard** — running without `RAG_PARAMETERS_PATH`, or with the production
   collection or DSN, must refuse to start.
5. **The fingerprint** — changing `top_k` in a candidate leaves the fingerprint unchanged (the
   collection is reused); changing `chunk_size` changes it (a fresh collection is built).
   Assert both in an offline test over `rag_config.ingest_fingerprint`.
6. **Corpus freezing** — two sweep configurations must score the same manifest. Assert the
   snapshot is replayed rather than re-discovered.
7. **Label a handful by hand** (3-5 entries via `pool`) and confirm tier-1 scores match manual
   inspection.
8. **MLflow** — the `RAG-Evaluation` experiment shows the parent run with one child per configuration,
   all params present, per-judgement rationales visible. Then unset `MLFLOW_TRACKING_URI`
   and confirm tier 1 stays silent and still succeeds.
9. **The first real ablation** — same labeled entries, `--rerank` on and off, nDCG delta
   reported, both runs diffable in the MLflow UI.
