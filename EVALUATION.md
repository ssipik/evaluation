# Evaluation

How this pipeline is measured and tuned: the golden set, the scorers, and the
loop that turns a parameter change into a number you can compare.

`ARCHITECTURE.md` block 12 is the authority on *why* the stack is MLflow; this
file is the procedure. `README.md` next to it is how to run the commands.

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
| `relevant_documents` | tier 1 | the documents that must be retrieved, each a `source_system` + `url` pair |
| `expect_refusal` | tier 3 | true for negative controls |

`reference` is written by hand on purpose: a reference generated from the same retrieval the
system uses measures self-consistency, not correctness. Synthetic generation is fine for the
*questions*, never for the references. `generate` therefore leaves `reference` empty and puts
the answer it proposed in a comment — a human moves it into the field or rewrites it.

**Two ways to get entries, and they meet in the middle.**

*Question first — pooling.* Do not read the corpus. Run the system at a high `k`
(20) under a few configurations, take the union of what came back, and label those by hand.
The method is called **pooling**, and it is how TREC builds its relevance judgments: the
top-ranked documents from each run are unioned into a pool, and only the pool is judged
(Voorhees, *Overview of TREC 2002*, <https://trec.nist.gov/pubs/trec11/papers/OVERVIEW.11.pdf>
— top 100 per run, average pool 769 documents).

*Document first — generation* (`generate`, added 2026-08-20). Sample documents from the
ledger, hand each one to an LLM, and ask for a summary plus a question one of its points
answers. The label falls out for free: the question was written from that document, so
`relevant_documents` is exact and nothing needs pooling. The judgement moves rather than
disappearing — what a human now checks is whether the *question* is fair, because a model
left alone writes "What does the document say about liquidity?", which is unanswerable once
the document is out of view, and "What did the ECB announce?", which matches fifty documents.
The prompt in `generate.py` rules both out; the review step catches what slips through.

The two methods cover different gaps. Pooling finds the documents a question you care about
should have returned. Generation gives breadth cheaply — every corner of the corpus gets an
entry, including the documents nobody would think to ask about. Use generation to reach the
~50-100 target, pooling to add the labels a generated question missed (a question written from
document A is often also answered by document B, and only pooling finds B).

A third model family writes them: the agent answers, a second family judges, a third asks.
Same self-preference reasoning as the judge — a question written by the family that later
answers or grades it flatters both.

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

Compares the retrieved chunks against `relevant_documents` by **exact identifier match** —
set membership, nothing semantic. That is what makes it deterministic and free.

Matching is on `source_system` + `url`, not chunk id: stage 5 rewrites `manifest_id` in place
and chunk ids regenerate on every re-chunk, while both halves of that pair are stable — the
same reason `fetch_document` keys on the url. Both halves, because the url alone does not
identify a document: two sources can list the same url, and scoring them as one would credit a
hit on the wrong source. The cost is granularity (you score document retrieval, not paragraph
retrieval), and the gain is that labels survive the chunk-size experiment you most want to
run.

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
*chunks*; the golden set labels *documents* (`relevant_documents`). Comparing the two directly
would let three chunks of the same document count as three hits and push recall above 1. So
before any formula runs, the retrieved chunk list is **collapsed to a ranked list of distinct
(`source_system`, `url`) pairs, first occurrence winning**:

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

The context comes from spans typed `RETRIEVER` in the trace, and **this repo emits none
today** — see B0, which must be done first.

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

**A kept configuration is versioned in the MLflow registry, not in a note.** Every tier-1 run
logs a parameter carrier — a placeholder model whose only content is the run's own params —
so the run can be registered under a name and given an alias. That version is then the
address of that configuration: `curve --mlflow-model name@alias` scores against it, and
`promote --mlflow-model name@alias` writes it into the `parameters.yaml` `rag-config` ships,
stamping the address in `version:`. Committing that file is the last step, and the only
manual one.

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

- **The golden set does not change**, because `relevant_documents` labels documents, not chunks.
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
