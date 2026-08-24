# groundtruth

[![CI](https://github.com/VSSOO7/groundtruth/actions/workflows/ci.yml/badge.svg)](https://github.com/VSSOO7/groundtruth/actions)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PostgreSQL pgvector](https://img.shields.io/badge/Postgres_16-pgvector-336791.svg)
![XGBoost LambdaMART](https://img.shields.io/badge/Reranker-XGBoost_LambdaMART-orange.svg)
![Demo](https://img.shields.io/badge/Demo-Interactive_Streamlit_App-2563eb.svg)

**Production RAG over SEC filings — with a *trained* reranker and an evaluation harness that gates CI.**

Ask a question about a 10-K. Get an answer where **every claim carries a citation to
the exact chunk that supports it**, or an explicit refusal when the retrieved
evidence doesn't support an answer. A confident wrong answer about a company's
financials is worse than no answer, so the system is built to refuse.

---

## 🚀 Interactive Showcase App

Run the interactive web dashboard with a single command:

```bash
make demo
```

The interactive application includes:
- **🔍 Financial Analyst Q&A**: Real-time query execution over SEC 10-K filings with live confidence gauges, claim-by-claim citation verification, and timing waterfalls (`Embed` ➔ `Hybrid Search` ➔ `XGBoost Rerank` ➔ `Generate`).
- **⚡ Hybrid Retrieval & XGBoost Visualizer**: Side-by-side comparison of Sparse BM25 vs Dense HNSW vs RRF vs learned LambdaMART scores, with interactive 15-feature vector inspection.
- **📊 CI Benchmark & Eval Gate**: Live nDCG@10 evaluator and regression gate simulator.
- **📁 SEC Filing Explorer**: Browse indexed 10-K sections and metadata.

---

### Example API Query

```bash
curl -s localhost:8000/query -H 'content-type: application/json' -d '{
  "question": "What goodwill impairment did the company record and what drove it?",
  "cik": "0000000001"
}' | jq
```

```json
{
  "refused": false,
  "answer": "A goodwill impairment charge of $42.0 million was recorded in the fourth quarter of fiscal 2024, related to the Freight Brokerage reporting unit. The impairment resulted from a sustained decline in spot market rates and a corresponding reduction in the unit's projected cash flows.",
  "confidence": 0.91,
  "claims": [
    { "text": "Recorded a $42.0 million goodwill impairment in Q4 FY2024.", "chunk_ids": [7] },
    { "text": "Driven by sustained decline in spot market rates.", "chunk_ids": [7] }
  ],
  "citations": [
    { "chunk_id": 7, "company_name": "Northwind Logistics Inc.", "fiscal_year": 2024,
      "section_name": "Financial Statements", "source_url": "https://..." }
  ],
  "timings_ms": { "embed": 11.4, "retrieve": 23.8, "rerank": 4.1, "generate": 1840.2, "total": 1879.5 },
  "reranker_active": true
}
```

---

## Why this repo exists

Most RAG projects stop at "embed, cosine similarity, stuff into a prompt." That
demo works and tells you nothing about whether it works *well*. Two things here
are deliberately different, and they're the two that turn retrieval from a demo
into an engineering discipline:

**1. The reranker is trained, not downloaded.** Retrieval is a ranking problem, so
it's solved with a ranking model: an XGBoost LambdaMART ranker (`rank:ndcg`) over
15 hand-designed features. It learns signals specific to filings — exact-phrase and
numeric overlap matter far more in a 10-K than in web text — that a generic
cross-encoder never sees. Feature importances double as an interpretable ablation.

**2. Quality is measured, and the measurement blocks merges.** nDCG@10, Recall@50,
and MRR run against a graded golden set on every pull request. **If nDCG@10 drops
more than 2%, CI fails and the PR cannot merge.** That single gate is what makes
every other number in this README trustworthy.

---

## Architecture

```
                    ┌──────────────────────── ingestion (offline) ────────────────────────┐
   SEC EDGAR ──────▶│ rate-limited fetch → HTML→text → section-aware chunk → bge embed    │
   (10-K / 10-Q)    │                            ↓                                        │
                    │              COPY into a NEW corpus_snapshots row                   │
                    │              ↓ atomic is_active flip (blue/green)                   │
                    └────────────────────────────┬────────────────────────────────────────┘
                                                 ▼
                              ┌──────── Postgres 16 + pgvector ────────┐
                              │  chunks.embedding  vector(768)  HNSW   │
                              │  chunks.tsv        tsvector     GIN    │
                              │  documents         cik/year/section    │
                              └────────────────────┬───────────────────┘
                                                   │  ONE SQL statement
   query ──▶ bge embed ──▶ ┌────────────────────────▼────────────────────────┐
                           │ filter (cik/year/section) INSIDE the ANN scan   │
                           │   ├─ dense:  embedding <=> qvec    (top 100)    │
                           │   └─ sparse: ts_rank_cd(tsv, ...)  (top 100)    │
                           │        └─ Reciprocal Rank Fusion, k=60          │
                           └────────────────────────┬────────────────────────┘
                                                    ▼
                           ┌─── XGBoost LambdaMART reranker (15 features) ───┐
                           │  falls back to RRF order if no model artifact   │
                           └────────────────────────┬────────────────────────┘
                                                    ▼  dedupe ≤3/doc, top 8
                            ┌── LLM Generator (structured output + caching) ──┐
                            │  grounding contract: every claim cites ≥1 chunk │
                            │  hallucinated chunk_ids stripped → refuse       │
                            └────────────────────────┬────────────────────────┘
                                                    ▼
                                   FastAPI  /query  /healthz  /readyz  /metrics
                                            └─▶ Prometheus → Grafana
```

### Why Postgres instead of a dedicated vector DB

This is the design decision most worth defending, and it isn't "one less service
to run."

Metadata filters — *this company, this fiscal year, Item 1A only* — are applied
**inside** the ANN scan, in the same SQL statement. Bolt a vector database onto a
relational store and you're forced to **post-filter**: fetch the top-*k* by vector
similarity, then throw away rows that fail the filter. When the filter is
selective (one company out of 500), most of your *k* evaporates and recall
collapses — silently, because you still get results back, just worse ones.

Keeping vectors, full-text, and metadata in one store means dense ANN, BM25-style
FTS, and relational filtering happen in a single round trip, with the filter
pushed down. See [`retrieval/hybrid.py`](src/groundtruth/retrieval/hybrid.py).

### Blue/green index rebuilds

Re-chunking or swapping the embedding model means rebuilding the whole index.
Doing that in place would serve queries from a half-populated index. Instead every
chunk belongs to a `corpus_snapshots` row; a new snapshot is built while the old
one serves, then flipped `is_active` in one transaction. A partial-unique index
enforces at most one active snapshot, and a `CHECK` prevents activating a snapshot
that never finished building. Rollback is another flip.

---

## The two differentiators, in detail

### The learned reranker

Candidate generation is recall-oriented (100 candidates); reranking is
precision-oriented. The ranker scores every candidate and reorders — so a chunk
that RRF buried at #40 can still reach the top, which is the entire point of a
second stage.

The 15 features ([`retrieval/features.py`](src/groundtruth/retrieval/features.py)):

| Group | Features |
|---|---|
| Retriever signals | `dense_score`, `dense_rank_recip`, `sparse_score`, `sparse_rank_recip`, `rrf_score`, `both_retrievers_hit` |
| Optional stacking | `cross_encoder_score` — a cross-encoder is used as a *feature*, never as the final ranker |
| Lexical | `query_term_overlap`, `query_coverage`, `exact_phrase_hit`, `numeric_overlap` |
| Structural | `token_count_log`, `section_is_risk`, `section_is_mdna`, `section_is_financials` |

Three failure modes this code is built to avoid, each of which produces
*plausible-looking* numbers:

- **Query leakage across the split.** Ranking data has one group per query. The
  train/val split is by **query**, never by row — otherwise chunks from the same
  query land on both sides and the model memorizes rather than ranks. This is the
  single most common way LTR benchmarks get silently inflated.
- **Train/serve feature skew.** Exactly one function turns a candidate into a
  vector, imported by both the online path and the training script. Skew here is
  invisible: results still come back ranked, just worse.
- **Silent schema drift.** A `.features.json` sidecar pins feature order next to
  the model. If the trained layout no longer matches the code, loading **raises**
  rather than serving a skewed model.

### The eval harness

| Component | What it does |
|---|---|
| `eval/metrics.py` | nDCG (exponential gain), Recall@k, MRR, hit-rate, Cohen's κ — hand-written, because the CI gate depends on them and frameworks disagree on the gain convention |
| `eval/run_eval.py` | Scores the active snapshot, persists config + git SHA + metrics to `eval_runs` |
| `eval/judge.py` | LLM-as-judge for answer faithfulness / relevance / completeness |
| `eval/agreement.py` | **Cohen's κ between machine and human labels** |
| `eval/gate.py` | Fails CI on regression |
| `eval/fixture_corpus.json` | Hermetic synthetic corpus so CI needs no network and no API key |

Four choices that keep the numbers honest:

- **Eval and serving share one code path.** `QueryPipeline.run(generate=False)` is
  what the harness calls. If they diverged, every number here would describe
  something users never receive.
- **Retrieval is scored without generation.** nDCG measures ranking; a generation
  call per query would add cost and variance while measuring nothing about it.
- **`ndcg_at_k` builds the ideal ranking from *all* labeled items**, not just the
  retrieved ones — so a relevant chunk the retriever *missed* still counts against
  the score. Grading only what you retrieved is self-congratulatory.
- **The LLM judge is held accountable.** Labels are bootstrapped cheaply and a
  slice is hand-verified; `eval_labels` keeps both grades for the same pair
  (`label_source` is part of the primary key) so κ is computable. An unvalidated
  LLM judge is just a second opinion you happen to trust. Below "moderate"
  agreement, the human slice carries the decision — and the harness reports that
  slice separately, always.

---

## Results

> Reproduce with `make ablation`. Numbers below are the shape of the report the
> harness emits; fill them from your own run — this repo does not ship metrics it
> did not measure on your corpus.

| Configuration | nDCG@10 | Recall@50 | MRR |
|---|---|---|---|
| Dense only (HNSW cosine) | — | — | — |
| Sparse only (Postgres FTS) | — | — | — |
| Hybrid + RRF *(the baseline the reranker must beat)* | — | — | — |
| Hybrid + RRF + **learned reranker** | — | — | — |

`make train` prints the held-out nDCG@10 for the reranker **and** the RRF baseline
with the lift between them, so a reranker that isn't earning its latency is
obvious immediately.

---

## Quickstart

```bash
git clone https://github.com/VSSOO7/groundtruth.git && cd groundtruth
make install
make env          # then set SEC_USER_AGENT and ANTHROPIC_API_KEY in .env
make up           # Postgres + API + Prometheus + Grafana
```

`SEC_USER_AGENT` must be a real contact string (`groundtruth/0.1 (you@domain.com)`).
EDGAR returns **403** without one — the client fails fast with that message rather
than letting you debug an opaque error.

### Path A — the hermetic fixture (no network, no API key)

```bash
make fixture      # load the synthetic corpus as an active snapshot
make eval         # nDCG@10, Recall@50, MRR
make gate         # compare against the committed baseline
```

### Path B — real filings

```bash
make ingest       # SEC 10-Ks -> chunks -> embeddings -> pgvector (blue/green)
make labels       # bootstrap golden queries + graded labels
make trainset     # build the reranker training set
make train        # LambdaMART; prints nDCG lift over the RRF baseline
make eval-baseline  # pin the measured baseline so the CI gate is meaningful
make serve
```

The API serves **before** any of the training steps: with no model artifact the
reranker logs a warning and falls back to RRF order. Ranking quality degrades;
availability does not.

---

## Production concerns this repo actually handles

| Concern | Approach |
|---|---|
| **Liveness vs readiness** | `/healthz` never touches the DB — a database blip must not make the orchestrator kill every replica. `/readyz` requires the DB *and* an active snapshot. Getting these backwards is a classic self-inflicted outage. |
| **Zero-downtime reindex** | `corpus_snapshots` blue/green with an atomic flip and one-line rollback |
| **Graceful degradation** | Missing reranker → RRF fallback; missing API key → retrieval-only mode |
| **Hallucinated citations** | Chunk IDs the model invented are stripped; if nothing survives, the answer is demoted to a refusal |
| **Prompt caching** | The invariant system block carries `cache_control`; volatile context sits in `messages`, after the breakpoint. `cache_read` is logged so a silent cache invalidator shows up. |
| **Cost control** | Cheap model for bulk labeling, strong model for generation and judging |
| **Observability** | Per-stage latency (embed / retrieve / rerank / generate) as Prometheus histograms, provisioned Grafana dashboard |
| **Rate limits** | EDGAR limiter (~10 req/s) so a bulk ingest can't get the deployment IP-banned |
| **Secrets** | `.gitignore` blocks `.env`, `*.pem`, `*.key`, `secrets/`; nothing but `.env.example` is tracked |
| **Container hygiene** | Multi-stage build, non-root uid 10001, `HEALTHCHECK` on `/healthz` |
| **Reproducibility** | Every eval run persists its config + git SHA, so any metric traces back to what produced it |

---

## Repo layout

```
db/schema.sql                  storage design: HNSW + GIN + snapshots + eval tables
src/groundtruth/
  config.py                    typed settings; every quality knob is serialized into eval_runs
  embedding.py                 bge-base-en-v1.5, asymmetric query instruction prefix
  pipeline.py                  the ONE path shared by eval and serving
  ingestion/
    edgar.py                   rate-limited SEC client (fails fast on a bad User-Agent)
    chunker.py                 section-aware 10-K Item splitting, TOC-artifact defense
    ingest.py                  blue/green ingest with COPY bulk load
  retrieval/
    types.py                   dependency-free Candidate/Filters so features are testable
    hybrid.py                  single-statement dense + sparse + RRF with pushed-down filters
    features.py               ★ the 15-feature contract (order is the train/serve contract)
    reranker.py               ★ LambdaMART scoring, sidecar schema check, RRF fallback
  generation/
    schema.py                  GroundedAnswer — a validator rejects uncited claims
    generator.py               structured output, prompt caching, citation verification
  training/
    build_labels.py            LLM label bootstrapping (cheap model)
    build_training_set.py      labels + live retrieval -> training JSONL
    train_reranker.py         ★ grouped split, rank:ndcg, feature sidecar
  eval/
    metrics.py                ★ nDCG / Recall / MRR / Cohen's κ
    run_eval.py               ★ the harness
    judge.py                   LLM-as-judge
    agreement.py               κ between machine and human labels
    gate.py                   ⚡ the CI regression gate
    fixture.py                 hermetic fixture loader
  api/main.py                  FastAPI: liveness vs readiness, metrics, /query
.github/workflows/ci.yml       ruff → mypy → pytest → eval gate → image build
```

★ = the two differentiators  ⚡ = the gate that makes the metrics load-bearing

---

## Tests

```bash
make test     # 93 tests, no DB and no API key required
```

The suite targets the failures that *look fine*: a grouped split that leaks a
query, a chunker that mistakes a table-of-contents line for a section heading, an
answer that cites a chunk that was never retrieved, a regression gate that flips on
floating-point noise at its own boundary, and a fixture whose labels reference a
passage id that doesn't exist.

## License

MIT
