-- groundtruth schema: dense vectors, BM25 full-text, and relational metadata in ONE store.
-- Design note: keeping all three in Postgres means retrieval is a single SQL round-trip and
-- metadata filters are applied *before* the vector scan, not after. A separate vector DB would
-- force post-filtering, which silently degrades recall@k when filters are selective.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- Index versioning: enables blue/green rebuilds. A snapshot is only readable
-- once `built_at` is set, so queries never observe a half-populated index.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corpus_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    chunker_version  TEXT        NOT NULL,
    embedder_version TEXT        NOT NULL,
    embedding_dim    INT         NOT NULL,
    notes            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    built_at         TIMESTAMPTZ,               -- NULL => still building
    is_active        BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT active_requires_built CHECK (NOT is_active OR built_at IS NOT NULL)
);

-- At most one active snapshot at a time.
CREATE UNIQUE INDEX IF NOT EXISTS one_active_snapshot
    ON corpus_snapshots ((TRUE)) WHERE is_active;

-- ---------------------------------------------------------------------------
-- Source documents (one row per SEC filing)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    accession_no  TEXT NOT NULL UNIQUE,         -- SEC's stable filing identifier
    cik           TEXT NOT NULL,
    company_name  TEXT NOT NULL,
    ticker        TEXT,
    form_type     TEXT NOT NULL,                -- 10-K, 10-Q, ...
    fiscal_year   INT  NOT NULL,
    filed_date    DATE NOT NULL,
    source_url    TEXT NOT NULL,
    raw_sha256    TEXT NOT NULL,                -- detects upstream restatements
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_cik_year ON documents (cik, fiscal_year DESC);
CREATE INDEX IF NOT EXISTS documents_company_trgm ON documents USING gin (company_name gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- Chunks: the retrieval unit. Partitioned logically by snapshot_id so an old
-- index stays queryable while a new one builds.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    snapshot_id  BIGINT NOT NULL REFERENCES corpus_snapshots(id) ON DELETE CASCADE,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Section-aware chunking: 10-K item headings are the natural semantic boundary.
    item_section TEXT,                          -- '1', '1A', '7', '7A', '8', ...
    section_name TEXT,                          -- 'Risk Factors', 'MD&A', ...
    ordinal      INT    NOT NULL,               -- position within the document
    char_start   INT    NOT NULL,               -- provenance back into raw text
    char_end     INT    NOT NULL,

    text         TEXT   NOT NULL,
    token_count  INT    NOT NULL,
    embedding    vector(768) NOT NULL,

    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,

    UNIQUE (snapshot_id, document_id, ordinal)
);

-- Dense ANN. HNSW over cosine: better recall/latency than IVFFlat and no training step.
-- m/ef_construction tuned for ~1M chunks; raise ef_search at query time for more recall.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Sparse BM25-ish ranking via Postgres FTS.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Supports pre-filtered retrieval (by company / year / section).
CREATE INDEX IF NOT EXISTS chunks_snapshot_doc ON chunks (snapshot_id, document_id);
CREATE INDEX IF NOT EXISTS chunks_section ON chunks (snapshot_id, item_section);

-- ---------------------------------------------------------------------------
-- Evaluation: golden queries + per-run metrics. Persisting runs is what makes
-- the CI regression gate and the README ablation table possible.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_queries (
    id           BIGSERIAL PRIMARY KEY,
    query        TEXT NOT NULL,
    -- Provenance of the *query text*: was the question itself vetted by a human?
    label_source TEXT NOT NULL CHECK (label_source IN ('llm_bootstrap', 'human_verified')),
    difficulty   TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Graded relevance, one row per (query, chunk, grader).
--
-- `label_source` is part of the primary key on purpose. If a pair could hold only
-- one grade, a human re-grade would overwrite the machine grade and the two could
-- never be compared -- which would make the judge/human agreement check
-- (eval/metrics.cohens_kappa) impossible to compute. Keeping both lets us quantify
-- exactly how far the cheap labeler can be trusted.
CREATE TABLE IF NOT EXISTS eval_labels (
    query_id     BIGINT NOT NULL REFERENCES eval_queries(id) ON DELETE CASCADE,
    chunk_id     BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    relevance    INT    NOT NULL CHECK (relevance BETWEEN 0 AND 3),  -- graded, for nDCG
    label_source TEXT   NOT NULL CHECK (label_source IN ('llm_bootstrap', 'human_verified')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (query_id, chunk_id, label_source)
);

-- Fast lookup of the trusted slice, and of all grades for one query.
CREATE INDEX IF NOT EXISTS eval_labels_by_source ON eval_labels (label_source, query_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id           BIGSERIAL PRIMARY KEY,
    snapshot_id  BIGINT REFERENCES corpus_snapshots(id) ON DELETE SET NULL,
    git_sha      TEXT NOT NULL,
    config       JSONB NOT NULL,                -- retrieval + rerank + model settings
    metrics      JSONB NOT NULL,                -- {ndcg@10, recall@50, mrr, faithfulness, ...}
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS eval_runs_recent ON eval_runs (created_at DESC);

-- ---------------------------------------------------------------------------
-- Production feedback -> future reranker training data (the flywheel)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback (
    id           BIGSERIAL PRIMARY KEY,
    request_id   TEXT NOT NULL,
    query        TEXT NOT NULL,
    served_chunk_ids BIGINT[] NOT NULL,
    verdict      TEXT NOT NULL CHECK (verdict IN ('helpful', 'unhelpful', 'wrong')),
    comment      TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
