"""Stage 1 of retrieval: hybrid candidate generation with Reciprocal Rank Fusion.

Why hybrid: dense embeddings miss exact-match signals that matter enormously in
filings (ticker symbols, "Item 1A", statute names, dollar figures). BM25 catches
those but misses paraphrase. RRF fuses them without needing score calibration
between two incomparable scales -- it only uses *ranks*.

Why one SQL statement: metadata filters (company, year, section) are applied
inside the ANN scan rather than after it. Post-filtering a fixed top-k, which is
what a bolt-on vector DB forces, quietly destroys recall when filters are
selective -- ask for one company and you may get zero of its chunks back.
"""

from __future__ import annotations

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from groundtruth.config import Settings
from groundtruth.retrieval.types import Candidate, Filters

__all__ = ["Candidate", "Filters", "retrieve_candidates"]

# `<=>` is pgvector's cosine distance. ts_rank_cd is Postgres' cover-density
# ranking -- a BM25 approximation that accounts for term proximity.
_HYBRID_SQL = """
WITH filtered AS (
    SELECT c.id, c.embedding, c.tsv
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.snapshot_id = %(snapshot_id)s
      AND (%(cik)s::text        IS NULL OR d.cik = %(cik)s)
      AND (%(fiscal_year)s::int IS NULL OR d.fiscal_year = %(fiscal_year)s)
      AND (%(sections)s::text[] IS NULL OR c.item_section = ANY(%(sections)s))
),
dense AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(qvec)s) AS rank,
           1 - (embedding <=> %(qvec)s)                        AS score
    FROM filtered
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(k)s
),
sparse AS (
    SELECT f.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(f.tsv, q.query) DESC) AS rank,
           ts_rank_cd(f.tsv, q.query)                                   AS score
    FROM filtered f,
         websearch_to_tsquery('english', %(qtext)s) AS q(query)
    WHERE f.tsv @@ q.query
    ORDER BY ts_rank_cd(f.tsv, q.query) DESC
    LIMIT %(k)s
),
fused AS (
    SELECT COALESCE(dn.id, sp.id)                            AS chunk_id,
           COALESCE(1.0 / (%(rrf_k)s + dn.rank), 0.0)
             + COALESCE(1.0 / (%(rrf_k)s + sp.rank), 0.0)    AS rrf_score,
           dn.score  AS dense_score,
           dn.rank   AS dense_rank,
           sp.score  AS sparse_score,
           sp.rank   AS sparse_rank
    FROM dense dn
    FULL OUTER JOIN sparse sp ON sp.id = dn.id
)
SELECT f.chunk_id, f.rrf_score, f.dense_score, f.dense_rank,
       f.sparse_score, f.sparse_rank,
       c.text, c.token_count, c.item_section, c.section_name,
       c.char_start, c.char_end,
       d.company_name, d.ticker, d.fiscal_year, d.form_type,
       d.accession_no, d.source_url
FROM fused f
JOIN chunks c    ON c.id = f.chunk_id
JOIN documents d ON d.id = c.document_id
ORDER BY f.rrf_score DESC
LIMIT %(k)s;
"""


def retrieve_candidates(
    pool: ConnectionPool,
    settings: Settings,
    *,
    query_text: str,
    query_vector: list[float],
    snapshot_id: int,
    filters: Filters | None = None,
) -> list[Candidate]:
    """Return up to `settings.candidate_k` fused candidates, best RRF score first."""
    f = filters or Filters()
    params = {
        "snapshot_id": snapshot_id,
        "qvec": str(query_vector),  # pgvector accepts the '[1,2,3]' text form
        "qtext": query_text,
        "k": settings.candidate_k,
        "rrf_k": settings.rrf_k,
        "cik": f.cik,
        "fiscal_year": f.fiscal_year,
        "sections": f.sections,
    }

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        # Session-scoped: trades latency for recall on the HNSW graph walk.
        cur.execute("SET LOCAL hnsw.ef_search = %s", (settings.hnsw_ef_search,))
        cur.execute(_HYBRID_SQL, params)
        rows = cur.fetchall()

    return [Candidate(**row) for row in rows]
