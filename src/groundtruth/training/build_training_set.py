"""Materialize the reranker training set from golden labels + live retrieval.

Run: `python -m groundtruth.training.build_training_set --out data/training/reranker.jsonl`

This is the bridge between the labels in Postgres and the model in `models/`. For
every golden query it runs the *real* hybrid retriever, extracts features through
the *same* `features.py` the online path uses, and joins each candidate to its
graded relevance. The output is the JSONL that `train_reranker.load_dataset()`
expects: one line per (query, candidate) with `qid`, `chunk_id`, `relevance`,
`features`.

Two decisions that keep the trained model honest:

* **Features come from the one shared module, never recomputed here.** Any skew
  between train-time and serve-time features silently degrades ranking, so there
  is exactly one code path that turns a candidate into a vector.

* **Retrieved-but-unlabeled candidates become relevance-0 negatives.** A ranker
  learns from contrast. If we kept only the labeled positives, XGBoost would see
  almost no negatives and learn nothing about what to push *down*. Capping the
  number of negatives per query keeps the groups balanced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import structlog
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from groundtruth.config import Settings, get_settings
from groundtruth.embedding import SentenceTransformerEmbedder
from groundtruth.retrieval.features import FeatureContext, extract_features
from groundtruth.retrieval.hybrid import retrieve_candidates

log = structlog.get_logger(__name__)


def load_labeled_queries(conn: Connection) -> list[dict[str, Any]]:
    """Golden queries that have at least one graded label.

    DISTINCT ON collapses a pair graded by both the machine and a human down to
    the human grade -- training on both rows would duplicate the pair and let the
    noisier label pull the model in the opposite direction.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """WITH best AS (
                   SELECT DISTINCT ON (query_id, chunk_id)
                          query_id, chunk_id, relevance
                     FROM eval_labels
                    ORDER BY query_id, chunk_id,
                             (label_source = 'human_verified') DESC
               )
               SELECT q.id, q.query,
                      jsonb_object_agg(b.chunk_id::text, b.relevance) AS labels
                 FROM eval_queries q
                 JOIN best b ON b.query_id = q.id
                GROUP BY q.id, q.query
                ORDER BY q.id"""
        )
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "query": r["query"],
            "labels": {int(k): int(v) for k, v in r["labels"].items()},
        }
        for r in rows
    ]


def active_snapshot(conn: Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM corpus_snapshots WHERE is_active LIMIT 1")
        row = cur.fetchone()
    if not row:
        raise SystemExit("No active corpus snapshot. Ingest with --activate first.")
    return int(row[0])


def build_rows(
    pool: ConnectionPool,
    settings: Settings,
    embedder: SentenceTransformerEmbedder,
    queries: list[dict[str, Any]],
    snapshot_id: int,
    *,
    max_negatives: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in queries:
        qvec = embedder.embed_query(q["query"])
        candidates = retrieve_candidates(
            pool,
            settings,
            query_text=q["query"],
            query_vector=qvec,
            snapshot_id=snapshot_id,
            filters=None,
        )
        ctx = FeatureContext.build(q["query"])
        labels = q["labels"]

        negatives = 0
        emitted = 0
        for cand in candidates:
            relevance = labels.get(cand.chunk_id, 0)
            # Balance groups: keep every labeled candidate, but cap the unlabeled
            # relevance-0 negatives so a single query can't swamp the objective.
            if relevance == 0:
                if negatives >= max_negatives:
                    continue
                negatives += 1
            rows.append(
                {
                    "qid": q["id"],
                    "chunk_id": cand.chunk_id,
                    "relevance": relevance,
                    "features": extract_features(cand, ctx),
                }
            )
            emitted += 1

        # A query whose labeled positives never surfaced in retrieval teaches the
        # ranker nothing (no positive in the group) -- flag it rather than hide it.
        retrieved_positives = sum(1 for c in candidates if labels.get(c.chunk_id, 0) > 0)
        if retrieved_positives == 0:
            log.warning(
                "build.query_without_retrieved_positive",
                qid=q["id"],
                labeled_positives=sum(1 for v in labels.values() if v > 0),
                hint="raise candidate_k or check the query -- its positives were not retrieved",
            )
        log.info(
            "build.query",
            qid=q["id"],
            candidates=len(candidates),
            emitted=emitted,
            positives=retrieved_positives,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the reranker training JSONL")
    ap.add_argument("--out", type=Path, default=Path("data/training/reranker.jsonl"))
    ap.add_argument(
        "--max-negatives", type=int, default=60, help="max relevance-0 candidates kept per query"
    )
    args = ap.parse_args()

    settings = get_settings()
    conn = Connection.connect(settings.database_url, autocommit=True)
    register_vector(conn)

    snapshot_id = active_snapshot(conn)
    queries = load_labeled_queries(conn)
    if not queries:
        raise SystemExit("No labeled queries. Run build_labels first.")

    pool = ConnectionPool(settings.database_url, min_size=1, max_size=4, open=True)
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)

    rows = build_rows(
        pool,
        settings,
        embedder,
        queries,
        snapshot_id,
        max_negatives=args.max_negatives,
    )

    pool.close()
    conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    n_queries = len({r["qid"] for r in rows})
    n_pos = sum(1 for r in rows if r["relevance"] > 0)
    log.info(
        "build.complete",
        out=str(args.out),
        rows=len(rows),
        queries=n_queries,
        positives=n_pos,
        negatives=len(rows) - n_pos,
    )
    print(
        f"Wrote {len(rows)} rows ({n_pos} positive, {len(rows) - n_pos} negative) "
        f"across {n_queries} queries -> {args.out}"
    )


if __name__ == "__main__":
    main()
