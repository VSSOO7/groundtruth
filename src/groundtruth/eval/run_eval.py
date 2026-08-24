"""Eval harness: score a retrieval config against the golden set and persist the run.

Run: `python -m groundtruth.eval.run_eval --tag "rerank-v2"`

Design decisions that make these numbers trustworthy:

* **Retrieval is scored without calling the generator.** nDCG measures ranking, and
  a generation call per query would add cost and variance while measuring nothing
  about ranking. Generation quality is scored separately, on a smaller slice.

* **Every run is persisted with its config and git SHA.** A metric you can't trace
  back to the exact chunker, embedder, and reranker that produced it is an anecdote.
  This is what makes the ablation table in the README reproducible.

* **The human-verified slice is reported separately.** LLM-bootstrapped labels are
  plentiful but noisy; the human slice is small but trusted. Reporting only the
  blended number hides whether a gain is real or an artifact of judge bias.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

import structlog
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from groundtruth.config import get_settings
from groundtruth.embedding import SentenceTransformerEmbedder
from groundtruth.eval.metrics import hit_rate_at_k, mrr, ndcg_at_k, recall_at_k
from groundtruth.pipeline import QueryPipeline
from groundtruth.retrieval.reranker import Reranker

log = structlog.get_logger(__name__)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_golden(conn: Connection) -> list[dict[str, Any]]:
    """Golden queries with their graded labels, grouped per query.

    A (query, chunk) pair can carry both a machine grade and a human grade. We
    prefer the human one via DISTINCT ON, so the trusted label always wins and a
    pair is never counted twice. `has_human` marks queries with any verified
    label -- that is the slice reported separately, because agreement with humans
    is what tells us whether a gain is real or judge bias.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """WITH best AS (
                   SELECT DISTINCT ON (query_id, chunk_id)
                          query_id, chunk_id, relevance, label_source
                     FROM eval_labels
                    ORDER BY query_id, chunk_id,
                             (label_source = 'human_verified') DESC
               )
               SELECT q.id, q.query, q.difficulty,
                      bool_or(b.label_source = 'human_verified') AS has_human,
                      jsonb_object_agg(b.chunk_id::text, b.relevance) AS labels
                 FROM eval_queries q
                 JOIN best b ON b.query_id = q.id
                GROUP BY q.id, q.query, q.difficulty
                ORDER BY q.id"""
        )
        rows = cur.fetchall()

    return [
        {
            "id": r["id"],
            "query": r["query"],
            "difficulty": r["difficulty"],
            "has_human": bool(r["has_human"]),
            "labels": {int(k): float(v) for k, v in r["labels"].items()},
        }
        for r in rows
    ]


def active_snapshot(conn: Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM corpus_snapshots WHERE is_active LIMIT 1")
        row = cur.fetchone()
    if not row:
        raise SystemExit("No active corpus snapshot. Run ingest with --activate first.")
    return int(row[0])


def aggregate(per_query: list[dict[str, Any]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = ("ndcg@10", "recall@50", "mrr", "hit@10")
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the retrieval eval harness")
    ap.add_argument("--tag", default="", help="label for this run, shown in the leaderboard")
    ap.add_argument("--no-rerank", action="store_true", help="baseline: RRF order only")
    ap.add_argument("--json-out", default="", help="also write metrics to this path")
    args = ap.parse_args()

    settings = get_settings()
    conn = Connection.connect(settings.database_url, autocommit=True)
    register_vector(conn)

    snapshot_id = active_snapshot(conn)
    golden = load_golden(conn)
    if not golden:
        raise SystemExit("Golden set is empty. Build it before running eval.")

    pool = ConnectionPool(settings.database_url, min_size=1, max_size=4, open=True)
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
    reranker = (
        Reranker(None, ())  # disabled -> preserves RRF order
        if args.no_rerank
        else Reranker.load(settings.reranker_path)
    )
    pipeline = QueryPipeline(
        pool=pool, settings=settings, embedder=embedder, reranker=reranker, generator=None
    )

    per_query: list[dict[str, Any]] = []
    for item in golden:
        result = pipeline.run(item["query"], snapshot_id=snapshot_id, generate=False)
        ranked = result.retrieved_ids
        labels = item["labels"]
        per_query.append(
            {
                "query_id": item["id"],
                "has_human": item["has_human"],
                "ndcg@10": ndcg_at_k(ranked, labels, k=10),
                "recall@50": recall_at_k(ranked, labels, k=50),
                "mrr": mrr(ranked, labels),
                "hit@10": hit_rate_at_k(ranked, labels, k=10),
            }
        )

    human = [q for q in per_query if q["has_human"]]
    metrics = {
        "overall": aggregate(per_query),
        "human_verified": aggregate(human),
        "n_queries": len(per_query),
        "n_human_verified": len(human),
        "reranker_active": reranker.is_active,
    }

    config = {
        "tag": args.tag,
        "embedding_model": settings.embedding_model,
        "chunker_version": settings.chunker_version,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "candidate_k": settings.candidate_k,
        "final_k": settings.final_k,
        "rrf_k": settings.rrf_k,
        "hnsw_ef_search": settings.hnsw_ef_search,
        "reranker_active": reranker.is_active,
    }

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO eval_runs (snapshot_id, git_sha, config, metrics)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (snapshot_id, git_sha(), json.dumps(config), json.dumps(metrics)),
        )
        row = cur.fetchone()
        assert row is not None
        run_id = int(row[0])

    pool.close()
    conn.close()

    print(json.dumps({"run_id": run_id, "config": config, "metrics": metrics}, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"run_id": run_id, "config": config, "metrics": metrics}, fh, indent=2)


if __name__ == "__main__":
    main()
