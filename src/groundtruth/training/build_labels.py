"""Bootstrap the golden set: LLM-generated queries and graded relevance labels.

Run: `python -m groundtruth.training.build_labels --per-snapshot 200`

Hand-labeling enough (query, chunk) pairs to train a ranker is the bottleneck
every retrieval project hits. The pragmatic answer -- and the one worth defending
in an interview -- is a two-tier scheme:

* **Bulk labels are LLM-bootstrapped** with the cheap model. For a sampled chunk
  we ask the model to write a question that chunk answers, then grade a small pool
  of *other* retrieved chunks for that question on the 0-3 scale. Labels land in
  `eval_labels` with `label_source='llm_bootstrap'`.

* **A small slice is human-verified.** These are the same queries, re-graded by a
  person and stored as `label_source='human_verified'`. The eval harness reports
  this slice separately, and `eval/metrics.cohens_kappa` quantifies how far the
  cheap judge can be trusted against it.

The honest framing: bootstrapped labels are plentiful and noisy; the human slice
is scarce and trusted; the project never pretends the first is the second.
"""

from __future__ import annotations

import argparse
from typing import Any

import anthropic
import structlog
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from groundtruth.config import Settings, get_settings
from groundtruth.embedding import SentenceTransformerEmbedder
from groundtruth.retrieval.hybrid import retrieve_candidates

log = structlog.get_logger(__name__)

QUERY_SYSTEM = """\
You write evaluation questions for a retrieval system over SEC filings. Given one \
excerpt from a 10-K or 10-Q, write a single natural question that a financial \
analyst would ask and that THIS excerpt directly answers. The question must be \
answerable from the excerpt alone, must not quote it verbatim, and must name the \
company or fiscal year only if a user realistically would. Return just the \
question.
"""

GRADE_SYSTEM = """\
You grade how well an excerpt answers a question, on this scale:
 
3 -- directly and completely answers the question.
2 -- contains substantial relevant information but is partial.
1 -- touches the topic but does not really answer it.
0 -- unrelated to the question.
 
Judge only the excerpt in front of you. Return the integer grade and a one-clause \
reason.
"""


class GeneratedQuery(BaseModel):
    question: str


class Grade(BaseModel):
    relevance: int = Field(ge=0, le=3)
    reason: str


def sample_chunks(conn: Connection, snapshot_id: int, n: int) -> list[dict[str, Any]]:
    """A spread of chunks to seed questions from -- biased toward the sections
    users actually ask about so the golden set isn't dominated by boilerplate."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, text, item_section
                 FROM chunks
                WHERE snapshot_id = %s
                  AND token_count >= 80
                ORDER BY
                  CASE WHEN item_section IN ('1A','7','8') THEN 0 ELSE 1 END,
                  random()
                LIMIT %s""",
            (snapshot_id, n),
        )
        return cur.fetchall()


def active_snapshot(conn: Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM corpus_snapshots WHERE is_active LIMIT 1")
        row = cur.fetchone()
    if not row:
        raise SystemExit("No active corpus snapshot. Ingest with --activate first.")
    return int(row[0])


class Labeler:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)

    def make_question(self, chunk_text: str) -> str:
        resp = self._client.messages.parse(
            model=self._settings.cheap_model,
            max_tokens=self._settings.max_output_tokens,
            system=[{"type": "text", "text": QUERY_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Excerpt:\n{chunk_text}"}],
            output_format=GeneratedQuery,
        )
        if resp.parsed_output is None:
            raise RuntimeError("Failed to parse generated query.")
        return resp.parsed_output.question.strip()

    def grade(self, question: str, chunk_text: str) -> Grade:
        resp = self._client.messages.parse(
            model=self._settings.cheap_model,
            max_tokens=self._settings.max_output_tokens,
            system=[{"type": "text", "text": GRADE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[
                {"role": "user", "content": f"Question: {question}\n\nExcerpt:\n{chunk_text}"}
            ],
            output_format=Grade,
        )
        if resp.parsed_output is None:
            raise RuntimeError("Failed to parse grade.")
        return resp.parsed_output


def persist_query(conn: Connection, question: str, difficulty: str | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO eval_queries (query, label_source, difficulty)
               VALUES (%s, 'llm_bootstrap', %s)
               RETURNING id""",
            (question, difficulty),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def persist_label(conn: Connection, query_id: int, chunk_id: int, relevance: int) -> None:
    """Upsert a machine grade. Scoped to label_source so re-running the bootstrap
    can never clobber a human grade for the same pair."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO eval_labels (query_id, chunk_id, relevance, label_source)
               VALUES (%s, %s, %s, 'llm_bootstrap')
               ON CONFLICT (query_id, chunk_id, label_source)
               DO UPDATE SET relevance = EXCLUDED.relevance""",
            (query_id, chunk_id, relevance),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap eval queries and graded labels")
    ap.add_argument(
        "--per-snapshot",
        type=int,
        default=200,
        help="how many seed chunks to generate questions from",
    )
    ap.add_argument(
        "--grade-pool", type=int, default=12, help="candidates graded per generated question"
    )
    args = ap.parse_args()

    settings = get_settings()
    conn = Connection.connect(settings.database_url, autocommit=True)
    register_vector(conn)

    snapshot_id = active_snapshot(conn)
    seeds = sample_chunks(conn, snapshot_id, args.per_snapshot)
    if not seeds:
        raise SystemExit("No chunks to sample. Ingest a corpus first.")

    pool = ConnectionPool(settings.database_url, min_size=1, max_size=4, open=True)
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
    labeler = Labeler(settings)

    n_queries = 0
    n_labels = 0
    for seed in seeds:
        question = labeler.make_question(seed["text"])

        # Grade the seed chunk itself plus a pool of co-retrieved candidates, so
        # each query gets a mix of a near-certain positive and plausible negatives.
        qvec = embedder.embed_query(question)
        candidates = retrieve_candidates(
            pool,
            settings,
            query_text=question,
            query_vector=qvec,
            snapshot_id=snapshot_id,
            filters=None,
        )[: args.grade_pool]

        # difficulty is left NULL here: the schema only allows easy|medium|hard,
        # and bootstrapped questions aren't triaged -- a human sets it on the
        # verified slice.
        query_id = persist_query(conn, question, difficulty=None)

        # The seed chunk is graded unconditionally, so every query is guaranteed a
        # near-certain positive even when retrieval fails to surface its own seed.
        seed_grade = labeler.grade(question, seed["text"])
        persist_label(conn, query_id, seed["id"], seed_grade.relevance)
        n_labels += 1

        for cand in candidates:
            if cand.chunk_id == seed["id"]:
                continue
            g = labeler.grade(question, cand.text)
            persist_label(conn, query_id, cand.chunk_id, g.relevance)
            n_labels += 1

        n_queries += 1
        if n_queries % 20 == 0:
            log.info("labels.progress", queries=n_queries, labels=n_labels)

    pool.close()
    conn.close()

    log.info("labels.complete", queries=n_queries, labels=n_labels)
    print(
        f"Bootstrapped {n_queries} queries and {n_labels} labels "
        f"(label_source='llm_bootstrap').\n"
        f"Next: hand-verify a slice and re-insert it with label_source='human_verified', "
        f"then check judge/human agreement with eval/metrics.cohens_kappa."
    )


if __name__ == "__main__":
    main()
