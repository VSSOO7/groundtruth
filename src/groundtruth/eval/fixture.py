"""Load the hermetic eval fixture into Postgres.

Run: `python -m groundtruth.eval.fixture --load`

CI needs a corpus, but a CI job that fetches live SEC filings is a CI job that
fails when the SEC rate-limits it or a company files a restatement. So the gate
runs against `eval/fixture_corpus.json`: synthetic 10-K-style passages written for
this repo, with hand-assigned graded labels.

What this deliberately does *not* do: run the chunker. One chunk is inserted per
passage so that label -> chunk_id mapping is exact and stable. The chunker has its
own unit tests; mixing it in here would make the gate's numbers move whenever
chunking changed, which is the opposite of a controlled baseline.

Embeddings are real -- computed by the configured model -- because the whole point
is to exercise the actual HNSW + FTS + RRF + rerank path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog
from pgvector.psycopg import register_vector
from psycopg import Connection

from groundtruth.config import get_settings
from groundtruth.embedding import Embedder, SentenceTransformerEmbedder

log = structlog.get_logger(__name__)

FIXTURE_PATH = Path("eval/fixture_corpus.json")


def _count_tokens(text: str) -> int:
    """Whitespace approximation. The fixture never exercises the real tokenizer,
    and pulling in tiktoken here would add a download to a job that must stay
    hermetic; token_count only feeds a log-scaled length feature."""
    return len(text.split())


def load_fixture(conn: Connection, path: Path, embedder: Embedder) -> tuple[int, int, int]:
    """Insert the fixture as a fresh active snapshot. Returns (chunks, queries, labels)."""
    fixture = json.loads(path.read_text())
    settings = get_settings()

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO corpus_snapshots
                 (chunker_version, embedder_version, embedding_dim, notes)
               VALUES (%s, %s, %s, 'eval fixture: one chunk per passage')
               RETURNING id""",
            ("fixture-passthrough-v1", settings.embedding_model, settings.embedding_dim),
        )
        row = cur.fetchone()
        assert row is not None
        snapshot_id = int(row[0])

    passage_to_chunk: dict[str, int] = {}
    n_chunks = 0

    for doc in fixture["documents"]:
        passages = doc["passages"]
        embeddings = embedder.embed_passages([p["text"] for p in passages])

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO documents
                     (accession_no, cik, company_name, ticker, form_type,
                      fiscal_year, filed_date, source_url, raw_sha256)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (accession_no) DO UPDATE SET ingested_at = now()
                   RETURNING id""",
                (
                    doc["accession_no"],
                    doc["cik"],
                    doc["company_name"],
                    doc["ticker"],
                    doc["form_type"],
                    doc["fiscal_year"],
                    doc["filed_date"],
                    f"https://example.invalid/fixture/{doc['accession_no']}",
                    "fixture",
                ),
            )
            row = cur.fetchone()
            assert row is not None
            document_id = int(row[0])

            cursor_pos = 0
            for ordinal, (passage, emb) in enumerate(zip(passages, embeddings, strict=True)):
                text = passage["text"]
                cur.execute(
                    """INSERT INTO chunks
                         (snapshot_id, document_id, item_section, section_name, ordinal,
                          char_start, char_end, text, token_count, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       RETURNING id""",
                    (
                        snapshot_id,
                        document_id,
                        passage["item_section"],
                        passage["section_name"],
                        ordinal,
                        cursor_pos,
                        cursor_pos + len(text),
                        text,
                        _count_tokens(text),
                        emb,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                passage_to_chunk[passage["id"]] = int(row[0])
                cursor_pos += len(text) + 2
                n_chunks += 1

    n_queries = 0
    n_labels = 0
    for q in fixture["queries"]:
        with conn.cursor() as cur:
            # Fixture labels are hand-assigned, so they are the trusted slice --
            # the gate's human_verified numbers come from exactly these rows.
            cur.execute(
                """INSERT INTO eval_queries (query, label_source, difficulty)
                   VALUES (%s, 'human_verified', %s) RETURNING id""",
                (q["query"], q.get("difficulty")),
            )
            row = cur.fetchone()
            assert row is not None
            query_id = int(row[0])
            n_queries += 1

            for passage_id, relevance in q["labels"].items():
                chunk_id = passage_to_chunk.get(passage_id)
                if chunk_id is None:
                    raise SystemExit(
                        f"Fixture query {q['query']!r} labels unknown passage "
                        f"{passage_id!r}. Fix eval/fixture_corpus.json."
                    )
                cur.execute(
                    """INSERT INTO eval_labels
                         (query_id, chunk_id, relevance, label_source)
                       VALUES (%s, %s, %s, 'human_verified')
                       ON CONFLICT (query_id, chunk_id, label_source)
                       DO UPDATE SET relevance = EXCLUDED.relevance""",
                    (query_id, chunk_id, relevance),
                )
                n_labels += 1

    with conn.cursor() as cur:
        cur.execute("UPDATE corpus_snapshots SET is_active = FALSE WHERE is_active")
        cur.execute(
            "UPDATE corpus_snapshots SET is_active = TRUE, built_at = now() WHERE id = %s",
            (snapshot_id,),
        )

    log.info(
        "fixture.loaded",
        snapshot_id=snapshot_id,
        chunks=n_chunks,
        queries=n_queries,
        labels=n_labels,
    )
    return n_chunks, n_queries, n_labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the hermetic eval fixture")
    ap.add_argument("--load", action="store_true", required=True)
    ap.add_argument("--path", type=Path, default=FIXTURE_PATH)
    args = ap.parse_args()

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)

    conn = Connection.connect(settings.database_url, autocommit=False)
    register_vector(conn)
    try:
        n_chunks, n_queries, n_labels = load_fixture(conn, args.path, embedder)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(
        f"Fixture loaded: {n_chunks} chunks, {n_queries} golden queries, "
        f"{n_labels} labels. Snapshot is active."
    )


if __name__ == "__main__":
    main()
