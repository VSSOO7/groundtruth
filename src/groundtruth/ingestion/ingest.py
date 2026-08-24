"""Ingestion job: EDGAR -> sections -> chunks -> embeddings -> pgvector.

Run: `python -m groundtruth.ingestion.ingest --ciks 320193,789019 --years 2`

Blue/green indexing is the production-relevant part. A new corpus is built under a
fresh `snapshot_id` while the old one keeps serving; only after the build
completes is the snapshot flipped `is_active` in a single transaction. Queries
therefore never observe a half-built index, and a failed ingest leaves the live
index untouched. This is what lets you re-chunk or swap embedding models on a
running system without downtime -- the story behind the `corpus_snapshots` table.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable

import structlog
from pgvector.psycopg import register_vector
from psycopg import Connection

from groundtruth.config import Settings, get_settings
from groundtruth.embedding import Embedder, SentenceTransformerEmbedder
from groundtruth.ingestion.chunker import chunk_sections, normalize_whitespace, split_sections
from groundtruth.ingestion.edgar import EdgarClient, FilingRef, html_to_text

log = structlog.get_logger(__name__)


def _token_counter() -> Callable[[str], int]:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


def create_snapshot(conn: Connection, settings: Settings) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO corpus_snapshots
                 (chunker_version, embedder_version, embedding_dim, notes)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (
                settings.chunker_version,
                settings.embedding_model,
                settings.embedding_dim,
                f"chunk={settings.chunk_tokens}/{settings.chunk_overlap_tokens}",
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def activate_snapshot(conn: Connection, snapshot_id: int) -> None:
    """Atomically flip the active index. Old snapshot stays on disk for rollback."""
    with conn.cursor() as cur:
        cur.execute("UPDATE corpus_snapshots SET is_active = FALSE WHERE is_active")
        cur.execute(
            "UPDATE corpus_snapshots SET is_active = TRUE, built_at = now() WHERE id = %s",
            (snapshot_id,),
        )


def ingest_filing(
    conn: Connection,
    embedder: Embedder,
    snapshot_id: int,
    ref: FilingRef,
    raw_html: str,
    count_tokens: Callable[[str], int],
) -> int:
    text = normalize_whitespace(html_to_text(raw_html))
    sections = split_sections(text)
    chunks = chunk_sections(
        sections,
        max_tokens=get_settings().chunk_tokens,
        overlap_tokens=get_settings().chunk_overlap_tokens,
        count_tokens=count_tokens,
    )
    if not chunks:
        log.warning("ingest.no_chunks", accession=ref.accession_no)
        return 0

    embeddings = embedder.embed_passages([c.text for c in chunks])

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents
                 (accession_no, cik, company_name, ticker, form_type,
                  fiscal_year, filed_date, source_url, raw_sha256)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (accession_no) DO UPDATE SET ingested_at = now()
               RETURNING id""",
            (
                ref.accession_no,
                ref.cik,
                ref.company_name,
                None,
                ref.form_type,
                ref.fiscal_year,
                ref.filed_date,
                ref.url,
                hashlib.sha256(raw_html.encode()).hexdigest(),
            ),
        )
        row = cur.fetchone()
        assert row is not None
        document_id = row[0]

        with cur.copy(
            """COPY chunks
                 (snapshot_id, document_id, item_section, section_name, ordinal,
                  char_start, char_end, text, token_count, embedding)
               FROM STDIN"""
        ) as copy:
            for chunk, emb in zip(chunks, embeddings, strict=True):
                copy.write_row(
                    (
                        snapshot_id,
                        document_id,
                        chunk.item_section,
                        chunk.section_name,
                        chunk.ordinal,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.text,
                        chunk.token_count,
                        str(emb),
                    )
                )
    return len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest SEC filings into pgvector")
    ap.add_argument("--ciks", required=True, help="comma-separated CIK numbers")
    ap.add_argument("--form", default="10-K")
    ap.add_argument("--years", type=int, default=3, help="most-recent N filings per CIK")
    ap.add_argument("--activate", action="store_true", help="flip live index after build")
    args = ap.parse_args()

    settings = get_settings()
    count_tokens = _token_counter()
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)

    conn = Connection.connect(settings.database_url, autocommit=False)
    register_vector(conn)

    try:
        snapshot_id = create_snapshot(conn, settings)
        conn.commit()
        log.info("ingest.snapshot_created", snapshot_id=snapshot_id)

        total_chunks = 0
        with EdgarClient(settings.sec_user_agent) as edgar:
            for cik in (c.strip() for c in args.ciks.split(",")):
                refs = edgar.list_filings(cik, form_type=args.form, limit=args.years)
                log.info("ingest.cik", cik=cik, filings=len(refs))
                for ref in refs:
                    html = edgar.fetch_document(ref)
                    n = ingest_filing(conn, embedder, snapshot_id, ref, html, count_tokens)
                    conn.commit()  # per-filing commit: a later failure keeps earlier work
                    total_chunks += n
                    log.info(
                        "ingest.filing", company=ref.company_name, year=ref.fiscal_year, chunks=n
                    )

        if args.activate:
            activate_snapshot(conn, snapshot_id)
            conn.commit()
            log.info("ingest.activated", snapshot_id=snapshot_id)
        else:
            log.info(
                "ingest.built_inactive",
                snapshot_id=snapshot_id,
                hint="re-run with --activate or flip is_active manually to serve",
            )

        log.info("ingest.done", snapshot_id=snapshot_id, total_chunks=total_chunks)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
