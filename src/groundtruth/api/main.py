"""FastAPI service.

Health endpoint conventions matter here and are worth stating explicitly, because
getting them backwards is a classic production outage: `/healthz` answers "is this
process alive" (never touches the DB, so a database blip doesn't cause the
orchestrator to kill and restart every replica), while `/readyz` answers "should
this replica receive traffic" (checks the DB and that an active index snapshot
exists). Liveness must not depend on dependencies; readiness must.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from starlette.responses import Response

from groundtruth.config import get_settings
from groundtruth.embedding import SentenceTransformerEmbedder
from groundtruth.generation.generator import Generator
from groundtruth.pipeline import QueryPipeline
from groundtruth.retrieval.reranker import Reranker
from groundtruth.retrieval.types import Filters

log = structlog.get_logger(__name__)

QUERIES = Counter("gt_queries_total", "Queries served", ["refused", "reranker"])
LATENCY = Histogram("gt_stage_latency_ms", "Per-stage latency (ms)", ["stage"])

_state: dict[str, Any] = {}


def _active_snapshot_id(pool: ConnectionPool) -> int | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM corpus_snapshots WHERE is_active LIMIT 1")
        row = cur.fetchone()
        return int(row[0]) if row else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load everything expensive once, at startup -- never per request.

    The embedding model is hundreds of MB and takes seconds to load; doing it
    lazily on first request would make the first user pay for it and make
    autoscaling behave unpredictably.
    """
    settings = get_settings()
    pool = ConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        open=True,
    )
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
    reranker = Reranker.load(settings.reranker_path)
    generator = Generator(settings) if settings.anthropic_api_key else None
    if generator is None:
        log.warning("api.no_api_key", effect="retrieval_only_mode")

    _state.update(
        settings=settings,
        pool=pool,
        pipeline=QueryPipeline(
            pool=pool,
            settings=settings,
            embedder=embedder,
            reranker=reranker,
            generator=generator,
        ),
    )
    log.info("api.ready", reranker_active=reranker.is_active, generation=generator is not None)
    try:
        yield
    finally:
        pool.close()


app = FastAPI(title="groundtruth", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    cik: str | None = None
    fiscal_year: int | None = Field(default=None, ge=1993, le=2100)
    sections: list[str] | None = None


class CitationOut(BaseModel):
    chunk_id: int
    company_name: str
    fiscal_year: int
    section_name: str | None
    source_url: str
    excerpt: str


class QueryResponse(BaseModel):
    request_id: str
    refused: bool
    answer: str
    confidence: float
    claims: list[dict[str, Any]]
    citations: list[CitationOut]
    timings_ms: dict[str, float]
    reranker_active: bool


def get_pipeline() -> QueryPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="service starting")
    return cast(QueryPipeline, pipeline)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: deliberately does NOT touch the database."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    """Readiness: requires a reachable DB and an active index snapshot."""
    pool = _state.get("pool")
    if pool is None:
        raise HTTPException(status_code=503, detail="not initialized")
    try:
        snapshot_id = _active_snapshot_id(pool)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}") from exc
    if snapshot_id is None:
        raise HTTPException(status_code=503, detail="no active corpus snapshot; run ingest")
    return {"status": "ready", "snapshot_id": snapshot_id}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    request: Request,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> QueryResponse:
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    pool = _state["pool"]

    snapshot_id = _active_snapshot_id(pool)
    if snapshot_id is None:
        raise HTTPException(status_code=503, detail="no active corpus snapshot; run ingest")

    result = pipeline.run(
        req.question,
        snapshot_id=snapshot_id,
        filters=Filters(cik=req.cik, fiscal_year=req.fiscal_year, sections=req.sections),
    )

    QUERIES.labels(
        refused=str(result.answer.refused).lower(),
        reranker=str(result.reranker_active).lower(),
    ).inc()
    for stage, value in (
        ("embed", result.timings.embed_ms),
        ("retrieve", result.timings.retrieve_ms),
        ("rerank", result.timings.rerank_ms),
        ("generate", result.timings.generate_ms),
    ):
        LATENCY.labels(stage=stage).observe(value)

    # Only return citations the answer actually used -- not the whole context
    # window. A UI that shows all 8 retrieved chunks as "sources" implies the
    # answer rests on all of them, which is not what happened.
    cited = result.answer.cited_chunk_ids()
    citations = [
        CitationOut(
            chunk_id=r.candidate.chunk_id,
            company_name=r.candidate.company_name,
            fiscal_year=r.candidate.fiscal_year,
            section_name=r.candidate.section_name,
            source_url=r.candidate.source_url,
            excerpt=r.candidate.text[:400],
        )
        for r in result.contexts
        if r.candidate.chunk_id in cited
    ]

    return QueryResponse(
        request_id=request_id,
        refused=result.answer.refused,
        answer=result.answer.answer,
        confidence=result.answer.confidence,
        claims=[c.model_dump() for c in result.answer.claims],
        citations=citations,
        timings_ms={
            "embed": round(result.timings.embed_ms, 1),
            "retrieve": round(result.timings.retrieve_ms, 1),
            "rerank": round(result.timings.rerank_ms, 1),
            "generate": round(result.timings.generate_ms, 1),
            "total": round(result.timings.total_ms, 1),
        },
        reranker_active=result.reranker_active,
    )
