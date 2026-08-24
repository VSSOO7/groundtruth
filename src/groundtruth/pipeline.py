"""The query pipeline: the one place retrieval, reranking, and generation compose.

Kept separate from the FastAPI layer so it can be driven by the eval harness and
the API through the identical code path. If eval and serving diverged, every
number in the README would be measuring something the users never receive --
which is the most common way RAG benchmarks end up lying.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
from psycopg_pool import ConnectionPool

from groundtruth.config import Settings
from groundtruth.embedding import Embedder
from groundtruth.generation.generator import Generator
from groundtruth.generation.schema import GroundedAnswer
from groundtruth.retrieval.hybrid import retrieve_candidates
from groundtruth.retrieval.reranker import RerankedCandidate, Reranker, dedupe_by_document
from groundtruth.retrieval.types import Filters

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class Timings:
    """Per-stage latency in milliseconds. Surfaced in the API response and OTel."""

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.retrieve_ms + self.rerank_ms + self.generate_ms


@dataclass(slots=True)
class PipelineResult:
    answer: GroundedAnswer
    contexts: list[RerankedCandidate]
    timings: Timings
    candidate_count: int
    reranker_active: bool
    retrieved_ids: list[int] = field(default_factory=list)


class QueryPipeline:
    def __init__(
        self,
        *,
        pool: ConnectionPool,
        settings: Settings,
        embedder: Embedder,
        reranker: Reranker,
        generator: Generator | None,
    ):
        self._pool = pool
        self._settings = settings
        self._embedder = embedder
        self._reranker = reranker
        self._generator = generator

    def run(
        self,
        question: str,
        *,
        snapshot_id: int,
        filters: Filters | None = None,
        generate: bool = True,
    ) -> PipelineResult:
        """Execute the full pipeline.

        `generate=False` runs retrieval and reranking only -- used by the eval
        harness when scoring nDCG, where a generation call per query would cost
        real money and measure nothing about ranking quality.
        """
        t = Timings()

        start = time.perf_counter()
        qvec = self._embedder.embed_query(question)
        t.embed_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        candidates = retrieve_candidates(
            self._pool,
            self._settings,
            query_text=question,
            query_vector=qvec,
            snapshot_id=snapshot_id,
            filters=filters,
        )
        t.retrieve_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        # Rerank the full candidate set, not a pre-truncated slice. Eval metrics
        # (recall@50, nDCG@10) score the ranker's true ordering via retrieved_ids,
        # so that ordering must be complete. Dedup + truncation to final_k is a
        # generation-context policy -- it shapes only what the LLM sees, and must
        # not be mistaken for the ranking the metrics grade.
        reranked = self._reranker.rerank(question, candidates, top_k=None)
        ranked = dedupe_by_document(reranked)[: self._settings.final_k]
        t.rerank_ms = (time.perf_counter() - start) * 1000

        if not generate or self._generator is None:
            answer = GroundedAnswer(
                refused=True, answer="Generation skipped.", claims=[], confidence=0.0
            )
        else:
            start = time.perf_counter()
            answer = self._generator.answer(question, ranked)
            t.generate_ms = (time.perf_counter() - start) * 1000

        log.info(
            "pipeline.complete",
            candidates=len(candidates),
            returned=len(ranked),
            refused=answer.refused,
            reranker_active=self._reranker.is_active,
            total_ms=round(t.total_ms, 1),
        )

        return PipelineResult(
            answer=answer,
            contexts=ranked,
            timings=t,
            candidate_count=len(candidates),
            reranker_active=self._reranker.is_active,
            retrieved_ids=[r.candidate.chunk_id for r in reranked],
        )
