"""Dependency-free retrieval data types.

Deliberately imports nothing beyond the stdlib. `features.py` and the reranker
operate on these types, so unit-testing feature extraction and ranking logic
does not require Postgres drivers, a database, or a model runtime installed.
Keeping the pure data layer separate from the I/O layer is what makes the fast
CI path fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Candidate:
    """A retrieved chunk plus the raw retriever signals the reranker consumes."""

    chunk_id: int
    text: str
    token_count: int
    rrf_score: float
    dense_score: float | None
    dense_rank: int | None
    sparse_score: float | None
    sparse_rank: int | None
    item_section: str | None
    section_name: str | None
    company_name: str
    ticker: str | None
    fiscal_year: int
    form_type: str = "10-K"
    accession_no: str = ""
    source_url: str = ""
    char_start: int = 0
    char_end: int = 0


@dataclass(slots=True)
class Filters:
    """Pre-scan metadata filters. `None` means 'no constraint on this field'."""

    cik: str | None = None
    fiscal_year: int | None = None
    sections: list[str] | None = field(default=None)
