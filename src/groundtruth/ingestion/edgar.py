"""SEC EDGAR client.

EDGAR is free and needs no key, but it has two rules that trip up every first
integration and are worth encoding rather than rediscovering:

1. **A descriptive User-Agent is mandatory.** Requests without one get a 403.
   The SEC asks for a contact address in it (`SEC_USER_AGENT` in .env).

2. **10 requests/second, and they mean it.** Exceed it and your IP is throttled
   or blocked. The client enforces a conservative internal rate limit so a bulk
   ingest can't get the whole deployment banned.

Scope note: this fetches the primary filing document. Parsing EDGAR's full-text
index and share-class edge cases is deliberately out of scope for v1 -- the goal
is a defensible corpus of real filings, not a complete EDGAR mirror.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx

_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary}"


class _RateLimiter:
    """Simple thread-safe minimum-interval limiter (SEC allows ~10 req/s)."""

    def __init__(self, min_interval: float = 0.15):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()


@dataclass(slots=True)
class FilingRef:
    cik: str
    accession_no: str
    form_type: str
    fiscal_year: int
    filed_date: str
    primary_document: str
    company_name: str

    @property
    def url(self) -> str:
        return _ARCHIVE.format(
            cik=int(self.cik),
            accession_nodash=self.accession_no.replace("-", ""),
            primary=self.primary_document,
        )


class EdgarClient:
    def __init__(self, user_agent: str, *, timeout: float = 30.0):
        if "example.com" in user_agent or not user_agent.strip():
            # Fail early with an actionable message rather than eating a 403.
            raise ValueError(
                "Set SEC_USER_AGENT to a real contact string, e.g. "
                "'groundtruth/0.1 (you@domain.com)'. SEC returns 403 otherwise."
            )
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
        )
        self._limiter = _RateLimiter()

    def list_filings(self, cik: str, *, form_type: str = "10-K", limit: int = 5) -> list[FilingRef]:
        """Most-recent filings of a given form type for a CIK."""
        self._limiter.wait()
        resp = self._client.get(_SUBMISSIONS.format(cik=cik))
        resp.raise_for_status()
        data = resp.json()

        name = data.get("name", "Unknown")
        recent = data["filings"]["recent"]
        out: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form != form_type:
                continue
            filed = recent["filingDate"][i]
            out.append(
                FilingRef(
                    cik=cik,
                    accession_no=recent["accessionNumber"][i],
                    form_type=form,
                    fiscal_year=int(filed[:4]),
                    filed_date=filed,
                    primary_document=recent["primaryDocument"][i],
                    company_name=name,
                )
            )
            if len(out) >= limit:
                break
        return out

    def fetch_document(self, ref: FilingRef) -> str:
        """Fetch the primary filing document's raw HTML/text."""
        self._limiter.wait()
        resp = self._client.get(ref.url)
        resp.raise_for_status()
        return resp.text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def html_to_text(html: str) -> str:
    """Extract readable text from filing HTML.

    Uses selectolax (fast, lenient) and drops script/style. Table structure is
    flattened to text -- v1 accepts that financial tables lose their grid, which
    is why the chunker keeps whole paragraphs together and the reranker leans on
    numeric-overlap features to still surface the right figures.
    """
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    for tag in tree.css("script, style, ix\\:header"):
        tag.decompose()
    body = tree.body or tree.root
    return body.text(separator="\n") if body else ""
