"""Section-aware chunking for 10-K/10-Q filings.

Why not a fixed-size splitter: a 10-K's semantics live in its Item structure.
"Item 1A. Risk Factors" and "Item 7. MD&A" answer different questions, and a
naive character splitter produces chunks that straddle the boundary between them
-- so a risk question retrieves half a paragraph of accounting policy. Splitting
on Item headings first, then packing within each section, keeps every chunk
attributable to exactly one section and makes `item_section` a usable retrieval
filter and reranker feature.

The hard part is that Item headings are not reliably formatted. Real filings use
"ITEM 1A.", "Item 1A -", "Item&#160;1A.", bold tags, and table-of-contents entries
that look identical to the real heading. The heuristics below are deliberately
conservative and documented, because silently mis-detecting a section corrupts
every downstream metric.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Canonical 10-K item names. Used to label sections and to sanity-check ordering.
ITEM_NAMES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "10": "Directors and Executive Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

# Matches "Item 1A." / "ITEM 7 -" / "Item 9A:" at the start of a line.
# Anchored to line start because the phrase "see Item 1A" appears constantly in
# running prose and must not be treated as a section boundary.
_ITEM_RE = re.compile(
    r"^\s{0,4}item\s+(\d{1,2}[A-C]?)\s*[.:\-\u2013\u2014]?\s*(.{0,80})$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(slots=True)
class Section:
    item: str | None  # '1A', '7', ... or None for pre-Item front matter
    name: str | None
    text: str
    char_start: int  # offsets into the normalized document text
    char_end: int


@dataclass(slots=True)
class Chunk:
    text: str
    token_count: int
    item_section: str | None
    section_name: str | None
    ordinal: int
    char_start: int
    char_end: int


def normalize_whitespace(text: str) -> str:
    """Collapse the whitespace noise typical of HTML-extracted filings.

    Character offsets in `Section`/`Chunk` refer to the *normalized* string, so
    normalization must happen exactly once, before splitting. Callers should
    persist the normalized text if they want offsets to remain meaningful.
    """
    text = text.replace("\xa0", " ").replace("\u2028", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str, *, min_section_chars: int = 500) -> list[Section]:
    """Split a filing into Item sections.

    Table-of-contents defense: a real 10-K lists every Item twice -- once in the
    TOC, once as the actual heading. TOC entries are followed by almost no text,
    so a candidate heading whose body is shorter than `min_section_chars` is
    treated as a TOC artifact and skipped, and its text is merged into whatever
    section precedes it. This is a heuristic, not a parser; it is tuned to be
    conservative, preferring to under-split rather than emit a bogus section.

    Returns sections in document order. If no headings are found at all, returns
    a single unlabeled section spanning the whole document, so ingestion of
    non-standard filings degrades to plain chunking rather than failing.
    """
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [Section(item=None, name=None, text=text, char_start=0, char_end=len(text))]

    candidates: list[tuple[int, str, int]] = []  # (start_offset, item, heading_end)
    for m in matches:
        item = m.group(1).upper()
        if item not in ITEM_NAMES:
            continue  # 'Item 47' etc. -- not a real 10-K item
        candidates.append((m.start(), item, m.end()))

    if not candidates:
        return [Section(item=None, name=None, text=text, char_start=0, char_end=len(text))]

    sections: list[Section] = []

    # Front matter before the first real heading (cover page, TOC).
    if candidates[0][0] > 0:
        sections.append(
            Section(
                item=None,
                name=None,
                text=text[: candidates[0][0]],
                char_start=0,
                char_end=candidates[0][0],
            )
        )

    for i, (start, item, heading_end) in enumerate(candidates):
        end = candidates[i + 1][0] if i + 1 < len(candidates) else len(text)
        body = text[heading_end:end]

        if len(body.strip()) < min_section_chars:
            # TOC artifact or a stub cross-reference. Fold into the previous
            # section rather than creating a near-empty one.
            if sections:
                prev = sections[-1]
                sections[-1] = Section(
                    item=prev.item,
                    name=prev.name,
                    text=text[prev.char_start : end],
                    char_start=prev.char_start,
                    char_end=end,
                )
            continue

        sections.append(
            Section(
                item=item,
                name=ITEM_NAMES[item],
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )

    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Paragraph units for packing. Blank-line separated, blanks preserved out."""
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_sections(
    sections: list[Section],
    *,
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    """Pack each section's paragraphs into token-budgeted chunks.

    Chunks never span a section boundary -- that is the whole point of splitting
    first. Within a section, paragraphs are accumulated until the budget is hit;
    a trailing overlap carries context across the seam so a fact split across a
    boundary is still retrievable from at least one chunk.

    `count_tokens` is injected (rather than importing tiktoken here) so tests can
    pass a trivial word counter and so the tokenizer can be swapped without
    touching packing logic.
    """
    chunks: list[Chunk] = []
    ordinal = 0

    for sec in sections:
        paragraphs = _split_paragraphs(sec.text)
        if not paragraphs:
            continue

        buffer: list[str] = []
        buffer_tokens = 0
        # Offset bookkeeping is approximate within a section: we track the
        # section start plus consumed characters, which is exact as long as
        # paragraphs are re-joined with the separator they were split on.
        cursor = sec.char_start

        def flush(buf: list[str], start: int, section: Section = sec) -> int:
            nonlocal ordinal
            if not buf:
                return start
            body = "\n\n".join(buf)
            end = start + len(body)
            chunks.append(
                Chunk(
                    text=body,
                    token_count=count_tokens(body),
                    item_section=section.item,
                    section_name=section.name,
                    ordinal=ordinal,
                    char_start=start,
                    char_end=end,
                )
            )
            ordinal += 1
            return end

        for para in paragraphs:
            ptokens = count_tokens(para)

            # A single paragraph over budget becomes its own chunk. Splitting
            # mid-paragraph would cut tables and numbered risk factors in half;
            # an oversized chunk is the lesser evil and is rare in practice.
            if ptokens >= max_tokens:
                cursor = flush(buffer, cursor, sec)
                buffer, buffer_tokens = [], 0
                cursor = flush([para], cursor, sec)
                continue

            if buffer_tokens + ptokens > max_tokens:
                cursor = flush(buffer, cursor, sec)
                # Carry the tail of the flushed buffer as overlap.
                carry: list[str] = []
                carry_tokens = 0
                for prev in reversed(buffer):
                    t = count_tokens(prev)
                    if carry_tokens + t > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carry_tokens += t
                buffer, buffer_tokens = carry, carry_tokens

            buffer.append(para)
            buffer_tokens += ptokens

        flush(buffer, cursor, sec)

    return chunks
