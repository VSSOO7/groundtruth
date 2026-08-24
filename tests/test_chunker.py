"""Tests for section-aware chunking.

These pin the behaviours that silently corrupt every downstream metric when they
regress: table-of-contents entries being mistaken for real sections, chunks
straddling an Item boundary, and inline "see Item 1A" cross-references creating
phantom sections.

`count_tokens` is a word counter here -- packing logic is what's under test, not
the tokenizer.
"""

import pytest

from groundtruth.ingestion.chunker import (
    ITEM_NAMES,
    Section,
    chunk_sections,
    normalize_whitespace,
    split_sections,
)


def word_count(s: str) -> int:
    return len(s.split())


@pytest.fixture
def filing() -> str:
    """A filing with a TOC that lists items, followed by the real sections."""
    body_1 = "Acme designs widgets. " * 60
    body_1a = "Our supply chain may be disrupted. " * 60
    body_7 = "Revenue grew 12 percent to $1,200 million. " * 60
    return normalize_whitespace(
        "ACME CORP FORM 10-K\n\n"
        "TABLE OF CONTENTS\n\n"
        "Item 1. Business\n"
        "Item 1A. Risk Factors\n"
        "Item 7. Management Discussion\n\n"
        f"Item 1. Business\n\n{body_1}\n\n"
        f"Item 1A. Risk Factors\n\n{body_1a}\n\n"
        f"Item 7. Management Discussion\n\n{body_7}"
    )


def test_toc_entries_do_not_become_sections(filing):
    """The TOC lists Item 1 too; only the real section should be emitted."""
    items = [s.item for s in split_sections(filing)]
    assert items.count("1") == 1


def test_real_sections_detected(filing):
    items = [s.item for s in split_sections(filing)]
    assert "1A" in items
    assert "7" in items


def test_sections_are_labeled_with_canonical_names(filing):
    for s in split_sections(filing):
        if s.item:
            assert s.name == ITEM_NAMES[s.item]


def test_offsets_are_ordered_and_in_range(filing):
    for s in split_sections(filing):
        assert s.char_start < s.char_end <= len(filing)


def test_chunks_never_straddle_a_section_boundary(filing):
    sections = split_sections(filing)
    chunks = chunk_sections(sections, max_tokens=50, overlap_tokens=10, count_tokens=word_count)
    risk_chunks = [c for c in chunks if c.item_section == "1A"]
    assert risk_chunks, "expected chunks from the Risk Factors section"
    # No Risk Factors chunk may contain MD&A or Business body text.
    for c in risk_chunks:
        assert "Acme designs widgets" not in c.text
        assert "Revenue grew 12 percent" not in c.text


def test_ordinals_are_unique_and_sequential(filing):
    chunks = chunk_sections(
        split_sections(filing), max_tokens=50, overlap_tokens=10, count_tokens=word_count
    )
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_oversized_paragraph_becomes_its_own_chunk():
    """Splitting mid-paragraph would cut tables and numbered risks in half."""
    big = Section(item="7", name="MD&A", text="word " * 500, char_start=0, char_end=2500)
    chunks = chunk_sections([big], max_tokens=50, overlap_tokens=10, count_tokens=word_count)
    assert len(chunks) == 1
    assert chunks[0].token_count == 500


def test_document_without_headings_degrades_to_single_section():
    sections = split_sections("Just some text with no item headings. " * 30)
    assert len(sections) == 1
    assert sections[0].item is None


def test_inline_cross_reference_is_not_a_heading():
    """'see Item 1A for details' appears constantly in running prose."""
    sections = split_sections("We discuss this further; see Item 1A for details. " * 40)
    assert all(s.item is None for s in sections)


def test_unknown_item_numbers_are_ignored():
    sections = split_sections("Item 47. Nonexistent\n\n" + ("filler text " * 200))
    assert all(s.item is None for s in sections)


def test_empty_sections_produce_no_chunks():
    empty = Section(item="1", name="Business", text="   \n\n  ", char_start=0, char_end=7)
    assert chunk_sections([empty], max_tokens=50, overlap_tokens=10, count_tokens=word_count) == []


def test_normalize_whitespace_collapses_nbsp_and_blank_runs():
    out = normalize_whitespace("a\xa0b\n\n\n\n\nc")
    assert "\xa0" not in out
    assert "\n\n\n" not in out
