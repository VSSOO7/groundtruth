"""Validate the hermetic eval fixture.

The fixture is the CI gate's entire corpus, so a typo in a passage id would make
`fixture.py` abort mid-load — after CI has already spent minutes booting Postgres
and downloading the embedding model. Catching it in the fast lane is much cheaper.

These checks need no database and no model, so they run in the unit test lane.
"""

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixture_corpus.json"


@pytest.fixture
def fixture():
    return json.loads(FIXTURE.read_text())


def _passage_ids(fixture):
    return {p["id"] for doc in fixture["documents"] for p in doc["passages"]}


def test_fixture_parses():
    assert FIXTURE.exists(), f"missing fixture at {FIXTURE}"
    json.loads(FIXTURE.read_text())


def test_every_label_references_a_real_passage(fixture):
    """The failure mode that would abort the CI load. Worth its own test."""
    known = _passage_ids(fixture)
    for q in fixture["queries"]:
        for passage_id in q["labels"]:
            assert passage_id in known, (
                f"query {q['query']!r} labels unknown passage {passage_id!r}"
            )


def test_passage_ids_are_unique(fixture):
    """Duplicates would silently collapse in the passage -> chunk_id map."""
    ids = [p["id"] for doc in fixture["documents"] for p in doc["passages"]]
    assert len(ids) == len(set(ids))


def test_accession_numbers_are_unique(fixture):
    """documents.accession_no is UNIQUE; a dupe would upsert onto the same row."""
    accessions = [d["accession_no"] for d in fixture["documents"]]
    assert len(accessions) == len(set(accessions))


def test_every_query_has_at_least_one_relevant_passage(fixture):
    """A query with only relevance-0 labels has an undefined ideal DCG, so its
    nDCG is meaningless and it would silently drag the mean down."""
    for q in fixture["queries"]:
        assert any(v > 0 for v in q["labels"].values()), (
            f"query {q['query']!r} has no relevant passage"
        )


def test_relevance_grades_are_in_schema_range(fixture):
    """eval_labels CHECKs relevance BETWEEN 0 AND 3."""
    for q in fixture["queries"]:
        for passage_id, grade in q["labels"].items():
            assert isinstance(grade, int), f"{passage_id} grade must be an int"
            assert 0 <= grade <= 3, f"{passage_id} grade {grade} out of range"


def test_difficulty_values_match_schema_check(fixture):
    """eval_queries CHECKs difficulty IN ('easy','medium','hard') or NULL."""
    allowed = {"easy", "medium", "hard", None}
    for q in fixture["queries"]:
        assert q.get("difficulty") in allowed, (
            f"query {q['query']!r} has difficulty {q.get('difficulty')!r}"
        )


def test_item_sections_cover_the_reranker_section_features(fixture):
    """features.py has one-hot features for items 1A, 7, and 8. If the fixture
    contained none of them, those features would be constant-zero and the gate
    could not detect a regression in them."""
    sections = {p["item_section"] for doc in fixture["documents"] for p in doc["passages"]}
    assert {"1A", "7", "8"} <= sections


def test_negatives_exist_so_ranking_is_measurable(fixture):
    """If every labeled passage were relevant, nDCG would be 1.0 for any ordering
    and the gate would never fire."""
    all_grades = [g for q in fixture["queries"] for g in q["labels"].values()]
    assert any(g == 0 for g in all_grades), "fixture has no relevance-0 negatives"


def test_documents_have_required_columns(fixture):
    required = {
        "accession_no",
        "cik",
        "company_name",
        "ticker",
        "form_type",
        "fiscal_year",
        "filed_date",
        "passages",
    }
    for doc in fixture["documents"]:
        missing = required - doc.keys()
        assert not missing, f"{doc.get('accession_no')} missing {missing}"


def test_passages_have_required_columns(fixture):
    required = {"id", "item_section", "section_name", "text"}
    for doc in fixture["documents"]:
        for p in doc["passages"]:
            missing = required - p.keys()
            assert not missing, f"passage {p.get('id')} missing {missing}"


def test_passages_are_substantive(fixture):
    """Very short passages make the length feature degenerate and the FTS ranking
    unstable, which shows up as a flaky gate rather than an obvious failure."""
    for doc in fixture["documents"]:
        for p in doc["passages"]:
            assert len(p["text"].split()) >= 25, f"passage {p['id']} is too short"


def test_corpus_spans_multiple_companies(fixture):
    """Cross-company distractors are what make retrieval non-trivial -- with one
    company, the sparse retriever alone would ace every query."""
    assert len({d["company_name"] for d in fixture["documents"]}) >= 3


def test_query_count_is_enough_to_average_over(fixture):
    assert len(fixture["queries"]) >= 10
