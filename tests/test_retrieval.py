"""
tests/test_retrieval.py
-----------------------
Unit and integration tests for the Temporal-RegionRAG retrieval module:
1. Lexical ranking baseline (BM25 Okapi + field boosting).
2. Source reference retention (doc_id, page_number, date_str, bbox coordinates).
3. Temporal multi-date comparison retrieval (e.g. comparing 1998 vs 2002).
4. Long-term progression/trend retrieval across multiple timepoints.
5. Empty / irrelevant query handling (clean empty results for 'no evidence found').
6. Visual region highlighting and snippet cropping.
7. Completely offline and deterministic execution.
"""

import pytest
from PIL import Image

from utils.extraction import Observation
from utils.retrieval import (
    LexicalRetriever,
    RetrievalResult,
    tokenize,
    extract_temporal_anchors,
    is_comparison_query,
    highlight_retrieved_regions,
    crop_observation_region,
)
from utils.synthetic_docs import (
    create_synthetic_retrieval_corpus,
    create_synthetic_retrieval_documents,
)


@pytest.fixture
def corpus():
    """Synthetic longitudinal multi-document medical corpus."""
    return create_synthetic_retrieval_corpus()


@pytest.fixture
def retriever():
    """Lexical retriever instance."""
    return LexicalRetriever()


# ---------------------------------------------------------------------------
# 1. Query Processing & Tokenization Tests
# ---------------------------------------------------------------------------

class TestQueryProcessing:

    def test_tokenize_basic(self):
        tokens = tokenize("What was the blood pressure reading on 15 Jan 1998?")
        assert "blood" in tokens
        assert "pressure" in tokens
        assert "reading" in tokens
        # Common stopwords removed
        assert "what" not in tokens
        assert "was" not in tokens
        assert "the" not in tokens

    def test_tokenize_preserves_numbers_and_units(self):
        tokens = tokenize("Glucose 5.4 mmol/L and BP 140/90")
        assert "glucose" in tokens
        assert any("5.4" in t for t in tokens)
        assert any("mmol" in t for t in tokens)

    def test_extract_temporal_anchors_years(self):
        anchors = extract_temporal_anchors("Compare glucose in 1998 and 2002")
        assert "1998" in anchors
        assert "2002" in anchors

    def test_extract_temporal_anchors_dates(self):
        anchors = extract_temporal_anchors("What happened on 15 Jan 1998?")
        assert any("15 Jan 1998" in a or "1998" in a for a in anchors)

    def test_is_comparison_query_by_keywords(self):
        assert is_comparison_query("Compare blood pressure readings")
        assert is_comparison_query("What is the glucose trend over time?")
        assert is_comparison_query("Difference between visits")
        assert is_comparison_query("Cholesterol vs glucose")

    def test_is_comparison_query_by_multiple_dates(self):
        assert is_comparison_query("Blood pressure in 1998 and 2002")

    def test_is_not_comparison_query(self):
        assert not is_comparison_query("What was the heart rate on 15 Jan 1998?")
        assert not is_comparison_query("Check penicillin allergy")


# ---------------------------------------------------------------------------
# 2. Retrieval Ranking & Source Retention Tests
# ---------------------------------------------------------------------------

class TestRetrievalRanking:

    def test_single_concept_retrieval(self, retriever, corpus):
        results = retriever.retrieve("What was the heart rate?", corpus, top_k=5)
        assert len(results) > 0
        top = results[0]
        assert "heart" in top.text.lower() or "rate" in top.text.lower()
        assert top.rank == 1
        assert top.score > 0

    def test_source_references_preserved(self, retriever, corpus):
        """Retrieved results must retain doc_id, page_number, date_str, and bbox."""
        results = retriever.retrieve("blood pressure", corpus, top_k=3)
        assert len(results) > 0
        for r in results:
            assert isinstance(r.doc_id, str) and len(r.doc_id) > 0
            assert isinstance(r.page_number, int) and r.page_number >= 1
            assert isinstance(r.date_str, str) and len(r.date_str) > 0
            assert isinstance(r.bbox, tuple) and len(r.bbox) == 4
            x, y, w, h = r.bbox
            assert w > 0 and h > 0
            assert r.has_valid_coordinates()

    def test_date_specific_retrieval(self, retriever, corpus):
        """Query targeting 1998 should rank 1998 document highest."""
        results = retriever.retrieve("blood pressure on 15 Jan 1998", corpus, top_k=5)
        assert len(results) > 0
        top = results[0]
        assert "1998" in top.doc_id or "1998" in (top.date_str or "")
        assert top.value == 140.0 or "140/90" in top.text

    def test_allergy_retrieval(self, retriever, corpus):
        results = retriever.retrieve("penicillin allergy urticaria", corpus, top_k=3)
        assert len(results) > 0
        top = results[0]
        assert "penicillin" in top.text.lower()

    def test_result_to_dict_format(self, retriever, corpus):
        results = retriever.retrieve("cholesterol", corpus, top_k=1)
        assert len(results) == 1
        d = results[0].to_dict()
        assert "rank" in d
        assert "retrieval_score" in d
        assert "doc_id" in d
        assert "bbox_x" in d
        assert "bbox_y" in d


# ---------------------------------------------------------------------------
# 3. Multi-Date Comparison Tests (Requirement 4)
# ---------------------------------------------------------------------------

class TestTemporalMultiDateComparison:

    def test_compare_two_explicit_dates(self, retriever, corpus):
        """Query comparing 1998 and 2002 returns matching evidence for BOTH dates."""
        query = "Compare blood pressure in 1998 and 2002"
        results = retriever.retrieve(query, corpus, top_k=5)

        assert len(results) >= 2
        dates_retrieved = {r.date_str for r in results if r.date_str}
        assert any("1998" in d for d in dates_retrieved)
        assert any("2002" in d for d in dates_retrieved)

        # Check that both values (140/90 from 1998 and 130/80 from 2002) are present
        texts = [r.text for r in results]
        assert any("140/90" in t for t in texts)
        assert any("130/80" in t for t in texts)

    def test_trend_over_time_returns_diverse_dates(self, retriever, corpus):
        """Query for 'glucose trend over time' returns readings across distinct dates."""
        query = "Show fasting glucose trend over time"
        results = retriever.retrieve(query, corpus, top_k=5)

        assert len(results) >= 3
        # Should cover observations across multiple documents/dates
        doc_ids = {r.doc_id for r in results}
        assert len(doc_ids) >= 2

        # Verify values from progression: 5.4, 6.8, 7.4
        values = [r.value for r in results if r.value is not None]
        assert 5.4 in values or 6.8 in values or 7.4 in values


# ---------------------------------------------------------------------------
# 4. No Evidence Found Handling (Requirement 6)
# ---------------------------------------------------------------------------

class TestNoEvidenceFound:

    def test_completely_irrelevant_query(self, retriever, corpus):
        """Queries for absent medical concepts return empty list."""
        query = "Brain MRI scan showing glioblastoma multiforme"
        results = retriever.retrieve(query, corpus, min_score=0.1)
        assert results == []

    def test_empty_query(self, retriever, corpus):
        assert retriever.retrieve("", corpus) == []
        assert retriever.retrieve("   ", corpus) == []

    def test_empty_corpus(self, retriever):
        assert retriever.retrieve("blood pressure", []) == []


# ---------------------------------------------------------------------------
# 5. Visual Region Highlighting Tests (Requirement 5)
# ---------------------------------------------------------------------------

class TestVisualRegionHighlighting:

    def test_highlight_retrieved_regions(self, retriever):
        corpus, images = create_synthetic_retrieval_documents()
        results = retriever.retrieve("blood pressure on 15 Jan 1998", corpus, top_k=2)
        assert len(results) > 0

        doc_id = results[0].doc_id
        page_img = images[doc_id]

        annotated = highlight_retrieved_regions(page_img, results)
        assert isinstance(annotated, Image.Image)
        assert annotated.size == page_img.size

    def test_crop_observation_region(self, retriever):
        corpus, images = create_synthetic_retrieval_documents()
        results = retriever.retrieve("penicillin allergy", corpus, top_k=1)
        assert len(results) == 1

        res = results[0]
        page_img = images[res.doc_id]
        crop = crop_observation_region(page_img, res.bbox, padding=5)

        assert crop is not None
        assert isinstance(crop, Image.Image)
        assert crop.width > res.bbox[2]
        assert crop.height > res.bbox[3]

    def test_crop_invalid_coordinates_returns_none(self):
        img = Image.new("RGB", (100, 100))
        assert crop_observation_region(img, (0, 0, 0, 0)) is None
