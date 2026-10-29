"""
tests/test_visual_retrieval.py
------------------------------
Unit and integration tests for optional visual document retrieval
(utils/visual_retrieval.py):

1. VisualPageRetriever interface returning page IDs and relevance scores.
2. Visual-retrieval-enabled mode (ColPaliVisualRetriever with late-interaction
   MaxSim scoring and deterministic encoders).
3. EmbeddingCache (in-memory + on-disk .npy persistence; avoids re-encoding pages).
4. OCR-based page retriever fallback (OCRPageRetrieverFallback) when ColPali is
   not installed, disabled, or raises an exception.
5. Connecting retrieved pages to table regions, observations, dates, and source
   bounding-box coordinates (ConnectedPageEvidence & VisualRetrievalEngine).
6. Explicit granularity notice that page-level visual retrieval does not
   automatically identify the exact table cell or measurement.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from PIL import Image

from utils.synthetic_docs import create_synthetic_verification_documents
from utils.visual_retrieval import (
    PAGE_LEVEL_LIMITATION_NOTICE,
    ColPaliConfig,
    ColPaliVisualRetriever,
    DocumentPage,
    EmbeddingCache,
    OCRPageRetrieverFallback,
    VisualPageHit,
    VisualPageRetriever,
    VisualRetrievalEngine,
    annotate_connected_page,
    make_page_id,
    maxsim_score,
    parse_page_id,
    probe_visual_environment,
)


@pytest.fixture(scope="module")
def corpus_and_pages():
    corpus, page_dict = create_synthetic_verification_documents()
    doc_pages = [
        DocumentPage(doc_id=doc_id, page_number=idx, image=img)
        for doc_id, imgs in page_dict.items()
        for idx, img in enumerate(imgs, start=1)
    ]
    return corpus, page_dict, doc_pages


# ---------------------------------------------------------------------------
# 1. Interface & page ID helpers
# ---------------------------------------------------------------------------

class TestVisualRetrieverInterface:

    def test_make_and_parse_page_id(self):
        pid = make_page_id("doc_1998_admission", 1)
        assert pid == "doc_1998_admission::p1"
        doc_id, page_num = parse_page_id(pid)
        assert doc_id == "doc_1998_admission"
        assert page_num == 1

    def test_probe_visual_environment_returns_expected_keys(self):
        env = probe_visual_environment()
        for key in (
            "platform",
            "machine",
            "torch_version",
            "device",
            "colpali_installed",
            "transformers_installed",
            "colpali_ready",
            "free_disk_gib",
            "notes",
        ):
            assert key in env

    def test_maxsim_score_ranks_aligned_patches_higher(self):
        q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        p_match = np.array([[0.9, 0.1, 0.0], [0.1, 0.95, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        p_other = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.9]], dtype=np.float32)
        assert maxsim_score(q, p_match) > maxsim_score(q, p_other)


# ---------------------------------------------------------------------------
# 2. Embedding cache (memory + disk)
# ---------------------------------------------------------------------------

class TestEmbeddingCache:

    def test_disk_and_memory_cache_roundtrip(self, tmp_path: Path):
        cache = EmbeddingCache(cache_dir=tmp_path / "embeds", persist_to_disk=True)
        img = Image.new("RGB", (64, 64), color=(120, 140, 160))
        key = cache.compute_key(img, "vidore/colpali-v1.2")

        assert cache.get(key) is None
        vec = np.ones((8, 16), dtype=np.float32) * 0.5
        cache.put(key, vec)

        # In-memory hit
        loaded_mem = cache.get(key)
        assert loaded_mem is not None
        assert np.allclose(loaded_mem, vec)

        # Fresh cache instance reads from disk without re-encoding
        cache2 = EmbeddingCache(cache_dir=tmp_path / "embeds", persist_to_disk=True)
        loaded_disk = cache2.get(key)
        assert loaded_disk is not None
        assert np.allclose(loaded_disk, vec)


# ---------------------------------------------------------------------------
# 3. Visual-retrieval-enabled mode (ColPaliVisualRetriever + caching)
# ---------------------------------------------------------------------------

class TestVisualRetrievalEnabledMode:

    def test_colpali_retriever_ranks_pages_and_caches_embeddings(
        self, tmp_path: Path, corpus_and_pages
    ):
        _, _, doc_pages = corpus_and_pages
        encode_call_count: Dict[str, int] = {"images": 0, "queries": 0}

        # Deterministic encoder: encode page based on mean pixel signature; match 2005 page
        target_page = next(p for p in doc_pages if p.doc_id == "doc_2005_discharge")
        target_sig = float(np.asarray(target_page.image).mean())

        def fake_image_encoder(img: Image.Image) -> np.ndarray:
            encode_call_count["images"] += 1
            sig = float(np.asarray(img).mean())
            if abs(sig - target_sig) < 1e-4:
                return np.array([[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]], dtype=np.float32)
            return np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

        def fake_query_encoder(q: str) -> np.ndarray:
            encode_call_count["queries"] += 1
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        cache = EmbeddingCache(cache_dir=tmp_path / "colpali_cache", persist_to_disk=True)
        retriever = ColPaliVisualRetriever(
            config=ColPaliConfig(
                model_name="vidore/colpali-v1.2-mock",
                embedding_cache_dir=tmp_path / "colpali_cache",
                enabled=True,
            ),
            embedding_cache=cache,
            image_encoder_fn=fake_image_encoder,
            query_encoder_fn=fake_query_encoder,
        )

        assert isinstance(retriever, VisualPageRetriever)
        assert retriever.is_available()

        # First call encodes every page once
        hits1 = retriever.retrieve_pages("What was the HbA1c in 2005?", doc_pages, top_k=3)
        assert len(hits1) == 3
        assert hits1[0].page_id == "doc_2005_discharge::p1"
        assert hits1[0].rank == 1
        assert hits1[0].score > hits1[1].score
        assert all(not h.cached for h in hits1)
        first_image_calls = encode_call_count["images"]
        assert first_image_calls == len(doc_pages)

        # Second call reuses cached page embeddings without re-encoding any page images
        hits2 = retriever.retrieve_pages("What was the HbA1c in 2005?", doc_pages, top_k=3)
        assert encode_call_count["images"] == first_image_calls
        assert all(h.cached for h in hits2)
        assert hits2[0].page_id == "doc_2005_discharge::p1"


# ---------------------------------------------------------------------------
# 4. OCR-based page retriever fallback mode
# ---------------------------------------------------------------------------

class TestOCRFallbackMode:

    def test_ocr_fallback_returns_page_ids_and_scores(self, corpus_and_pages):
        corpus, _, doc_pages = corpus_and_pages
        fallback = OCRPageRetrieverFallback(observations=corpus)
        assert fallback.is_available()

        hits = fallback.retrieve_pages("Glycated Haemoglobin HbA1c in 2005", doc_pages, top_k=3)
        assert len(hits) >= 1
        assert isinstance(hits[0], VisualPageHit)
        assert hits[0].page_id == "doc_2005_discharge::p1"
        assert hits[0].doc_id == "doc_2005_discharge"
        assert hits[0].page_number == 1
        assert hits[0].score > 0
        assert hits[0].backend_name == "ocr-lexical-page-fallback"

    def test_engine_automatically_falls_back_when_colpali_disabled(self, corpus_and_pages):
        corpus, _, doc_pages = corpus_and_pages
        disabled_colpali = ColPaliVisualRetriever(config=ColPaliConfig(enabled=False))
        assert not disabled_colpali.is_available()

        engine = VisualRetrievalEngine(visual_retriever=disabled_colpali)
        outcome = engine.retrieve_and_connect(
            query="What was the heart rate on 15 Jan 1998?",
            pages=doc_pages,
            observations=corpus,
            top_k_pages=3,
            prefer_visual=True,
        )
        assert outcome.used_fallback
        assert outcome.backend_used == "ocr-lexical-page-fallback"
        assert outcome.fallback_reason is not None
        assert len(outcome.page_hits) >= 1
        assert outcome.page_hits[0].doc_id == "doc_1998_admission"

    def test_engine_falls_back_if_visual_retriever_raises(self, corpus_and_pages):
        corpus, _, doc_pages = corpus_and_pages

        class BrokenVisualRetriever(VisualPageRetriever):
            name = "broken-visual"

            def is_available(self) -> bool:
                return True

            def retrieve_pages(self, query, pages, top_k=5):
                raise RuntimeError("simulated GPU out-of-memory")

        engine = VisualRetrievalEngine(visual_retriever=BrokenVisualRetriever())
        outcome = engine.retrieve_and_connect(
            query="Blood pressure on 22 Apr 2002",
            pages=doc_pages,
            observations=corpus,
            top_k_pages=2,
            prefer_visual=True,
        )
        assert outcome.used_fallback
        assert "simulated GPU out-of-memory" in (outcome.fallback_reason or "")
        assert outcome.backend_used == "ocr-lexical-page-fallback"
        assert any(h.doc_id == "doc_2002_clinic" for h in outcome.page_hits)


# ---------------------------------------------------------------------------
# 5. Connecting pages to table regions, observations, dates, and coordinates
# ---------------------------------------------------------------------------

class TestPageToRegionConnection:

    def test_connected_pages_link_tables_observations_dates_and_bboxes(
        self, tmp_path: Path, corpus_and_pages
    ):
        corpus, page_dict, doc_pages = corpus_and_pages

        # Test in visual-retrieval-enabled mode with a deterministic encoder targeting 2002 clinic
        target_img = page_dict["doc_2002_clinic"][0]
        target_sig = float(np.asarray(target_img).mean())

        visual = ColPaliVisualRetriever(
            config=ColPaliConfig(embedding_cache_dir=tmp_path / "e", enabled=True),
            embedding_cache=EmbeddingCache(tmp_path / "e"),
            image_encoder_fn=lambda img: (
                np.array([[1.0, 0.0]], dtype=np.float32)
                if abs(float(np.asarray(img).mean()) - target_sig) < 1e-4
                else np.array([[0.0, 1.0]], dtype=np.float32)
            ),
            query_encoder_fn=lambda q: np.array([[1.0, 0.0]], dtype=np.float32),
        )
        engine = VisualRetrievalEngine(visual_retriever=visual)
        outcome = engine.retrieve_and_connect(
            query="What was the blood pressure in 2002?",
            pages=doc_pages,
            observations=corpus,
            top_k_pages=2,
            prefer_visual=True,
        )

        assert not outcome.used_fallback
        assert outcome.backend_used == "colpali-late-interaction"
        top_page = outcome.connected_pages[0]
        assert top_page.page_id == "doc_2002_clinic::p1"

        # Connected table regions from OpenCV table detector
        assert len(top_page.table_regions) >= 1
        for (tx, ty, tw, th) in top_page.table_regions:
            assert tw > 0 and th > 0

        # Connected observations & coordinates on that page
        assert len(top_page.observations) >= 5
        assert all(len(o.bbox) == 4 and o.bbox[2] > 0 for o in top_page.observations)

        # Connected dates on that page
        assert any("2002-04-22" in d for d in top_page.dates_on_page)

        # Connected verified observation for the query ("blood pressure in 2002")
        assert len(top_page.verified_evidence) == 1
        ev = top_page.verified_evidence[0]
        assert "130/80" in ev.text
        assert ev.doc_id == "doc_2002_clinic"
        assert ev.has_valid_coordinates()

        # Annotated image renders both table regions and observation boxes
        annotated = annotate_connected_page(target_img, top_page)
        assert isinstance(annotated, Image.Image)
        assert annotated.size == target_img.size

    def test_page_level_limitation_notice_is_explicit(self, corpus_and_pages):
        corpus, _, doc_pages = corpus_and_pages
        engine = VisualRetrievalEngine()
        outcome = engine.retrieve_and_connect(
            query="blood pressure",
            pages=doc_pages,
            observations=corpus,
            top_k_pages=2,
        )
        assert "does not automatically identify the exact table cell" in outcome.limitation_notice
        assert outcome.limitation_notice == PAGE_LEVEL_LIMITATION_NOTICE
        for h in outcome.page_hits:
            assert h.to_dict()["granularity"] == "page_level_only"
            assert h.granularity_note == PAGE_LEVEL_LIMITATION_NOTICE


# ---------------------------------------------------------------------------
# 6. Streamlit UI integration test for pages/04_Retrieve.py
# ---------------------------------------------------------------------------

class TestRetrieveStreamlitPageIntegration:

    def test_retrieve_page_renders_and_runs_visual_retrieval_without_exception(self):
        from streamlit.testing.v1 import AppTest

        page_path = str(Path(__file__).resolve().parent.parent / "pages" / "04_Retrieve.py")
        at = AppTest.from_file(page_path, default_timeout=20)
        at.run()
        assert not at.exception
        assert len(at.dataframe) >= 2  # Structured evidence table + visual page hits table

        # Enter a new query and run retrieval
        at.text_input(key="current_user_query").set_value("What was the blood pressure in 2002?").run()
        assert not at.exception
        assert any(PAGE_LEVEL_LIMITATION_NOTICE in i.value for i in at.info)



