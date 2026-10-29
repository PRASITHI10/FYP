"""
tests/test_qa_pipeline.py
-------------------------
Integration tests for the evidence-grounded Q&A pipeline:
retrieval -> temporal verification -> answer generation (template / optional LLM).

Covers: evidence-only answers, provenance, chronological comparison, insufficient
evidence, conflicts, uncertain OCR, missing timestamps, LLM fallback & grounding
guard, source-page highlighting, and an end-to-end run from a synthetic PDF
through DocumentProcessor.
"""

import pytest
from PIL import Image

from utils.qa_pipeline import (
    RESEARCH_DISCLAIMER,
    CallableLLMAnswerGenerator,
    EvidenceQAPipeline,
    TemplateAnswerGenerator,
    build_llm_prompt,
    evidence_row,
    get_source_page_image,
    highlight_evidence_regions,
    validate_grounded_answer,
)
from utils.synthetic_docs import (
    create_synthetic_retrieval_corpus,
    create_synthetic_verification_documents,
)
from utils.temporal_verification import VerificationStatus


@pytest.fixture(scope="module")
def docs():
    return create_synthetic_verification_documents()


@pytest.fixture
def pipeline():
    return EvidenceQAPipeline()


# ---------------------------------------------------------------------------
# Evidence-only answers and provenance
# ---------------------------------------------------------------------------

class TestEvidenceGroundedAnswers:

    def test_clean_answer_cites_only_supplied_evidence(self, pipeline):
        corpus = create_synthetic_retrieval_corpus()
        res = pipeline.answer("What was the heart rate on 15 Jan 1998?", corpus)
        assert not res.insufficient_evidence
        assert "Heart Rate: 88 bpm" in res.answer.text
        assert "[E1]" in res.answer.text
        # 2002 / 2005 heart rates must not leak into a 1998 answer
        assert "72 bpm" not in res.answer.text
        assert "70 bpm" not in res.answer.text
        assert res.answer.generator_name == "evidence-template"

    def test_every_cited_item_has_provenance(self, pipeline, docs):
        corpus, _ = docs
        res = pipeline.answer("Show fasting blood glucose trend over time", corpus)
        assert res.evidence
        for i, ev in enumerate(res.evidence, start=1):
            row = evidence_row(ev, f"[E{i}]")
            assert row["Document"]
            assert row["Page"] >= 1
            assert row["Region (x,y,w,h)"] != ""
            assert row["Recorded text"] == ev.text
            assert row["Status"] in {"VERIFIED", "UNCERTAIN", "CONFLICTING"}

    def test_narrative_mention_is_not_a_measurement(self, pipeline):
        corpus = create_synthetic_retrieval_corpus()
        res = pipeline.answer("What was the blood pressure on 15 Jan 1998?", corpus)
        verified_texts = [e.text for e in res.package.verified]
        assert verified_texts == ["Blood Pressure: 140/90 mmHg"]
        assert any("Related mention only" in r.reason for r in res.package.rejected)

    def test_disclaimer_present(self, pipeline):
        res = pipeline.answer("heart rate", create_synthetic_retrieval_corpus())
        assert "not a clinical diagnostic tool" in res.disclaimer
        assert res.disclaimer == RESEARCH_DISCLAIMER


# ---------------------------------------------------------------------------
# Chronological comparison
# ---------------------------------------------------------------------------

class TestChronologicalComparison:

    def test_trend_rows_are_time_ordered(self, pipeline, docs):
        corpus, _ = docs
        res = pipeline.answer("Show fasting blood glucose trend over time", corpus)
        assert len(res.comparison_rows) >= 3
        keys = [e.normalized_date.sort_key for e in res.evidence]
        assert keys == sorted(keys)
        dates = [r["Date"] for r in res.comparison_rows]
        assert dates.index("1998-01-15") < dates.index("2002-04-22") < dates.index("2005-11-10")
        assert "no trend or causal interpretation" in res.answer.text

    def test_two_date_comparison_includes_both_dates(self, pipeline):
        corpus = create_synthetic_retrieval_corpus()
        res = pipeline.answer("Compare blood pressure in 1998 and 2002", corpus)
        texts = [r["Recorded text"] for r in res.comparison_rows]
        assert "Blood Pressure: 140/90 mmHg" in texts
        assert "Blood Pressure: 130/80 mmHg" in texts
        assert not any("124/78" in t for t in texts)  # 2005 is out of scope


# ---------------------------------------------------------------------------
# Insufficient / conflicting / uncertain evidence reporting
# ---------------------------------------------------------------------------

class TestEvidenceQualityReporting:

    def test_insufficient_evidence_absent_concept(self, pipeline, docs):
        corpus, _ = docs
        res = pipeline.answer("What were the brain MRI results?", corpus)
        assert res.insufficient_evidence
        assert "Insufficient evidence" in res.answer.text
        assert res.evidence == []

    def test_insufficient_evidence_out_of_period(self, pipeline):
        corpus = create_synthetic_retrieval_corpus()
        res = pipeline.answer("What was the blood pressure in 2010?", corpus)
        assert res.insufficient_evidence
        assert "Insufficient evidence" in res.answer.text
        assert "130" not in res.answer.text.split("rejected")[0]

    def test_empty_question_and_empty_corpus(self, pipeline, docs):
        corpus, _ = docs
        assert pipeline.answer("", corpus).insufficient_evidence
        assert pipeline.answer("blood pressure", []).insufficient_evidence

    def test_conflicts_reported_without_choosing(self, pipeline, docs):
        corpus, _ = docs
        res = pipeline.answer("Check blood pressure on 15 Jan 1998", corpus)
        assert res.has_conflicts
        assert "Conflicting evidence" in res.answer.text
        assert "None of them is selected as correct" in res.answer.text
        assert "140/90" in res.answer.text and "110/70" in res.answer.text
        # conflicting values are never listed as verified
        assert all(e.status != VerificationStatus.VERIFIED for e in res.package.conflicting)

    def test_uncertain_ocr_flagged(self, pipeline, docs):
        corpus, _ = docs
        res = pipeline.answer("Check blood pressure on 15 Jan 1998", corpus)
        assert res.has_uncertain_ocr
        assert "Uncertain OCR quality" in res.answer.text
        assert "138/88" not in [e.text for e in res.package.verified]

    def test_missing_timestamp_reported(self, pipeline, docs):
        corpus, _ = docs
        res = pipeline.answer("What was the heart rate in 1998?", corpus)
        assert res.missing_timestamp_count == 1
        assert "no timestamp" in res.answer.text

    def test_ambiguous_date_not_verified(self, pipeline, docs):
        corpus, _ = docs
        res = pipeline.answer("Show glucose levels in 1998", corpus)
        amb = [e for e in res.package.uncertain if e.normalized_date.is_ambiguous]
        assert amb and "Ambiguous numeric date" in res.answer.text


# ---------------------------------------------------------------------------
# Optional LLM behind a clean interface
# ---------------------------------------------------------------------------

class TestOptionalLLM:

    Q = "What was the heart rate on 15 Jan 1998?"

    def test_unconfigured_llm_falls_back(self, pipeline):
        res = pipeline.answer(self.Q, create_synthetic_retrieval_corpus(),
                              generator=CallableLLMAnswerGenerator(None))
        assert res.answer.used_fallback
        assert res.answer.generator_name == "evidence-template"
        assert "not available" in res.answer.fallback_reason

    def test_failing_llm_falls_back(self, pipeline):
        def boom(_):
            raise ConnectionError("offline")
        res = pipeline.answer(self.Q, create_synthetic_retrieval_corpus(),
                              generator=CallableLLMAnswerGenerator(boom))
        assert res.answer.used_fallback and "offline" in res.answer.fallback_reason
        assert "Heart Rate: 88 bpm" in res.answer.text

    def test_hallucinated_number_rejected(self, pipeline):
        fake = CallableLLMAnswerGenerator(lambda p: "The heart rate was 95 bpm [E1].")
        res = pipeline.answer(self.Q, create_synthetic_retrieval_corpus(), generator=fake)
        assert res.answer.used_fallback
        assert "not present in evidence" in res.answer.fallback_reason
        assert "95" not in res.answer.text

    def test_uncited_llm_answer_rejected(self, pipeline):
        fake = CallableLLMAnswerGenerator(lambda p: "The heart rate was 88 bpm.")
        res = pipeline.answer(self.Q, create_synthetic_retrieval_corpus(), generator=fake)
        assert res.answer.used_fallback and "citations" in res.answer.fallback_reason

    def test_grounded_llm_answer_accepted(self, pipeline):
        fake = CallableLLMAnswerGenerator(
            lambda p: "On 15 Jan 1998 the recorded heart rate was 88 bpm [E1].", name="fake-llm")
        res = pipeline.answer(self.Q, create_synthetic_retrieval_corpus(), generator=fake)
        assert not res.answer.used_fallback
        assert res.answer.generator_name == "fake-llm"

    def test_llm_not_called_without_evidence(self, pipeline):
        calls = []
        fake = CallableLLMAnswerGenerator(lambda p: calls.append(p) or "x [E1]")
        res = pipeline.answer("brain MRI results", create_synthetic_retrieval_corpus(), generator=fake)
        assert calls == []
        assert res.insufficient_evidence and res.answer.used_fallback

    def test_prompt_contains_only_package_evidence(self, pipeline):
        res = pipeline.answer(self.Q, create_synthetic_retrieval_corpus())
        prompt = build_llm_prompt(res.package)
        assert "Heart Rate: 88 bpm" in prompt
        assert "Heart Rate: 72 bpm" not in prompt
        assert "ONLY the evidence" in prompt

    def test_validator_accepts_evidence_numbers(self, pipeline):
        res = pipeline.answer(self.Q, create_synthetic_retrieval_corpus())
        ok, _ = validate_grounded_answer("Heart rate 88 bpm on 1998-01-15 [E1].", res.package)
        assert ok


# ---------------------------------------------------------------------------
# Source page inspection
# ---------------------------------------------------------------------------

class TestSourceInspection:

    def test_every_evidence_item_has_a_page_image(self, pipeline, docs):
        corpus, pages = docs
        res = pipeline.answer("Check blood pressure on 15 Jan 1998", corpus)
        for ev in res.evidence:
            img = get_source_page_image(pages, ev.doc_id, ev.page_number)
            assert isinstance(img, Image.Image)
            x, y, w, h = ev.bbox
            assert x + w <= img.width and y + h <= img.height

    def test_highlight_changes_region_pixels(self, pipeline, docs):
        corpus, pages = docs
        res = pipeline.answer("What was the heart rate on 15 Jan 1998?", corpus)
        ev = res.package.verified[0]
        page = get_source_page_image(pages, ev.doc_id, ev.page_number)
        out = highlight_evidence_regions(page, [ev], labels=["E1"])
        assert out.size == page.size
        x, y, w, h = ev.bbox
        assert out.getpixel((x + 1, y + h // 2)) != page.getpixel((x + 1, y + h // 2))

    def test_missing_page_returns_none(self):
        assert get_source_page_image({}, "nope", 1) is None
        assert get_source_page_image({"d": [Image.new("RGB", (5, 5))]}, "d", 3) is None


# ---------------------------------------------------------------------------
# End-to-end: synthetic PDF -> extraction -> Q&A
# ---------------------------------------------------------------------------

class TestEndToEndFromPdf:

    def test_pdf_extraction_to_answer(self, pipeline):
        from utils.extraction import DocumentProcessor
        from utils.pdf_utils import pdf_to_images
        from utils.synthetic_docs import make_text_pdf

        pdf = make_text_pdf()
        obs = DocumentProcessor(doc_id="e2e.pdf", render_dpi=100).process_pdf(pdf)
        pages = {"e2e.pdf": pdf_to_images(pdf, dpi=100)}

        res = pipeline.answer("What was the haemoglobin?", obs)
        assert res.evidence, res.answer.text
        ev = res.evidence[0]
        assert ev.doc_id == "e2e.pdf"
        assert "Haemoglobin" in ev.text
        img = get_source_page_image(pages, ev.doc_id, ev.page_number)
        assert img is not None
        x, y, w, h = ev.bbox
        assert 0 <= x < img.width and 0 <= y < img.height


# ---------------------------------------------------------------------------
# Streamlit UI integration tests (pages/06_Ask.py via AppTest)
# ---------------------------------------------------------------------------

class TestStreamlitAskPageUI:

    @staticmethod
    def _ask_page_path() -> str:
        from pathlib import Path
        return str(Path(__file__).resolve().parent.parent / "pages" / "06_Ask.py")

    def test_ask_page_conflict_and_source_inspection(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(self._ask_page_path(), default_timeout=60).run()
        assert not at.exception
        assert any("not a clinical diagnostic tool" in e.value for e in at.error)

        # Click "BP on 15 Jan 1998 (conflict)" example button
        at.button[2].click().run()
        assert not at.exception
        assert any("Conflicting evidence" in e.value for e in at.error)
        assert any("Uncertain OCR" in w.value for w in at.warning)
        assert len(at.dataframe) >= 1
        assert len(at.selectbox) == 1

        # Switch inspected source region item
        at.selectbox[0].select(1).run()
        assert not at.exception

    def test_ask_page_chronological_comparison_and_optional_llm_fallback(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(self._ask_page_path(), default_timeout=60).run()
        # Switch sidebar radio to "Optional language model" (none configured -> template fallback)
        at.sidebar.radio[0].set_value("Optional language model").run()
        # Click "Glucose over time" example button
        at.button[0].click().run()
        assert not at.exception
        assert any("Chronological comparison" in m.value for m in at.markdown)
        assert any("fallback" in c.value for c in at.caption)

    def test_ask_page_insufficient_evidence(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(self._ask_page_path(), default_timeout=60).run()
        # Click "Brain MRI (no evidence)" example button
        at.button[4].click().run()
        assert not at.exception
        assert any("Insufficient evidence" in w.value for w in at.warning)

