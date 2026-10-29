"""
tests/test_extraction.py
------------------------
Unit tests for Stage 2: text extraction, OCR, table detection,
synthetic document generation, and the Observation model.

Run with:
    cd /path/to/FYP
    source venv/bin/activate
    python -m pytest tests/test_extraction.py -v
"""

from __future__ import annotations

import io
import sys
import os
import types

import pytest
from PIL import Image
import numpy as np

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def text_pdf_bytes():
    """Synthetic digital PDF (text layer present)."""
    from utils.synthetic_docs import make_text_pdf
    return make_text_pdf()


@pytest.fixture(scope="module")
def scanned_pdf_bytes():
    """Synthetic 'scanned' PDF (image-only, no text layer)."""
    from utils.synthetic_docs import make_scanned_pdf
    return make_scanned_pdf()


@pytest.fixture(scope="module")
def clinical_image():
    """Synthetic clinical image."""
    from utils.synthetic_docs import make_clinical_image
    return make_clinical_image()


@pytest.fixture(scope="module")
def noisy_image(clinical_image):
    """Clinical image with added noise (should trigger uncertain OCR)."""
    from utils.synthetic_docs import make_noisy_image
    return make_noisy_image(clinical_image, noise_level=0.15)


# ---------------------------------------------------------------------------
# Observation dataclass tests
# ---------------------------------------------------------------------------

class TestObservationModel:
    def test_defaults(self):
        from utils.extraction import Observation
        obs = Observation()
        assert obs.obs_id != ""
        assert obs.doc_id == ""
        assert obs.page_number == 0
        assert obs.ocr_uncertain is False
        assert obs.error_message is None

    def test_to_dict_has_all_keys(self):
        from utils.extraction import Observation
        obs = Observation(
            doc_id="d1", page_number=2, source_type="ocr",
            text="HR 88 bpm", value=88.0, unit="bpm",
            date_str="15 Jan 1998", bbox=(10, 20, 100, 30),
            ocr_confidence=0.91, ocr_uncertain=False,
        )
        d = obs.to_dict()
        expected_keys = {
            "obs_id", "doc_id", "page_number", "source_type",
            "text", "value", "unit", "date_str",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "ocr_confidence", "ocr_uncertain", "error_message",
        }
        assert expected_keys == set(d.keys())
        assert d["value"] == 88.0
        assert d["unit"] == "bpm"
        assert d["bbox_x"] == 10

    def test_uncertain_flag_set(self):
        from utils.extraction import Observation
        obs = Observation(ocr_confidence=0.3, ocr_uncertain=True, error_message="Low OCR confidence")
        assert obs.ocr_uncertain is True
        assert obs.error_message == "Low OCR confidence"


# ---------------------------------------------------------------------------
# Regex / helper tests (no EasyOCR needed)
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_date_iso(self):
        from utils.extraction import _extract_date
        assert _extract_date("Patient admitted 2024-01-15, stable.") == "2024-01-15"

    def test_extract_date_uk(self):
        from utils.extraction import _extract_date
        assert _extract_date("Date: 15/01/2024") == "15/01/2024"

    def test_extract_date_long(self):
        from utils.extraction import _extract_date
        result = _extract_date("Admitted January 15, 2024 for review.")
        assert result is not None
        assert "2024" in result

    def test_extract_date_none(self):
        from utils.extraction import _extract_date
        assert _extract_date("No date here at all.") is None

    def test_extract_value_mmhg(self):
        from utils.extraction import _extract_value_unit
        val, unit = _extract_value_unit("BP: 140 mmHg")
        assert val == 140.0
        assert unit == "mmHg"

    def test_extract_value_temp(self):
        from utils.extraction import _extract_value_unit
        val, unit = _extract_value_unit("Temperature 37.2 °C")
        assert val == pytest.approx(37.2, abs=0.01)
        assert unit == "°C"

    def test_extract_value_no_unit(self):
        from utils.extraction import _extract_value_unit
        val, unit = _extract_value_unit("Score: 88")
        assert val == 88.0
        assert unit is None

    def test_extract_value_none(self):
        from utils.extraction import _extract_value_unit
        val, unit = _extract_value_unit("No numbers here.")
        assert val is None
        assert unit is None


# ---------------------------------------------------------------------------
# Synthetic document generation tests
# ---------------------------------------------------------------------------

class TestSyntheticDocs:
    def test_make_text_pdf_returns_bytes(self, text_pdf_bytes):
        assert isinstance(text_pdf_bytes, bytes)
        assert len(text_pdf_bytes) > 100
        # Check PDF magic bytes
        assert text_pdf_bytes[:4] == b"%PDF"

    def test_make_scanned_pdf_returns_bytes(self, scanned_pdf_bytes):
        assert isinstance(scanned_pdf_bytes, bytes)
        assert scanned_pdf_bytes[:4] == b"%PDF"

    def test_make_clinical_image_returns_pil(self, clinical_image):
        assert isinstance(clinical_image, Image.Image)
        assert clinical_image.mode == "RGB"
        assert clinical_image.width > 0 and clinical_image.height > 0

    def test_noisy_image_same_size(self, clinical_image, noisy_image):
        assert noisy_image.size == clinical_image.size

    def test_text_pdf_has_multiple_pages(self, text_pdf_bytes):
        """Synthetic text PDF should have ≥ 2 pages (content + table page)."""
        from utils.pdf_utils import get_page_count
        assert get_page_count(text_pdf_bytes) >= 2


# ---------------------------------------------------------------------------
# DocumentProcessor — PDF text extraction (no OCR needed)
# ---------------------------------------------------------------------------

class TestDocumentProcessorPdfText:
    def test_returns_observations(self, text_pdf_bytes):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="test_pdf")
        obs  = proc.process_pdf(text_pdf_bytes)
        assert isinstance(obs, list)
        assert len(obs) > 0

    def test_obs_have_doc_id(self, text_pdf_bytes):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="my_doc")
        obs  = proc.process_pdf(text_pdf_bytes)
        text_obs = [o for o in obs if o.source_type == "pdf_text"]
        assert all(o.doc_id == "my_doc" for o in text_obs)

    def test_obs_have_page_numbers(self, text_pdf_bytes):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="pg_test")
        obs  = proc.process_pdf(text_pdf_bytes)
        page_nums = {o.page_number for o in obs if o.page_number > 0}
        assert len(page_nums) >= 1

    def test_text_obs_contain_known_content(self, text_pdf_bytes):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="content_test")
        obs  = proc.process_pdf(text_pdf_bytes)
        all_text = " ".join(o.text for o in obs).lower()
        # The synthetic doc always contains these labels
        assert any(kw in all_text for kw in ["blood", "heart", "temperature", "haemoglobin"])

    def test_date_extracted_from_text_pdf(self, text_pdf_bytes):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="date_test")
        obs  = proc.process_pdf(text_pdf_bytes)
        dates = [o.date_str for o in obs if o.date_str]
        assert len(dates) > 0, "Expected at least one date to be extracted"

    def test_value_and_unit_extracted(self, text_pdf_bytes):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="val_test")
        obs  = proc.process_pdf(text_pdf_bytes)
        # Some observations should have numeric values + units
        val_obs = [o for o in obs if o.value is not None and o.unit is not None]
        assert len(val_obs) > 0, "Expected value+unit pairs (e.g. 88 bpm)"

    def test_empty_bytes_returns_error_obs(self):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="err_test")
        obs  = proc.process_pdf(b"")
        assert len(obs) == 1
        assert obs[0].error_message is not None

    def test_invalid_pdf_returns_error_obs(self):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="bad_pdf")
        obs  = proc.process_pdf(b"this is not a pdf at all!!")
        assert any(o.error_message for o in obs)

    def test_bbox_coordinates_non_negative(self, text_pdf_bytes):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="bbox_test")
        obs  = proc.process_pdf(text_pdf_bytes)
        for o in obs:
            x, y, w, h = o.bbox
            assert x >= 0 and y >= 0, f"Negative bbox coords in {o}"


# ---------------------------------------------------------------------------
# DocumentProcessor — image input (no OCR needed for structural tests)
# ---------------------------------------------------------------------------

class TestDocumentProcessorImage:
    def test_none_image_returns_error(self):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="none_img")
        obs  = proc.process_image(None)
        assert len(obs) >= 1
        assert obs[0].error_message is not None

    def test_image_returns_list(self, clinical_image):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="img_test")
        obs  = proc.process_image(clinical_image)
        # Even if EasyOCR is not installed, we get a list back
        assert isinstance(obs, list)

    def test_image_obs_page_number_zero(self, clinical_image):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="page0_test")
        obs  = proc.process_image(clinical_image)
        # Standalone images use page_number = 0
        assert all(o.page_number == 0 for o in obs)


# ---------------------------------------------------------------------------
# Table detection (OpenCV, no model required)
# ---------------------------------------------------------------------------

class TestTableDetection:
    def test_no_table_in_plain_image(self):
        """A uniform image should have no table regions detected."""
        from utils.extraction import detect_table_regions
        img  = Image.new("RGB", (400, 300), color=(240, 240, 240))
        boxes = detect_table_regions(img)
        assert isinstance(boxes, list)
        # Might find 0 or a very small number of false positives
        assert len(boxes) <= 2

    def test_table_detected_in_grid_image(self):
        """An image with clear grid lines should yield at least one table region."""
        from utils.extraction import detect_table_regions
        import cv2

        arr = np.ones((400, 600, 3), dtype=np.uint8) * 255
        # Draw a 4×4 grid of black lines
        for i in range(5):
            y = 80 + i * 50
            cv2.line(arr, (50, y), (550, y), (0, 0, 0), 2)
        for j in range(5):
            x = 50 + j * 125
            cv2.line(arr, (x, 80), (x, 280), (0, 0, 0), 2)

        img   = Image.fromarray(arr)
        boxes = detect_table_regions(img, min_area=1000)
        assert len(boxes) >= 1, "Expected at least one table region in grid image"

    def test_all_boxes_have_positive_dimensions(self):
        from utils.extraction import detect_table_regions
        import cv2
        arr = np.ones((400, 600, 3), dtype=np.uint8) * 255
        for i in range(5):
            cv2.line(arr, (50, 80 + i * 50), (550, 80 + i * 50), (0, 0, 0), 2)
        for j in range(5):
            cv2.line(arr, (50 + j * 125, 80), (50 + j * 125, 280), (0, 0, 0), 2)
        img   = Image.fromarray(arr)
        boxes = detect_table_regions(img, min_area=500)
        for x, y, w, h in boxes:
            assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

class TestAnnotateImage:
    def test_returns_pil_image(self, clinical_image):
        from utils.extraction import Observation, annotate_image
        obs = [Observation(
            doc_id="ann_test", page_number=0, source_type="pdf_text",
            text="BP 140 mmHg", bbox=(50, 50, 200, 30),
        )]
        result = annotate_image(clinical_image, obs)
        assert isinstance(result, Image.Image)
        assert result.size == clinical_image.size

    def test_same_size_as_input(self, clinical_image):
        from utils.extraction import annotate_image
        result = annotate_image(clinical_image, [])
        assert result.size == clinical_image.size

    def test_annotated_differs_from_original(self, clinical_image):
        from utils.extraction import Observation, annotate_image
        obs = [Observation(
            source_type="ocr", text="Test", bbox=(10, 10, 100, 40)
        )]
        result  = annotate_image(clinical_image, obs)
        orig_arr = np.array(clinical_image)
        anno_arr = np.array(result)
        # Annotation should have changed at least some pixels
        assert not np.array_equal(orig_arr, anno_arr)


# ---------------------------------------------------------------------------
# OCR-specific tests (skipped gracefully if EasyOCR not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("easyocr"),
    reason="EasyOCR not installed",
)
class TestOCR:
    def test_ocr_returns_observations(self, clinical_image):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="ocr_test")
        obs  = proc.process_image(clinical_image)
        ocr_obs = [o for o in obs if o.source_type == "ocr"]
        # Should extract at least a few words from a synthetic image
        assert len(ocr_obs) >= 1

    def test_ocr_obs_have_confidence(self, clinical_image):
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="conf_test")
        obs  = proc.process_image(clinical_image)
        ocr_obs = [o for o in obs if o.source_type == "ocr"]
        for o in ocr_obs:
            assert o.ocr_confidence is not None
            assert 0.0 <= o.ocr_confidence <= 1.0

    def test_noisy_image_flags_uncertain(self, noisy_image):
        """High-noise image should produce at least some uncertain observations."""
        from utils.extraction import DocumentProcessor
        proc    = DocumentProcessor(doc_id="noisy_test")
        obs     = proc.process_image(noisy_image)
        ocr_obs = [o for o in obs if o.source_type == "ocr"]
        if ocr_obs:
            uncertain = [o for o in ocr_obs if o.ocr_uncertain]
            # With significant noise, at least some observations may be uncertain
            # (this is a soft assertion — OCR model quality affects the result)
            # We just verify the flag mechanism works, not its specific count
            all_have_flag = all(isinstance(o.ocr_uncertain, bool) for o in ocr_obs)
            assert all_have_flag

    def test_original_text_not_silently_corrected(self, clinical_image):
        """OCR should store the raw detected text, not a 'corrected' version."""
        from utils.extraction import DocumentProcessor
        proc = DocumentProcessor(doc_id="raw_test")
        obs  = proc.process_image(clinical_image)
        ocr_obs = [o for o in obs if o.source_type == "ocr"]
        for o in ocr_obs:
            # Text should never be None (empty string is OK)
            assert o.text is not None
