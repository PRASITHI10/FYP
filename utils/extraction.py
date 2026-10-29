"""
utils/extraction.py
-------------------
Core document-processing module for Temporal-RegionRAG (Stage 2).

Responsibilities
----------------
1. Extract text + page numbers from digital PDFs via PyMuPDF.
2. Run EasyOCR on image regions and scanned (image-based) PDF pages.
3. Detect table regions via rule-based OpenCV contour analysis.
4. Return each extracted unit as an ``Observation`` dataclass that carries:
   - text, value (if parseable), unit, date/time (if found)
   - document ID, page number, bounding-box (x, y, w, h in pixels)
   - ocr_uncertain flag, error_message
5. Never silently correct OCR; flag uncertain results instead.
6. Handle missing dates, unreadable text, and processing failures gracefully.

Usage
-----
    from utils.extraction import DocumentProcessor
    proc = DocumentProcessor(doc_id="patient_001")
    obs  = proc.process_pdf(pdf_bytes)          # list[Observation]
    obs  = proc.process_image(pil_image)        # list[Observation]
"""

from __future__ import annotations

import io
import os
import re
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully so the rest of the app still works
# ---------------------------------------------------------------------------

try:
    import pymupdf as fitz
    _PYMUPDF_OK = True
except ImportError:
    try:
        import fitz
        _PYMUPDF_OK = True
    except ImportError:
        _PYMUPDF_OK = False
        logger.warning("PyMuPDF not available — PDF text extraction disabled.")

try:
    import easyocr
    _EASYOCR_OK = True
except ImportError:
    _EASYOCR_OK = False
    logger.warning("EasyOCR not installed — OCR features disabled.")

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    logger.warning("OpenCV not available — table detection disabled.")

# ---------------------------------------------------------------------------
# Regex patterns for lightweight extraction
# ---------------------------------------------------------------------------

# Date patterns (various formats found in historical medical records)
_DATE_PATTERNS = [
    # ISO:  2024-01-15, 1998-07-22
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    # UK:   15/01/2024, 22.07.1998
    re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b"),
    # Long: January 15, 2024 / 15 Jan 1998
    re.compile(
        r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
]

# Numeric value + optional unit (e.g. "120 mmHg", "140/90 mmHg", "98.6 °F", "36.5°C")
_VALUE_PATTERN = re.compile(
    r"([-+]?\d+(?:[.,]\d+)?)(?:/\d+(?:[.,]\d+)?)?\s*"
    r"(mmHg|bpm|°C|°F|mg/dL|g/dL|mEq/L|U/L|IU/L|kg|lbs?|cm|mm|%|"
    r"mL|L|mg|µg|ng|pg|mmol/L|µmol/L|beats?/min)?\b",
    re.IGNORECASE,
)

# OCR confidence threshold below which we flag uncertainty
_OCR_CONFIDENCE_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Observation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    One extracted unit from a document.

    All spatial coordinates are in pixels relative to the rendered page image
    at the DPI used during extraction.  (x, y) = top-left corner, then w × h.
    """

    obs_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    doc_id:       str = ""
    page_number:  int = 0           # 1-indexed; 0 = standalone image
    source_type:  str = ""          # "pdf_text" | "ocr" | "table_cell"

    text:         str = ""
    value:        Optional[float] = None
    unit:         Optional[str]  = None
    date_str:     Optional[str]  = None   # raw string from document

    bbox:         Tuple[int,int,int,int] = (0, 0, 0, 0)  # x, y, w, h (pixels)

    ocr_confidence: Optional[float] = None   # 0–1; None for non-OCR sources
    ocr_uncertain:  bool = False             # True when confidence < threshold
    error_message:  Optional[str] = None     # set on partial/failed extraction

    def to_dict(self) -> dict:
        return {
            "obs_id":        self.obs_id,
            "doc_id":        self.doc_id,
            "page_number":   self.page_number,
            "source_type":   self.source_type,
            "text":          self.text,
            "value":         self.value,
            "unit":          self.unit,
            "date_str":      self.date_str,
            "bbox_x":        self.bbox[0],
            "bbox_y":        self.bbox[1],
            "bbox_w":        self.bbox[2],
            "bbox_h":        self.bbox[3],
            "ocr_confidence": self.ocr_confidence,
            "ocr_uncertain": self.ocr_uncertain,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_date(text: str) -> Optional[str]:
    """Return the first date-like string found in *text*, or None."""
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def _extract_value_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Try to parse a numeric value and optional unit from *text*.
    Returns (value, unit) or (None, None) when nothing is found.
    """
    m = _VALUE_PATTERN.search(text)
    if not m:
        return None, None
    try:
        val = float(m.group(1).replace(",", "."))
    except ValueError:
        return None, None
    unit = m.group(2) or None
    return val, unit


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    if _CV2_OK:
        import cv2
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb[:, :, ::-1]  # simple channel swap fallback


# ---------------------------------------------------------------------------
# EasyOCR reader  (lazy singleton — model loads once)
# ---------------------------------------------------------------------------

_ocr_reader = None

# Default model directory inside the project to allow sandboxed / offline execution
_DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent.parent / ".easyocr_models")


def _get_ocr_reader(languages: List[str] = None):
    """Return (or create) the shared EasyOCR Reader instance."""
    global _ocr_reader
    if not _EASYOCR_OK:
        return None
    if _ocr_reader is None:
        langs = languages or ["en"]
        logger.info("Loading EasyOCR model (first run may take a moment) …")
        model_dir = os.environ.get("EASYOCR_MODEL_DIR", _DEFAULT_MODEL_DIR)
        try:
            os.makedirs(model_dir, exist_ok=True)
            _ocr_reader = easyocr.Reader(
                langs,
                gpu=False,
                verbose=False,
                model_storage_directory=model_dir,
                user_network_directory=model_dir,
            )
        except Exception as exc:
            logger.warning("Failed to initialize EasyOCR Reader: %s", exc)
            _ocr_reader = None
    return _ocr_reader


# ---------------------------------------------------------------------------
# Table detection (OpenCV contour-based, no heavyweight models)
# ---------------------------------------------------------------------------

def detect_table_regions(
    img: Image.Image,
    min_area: int = 5000,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect rectangular table-like regions in *img* using morphological
    operations and contour analysis.

    Returns a list of (x, y, w, h) bounding boxes for detected regions.
    Falls back to an empty list if OpenCV is unavailable or the image has
    no clear table structure.
    """
    if not _CV2_OK:
        logger.warning("OpenCV unavailable — table detection skipped.")
        return []

    import cv2

    bgr = _pil_to_bgr(img)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(
        grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # Combine — where both horizontal and vertical lines exist → table
    combined = cv2.add(horiz, vert)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, dilate_k, iterations=4)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    img_area = img.width * img.height
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filter noise and trivially small or full-image boxes
        if area < min_area or area > img_area * 0.95:
            continue
        # Must be roughly rectangular (aspect ratio sanity)
        if w < 20 or h < 10:
            continue
        boxes.append((x, y, w, h))

    return boxes


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    Orchestrates text extraction, OCR, and table detection for a single
    document.

    Parameters
    ----------
    doc_id : str
        Identifier for the document being processed (e.g. filename stem).
    ocr_languages : list[str]
        Language codes passed to EasyOCR.  Default ['en'].
    render_dpi : int
        DPI used when rendering PDF pages to images for OCR.
    """

    def __init__(
        self,
        doc_id: str = "unknown",
        ocr_languages: Optional[List[str]] = None,
        render_dpi: int = 150,
    ):
        self.doc_id        = doc_id
        self.ocr_languages = ocr_languages or ["en"]
        self.render_dpi    = render_dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_bytes: bytes) -> List[Observation]:
        """
        Process a PDF from raw bytes.

        Strategy:
        - Pages with embedded text → extract via PyMuPDF (fast, accurate).
        - Pages with no text (scanned) → render to image and run EasyOCR.
        - Table regions are detected on the rendered page image in both cases.
        """
        if not pdf_bytes:
            return [self._error_obs(0, "Empty PDF bytes supplied.")]

        if not _PYMUPDF_OK:
            return [self._error_obs(0, "PyMuPDF not installed.")]

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            return [self._error_obs(0, f"Cannot open PDF: {exc}")]

        all_obs: List[Observation] = []
        zoom = self.render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page     = doc.load_page(page_idx)

            # Render page image (used for table detection + OCR fallback)
            try:
                pix    = page.get_pixmap(matrix=matrix, alpha=False)
                img_io = io.BytesIO(pix.tobytes("png"))
                page_img = Image.open(img_io).convert("RGB")
            except Exception as exc:
                all_obs.append(
                    self._error_obs(page_num, f"Render failed: {exc}")
                )
                continue

            # --- Text extraction ---
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            page_obs  = self._extract_text_blocks(text_dict, page_num, zoom)

            if not page_obs:
                # Scanned page — fall back to OCR on the full rendered image
                page_obs = self._run_ocr_on_image(page_img, page_num)

            # --- Table detection on rendered image ---
            table_obs = self._detect_and_crop_tables(page_img, page_num)

            all_obs.extend(page_obs)
            all_obs.extend(table_obs)

        doc.close()
        return all_obs

    def process_image(self, img: Image.Image) -> List[Observation]:
        """
        Process a standalone image (JPG, PNG, etc.).
        Runs EasyOCR + table detection.
        """
        if img is None:
            return [self._error_obs(0, "No image provided.")]

        try:
            img = img.convert("RGB")
        except Exception as exc:
            return [self._error_obs(0, f"Image conversion failed: {exc}")]

        obs: List[Observation] = []
        obs.extend(self._run_ocr_on_image(img, page_number=0))
        obs.extend(self._detect_and_crop_tables(img, page_number=0))
        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text_blocks(
        self,
        text_dict: dict,
        page_number: int,
        zoom: float,
    ) -> List[Observation]:
        """Parse PyMuPDF text dict → Observation list."""
        obs_list: List[Observation] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                block_text += "\n"
            block_text = block_text.strip()
            if not block_text:
                continue

            # PyMuPDF bbox in PDF units → scale to pixel coords
            bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))
            x  = int(bx0 * zoom)
            y  = int(by0 * zoom)
            bw = int((bx1 - bx0) * zoom)
            bh = int((by1 - by0) * zoom)

            val, unit = _extract_value_unit(block_text)
            date_str  = _extract_date(block_text)

            obs_list.append(Observation(
                doc_id      = self.doc_id,
                page_number = page_number,
                source_type = "pdf_text",
                text        = block_text,
                value       = val,
                unit        = unit,
                date_str    = date_str,
                bbox        = (x, y, bw, bh),
            ))
        return obs_list

    def _run_ocr_on_image(
        self,
        img: Image.Image,
        page_number: int,
    ) -> List[Observation]:
        """Run EasyOCR on a PIL image; return Observation list."""
        if not _EASYOCR_OK:
            return [self._error_obs(
                page_number,
                "EasyOCR not installed — OCR unavailable.",
                source_type="ocr",
            )]

        reader = _get_ocr_reader(self.ocr_languages)
        if reader is None:
            return [self._error_obs(
                page_number, "OCR reader failed to initialise.", source_type="ocr"
            )]

        try:
            img_np = np.array(img)
            results = reader.readtext(img_np, detail=1, paragraph=False)
        except Exception as exc:
            return [self._error_obs(
                page_number, f"OCR failed: {exc}", source_type="ocr"
            )]

        obs_list: List[Observation] = []
        for (bbox_pts, text, conf) in results:
            text = text.strip()
            if not text:
                continue

            # EasyOCR bbox: list of 4 corner points [[x,y], ...]
            try:
                xs = [int(p[0]) for p in bbox_pts]
                ys = [int(p[1]) for p in bbox_pts]
                x, y = min(xs), min(ys)
                bw = max(xs) - x
                bh = max(ys) - y
            except Exception:
                x, y, bw, bh = 0, 0, 0, 0

            val, unit = _extract_value_unit(text)
            date_str  = _extract_date(text)
            uncertain = bool(conf < _OCR_CONFIDENCE_THRESHOLD)

            obs_list.append(Observation(
                doc_id         = self.doc_id,
                page_number    = page_number,
                source_type    = "ocr",
                text           = text,
                value          = val,
                unit           = unit,
                date_str       = date_str,
                bbox           = (x, y, bw, bh),
                ocr_confidence = round(float(conf), 4),
                ocr_uncertain  = uncertain,
                error_message  = "Low OCR confidence" if uncertain else None,
            ))
        return obs_list

    def _detect_and_crop_tables(
        self,
        img: Image.Image,
        page_number: int,
    ) -> List[Observation]:
        """Detect table bounding boxes and create one Observation per table."""
        try:
            boxes = detect_table_regions(img)
        except Exception as exc:
            logger.warning("Table detection failed: %s", exc)
            return []

        obs_list: List[Observation] = []
        for (x, y, w, h) in boxes:
            # Crop the table region and optionally OCR it
            try:
                crop = img.crop((x, y, x + w, y + h))
                cell_obs = self._run_ocr_on_image(crop, page_number)
                # Adjust bbox coords back to full-page space
                for o in cell_obs:
                    ox, oy, ow, oh = o.bbox
                    o.bbox        = (x + ox, y + oy, ow, oh)
                    o.source_type = "table_cell"
                obs_list.extend(cell_obs)
            except Exception as exc:
                obs_list.append(self._error_obs(
                    page_number,
                    f"Table OCR crop failed: {exc}",
                    source_type="table_cell",
                    bbox=(x, y, w, h),
                ))
        return obs_list

    def _error_obs(
        self,
        page_number: int,
        message: str,
        source_type: str = "error",
        bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Observation:
        """Create an Observation that records a processing failure."""
        return Observation(
            doc_id        = self.doc_id,
            page_number   = page_number,
            source_type   = source_type,
            text          = "",
            bbox          = bbox,
            error_message = message,
        )


# ---------------------------------------------------------------------------
# Annotation helper  (draw bboxes on a copy of the page image)
# ---------------------------------------------------------------------------

_COLOUR_MAP = {
    "pdf_text":   "#3B82F6",   # blue
    "ocr":        "#22C55E",   # green
    "table_cell": "#F97316",   # orange
    "error":      "#EF4444",   # red
}


def annotate_image(
    img: Image.Image,
    observations: List[Observation],
    show_text: bool = True,
) -> Image.Image:
    """
    Return a copy of *img* with bounding boxes drawn for each Observation.

    Parameters
    ----------
    img : PIL.Image.Image
        The original (un-annotated) page image.
    observations : list[Observation]
        Observations whose bboxes are to be drawn.
    show_text : bool
        If True, draw a short label inside each box.
    """
    annotated = img.copy().convert("RGBA")
    overlay   = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw      = ImageDraw.Draw(overlay)

    for obs in observations:
        x, y, w, h = obs.bbox
        if w <= 0 or h <= 0:
            continue
        colour_hex = _COLOUR_MAP.get(obs.source_type, "#888888")
        # Convert hex to RGBA
        r, g, b = (
            int(colour_hex[1:3], 16),
            int(colour_hex[3:5], 16),
            int(colour_hex[5:7], 16),
        )
        fill_colour   = (r, g, b, 40)   # semi-transparent fill
        border_colour = (r, g, b, 200)

        draw.rectangle([x, y, x + w, y + h], fill=fill_colour, outline=border_colour, width=2)

        if show_text and obs.text:
            label = obs.text[:30] + ("…" if len(obs.text) > 30 else "")
            if obs.ocr_uncertain:
                label = "⚠ " + label
            draw.text((x + 3, y + 2), label, fill=(r, g, b, 230))

    annotated = Image.alpha_composite(annotated, overlay)
    return annotated.convert("RGB")
