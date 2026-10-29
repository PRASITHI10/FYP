"""
utils/synthetic_docs.py
-----------------------
Generate synthetic PDF and image documents for testing without requiring
any external files, real patient data, or large AI models.

All documents are created entirely in-memory using PyMuPDF and Pillow.
"""

from __future__ import annotations

import io
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.extraction import Observation

try:
    import pymupdf as fitz
    _FITZ_OK = True
except ImportError:
    try:
        import fitz
        _FITZ_OK = True
    except ImportError:
        _FITZ_OK = False


# ---------------------------------------------------------------------------
# Synthetic PDF builders
# ---------------------------------------------------------------------------

def make_text_pdf(
    title: str = "Patient Record — Historical Medical Document",
    observations: list[dict] | None = None,
) -> bytes:
    """
    Create a synthetic digital (text-layer) PDF in-memory.

    Parameters
    ----------
    title : str
        Title text written at the top of page 1.
    observations : list[dict]
        Each dict may contain: text, value, unit, date.
        Defaults to a set of realistic clinical observations.

    Returns
    -------
    bytes
        Raw PDF bytes ready to be passed to DocumentProcessor.process_pdf().
    """
    if not _FITZ_OK:
        raise RuntimeError("PyMuPDF required to generate synthetic PDFs.")

    if observations is None:
        observations = _default_observations()

    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)   # A4

    y = 60
    # Title
    page.insert_text(
        (50, y), title,
        fontsize=14, fontname="helv", color=(0, 0, 0.4)
    )
    y += 30

    # Divider line
    page.draw_line((50, y), (545, y), color=(0.5, 0.5, 0.5), width=0.5)
    y += 18

    for obs in observations:
        line = _format_obs_line(obs)
        page.insert_text((60, y), line, fontsize=10, fontname="cour")
        y += 18
        if y > 800:
            page = doc.new_page(width=595, height=842)
            y = 50

    # Second page: simple table
    page2 = doc.new_page(width=595, height=842)
    _draw_text_table(page2, observations[:6])

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def make_scanned_pdf(
    rows: list[str] | None = None,
) -> bytes:
    """
    Create a synthetic 'scanned' PDF — the text is rendered as pixels in an
    image and then embedded as an image page (no text layer), so PyMuPDF
    text extraction returns nothing and EasyOCR must be used instead.

    Returns
    -------
    bytes
        Raw PDF bytes.
    """
    if not _FITZ_OK:
        raise RuntimeError("PyMuPDF required to generate synthetic PDFs.")

    if rows is None:
        rows = [
            "Patient: J. Smith                Date: 15 Jan 1998",
            "Blood Pressure: 140/90 mmHg      HR: 88 bpm",
            "Temperature: 37.2 °C             RR: 18",
            "Haemoglobin: 12.5 g/dL",
            "Impression: Hypertension, stable. Review in 3 months.",
        ]

    # Render text into a Pillow image
    img = _text_to_image(rows, width=595, height=842)

    # Embed the image as a PDF page
    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)
    img_bytes = _pil_to_png_bytes(img)
    page.insert_image(page.rect, stream=img_bytes)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Synthetic image builders
# ---------------------------------------------------------------------------

def make_clinical_image(
    width: int = 800,
    height: int = 600,
    include_table: bool = True,
) -> Image.Image:
    """
    Return a synthetic clinical document image (PIL RGB) with:
    - A header
    - Several rows of key-value observations
    - Optionally a simple hand-drawn table with grid lines
    """
    img  = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, width, 60], fill=(30, 60, 100))
    draw.text(
        (20, 15),
        "TEMPORAL-REGIONRAG  |  Synthetic Medical Record",
        fill=(240, 240, 240),
    )

    # Observation rows
    obs = _default_observations()
    y   = 80
    for o in obs[:8]:
        line = _format_obs_line(o)
        draw.text((30, y), line, fill=(30, 30, 30))
        y += 28

    if include_table and y + 150 < height:
        _draw_pil_table(draw, obs[:4], x=30, y=y + 10, col_width=180, row_height=30)

    return img


def make_noisy_image(base_image: Image.Image, noise_level: float = 0.05) -> Image.Image:
    """
    Add salt-and-pepper noise to a PIL image to simulate scan artefacts.
    Useful for testing uncertain OCR paths.
    """
    arr   = np.array(base_image, dtype=np.float32)
    noise = np.random.choice(
        [0, 255, -1],   # -1 = no change
        size=arr.shape,
        p=[noise_level / 2, noise_level / 2, 1 - noise_level],
    )
    mask  = noise >= 0
    arr[mask] = noise[mask]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_observations() -> list[dict]:
    return [
        {"text": "Date of Admission",  "value": None,   "unit": None,    "date": "15 Jan 1998"},
        {"text": "Blood Pressure",      "value": "140/90","unit": "mmHg",  "date": None},
        {"text": "Heart Rate",          "value": 88,      "unit": "bpm",   "date": None},
        {"text": "Temperature",         "value": 37.2,    "unit": "°C",    "date": None},
        {"text": "Respiratory Rate",    "value": 18,      "unit": None,    "date": None},
        {"text": "Haemoglobin",         "value": 12.5,    "unit": "g/dL",  "date": None},
        {"text": "Blood Glucose",       "value": 5.4,     "unit": "mmol/L","date": None},
        {"text": "Sodium",              "value": 138,     "unit": "mEq/L", "date": None},
        {"text": "Potassium",           "value": 4.1,     "unit": "mEq/L", "date": None},
        {"text": "Review Date",         "value": None,    "unit": None,    "date": "22 Apr 1998"},
    ]


def _format_obs_line(obs: dict) -> str:
    parts = [obs.get("text", "")]
    if obs.get("date"):
        parts.append(f"[{obs['date']}]")
    if obs.get("value") is not None:
        val_str = str(obs["value"])
        if obs.get("unit"):
            val_str += f" {obs['unit']}"
        parts.append(f": {val_str}")
    return "   ".join(parts)


def _draw_text_table(page, observations: list[dict]) -> None:
    """Draw a simple grid table on a PyMuPDF page."""
    x0, y0 = 50, 100
    col_w, row_h = 240, 22
    headers = ["Observation", "Value", "Unit", "Date"]

    for ci, hdr in enumerate(headers):
        page.draw_rect(
            fitz.Rect(x0 + ci * col_w, y0, x0 + (ci + 1) * col_w, y0 + row_h),
            color=(0, 0, 0), fill=(0.85, 0.85, 0.95), width=0.5,
        )
        page.insert_text(
            (x0 + ci * col_w + 4, y0 + 14),
            hdr, fontsize=9, fontname="helv", color=(0, 0, 0)
        )

    for ri, obs in enumerate(observations):
        ry  = y0 + (ri + 1) * row_h
        row = [
            obs.get("text", ""),
            str(obs.get("value", "")),
            obs.get("unit", ""),
            obs.get("date", ""),
        ]
        fill = (1, 1, 1) if ri % 2 == 0 else (0.95, 0.95, 1.0)
        for ci, cell in enumerate(row):
            page.draw_rect(
                fitz.Rect(x0 + ci * col_w, ry, x0 + (ci + 1) * col_w, ry + row_h),
                color=(0.7, 0.7, 0.7), fill=fill, width=0.3,
            )
            page.insert_text(
                (x0 + ci * col_w + 4, ry + 14),
                str(cell)[:28], fontsize=8, fontname="cour",
            )


def _draw_pil_table(
    draw: ImageDraw.ImageDraw,
    observations: list[dict],
    x: int, y: int,
    col_width: int = 180,
    row_height: int = 30,
) -> None:
    """Draw a simple grid table onto a PIL ImageDraw canvas."""
    headers = ["Observation", "Value", "Unit", "Date"]
    n_cols  = len(headers)

    # Header row
    for ci, h in enumerate(headers):
        cx = x + ci * col_width
        draw.rectangle([cx, y, cx + col_width, y + row_height], fill=(30, 60, 100), outline=(0, 0, 0))
        draw.text((cx + 4, y + 8), h, fill=(255, 255, 255))

    for ri, obs in enumerate(observations):
        ry  = y + (ri + 1) * row_height
        row = [
            obs.get("text", ""),
            str(obs.get("value", "")),
            obs.get("unit", ""),
            obs.get("date", "") or "",
        ]
        fill = (255, 255, 255) if ri % 2 == 0 else (235, 235, 245)
        for ci, cell in enumerate(row):
            cx = x + ci * col_width
            draw.rectangle([cx, ry, cx + col_width, ry + row_height], fill=fill, outline=(180, 180, 180))
            draw.text((cx + 4, ry + 8), str(cell)[:22], fill=(30, 30, 30))


def _text_to_image(rows: list[str], width: int, height: int) -> Image.Image:
    """Render text rows onto a white PIL image (simulates a scanned page)."""
    img  = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Add slight background noise to simulate scanner artefacts
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 3, arr.shape)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    y = 40
    for row in rows:
        draw.text((30, y), row, fill=(20, 20, 20))
        y += 35

    return img


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Synthetic Longitudinal Retrieval Benchmark Corpus
# ---------------------------------------------------------------------------

def create_synthetic_retrieval_corpus() -> List[Observation]:
    """
    Generate a benchmark corpus of structured observations across multiple
    historical medical documents and distinct dates (1998, 2002, 2005).

    Used for testing and demonstrating:
    - Single-observation query matching with spatial coordinate retention.
    - Multi-date temporal comparison (e.g. 'Compare blood pressure in 1998 and 2002').
    - Long-term clinical progression (e.g. 'Show glucose trend over time').
    - Querying unmentioned concepts to verify 'no evidence found' handling.
    """
    docs_data = [
        {
            "doc_id": "doc_1998_admission",
            "date": "15 Jan 1998",
            "page": 1,
            "items": [
                ("Admission Note: Patient admitted with hypertension and severe headache.", None, None, (50, 80, 520, 24)),
                ("Blood Pressure: 140/90 mmHg", 140.0, "mmHg", (50, 115, 230, 24)),
                ("Heart Rate: 88 bpm", 88.0, "bpm", (50, 150, 170, 24)),
                ("Body Temperature: 37.2 °C", 37.2, "°C", (50, 185, 190, 24)),
                ("Fasting Blood Glucose: 5.4 mmol/L", 5.4, "mmol/L", (50, 220, 260, 24)),
                ("Total Serum Cholesterol: 185 mg/dL", 185.0, "mg/dL", (50, 255, 260, 24)),
                ("Documented Allergy: Penicillin causes severe urticaria", None, None, (50, 290, 420, 24)),
            ],
        },
        {
            "doc_id": "doc_2002_clinic",
            "date": "22 Apr 2002",
            "page": 1,
            "items": [
                ("Follow-up Clinic Note: Cardiovascular status check.", None, None, (50, 80, 460, 24)),
                ("Blood Pressure: 130/80 mmHg", 130.0, "mmHg", (50, 115, 230, 24)),
                ("Heart Rate: 72 bpm", 72.0, "bpm", (50, 150, 170, 24)),
                ("Fasting Blood Glucose: 6.8 mmol/L", 6.8, "mmol/L", (50, 185, 260, 24)),
                ("Total Serum Cholesterol: 210 mg/dL", 210.0, "mg/dL", (50, 220, 260, 24)),
                ("Prescription: Started Lisinopril 10 mg daily for hypertension", None, None, (50, 255, 460, 24)),
            ],
        },
        {
            "doc_id": "doc_2005_discharge",
            "date": "10 Nov 2005",
            "page": 1,
            "items": [
                ("Annual Diabetic & Cardio Review: Comprehensive evaluation.", None, None, (50, 80, 500, 24)),
                ("Blood Pressure: 124/78 mmHg", 124.0, "mmHg", (50, 115, 230, 24)),
                ("Heart Rate: 70 bpm", 70.0, "bpm", (50, 150, 170, 24)),
                ("Glycated Haemoglobin HbA1c: 7.1 %", 7.1, "%", (50, 185, 280, 24)),
                ("Fasting Blood Glucose: 7.4 mmol/L", 7.4, "mmol/L", (50, 220, 260, 24)),
                ("Total Serum Cholesterol: 235 mg/dL", 235.0, "mg/dL", (50, 255, 260, 24)),
                ("Treatment Adjustment: Titrate Metformin to 1000 mg twice daily", None, None, (50, 290, 490, 24)),
            ],
        },
    ]

    corpus: List[Observation] = []
    for d in docs_data:
        doc_id = d["doc_id"]
        doc_date = d["date"]
        page_num = d["page"]
        for text, val, unit, bbox in d["items"]:
            corpus.append(
                Observation(
                    doc_id=doc_id,
                    page_number=page_num,
                    source_type="pdf_text",
                    text=text,
                    value=val,
                    unit=unit,
                    date_str=doc_date,
                    bbox=bbox,
                    ocr_confidence=1.0,
                    ocr_uncertain=False,
                )
            )
    return corpus


def create_synthetic_retrieval_documents() -> Tuple[List[Observation], dict[str, Image.Image]]:
    """
    Generate both the structured Observations and rendered document page images
    where text is drawn at the exact bounding box coordinates.
    Returns (observations, {doc_id: PIL_Image}).
    """
    corpus = create_synthetic_retrieval_corpus()
    images: dict[str, Image.Image] = {}

    # Group by doc_id
    doc_map: dict[str, list[Observation]] = {}
    for obs in corpus:
        doc_map.setdefault(obs.doc_id, []).append(obs)

    for doc_id, obs_list in doc_map.items():
        doc_date = obs_list[0].date_str if obs_list else ""
        img_w, img_h = 650, 400
        img = Image.new("RGB", (img_w, img_h), color=(252, 252, 254))
        draw = ImageDraw.Draw(img)

        # Header banner
        draw.rectangle([0, 0, img_w, 55], fill=(35, 55, 95))
        draw.text((25, 12), f"HISTORICAL MEDICAL RECORD  |  {doc_id}", fill=(255, 255, 255))
        draw.text((25, 32), f"Date of Record: {doc_date}", fill=(200, 220, 255))

        # Draw each observation text at its bbox
        for obs in obs_list:
            bx, by, bw, bh = obs.bbox
            draw.text((bx, by + 4), obs.text, fill=(25, 25, 30))

        images[doc_id] = img

    return corpus, images


# ---------------------------------------------------------------------------
# Synthetic Verification Benchmark Corpus (includes edge cases)
# ---------------------------------------------------------------------------

def create_synthetic_verification_corpus() -> List[Observation]:
    """
    Generate a benchmark corpus designed for testing the temporal verification engine:
    - Clean longitudinal records (1998, 2002, 2005).
    - Conflicting measurement on identical encounter (BP 110/70 vs 140/90 on 15 Jan 1998).
    - Duplicate observation (identical BP entry).
    - Ambiguous numeric date format (04/05/1998).
    - Missing timestamp (undated heart rate).
    - Uncertain OCR quality (low confidence flag).
    """
    base_corpus = create_synthetic_retrieval_corpus()

    edge_cases = [
        # 1. Conflicting observation on 15 Jan 1998 (same encounter, contradictory BP)
        Observation(
            obs_id="obs_conflict_bp",
            doc_id="doc_1998_nursing_note",
            page_number=1,
            source_type="pdf_text",
            text="Blood Pressure: 110/70 mmHg (discrepant reading)",
            value=110.0,
            unit="mmHg",
            date_str="15 Jan 1998",
            bbox=(50, 320, 260, 24),
            ocr_confidence=1.0,
            ocr_uncertain=False,
        ),
        # 2. Duplicate observation on 15 Jan 1998 (exact same value/unit/date)
        Observation(
            obs_id="obs_duplicate_bp",
            doc_id="doc_1998_triage",
            page_number=1,
            source_type="pdf_text",
            text="Blood Pressure: 140/90 mmHg",
            value=140.0,
            unit="mmHg",
            date_str="15 Jan 1998",
            bbox=(50, 350, 230, 24),
            ocr_confidence=1.0,
            ocr_uncertain=False,
        ),
        # 3. Ambiguous date format: 04/05/1998 (could be 4 May or 5 April)
        Observation(
            obs_id="obs_ambiguous_date",
            doc_id="doc_ambiguous_lab",
            page_number=1,
            source_type="pdf_text",
            text="Fasting Blood Glucose: 6.1 mmol/L",
            value=6.1,
            unit="mmol/L",
            date_str="04/05/1998",
            bbox=(50, 110, 260, 24),
            ocr_confidence=1.0,
            ocr_uncertain=False,
        ),
        # 4. Missing timestamp (undated record)
        Observation(
            obs_id="obs_missing_timestamp",
            doc_id="doc_undated_vitals",
            page_number=1,
            source_type="ocr",
            text="Heart Rate: 84 bpm (routine check)",
            value=84.0,
            unit="bpm",
            date_str=None,
            bbox=(50, 140, 240, 24),
            ocr_confidence=0.92,
            ocr_uncertain=False,
        ),
        # 5. Low OCR confidence (transcription uncertainty)
        Observation(
            obs_id="obs_low_conf_ocr",
            doc_id="doc_faded_scan_1998",
            page_number=1,
            source_type="ocr",
            text="Blood Pressure: 138/88 mmHg",
            value=138.0,
            unit="mmHg",
            date_str="15 Jan 1998",
            bbox=(50, 170, 230, 24),
            ocr_confidence=0.45,
            ocr_uncertain=True,
            error_message="Low OCR confidence",
        ),
    ]

    return base_corpus + edge_cases




def render_observation_pages(
    observations: List[Observation],
    width: int = 650,
    height: int = 400,
) -> dict[str, List[Image.Image]]:
    """
    Render one synthetic page per (doc_id, page) with each observation's text drawn
    at its bounding box, so evidence regions can be inspected visually.
    Returns {doc_id: [page_1_image, page_2_image, ...]}.
    """
    by_doc: dict[str, dict[int, list[Observation]]] = {}
    for o in observations:
        by_doc.setdefault(o.doc_id, {}).setdefault(max(o.page_number, 1), []).append(o)

    out: dict[str, List[Image.Image]] = {}
    for doc_id, pages in by_doc.items():
        n_pages = max(pages.keys())
        imgs: List[Image.Image] = []
        for p in range(1, n_pages + 1):
            img = Image.new("RGB", (width, height), color=(252, 252, 254))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, 55], fill=(35, 55, 95))
            draw.text((25, 12), f"SYNTHETIC RECORD  |  {doc_id}  |  page {p}", fill=(255, 255, 255))
            dates = {o.date_str for o in pages.get(p, []) if o.date_str}
            draw.text((25, 32), f"Date of Record: {', '.join(sorted(dates)) or 'not stated'}",
                      fill=(200, 220, 255))
            page_obs = pages.get(p, [])
            if len(page_obs) >= 3:
                ys = [o.bbox[1] for o in page_obs]
                y_top = max(65, min(ys) - 4)
                y_bot = min(height - 10, max(o.bbox[1] + o.bbox[3] for o in page_obs) + 6)
                draw.rectangle([42, y_top, 590, y_bot], outline=(40, 40, 40), width=2)
                draw.line([360, y_top, 360, y_bot], fill=(60, 60, 60), width=2)
                for o in page_obs:
                    row_y = o.bbox[1] + o.bbox[3] + 2
                    if y_top < row_y < y_bot:
                        draw.line([42, row_y, 590, row_y], fill=(60, 60, 60), width=1)
            for o in page_obs:
                bx, by, _, _ = o.bbox
                draw.text((bx, by + 4), o.text, fill=(25, 25, 30))
            imgs.append(img)
        out[doc_id] = imgs
    return out


def create_synthetic_verification_documents() -> Tuple[List[Observation], dict[str, List[Image.Image]]]:
    """Verification corpus (with edge cases) plus rendered source pages for every document."""
    corpus = create_synthetic_verification_corpus()
    return corpus, render_observation_pages(corpus)
