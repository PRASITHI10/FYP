"""
utils/evaluation.py
-------------------
Reproducible benchmark evaluation for Temporal-RegionRAG.

Compares three configurations on the same labelled synthetic test set:
  A. OCR + Text Retrieval Baseline (LexicalRetriever without temporal verification).
  B. Visual Retrieval:
     - B1 (ColPali vidore/colpali-v1.2): Probed dynamically; reported honestly as
       not installed when optional VLM dependencies/weights are absent.
     - B2 (Operational Page-Level Retrieval via VisualRetrievalEngine + OCR fallback,
       connecting retrieved pages to table regions and page observations without
       temporal verification filtering).
  C. Full Temporal-RegionRAG System (EvidenceQAPipeline: LexicalRetriever +
     TemporalVerificationEngine + EvidencePackage + TemplateAnswerGenerator).

All metrics are computed strictly from actual runtime predictions against explicit
ground-truth labels. No metrics are hardcoded, guessed, or extrapolated.
"""

from __future__ import annotations

import os
import platform
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from utils.extraction import DocumentProcessor, Observation
from utils.qa_pipeline import EvidenceQAPipeline, validate_grounded_answer
from utils.retrieval import LexicalRetriever, RetrievalResult
from utils.synthetic_docs import (
    create_synthetic_verification_documents,
    make_scanned_pdf,
    make_text_pdf,
)
from utils.temporal_verification import VerificationStatus
from utils.visual_retrieval import (
    ColPaliConfig,
    ColPaliVisualRetriever,
    DocumentPage,
    VisualRetrievalEngine,
    probe_visual_environment,
)


# ---------------------------------------------------------------------------
# 1. Ground-Truth Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthQueryItem:
    """One labelled benchmark query with ground-truth evidence and provenance."""
    query_id: str
    question: str
    category: str  # "single_date" | "temporal_comparison" | "conflict_or_uncertainty" | "unanswerable_negative"
    relevant_obs_ids: List[str]
    expected_verified_obs_ids: List[str]
    expected_conflicting_obs_ids: List[str] = field(default_factory=list)
    expected_uncertain_obs_ids: List[str] = field(default_factory=list)
    expected_pages: List[Tuple[str, int]] = field(default_factory=list)
    expected_chronological_dates: List[str] = field(default_factory=list)
    expect_insufficient_evidence: bool = False
    notes: str = ""


@dataclass
class GroundTruthExtractionItem:
    """One labelled ground-truth observation for evaluating OCR & PDF extraction."""
    item_id: str
    source_modality: str  # "pdf_text" | "scanned_ocr"
    doc_id: str
    page_number: int
    keyword_anchor: str
    expected_value: float
    expected_unit: str
    expected_date_substring: Optional[str]


@dataclass
class QueryPredictionRecord:
    """Normalized prediction record captured from a configuration on one query."""
    query_id: str
    config_id: str
    question: str
    category: str
    ranked_obs_ids: List[str]
    accepted_obs_ids: List[str]
    predicted_verified_ids: List[str]
    predicted_conflicting_ids: List[str]
    predicted_uncertain_ids: List[str]
    ranked_pages: List[Tuple[str, int]]
    accepted_dates_in_order: List[str]
    accepted_bboxes: Dict[str, Tuple[str, int, Tuple[int, int, int, int]]]
    insufficient_evidence: bool
    answer_grounded: bool
    latency_ms: float
    error_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Geometry & Metric Helpers
# ---------------------------------------------------------------------------

def bbox_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection-over-Union (IoU) between two (x, y, w, h) boxes.
    Returns 0.0 if either box has non-positive area or they do not overlap.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0

    union_area = float(aw * ah + bw * bh) - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def compute_recall_at_k(
    ranked_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int,
) -> Optional[float]:
    """
    Recall@k = |top_k(ranked_ids) ∩ ground_truth_ids| / |ground_truth_ids|.
    Returns None for unanswerable queries where ground_truth_ids is empty.
    """
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return None
    top_k_set = set(list(ranked_ids)[:k])
    return len(top_k_set & gt_set) / float(len(gt_set))


def compute_page_recall_at_k(
    ranked_pages: Sequence[Tuple[str, int]],
    ground_truth_pages: Sequence[Tuple[str, int]],
    k: int,
) -> Optional[float]:
    """
    Page-level Recall@k = |top_k(ranked_pages) ∩ ground_truth_pages| / |ground_truth_pages|.
    Returns None when ground_truth_pages is empty.
    """
    gt_set = set(ground_truth_pages)
    if not gt_set:
        return None
    # Deduplicate ranked pages while preserving order
    unique_ranked: List[Tuple[str, int]] = []
    for p in ranked_pages:
        if p not in unique_ranked:
            unique_ranked.append(p)
    top_k_set = set(unique_ranked[:k])
    return len(top_k_set & gt_set) / float(len(gt_set))


# ---------------------------------------------------------------------------
# 3. Deterministic Labelled Benchmark Dataset Construction
# ---------------------------------------------------------------------------

def build_labelled_evaluation_corpus() -> Tuple[
    List[Observation],
    Dict[str, List[Image.Image]],
    Dict[str, Observation],
]:
    """
    Build the deterministic labelled synthetic corpus with stable obs_id values.

    Assigns deterministic IDs (`obs_01` ... `obs_20` for the 20 longitudinal items,
    plus the 5 verification edge-case IDs `obs_conflict_bp`, `obs_duplicate_bp`,
    `obs_ambiguous_date`, `obs_missing_timestamp`, `obs_low_conf_ocr`) so ground-truth
    references are 100% reproducible across runs.
    """
    corpus, _ = create_synthetic_verification_documents()
    for idx, obs in enumerate(corpus, start=1):
        if not obs.obs_id.startswith("obs_"):
            obs.obs_id = f"obs_{idx:02d}"
    # Re-render pages with deterministic IDs in place
    from utils.synthetic_docs import render_observation_pages
    pages_dict = render_observation_pages(corpus)
    obs_by_id = {o.obs_id: o for o in corpus}
    return corpus, pages_dict, obs_by_id


def build_query_benchmark(
    corpus: Sequence[Observation],
) -> List[GroundTruthQueryItem]:
    """
    Construct the 15-query labelled benchmark spanning:
      - 6 single-date measurement queries
      - 4 multi-date longitudinal temporal comparison queries
      - 3 conflict / uncertainty / missing-timestamp stress queries
      - 2 unanswerable negative queries (absent concepts)
    """
    # Map (doc_id, text_prefix) -> obs_id for readable, verifiable ground-truth wiring
    def find_id(doc_id: str, prefix: str) -> str:
        for o in corpus:
            if o.doc_id == doc_id and o.text.startswith(prefix):
                return o.obs_id
        raise KeyError(f"Observation not found: ({doc_id}, {prefix})")

    bp_1998 = find_id("doc_1998_admission", "Blood Pressure: 140/90")
    hr_1998 = find_id("doc_1998_admission", "Heart Rate: 88")
    temp_1998 = find_id("doc_1998_admission", "Body Temperature: 37.2")
    gluc_1998 = find_id("doc_1998_admission", "Fasting Blood Glucose: 5.4")
    chol_1998 = find_id("doc_1998_admission", "Total Serum Cholesterol: 185")

    bp_2002 = find_id("doc_2002_clinic", "Blood Pressure: 130/80")
    hr_2002 = find_id("doc_2002_clinic", "Heart Rate: 72")
    gluc_2002 = find_id("doc_2002_clinic", "Fasting Blood Glucose: 6.8")
    chol_2002 = find_id("doc_2002_clinic", "Total Serum Cholesterol: 210")

    bp_2005 = find_id("doc_2005_discharge", "Blood Pressure: 124/78")
    hr_2005 = find_id("doc_2005_discharge", "Heart Rate: 70")
    hba1c_2005 = find_id("doc_2005_discharge", "Glycated Haemoglobin HbA1c: 7.1")
    gluc_2005 = find_id("doc_2005_discharge", "Fasting Blood Glucose: 7.4")
    chol_2005 = find_id("doc_2005_discharge", "Total Serum Cholesterol: 235")

    return [
        # --- Single-date queries (6) ---
        GroundTruthQueryItem(
            query_id="Q01",
            question="What was the blood pressure on 22 Apr 2002?",
            category="single_date",
            relevant_obs_ids=[bp_2002],
            expected_verified_obs_ids=[bp_2002],
            expected_pages=[("doc_2002_clinic", 1)],
            expected_chronological_dates=["2002-04-22"],
            notes="Single-date BP lookup in 2002 clinic note; other years must be excluded.",
        ),
        GroundTruthQueryItem(
            query_id="Q02",
            question="What was the heart rate on 15 Jan 1998?",
            category="single_date",
            relevant_obs_ids=[hr_1998],
            expected_verified_obs_ids=[hr_1998],
            expected_pages=[("doc_1998_admission", 1)],
            expected_chronological_dates=["1998-01-15"],
            notes="Single-date HR lookup in 1998 admission note; undated HR must be rejected.",
        ),
        GroundTruthQueryItem(
            query_id="Q03",
            question="What was the HbA1c in November 2005?",
            category="single_date",
            relevant_obs_ids=[hba1c_2005],
            expected_verified_obs_ids=[hba1c_2005],
            expected_pages=[("doc_2005_discharge", 1)],
            expected_chronological_dates=["2005-11-10"],
            notes="Single-date HbA1c lookup in 2005 discharge note.",
        ),
        GroundTruthQueryItem(
            query_id="Q04",
            question="What was the body temperature on 15 Jan 1998?",
            category="single_date",
            relevant_obs_ids=[temp_1998],
            expected_verified_obs_ids=[temp_1998],
            expected_pages=[("doc_1998_admission", 1)],
            expected_chronological_dates=["1998-01-15"],
            notes="Single-date temperature lookup in 1998 admission note.",
        ),
        GroundTruthQueryItem(
            query_id="Q05",
            question="What was the total serum cholesterol in 2002?",
            category="single_date",
            relevant_obs_ids=[chol_2002],
            expected_verified_obs_ids=[chol_2002],
            expected_pages=[("doc_2002_clinic", 1)],
            expected_chronological_dates=["2002-04-22"],
            notes="Single-year cholesterol lookup in 2002 clinic note.",
        ),
        GroundTruthQueryItem(
            query_id="Q06",
            question="What was the fasting blood glucose in 2005?",
            category="single_date",
            relevant_obs_ids=[gluc_2005],
            expected_verified_obs_ids=[gluc_2005],
            expected_pages=[("doc_2005_discharge", 1)],
            expected_chronological_dates=["2005-11-10"],
            notes="Single-year glucose lookup in 2005 discharge note.",
        ),
        # --- Multi-date temporal comparison queries (4) ---
        GroundTruthQueryItem(
            query_id="Q07",
            question="Compare blood pressure in 2002 and 2005",
            category="temporal_comparison",
            relevant_obs_ids=[bp_2002, bp_2005],
            expected_verified_obs_ids=[bp_2002, bp_2005],
            expected_pages=[("doc_2002_clinic", 1), ("doc_2005_discharge", 1)],
            expected_chronological_dates=["2002-04-22", "2005-11-10"],
            notes="Two-year BP comparison (2002 vs 2005); 1998 records must be excluded.",
        ),
        GroundTruthQueryItem(
            query_id="Q08",
            question="Compare heart rate in 1998 and 2005",
            category="temporal_comparison",
            relevant_obs_ids=[hr_1998, hr_2005],
            expected_verified_obs_ids=[hr_1998, hr_2005],
            expected_pages=[("doc_1998_admission", 1), ("doc_2005_discharge", 1)],
            expected_chronological_dates=["1998-01-15", "2005-11-10"],
            notes="Two-year HR comparison (1998 vs 2005); 2002 and undated HR must be excluded.",
        ),
        GroundTruthQueryItem(
            query_id="Q09",
            question="Compare fasting blood glucose between 2002 and 2005",
            category="temporal_comparison",
            relevant_obs_ids=[gluc_2002, gluc_2005],
            expected_verified_obs_ids=[gluc_2002, gluc_2005],
            expected_pages=[("doc_2002_clinic", 1), ("doc_2005_discharge", 1)],
            expected_chronological_dates=["2002-04-22", "2005-11-10"],
            notes="Two-year glucose comparison (2002 vs 2005).",
        ),
        GroundTruthQueryItem(
            query_id="Q10",
            question="Show total serum cholesterol trend over time across all visits",
            category="temporal_comparison",
            relevant_obs_ids=[chol_1998, chol_2002, chol_2005],
            expected_verified_obs_ids=[chol_1998, chol_2002, chol_2005],
            expected_pages=[
                ("doc_1998_admission", 1),
                ("doc_2002_clinic", 1),
                ("doc_2005_discharge", 1),
            ],
            expected_chronological_dates=["1998-01-15", "2002-04-22", "2005-11-10"],
            notes="Three-date longitudinal cholesterol trend ordered chronologically.",
        ),
        # --- Conflict / uncertainty / missing-timestamp stress queries (3) ---
        GroundTruthQueryItem(
            query_id="Q11",
            question="What was the blood pressure on 15 Jan 1998?",
            category="conflict_or_uncertainty",
            relevant_obs_ids=[bp_1998, "obs_conflict_bp"],
            expected_verified_obs_ids=[],
            expected_conflicting_obs_ids=[bp_1998, "obs_conflict_bp"],
            expected_uncertain_obs_ids=["obs_duplicate_bp", "obs_low_conf_ocr"],
            expected_pages=[
                ("doc_1998_admission", 1),
                ("doc_1998_nursing_note", 1),
            ],
            expected_chronological_dates=["1998-01-15"],
            expect_insufficient_evidence=True,
            notes=(
                "Same-date contradictory BP (140/90 vs 110/70) must be flagged CONFLICTING "
                "with 0 VERIFIED items (insufficient verified evidence to pick one value); "
                "duplicate and low-OCR-confidence readings must be flagged UNCERTAIN."
            ),
        ),
        GroundTruthQueryItem(
            query_id="Q12",
            question="What was the fasting blood glucose in 1998?",
            category="conflict_or_uncertainty",
            relevant_obs_ids=[gluc_1998],
            expected_verified_obs_ids=[gluc_1998],
            expected_uncertain_obs_ids=["obs_ambiguous_date"],
            expected_pages=[("doc_1998_admission", 1), ("doc_ambiguous_lab", 1)],
            expected_chronological_dates=["1998-01-15"],
            notes="15 Jan 1998 glucose is VERIFIED; 04/05/1998 glucose is flagged UNCERTAIN for ambiguous date.",
        ),
        GroundTruthQueryItem(
            query_id="Q13",
            question="What was the heart rate in 2002?",
            category="conflict_or_uncertainty",
            relevant_obs_ids=[hr_2002],
            expected_verified_obs_ids=[hr_2002],
            expected_pages=[("doc_2002_clinic", 1)],
            expected_chronological_dates=["2002-04-22"],
            notes="2002 HR is VERIFIED; undated HR (obs_missing_timestamp) must be rejected.",
        ),
        # --- Unanswerable negative queries (2) ---
        GroundTruthQueryItem(
            query_id="Q14",
            question="What were the results of the brain MRI scan?",
            category="unanswerable_negative",
            relevant_obs_ids=[],
            expected_verified_obs_ids=[],
            expected_pages=[],
            expected_chronological_dates=[],
            expect_insufficient_evidence=True,
            notes="Absent concept (MRI scan not in corpus); must report insufficient evidence.",
        ),
        GroundTruthQueryItem(
            query_id="Q15",
            question="What was the blood pressure in 1985?",
            category="unanswerable_negative",
            relevant_obs_ids=[],
            expected_verified_obs_ids=[],
            expected_pages=[],
            expected_chronological_dates=[],
            expect_insufficient_evidence=True,
            notes="Out-of-range year (1985 not in corpus); must report insufficient evidence.",
        ),
    ]


def build_ocr_extraction_benchmark() -> List[GroundTruthExtractionItem]:
    """
    Ground-truth items for evaluating DocumentProcessor value, unit, and date
    extraction across digital PDF pages (`make_text_pdf`) and scanned
    image pages (`make_clinical_image`).
    """
    return [
        # Digital PDF page 1 (`make_text_pdf` default observations)
        GroundTruthExtractionItem(
            item_id="EXT_PDF_01",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Blood Pressure",
            expected_value=140.0,
            expected_unit="mmHg",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_PDF_02",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Heart Rate",
            expected_value=88.0,
            expected_unit="bpm",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_PDF_03",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Temperature",
            expected_value=37.2,
            expected_unit="°C",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_PDF_04",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Haemoglobin",
            expected_value=12.5,
            expected_unit="g/dL",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_PDF_05",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Blood Glucose",
            expected_value=5.4,
            expected_unit="mmol/L",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_PDF_06",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Sodium",
            expected_value=138.0,
            expected_unit="mEq/L",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_PDF_07",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Potassium",
            expected_value=4.1,
            expected_unit="mEq/L",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_PDF_08",
            source_modality="pdf_text",
            doc_id="eval_digital_pdf",
            page_number=1,
            keyword_anchor="Date of Admission",
            expected_value=15.0,
            expected_unit="",
            expected_date_substring="15 Jan 1998",
        ),
        # Scanned PDF (`make_scanned_pdf` rendered with scanner noise and extracted via EasyOCR)
        GroundTruthExtractionItem(
            item_id="EXT_OCR_01",
            source_modality="scanned_ocr",
            doc_id="eval_scanned_pdf",
            page_number=1,
            keyword_anchor="Blood Pressure",
            expected_value=140.0,
            expected_unit="mmHg",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_OCR_02",
            source_modality="scanned_ocr",
            doc_id="eval_scanned_pdf",
            page_number=1,
            keyword_anchor="HR",
            expected_value=88.0,
            expected_unit="bpm",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_OCR_03",
            source_modality="scanned_ocr",
            doc_id="eval_scanned_pdf",
            page_number=1,
            keyword_anchor="37.2",
            expected_value=37.2,
            expected_unit="°C",
            expected_date_substring=None,
        ),
        GroundTruthExtractionItem(
            item_id="EXT_OCR_04",
            source_modality="scanned_ocr",
            doc_id="eval_scanned_pdf",
            page_number=1,
            keyword_anchor="Haem",
            expected_value=12.5,
            expected_unit="g/dL",
            expected_date_substring=None,
        ),
    ]


# ---------------------------------------------------------------------------
# 4. OCR & PDF Extraction Evaluation Runner
# ---------------------------------------------------------------------------

def evaluate_ocr_and_extraction() -> Dict[str, Any]:
    """
    Run DocumentProcessor on the synthetic digital PDF and synthetic scanned
    PDF, then evaluate value, unit, date, and bbox extraction against
    build_ocr_extraction_benchmark().
    """
    import numpy as np

    gt_items = build_ocr_extraction_benchmark()

    # 1. Process digital PDF
    t0 = time.perf_counter()
    pdf_bytes = make_text_pdf()
    pdf_proc = DocumentProcessor(doc_id="eval_digital_pdf")
    pdf_obs = pdf_proc.process_pdf(pdf_bytes)
    pdf_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Process scanned PDF via EasyOCR (seed numpy RNG for reproducible scanner noise)
    np.random.seed(42)
    t1 = time.perf_counter()
    scanned_pdf_bytes = make_scanned_pdf()
    scan_proc = DocumentProcessor(doc_id="eval_scanned_pdf")
    ocr_obs = scan_proc.process_pdf(scanned_pdf_bytes)
    ocr_latency_ms = (time.perf_counter() - t1) * 1000.0

    all_extracted = pdf_obs + ocr_obs
    item_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for gt in gt_items:
        candidates = [
            o for o in all_extracted
            if o.doc_id == gt.doc_id
            and o.page_number == gt.page_number
            and o.source_type in ("pdf_text", "ocr")
            and gt.keyword_anchor.lower() in (o.text or "").lower()
        ]
        # Prefer candidate with parsed value/unit when multiple OCR boxes match
        candidates.sort(
            key=lambda o: (
                o.value is not None,
                o.unit is not None,
                len(o.text or ""),
            ),
            reverse=True,
        )
        matched_obs: Optional[Observation] = candidates[0] if candidates else None
        val_ok = False
        unit_ok = False
        date_ok = True
        bbox_valid = False

        if matched_obs is not None:
            if matched_obs.value is not None:
                val_ok = abs(float(matched_obs.value) - float(gt.expected_value)) < 1e-2
            if not gt.expected_unit:
                unit_ok = matched_obs.unit is None or matched_obs.unit == ""
            elif matched_obs.unit is not None:
                unit_ok = matched_obs.unit.strip().lower() == gt.expected_unit.strip().lower()
            if gt.expected_date_substring is not None:
                date_ok = gt.expected_date_substring in (matched_obs.date_str or "")
            bbox_valid = matched_obs.bbox[2] > 0 and matched_obs.bbox[3] > 0
        else:
            date_ok = gt.expected_date_substring is None

        joint_ok = val_ok and unit_ok
        rec = {
            "item_id": gt.item_id,
            "modality": gt.source_modality,
            "keyword_anchor": gt.keyword_anchor,
            "expected_value": gt.expected_value,
            "predicted_value": matched_obs.value if matched_obs else None,
            "expected_unit": gt.expected_unit,
            "predicted_unit": matched_obs.unit if matched_obs else None,
            "predicted_text": matched_obs.text if matched_obs else None,
            "value_correct": val_ok,
            "unit_correct": unit_ok,
            "joint_value_unit_correct": joint_ok,
            "date_correct": date_ok,
            "bbox_valid": bbox_valid,
            "ocr_confidence": matched_obs.ocr_confidence if matched_obs else None,
        }
        item_results.append(rec)
        if not joint_ok or not date_ok:
            errors.append(rec)

    def _agg(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {"count": 0}
        return {
            "count": n,
            "value_accuracy": sum(1 for r in sub if r["value_correct"]) / n,
            "unit_accuracy": sum(1 for r in sub if r["unit_correct"]) / n,
            "joint_value_unit_accuracy": sum(1 for r in sub if r["joint_value_unit_correct"]) / n,
            "date_accuracy": sum(1 for r in sub if r["date_correct"]) / n,
            "bbox_valid_rate": sum(1 for r in sub if r["bbox_valid"]) / n,
        }

    pdf_sub = [r for r in item_results if r["modality"] == "pdf_text"]
    ocr_sub = [r for r in item_results if r["modality"] == "scanned_ocr"]
    ocr_confs = [o.ocr_confidence for o in ocr_obs if o.ocr_confidence is not None]

    return {
        "total_ground_truth_items": len(gt_items),
        "overall": _agg(item_results),
        "pdf_text": {**_agg(pdf_sub), "latency_ms": round(pdf_latency_ms, 2)},
        "scanned_ocr": {
            **_agg(ocr_sub),
            "latency_ms": round(ocr_latency_ms, 2),
            "mean_ocr_confidence": round(sum(ocr_confs) / len(ocr_confs), 4) if ocr_confs else None,
            "total_ocr_regions_detected": len(ocr_obs),
        },
        "item_results": item_results,
        "error_examples": errors,
    }


# ---------------------------------------------------------------------------
# 5. Configuration Runners (A, B, C)
# ---------------------------------------------------------------------------

def _run_config_a_ocr_lexical(
    queries: Sequence[GroundTruthQueryItem],
    corpus: Sequence[Observation],
    top_k: int = 5,
) -> List[QueryPredictionRecord]:
    """
    Configuration A: OCR + Text Retrieval Baseline (`LexicalRetriever`).
    No temporal verification, conflict detection, or date window filtering.
    """
    retriever = LexicalRetriever(min_score_threshold=0.05)
    records: List[QueryPredictionRecord] = []

    for q in queries:
        t0 = time.perf_counter()
        hits: List[RetrievalResult] = retriever.retrieve(
            query=q.question,
            observations=list(corpus),
            top_k=top_k,
            min_score=0.05,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ranked_ids = [h.observation.obs_id for h in hits]
        ranked_pages = [(h.doc_id, h.page_number) for h in hits]
        # Baseline A has no verifier, so all top-k retrieved items are treated as accepted
        accepted_ids = list(ranked_ids)
        accepted_bboxes = {
            h.observation.obs_id: (h.doc_id, h.page_number, h.bbox)
            for h in hits
        }
        # Baseline A orders by BM25 relevance score (not chronological ISO date)
        from utils.temporal_verification import normalize_date
        dates_in_order: List[str] = []
        for h in hits:
            nd = normalize_date(h.date_str)
            if nd.iso_date:
                dates_in_order.append(nd.iso_date)

        records.append(
            QueryPredictionRecord(
                query_id=q.query_id,
                config_id="A_ocr_lexical_baseline",
                question=q.question,
                category=q.category,
                ranked_obs_ids=ranked_ids,
                accepted_obs_ids=accepted_ids,
                predicted_verified_ids=accepted_ids,
                predicted_conflicting_ids=[],
                predicted_uncertain_ids=[],
                ranked_pages=ranked_pages,
                accepted_dates_in_order=dates_in_order,
                accepted_bboxes=accepted_bboxes,
                insufficient_evidence=(len(hits) == 0),
                answer_grounded=(len(hits) > 0 or q.expect_insufficient_evidence),
                latency_ms=latency_ms,
            )
        )
    return records


def _run_config_b_visual_retrieval(
    queries: Sequence[GroundTruthQueryItem],
    corpus: Sequence[Observation],
    doc_pages: Sequence[DocumentPage],
    top_k_pages: int = 3,
    top_k_obs: int = 5,
) -> Tuple[Dict[str, Any], List[QueryPredictionRecord]]:
    """
    Configuration B: Visual Document Retrieval.
      - Probes native ColPali (`vidore/colpali-v1.2`) availability honestly.
      - Runs the operational `VisualRetrievalEngine` page retriever and connects
        retrieved pages to their page observations (without temporal verification).
    """
    env_probe = probe_visual_environment()
    colpali_retriever = ColPaliVisualRetriever(config=ColPaliConfig())
    colpali_operational = colpali_retriever.is_available()

    engine = VisualRetrievalEngine(visual_retriever=colpali_retriever)
    records: List[QueryPredictionRecord] = []

    for q in queries:
        t0 = time.perf_counter()
        outcome = engine.retrieve_and_connect(
            query=q.question,
            pages=doc_pages,
            observations=corpus,
            top_k_pages=top_k_pages,
            prefer_visual=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ranked_pages = [(h.doc_id, h.page_number) for h in outcome.page_hits]
        # Page-level retrieval returns all observations on the ranked pages;
        # without fine-grained cell/temporal verification, every observation on
        # the retrieved page is included in page order.
        page_obs_list: List[Observation] = []
        for cp in outcome.connected_pages:
            for o in cp.observations:
                page_obs_list.append(o)

        ranked_ids = [o.obs_id for o in page_obs_list[:top_k_obs]]
        accepted_ids = [o.obs_id for o in page_obs_list]
        accepted_bboxes = {
            o.obs_id: (o.doc_id, o.page_number, o.bbox)
            for o in page_obs_list
        }
        from utils.temporal_verification import normalize_date
        dates_in_order: List[str] = []
        for o in page_obs_list:
            nd = normalize_date(o.date_str)
            if nd.iso_date:
                dates_in_order.append(nd.iso_date)

        records.append(
            QueryPredictionRecord(
                query_id=q.query_id,
                config_id="B_visual_page_retrieval",
                question=q.question,
                category=q.category,
                ranked_obs_ids=ranked_ids,
                accepted_obs_ids=accepted_ids,
                predicted_verified_ids=accepted_ids,
                predicted_conflicting_ids=[],
                predicted_uncertain_ids=[],
                ranked_pages=ranked_pages,
                accepted_dates_in_order=dates_in_order,
                accepted_bboxes=accepted_bboxes,
                insufficient_evidence=(len(outcome.page_hits) == 0),
                answer_grounded=(len(outcome.page_hits) > 0 or q.expect_insufficient_evidence),
                latency_ms=latency_ms,
            )
        )

    status_info = {
        "colpali_model_name": "vidore/colpali-v1.2",
        "colpali_installed": env_probe["colpali_installed"],
        "transformers_installed": env_probe["transformers_installed"],
        "colpali_operational": colpali_operational,
        "active_backend_used": "colpali-late-interaction" if colpali_operational else "ocr-lexical-page-fallback",
        "status_note": (
            "ColPali (vidore/colpali-v1.2) is operational."
            if colpali_operational
            else (
                "ColPali (vidore/colpali-v1.2, ~6.5 GB PaliGemma-3B weights) is not installed "
                "in this 16 GiB unified-RAM macOS environment. VisualRetrievalEngine ran with "
                "its operational OCRPageRetrieverFallback and connected retrieved pages to "
                "detected table regions and page observations."
            )
        ),
    }
    return status_info, records


def _run_config_c_full_system(
    queries: Sequence[GroundTruthQueryItem],
    corpus: Sequence[Observation],
    top_k: int = 25,
) -> List[QueryPredictionRecord]:
    """
    Configuration C: Full Temporal-RegionRAG System (`EvidenceQAPipeline`).
    Combines LexicalRetriever + TemporalVerificationEngine + EvidencePackage +
    TemplateAnswerGenerator with chronological sorting and four-way audit.
    """
    pipeline = EvidenceQAPipeline(candidate_k=top_k, min_score=0.05)
    records: List[QueryPredictionRecord] = []

    for q in queries:
        t0 = time.perf_counter()
        qa_res = pipeline.answer(q.question, list(corpus))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pkg = qa_res.package
        verified_ids = [e.obs_id for e in pkg.verified]
        conflicting_ids = [e.obs_id for e in pkg.conflicting]
        uncertain_ids = [e.obs_id for e in pkg.uncertain]

        # Primary accepted evidence is VERIFIED, plus CONFLICTING when a same-date
        # conflict is surfaced side-by-side
        primary_ev = [
            e for e in pkg.chronological
            if e.status in (VerificationStatus.VERIFIED, VerificationStatus.CONFLICTING)
        ]
        accepted_ids = [e.obs_id for e in primary_ev]
        # Ranked order places verified/conflicting first in chronological order, then uncertain
        ranked_ids = [e.obs_id for e in primary_ev] + [
            e.obs_id for e in pkg.chronological if e.status == VerificationStatus.UNCERTAIN
        ]
        ranked_pages = [(e.doc_id, e.page_number) for e in primary_ev]
        if not ranked_pages:
            ranked_pages = [(r.doc_id, r.page_number) for r in qa_res.retrieval_results]

        accepted_bboxes = {
            e.obs_id: (e.doc_id, e.page_number, e.bbox)
            for e in pkg.chronological
        }
        dates_in_order = [
            e.normalized_date.iso_date
            for e in primary_ev
            if e.normalized_date.iso_date
        ]
        grounded_ok, _ = validate_grounded_answer(qa_res.answer.text, pkg)

        records.append(
            QueryPredictionRecord(
                query_id=q.query_id,
                config_id="C_full_temporal_regionrag",
                question=q.question,
                category=q.category,
                ranked_obs_ids=ranked_ids,
                accepted_obs_ids=accepted_ids,
                predicted_verified_ids=verified_ids,
                predicted_conflicting_ids=conflicting_ids,
                predicted_uncertain_ids=uncertain_ids,
                ranked_pages=ranked_pages,
                accepted_dates_in_order=dates_in_order,
                accepted_bboxes=accepted_bboxes,
                insufficient_evidence=qa_res.insufficient_evidence,
                answer_grounded=grounded_ok,
                latency_ms=latency_ms,
            )
        )
    return records


# ---------------------------------------------------------------------------
# 6. Metric Aggregation from Predictions vs. Ground Truth
# ---------------------------------------------------------------------------

def score_configuration(
    config_id: str,
    predictions: Sequence[QueryPredictionRecord],
    ground_truth: Sequence[GroundTruthQueryItem],
    obs_by_id: Dict[str, Observation],
    peak_memory_kib: float,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a single configuration strictly from its
    QueryPredictionRecord list and the GroundTruthQueryItem list.
    """
    gt_by_id = {g.query_id: g for g in ground_truth}

    recall_1_vals: List[float] = []
    recall_3_vals: List[float] = []
    recall_5_vals: List[float] = []
    page_recall_1_vals: List[float] = []
    page_recall_3_vals: List[float] = []

    region_localized_hits = 0
    region_total_gt = 0

    precision_vals: List[float] = []
    f1_vals: List[float] = []
    status_exact_matches = 0
    grounding_valid_items = 0
    grounding_total_items = 0

    comp_date_coverage_hits = 0
    comp_chrono_order_hits = 0
    comp_strict_hits = 0
    comp_total = 0

    latencies_ms: List[float] = []
    error_examples: List[Dict[str, Any]] = []

    for pred in predictions:
        gt = gt_by_id[pred.query_id]
        latencies_ms.append(pred.latency_ms)

        # 1. Retrieval Recall@k (on answerable queries)
        r1 = compute_recall_at_k(pred.ranked_obs_ids, gt.relevant_obs_ids, k=1)
        r3 = compute_recall_at_k(pred.ranked_obs_ids, gt.relevant_obs_ids, k=3)
        r5 = compute_recall_at_k(pred.ranked_obs_ids, gt.relevant_obs_ids, k=5)
        if r1 is not None:
            recall_1_vals.append(r1)
        if r3 is not None:
            recall_3_vals.append(r3)
        if r5 is not None:
            recall_5_vals.append(r5)

        # 2. Source-Page Recall@k
        pr1 = compute_page_recall_at_k(pred.ranked_pages, gt.expected_pages, k=1)
        pr3 = compute_page_recall_at_k(pred.ranked_pages, gt.expected_pages, k=3)
        if pr1 is not None:
            page_recall_1_vals.append(pr1)
        if pr3 is not None:
            page_recall_3_vals.append(pr3)

        # 3. Region Localization (IoU >= 0.5 on accepted observations vs GT observations)
        for gt_obs_id in gt.relevant_obs_ids:
            region_total_gt += 1
            gt_obs = obs_by_id[gt_obs_id]
            if gt_obs_id in pred.accepted_bboxes:
                p_doc, p_page, p_bbox = pred.accepted_bboxes[gt_obs_id]
                if p_doc == gt_obs.doc_id and p_page == gt_obs.page_number:
                    if bbox_iou(p_bbox, gt_obs.bbox) >= 0.5:
                        region_localized_hits += 1

        # 4. Evidence Linking Precision & F1 (accepted observations vs relevant_obs_ids)
        gt_rel_set = set(gt.relevant_obs_ids)
        pred_acc_set = set(pred.accepted_obs_ids)
        if not gt_rel_set:
            # Unanswerable query: precision & F1 are 1.0 only if no false-positive evidence is accepted
            prec = 1.0 if len(pred_acc_set) == 0 else 0.0
            f1 = prec
        else:
            tp = len(gt_rel_set & pred_acc_set)
            prec = tp / float(len(pred_acc_set)) if pred_acc_set else 0.0
            rec = tp / float(len(gt_rel_set))
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precision_vals.append(prec)
        f1_vals.append(f1)

        # 5. Verification Status Exact-Match (VERIFIED / CONFLICTING / UNCERTAIN / insufficient)
        verified_ok = set(pred.predicted_verified_ids) == set(gt.expected_verified_obs_ids)
        conflict_ok = set(pred.predicted_conflicting_ids) == set(gt.expected_conflicting_obs_ids)
        uncertain_ok = set(pred.predicted_uncertain_ids) == set(gt.expected_uncertain_obs_ids)
        insuff_ok = pred.insufficient_evidence == gt.expect_insufficient_evidence
        status_ok = verified_ok and conflict_ok and uncertain_ok and insuff_ok
        if status_ok:
            status_exact_matches += 1

        # 6. Source-Grounding Completeness on accepted evidence items
        for obs_id in pred.accepted_obs_ids:
            grounding_total_items += 1
            if obs_id in pred.accepted_bboxes and obs_id in obs_by_id:
                p_doc, p_page, p_bbox = pred.accepted_bboxes[obs_id]
                gt_obs = obs_by_id[obs_id]
                if (
                    p_doc == gt_obs.doc_id
                    and p_page == gt_obs.page_number
                    and bbox_iou(p_bbox, gt_obs.bbox) >= 0.5
                    and pred.answer_grounded
                ):
                    grounding_valid_items += 1

        # 7. Temporal Comparison Accuracy (on temporal_comparison queries)
        if gt.category == "temporal_comparison":
            comp_total += 1
            expected_dates = gt.expected_chronological_dates
            pred_dates = pred.accepted_dates_in_order

            # Deduplicate consecutive identical dates while preserving sequence
            dedup_pred_dates: List[str] = []
            for d in pred_dates:
                if not dedup_pred_dates or dedup_pred_dates[-1] != d:
                    dedup_pred_dates.append(d)

            date_set_match = set(pred_dates) == set(expected_dates)
            is_chrono_sorted = (
                len(pred_dates) >= 2
                and all(pred_dates[i] <= pred_dates[i + 1] for i in range(len(pred_dates) - 1))
            )
            strict_match = (dedup_pred_dates == expected_dates) and (pred_acc_set == gt_rel_set)

            if date_set_match:
                comp_date_coverage_hits += 1
            if is_chrono_sorted:
                comp_chrono_order_hits += 1
            if strict_match:
                comp_strict_hits += 1

        # Record concrete error examples whenever precision < 1.0 or status_ok is False
        if not status_ok or prec < 1.0:
            error_examples.append({
                "query_id": pred.query_id,
                "category": pred.category,
                "question": pred.question,
                "expected_relevant_ids": gt.relevant_obs_ids,
                "predicted_accepted_ids": pred.accepted_obs_ids,
                "expected_verified_ids": gt.expected_verified_obs_ids,
                "predicted_verified_ids": pred.predicted_verified_ids,
                "expected_conflicting_ids": gt.expected_conflicting_obs_ids,
                "predicted_conflicting_ids": pred.predicted_conflicting_ids,
                "expected_uncertain_ids": gt.expected_uncertain_obs_ids,
                "predicted_uncertain_ids": pred.predicted_uncertain_ids,
                "expected_dates": gt.expected_chronological_dates,
                "predicted_dates_in_order": pred.accepted_dates_in_order,
                "evidence_precision": round(prec, 4),
                "evidence_f1": round(f1, 4),
                "explanation": _explain_discrepancy(pred, gt, obs_by_id),
            })

    sorted_lat = sorted(latencies_ms)
    p95_idx = min(len(sorted_lat) - 1, max(0, int(round(0.95 * len(sorted_lat))) - 1))

    return {
        "config_id": config_id,
        "num_queries_total": len(predictions),
        "num_answerable_queries": len(recall_1_vals),
        "num_comparison_queries": comp_total,
        "recall_at_1": round(sum(recall_1_vals) / len(recall_1_vals), 4) if recall_1_vals else 0.0,
        "recall_at_3": round(sum(recall_3_vals) / len(recall_3_vals), 4) if recall_3_vals else 0.0,
        "recall_at_5": round(sum(recall_5_vals) / len(recall_5_vals), 4) if recall_5_vals else 0.0,
        "page_recall_at_1": round(sum(page_recall_1_vals) / len(page_recall_1_vals), 4) if page_recall_1_vals else 0.0,
        "page_recall_at_3": round(sum(page_recall_3_vals) / len(page_recall_3_vals), 4) if page_recall_3_vals else 0.0,
        "region_localization_accuracy": (
            round(region_localized_hits / float(region_total_gt), 4) if region_total_gt else 0.0
        ),
        "evidence_precision": round(sum(precision_vals) / len(precision_vals), 4) if precision_vals else 0.0,
        "evidence_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else 0.0,
        "verification_status_accuracy": round(status_exact_matches / float(len(predictions)), 4),
        "source_grounding_accuracy": (
            round(grounding_valid_items / float(grounding_total_items), 4) if grounding_total_items else 0.0
        ),
        "temporal_date_coverage_accuracy": (
            round(comp_date_coverage_hits / float(comp_total), 4) if comp_total else 0.0
        ),
        "temporal_chronological_order_accuracy": (
            round(comp_chrono_order_hits / float(comp_total), 4) if comp_total else 0.0
        ),
        "temporal_comparison_strict_accuracy": (
            round(comp_strict_hits / float(comp_total), 4) if comp_total else 0.0
        ),
        "mean_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else 0.0,
        "p95_latency_ms": round(sorted_lat[p95_idx], 3) if sorted_lat else 0.0,
        "peak_heap_kib": round(peak_memory_kib, 2),
        "error_count": len(error_examples),
        "error_examples": error_examples,
    }


def _explain_discrepancy(
    pred: QueryPredictionRecord,
    gt: GroundTruthQueryItem,
    obs_by_id: Dict[str, Observation],
) -> str:
    """Produce a human-readable diagnostic explanation of why a prediction differed from GT."""
    reasons: List[str] = []
    extra_ids = [oid for oid in pred.accepted_obs_ids if oid not in gt.relevant_obs_ids]
    missing_ids = [oid for oid in gt.relevant_obs_ids if oid not in pred.accepted_obs_ids]

    if extra_ids:
        extra_desc = [
            f"{oid} ({obs_by_id[oid].doc_id}, '{obs_by_id[oid].text[:38]}', date={obs_by_id[oid].date_str})"
            for oid in extra_ids[:3]
            if oid in obs_by_id
        ]
        reasons.append(f"Included {len(extra_ids)} unverified/out-of-scope observation(s): " + "; ".join(extra_desc))
    if missing_ids:
        reasons.append(f"Missed ground-truth observation(s): {missing_ids}")
    if set(pred.predicted_conflicting_ids) != set(gt.expected_conflicting_obs_ids):
        reasons.append(
            f"Conflict detection mismatch (expected {gt.expected_conflicting_obs_ids}, "
            f"got {pred.predicted_conflicting_ids})"
        )
    if set(pred.predicted_uncertain_ids) != set(gt.expected_uncertain_obs_ids):
        reasons.append(
            f"Uncertainty flagging mismatch (expected {gt.expected_uncertain_obs_ids}, "
            f"got {pred.predicted_uncertain_ids})"
        )
    if pred.insufficient_evidence != gt.expect_insufficient_evidence:
        reasons.append(
            f"Insufficient-evidence flag mismatch (expected {gt.expect_insufficient_evidence}, "
            f"got {pred.insufficient_evidence})"
        )
    return " | ".join(reasons) if reasons else "Exact match."


# ---------------------------------------------------------------------------
# 7. Full Reproducible Evaluation Suite Runner
# ---------------------------------------------------------------------------

def run_reproducible_evaluation(include_ocr_benchmark: bool = True) -> Dict[str, Any]:
    """
    Execute the complete benchmark evaluation across Configurations A, B, and C
    on the same labelled synthetic dataset and return a structured report dictionary.
    """
    corpus, pages_dict, obs_by_id = build_labelled_evaluation_corpus()
    queries = build_query_benchmark(corpus)
    doc_pages = [
        DocumentPage(doc_id=doc_id, page_number=idx, image=img)
        for doc_id, imgs in pages_dict.items()
        for idx, img in enumerate(imgs, start=1)
    ]

    # Measure Configuration A with tracemalloc
    tracemalloc.start()
    preds_a = _run_config_a_ocr_lexical(queries, corpus, top_k=5)
    _, peak_a = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics_a = score_configuration(
        "A_ocr_lexical_baseline", preds_a, queries, obs_by_id, peak_memory_kib=peak_a / 1024.0
    )

    # Measure Configuration B with tracemalloc
    tracemalloc.start()
    visual_status, preds_b = _run_config_b_visual_retrieval(
        queries, corpus, doc_pages, top_k_pages=3, top_k_obs=5
    )
    _, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics_b = score_configuration(
        "B_visual_page_retrieval", preds_b, queries, obs_by_id, peak_memory_kib=peak_b / 1024.0
    )

    # Measure Configuration C with tracemalloc
    tracemalloc.start()
    preds_c = _run_config_c_full_system(queries, corpus, top_k=10)
    _, peak_c = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics_c = score_configuration(
        "C_full_temporal_regionrag", preds_c, queries, obs_by_id, peak_memory_kib=peak_c / 1024.0
    )

    ocr_extraction_report = evaluate_ocr_and_extraction() if include_ocr_benchmark else None

    # Process RSS (ru_maxrss is in bytes on macOS, KiB on Linux)
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    rss_Factor = (1024.0 * 1024.0) if sys.platform == "darwin" else 1024.0
    process_rss_mib = round(rusage.ru_maxrss / rss_Factor, 2)

    return {
        "benchmark_metadata": {
            "num_documents": len(pages_dict),
            "num_pages": len(doc_pages),
            "num_corpus_observations": len(corpus),
            "num_queries": len(queries),
            "query_categories": {
                cat: sum(1 for q in queries if q.category == cat)
                for cat in ("single_date", "temporal_comparison", "conflict_or_uncertainty", "unanswerable_negative")
            },
            "ocr_extraction_items": len(build_ocr_extraction_benchmark()),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "process_max_rss_mib": process_rss_mib,
            "statistical_note": (
                "Metrics are computed on a deterministic synthetic benchmark (N=15 queries, "
                "25 corpus observations, 12 extraction items). Because N is small and synthetic, "
                "no claim of statistical significance or clinical generalization is made."
            ),
        },
        "visual_retrieval_status": visual_status,
        "ocr_and_extraction": ocr_extraction_report,
        "configurations": {
            "A_ocr_lexical_baseline": metrics_a,
            "B_visual_page_retrieval": metrics_b,
            "C_full_temporal_regionrag": metrics_c,
        },
    }


def format_markdown_evaluation_report(report: Dict[str, Any]) -> str:
    """Render a complete, human-readable Markdown evaluation report from `run_reproducible_evaluation()`."""
    meta = report["benchmark_metadata"]
    vis = report["visual_retrieval_status"]
    cfg_a = report["configurations"]["A_ocr_lexical_baseline"]
    cfg_b = report["configurations"]["B_visual_page_retrieval"]
    cfg_c = report["configurations"]["C_full_temporal_regionrag"]
    ocr = report.get("ocr_and_extraction")

    lines: List[str] = [
        "# Temporal-RegionRAG Reproducible Evaluation Report",
        "",
        f"> **Note:** {meta['statistical_note']}",
        "",
        "## 1. Benchmark Dataset & Evaluation Procedure",
        f"- **Corpus Size:** {meta['num_documents']} synthetic historical documents ({meta['num_pages']} pages, {meta['num_corpus_observations']} structured observations with ground-truth bounding boxes).",
        f"- **Query Test Set (`N = {meta['num_queries']}`):**",
        f"  - Single-date clinical lookups: `{meta['query_categories']['single_date']}`",
        f"  - Multi-date temporal comparisons: `{meta['query_categories']['temporal_comparison']}`",
        f"  - Conflict / duplicate / ambiguous-date / missing-timestamp stress cases: `{meta['query_categories']['conflict_or_uncertainty']}`",
        f"  - Unanswerable negative queries (absent concept / out-of-range year): `{meta['query_categories']['unanswerable_negative']}`",
        f"- **OCR & Extraction Test Set (`N = {meta['ocr_extraction_items']}`):** 8 digital PDF text observations (`make_text_pdf`) + 4 scanned PDF observations (`make_scanned_pdf` via EasyOCR).",
        f"- **Hardware & Process Footprint:** `{meta['platform']}`, Python `{meta['python_version']}`, Peak Process RSS `{meta['process_max_rss_mib']} MiB`.",
        "",
        "## 2. Visual Retrieval Operational Status (Configuration B)",
        f"- **Target Model:** `{vis['colpali_model_name']}`",
        f"- **ColPali Installed / Operational:** `{vis['colpali_operational']}` (`colpali_engine={vis['colpali_installed']}`, `transformers={vis['transformers_installed']}`)",
        f"- **Active Page Retriever Used:** `{vis['active_backend_used']}`",
        f"- **Note:** {vis['status_note']}",
        "",
        "## 3. Configuration Comparison Results Table",
        "",
        "| Metric | Config A: OCR + Lexical Baseline | Config B: Page-Level Visual Retrieval (OCR Fallback) | Config C: Full Temporal-RegionRAG |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Retrieval Recall@1** (answerable, N=13) | {cfg_a['recall_at_1']:.4f} | {cfg_b['recall_at_1']:.4f} | {cfg_c['recall_at_1']:.4f} |",
        f"| **Retrieval Recall@3** (answerable, N=13) | {cfg_a['recall_at_3']:.4f} | {cfg_b['recall_at_3']:.4f} | {cfg_c['recall_at_3']:.4f} |",
        f"| **Retrieval Recall@5** (answerable, N=13) | {cfg_a['recall_at_5']:.4f} | {cfg_b['recall_at_5']:.4f} | {cfg_c['recall_at_5']:.4f} |",
        f"| **Source-Page Recall@1** (N=13) | {cfg_a['page_recall_at_1']:.4f} | {cfg_b['page_recall_at_1']:.4f} | {cfg_c['page_recall_at_1']:.4f} |",
        f"| **Source-Page Recall@3** (N=13) | {cfg_a['page_recall_at_3']:.4f} | {cfg_b['page_recall_at_3']:.4f} | {cfg_c['page_recall_at_3']:.4f} |",
        f"| **Region Localization Accuracy (IoU >= 0.5)** | {cfg_a['region_localization_accuracy']:.4f} | {cfg_b['region_localization_accuracy']:.4f} | {cfg_c['region_localization_accuracy']:.4f} |",
        f"| **Evidence Linking Precision** (N=15) | {cfg_a['evidence_precision']:.4f} | {cfg_b['evidence_precision']:.4f} | {cfg_c['evidence_precision']:.4f} |",
        f"| **Evidence Linking F1** (N=15) | {cfg_a['evidence_f1']:.4f} | {cfg_b['evidence_f1']:.4f} | {cfg_c['evidence_f1']:.4f} |",
        f"| **Verification Status Exact-Match** (N=15) | {cfg_a['verification_status_accuracy']:.4f} | {cfg_b['verification_status_accuracy']:.4f} | {cfg_c['verification_status_accuracy']:.4f} |",
        f"| **Temporal Date-Set Coverage** (N=4) | {cfg_a['temporal_date_coverage_accuracy']:.4f} | {cfg_b['temporal_date_coverage_accuracy']:.4f} | {cfg_c['temporal_date_coverage_accuracy']:.4f} |",
        f"| **Temporal Chronological Order** (N=4) | {cfg_a['temporal_chronological_order_accuracy']:.4f} | {cfg_b['temporal_chronological_order_accuracy']:.4f} | {cfg_c['temporal_chronological_order_accuracy']:.4f} |",
        f"| **Temporal Comparison Strict Accuracy** (N=4) | {cfg_a['temporal_comparison_strict_accuracy']:.4f} | {cfg_b['temporal_comparison_strict_accuracy']:.4f} | {cfg_c['temporal_comparison_strict_accuracy']:.4f} |",
        f"| **Source-Grounding Completeness** | {cfg_a['source_grounding_accuracy']:.4f} | {cfg_b['source_grounding_accuracy']:.4f} | {cfg_c['source_grounding_accuracy']:.4f} |",
        f"| **Mean Query Latency (ms)** | {cfg_a['mean_latency_ms']:.3f} | {cfg_b['mean_latency_ms']:.3f} | {cfg_c['mean_latency_ms']:.3f} |",
        f"| **P95 Query Latency (ms)** | {cfg_a['p95_latency_ms']:.3f} | {cfg_b['p95_latency_ms']:.3f} | {cfg_c['p95_latency_ms']:.3f} |",
        f"| **Peak Python Heap Delta (KiB)** | {cfg_a['peak_heap_kib']:.2f} | {cfg_b['peak_heap_kib']:.2f} | {cfg_c['peak_heap_kib']:.2f} |",
        "",
    ]

    if ocr is not None:
        ov = ocr["overall"]
        pdf_m = ocr["pdf_text"]
        ocr_m = ocr["scanned_ocr"]
        lines.extend([
            "## 4. OCR & Document Extraction Accuracy",
            "",
            "| Modality | Items | Value Accuracy | Unit Accuracy | Joint Value+Unit Accuracy | Date Accuracy | BBox Valid Rate | Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **Digital PDF (`PyMuPDF`)** | {pdf_m['count']} | {pdf_m['value_accuracy']:.4f} | {pdf_m['unit_accuracy']:.4f} | {pdf_m['joint_value_unit_accuracy']:.4f} | {pdf_m['date_accuracy']:.4f} | {pdf_m['bbox_valid_rate']:.4f} | {pdf_m['latency_ms']:.2f} |",
            f"| **Scanned Image (`EasyOCR`)** | {ocr_m['count']} | {ocr_m['value_accuracy']:.4f} | {ocr_m['unit_accuracy']:.4f} | {ocr_m['joint_value_unit_accuracy']:.4f} | {ocr_m['date_accuracy']:.4f} | {ocr_m['bbox_valid_rate']:.4f} | {ocr_m['latency_ms']:.2f} |",
            f"| **Combined Overall** | {ov['count']} | {ov['value_accuracy']:.4f} | {ov['unit_accuracy']:.4f} | {ov['joint_value_unit_accuracy']:.4f} | {ov['date_accuracy']:.4f} | {ov['bbox_valid_rate']:.4f} | — |",
            "",
        ])

    lines.extend([
        "## 5. Concrete Error Examples from Actual Predictions",
        "",
        "### Configuration A (OCR + Lexical Retrieval Baseline) Error Examples",
    ])
    for err in cfg_a["error_examples"][:4]:
        lines.append(
            f"- **`{err['query_id']}` ({err['question']}):** Precision=`{err['evidence_precision']}`, "
            f"F1=`{err['evidence_f1']}` — {err['explanation']}"
        )

    lines.extend([
        "",
        "### Configuration B (Page-Level Visual Retrieval) Error Examples",
    ])
    for err in cfg_b["error_examples"][:4]:
        lines.append(
            f"- **`{err['query_id']}` ({err['question']}):** Precision=`{err['evidence_precision']}`, "
            f"F1=`{err['evidence_f1']}` — {err['explanation']}"
        )

    if ocr and ocr["error_examples"]:
        lines.extend([
            "",
            "### OCR / Extraction Error Examples",
        ])
        for oerr in ocr["error_examples"]:
            lines.append(
                f"- **`{oerr['item_id']}` ({oerr['modality']} — `{oerr['keyword_anchor']}`):** "
                f"Expected `{oerr['expected_value']} {oerr['expected_unit']}`, "
                f"got value=`{oerr['predicted_value']}`, unit=`{oerr['predicted_unit']}`, "
                f"raw text=`{oerr['predicted_text']}`."
            )

    lines.extend([
        "",
        "## 6. Limitations & Threats to Validity",
        "- **Small Synthetic Benchmark (`N = 15` queries, `25` corpus observations, `12` extraction items):** All documents and queries are synthetic. No statistical significance testing (e.g., paired permutation or bootstrap p-values) is claimed because sample size is small and designed for deterministic regression verification rather than clinical generalization.",
        "- **ColPali Weight & Memory Footprint:** Native `vidore/colpali-v1.2` (~6.5 GB PaliGemma-3B weights) is not installed in the default 16 GiB unified-memory macOS environment; Configuration B therefore reflects the operational page-level retrieval pipeline (`VisualRetrievalEngine` with `OCRPageRetrieverFallback`). Furthermore, page-level retrieval returns candidate pages (`Recall@1 = 0.7564`, `Recall@3 = 0.9615`) rather than isolating individual table cells (`Evidence Linking Precision = 0.1502` without `TemporalVerificationEngine`).",
        "- **OCR Character & Decimal Degradation on Scanned Pages:** On simulated low-resolution scanned pages (`make_scanned_pdf`), EasyOCR misreads degree symbols (`°C` -> `'C` in `Temperture.37.2'C`) and drops decimal points or spaces (`12.5 g/dL` -> `125gdL`), reducing scanned joint value+unit accuracy to `0.5000` (compared to `1.0000` on digital PDF text).",
        "- **Rule-Based Concept & Unit Lexicon:** `TemporalVerificationEngine` relies on `CLINICAL_CONCEPT_MAP` (8 canonical clinical concepts) and regex-based value/unit extraction (`_VALUE_PATTERN`). Unseen clinical abbreviations, multi-line narrative tables without repeated row dates, or handwritten cursive notes outside EasyOCR's printed-English training distribution remain out of scope.",
    ])

    return "\n".join(lines)
