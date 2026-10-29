"""
utils/temporal_verification.py
------------------------------
Core research module: Query-Conditioned Temporal Verification and Evidence Linking
for Historical Medical Documents.

Responsibilities:
1. Parse user queries to identify target clinical concepts and requested temporal periods/dates.
2. Normalize dates and timestamps (ISO-8601, UK, US, named months) while preserving the original verbatim date string.
3. Chronologically sort and index observations along the patient timeline.
4. Verify whether each candidate observation matches the requested measurement/concept, date, unit, and source context.
5. Detect missing timestamps, conflicting values on identical encounters, duplicate records, and ambiguous date formats.
6. Link every accepted observation directly to its source document ID, page number, and spatial bounding box.
7. Separate evidence into four distinct categories: VERIFIED, UNCERTAIN, CONFLICTING, and REJECTED, each with an explicit recorded reason.
8. Maintain strict scientific restraint:
   - Never invent dates, values, units, or missing evidence.
   - Do not assume that two similar values refer to the same measurement.
   - Do not infer correlation implies causation.
   - Do not claim clinical validation without empirical experimental benchmarking.

Architecture:
- Completely decoupled from the Streamlit UI.
- Deterministic, rule-based, fully offline execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from utils.extraction import Observation


# ---------------------------------------------------------------------------
# Clinical Concept Lexicon (Canonical Mapping & Standard Units)
# ---------------------------------------------------------------------------

CLINICAL_CONCEPT_MAP: Dict[str, Dict[str, any]] = {
    "blood_pressure": {
        "aliases": ["blood pressure", "bp", "systolic", "diastolic", "hypertension", "normotensive"],
        "standard_units": ["mmhg"],
        "display_name": "Blood Pressure",
    },
    "heart_rate": {
        "aliases": ["heart rate", "hr", "pulse", "pulse rate", "cardiac rate", "bpm"],
        "standard_units": ["bpm", "beats/min", "beats per minute"],
        "display_name": "Heart Rate",
    },
    "glucose": {
        "aliases": ["glucose", "blood glucose", "fasting glucose", "fbg", "blood sugar", "glycemia"],
        "standard_units": ["mmol/l", "mg/dl"],
        "display_name": "Fasting Blood Glucose",
    },
    "cholesterol": {
        "aliases": ["cholesterol", "total cholesterol", "serum cholesterol", "lipid", "hyperlipidemia"],
        "standard_units": ["mg/dl", "mmol/l"],
        "display_name": "Total Serum Cholesterol",
    },
    "hba1c": {
        "aliases": ["hba1c", "glycated haemoglobin", "glycated hemoglobin", "hemoglobin a1c", "a1c"],
        "standard_units": ["%"],
        "display_name": "Glycated Haemoglobin (HbA1c)",
    },
    "temperature": {
        "aliases": ["temperature", "body temperature", "temp", "febrile", "afebrile"],
        "standard_units": ["°c", "°f", "c", "f"],
        "display_name": "Body Temperature",
    },
    "respiratory_rate": {
        "aliases": ["respiratory rate", "respiration rate", "rr", "breaths/min"],
        "standard_units": ["breaths/min", "/min"],
        "display_name": "Respiratory Rate",
    },
    "allergy": {
        "aliases": ["allergy", "allergies", "allergic", "urticaria", "anaphylaxis", "adverse reaction"],
        "standard_units": [],
        "display_name": "Allergy Status",
    },
}

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NormalizedDate:
    """
    Structured representation of a parsed date.
    Preserves verbatim raw text while providing standardized ISO and sort keys.
    """
    raw_text: Optional[str] = None
    iso_date: Optional[str] = None        # e.g. "1998-01-15", "1998-01", "1998"
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    sort_key: float = float("inf")        # Chronological sort order
    granularity: str = "missing"          # "day", "month", "year", "missing"
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    is_missing: bool = False

    def __str__(self) -> str:
        return self.iso_date or self.raw_text or "Undated"


@dataclass
class TemporalConstraint:
    """Target temporal constraint identified from a user query."""
    constraint_type: str                  # "year", "exact_date", "range", "unbounded"
    target_year: Optional[int] = None
    target_date_iso: Optional[str] = None
    start_date_iso: Optional[str] = None
    end_date_iso: Optional[str] = None
    raw_query_segment: str = ""


@dataclass
class QuerySpecification:
    """Structured representation of a parsed clinical question."""
    raw_query: str
    target_concepts: List[str]            # Canonical concept keys, e.g. ["blood_pressure"]
    target_concept_labels: List[str]      # Human-readable labels
    temporal_constraints: List[TemporalConstraint]
    is_comparison: bool = False


class VerificationStatus(str, Enum):
    """Four-way verification categorization for evidence grounding."""
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTING = "CONFLICTING"
    REJECTED = "REJECTED"


@dataclass
class VerifiedEvidence:
    """
    A grounded clinical observation with verified status, recorded explanation,
    chronological timestamp, and exact source provenance.
    """
    observation: Observation
    status: VerificationStatus
    reason: str
    normalized_date: NormalizedDate
    concept_matched: Optional[str] = None
    unit_valid: bool = True
    temporal_match: bool = True
    conflict_details: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of_id: Optional[str] = None

    # Provenance properties linking directly to source
    @property
    def obs_id(self) -> str:
        return self.observation.obs_id

    @property
    def doc_id(self) -> str:
        return self.observation.doc_id

    @property
    def page_number(self) -> int:
        return self.observation.page_number

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.observation.bbox

    @property
    def text(self) -> str:
        return self.observation.text

    @property
    def value(self) -> Optional[float]:
        return self.observation.value

    @property
    def unit(self) -> Optional[str]:
        return self.observation.unit

    @property
    def raw_date_str(self) -> Optional[str]:
        return self.observation.date_str

    @property
    def source_type(self) -> str:
        return self.observation.source_type

    def has_valid_coordinates(self) -> bool:
        _, _, w, h = self.bbox
        return w > 0 and h > 0

    def to_dict(self) -> dict:
        d = self.observation.to_dict()
        d["verification_status"] = self.status.value
        d["verification_reason"] = self.reason
        d["normalized_date"] = self.normalized_date.iso_date or ""
        d["date_granularity"] = self.normalized_date.granularity
        d["date_is_ambiguous"] = self.normalized_date.is_ambiguous
        d["date_ambiguity_reason"] = self.normalized_date.ambiguity_reason or ""
        d["concept_matched"] = self.concept_matched or ""
        d["unit_valid"] = self.unit_valid
        d["temporal_match"] = self.temporal_match
        d["is_duplicate"] = self.is_duplicate
        d["conflict_details"] = self.conflict_details or ""
        return d


@dataclass
class VerificationReport:
    """Complete summary of a query-conditioned temporal verification run."""
    query_spec: QuerySpecification
    verified: List[VerifiedEvidence]
    uncertain: List[VerifiedEvidence]
    conflicting: List[VerifiedEvidence]
    rejected: List[VerifiedEvidence]
    chronological_evidence: List[VerifiedEvidence]  # All non-rejected evidence in temporal order
    total_candidates: int
    detected_anomalies: List[str]


# ---------------------------------------------------------------------------
# 1. Date Normalization & Ambiguity Detection
# ---------------------------------------------------------------------------

def normalize_date(date_text: Optional[str]) -> NormalizedDate:
    """
    Parse and normalize a raw date string into an ISO representation and sort key.
    Detects missing dates and flags ambiguous numeric formats (e.g. DD/MM vs MM/DD).
    Never invents missing day or month values.
    """
    if not date_text or not date_text.strip():
        return NormalizedDate(
            raw_text=date_text,
            granularity="missing",
            is_missing=True,
            ambiguity_reason="No date text provided in source record",
        )

    raw = date_text.strip()

    # Pattern 1: ISO YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
    if m_iso:
        y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            sort_val = y * 10000 + m * 100 + d
            return NormalizedDate(
                raw_text=raw,
                iso_date=f"{y:04d}-{m:02d}-{d:02d}",
                year=y,
                month=m,
                day=d,
                sort_key=float(sort_val),
                granularity="day",
            )

    # Pattern 2: Named Month (e.g. "15 Jan 1998", "January 15, 1998", "15 January 1998")
    m_named = re.search(
        r"\b(?:(\d{1,2})\s+)?([A-Za-z]{3,9})\s+(\d{1,2})?,?\s*(\d{4})\b",
        raw,
    )
    if m_named:
        day_pre, month_name, day_post, year_str = m_named.groups()
        m_key = month_name.lower()
        if m_key in MONTH_MAP:
            m_val = MONTH_MAP[m_key]
            y_val = int(year_str)
            d_val = int(day_pre or day_post) if (day_pre or day_post) else None

            if d_val and 1 <= d_val <= 31:
                sort_val = y_val * 10000 + m_val * 100 + d_val
                return NormalizedDate(
                    raw_text=raw,
                    iso_date=f"{y_val:04d}-{m_val:02d}-{d_val:02d}",
                    year=y_val,
                    month=m_val,
                    day=d_val,
                    sort_key=float(sort_val),
                    granularity="day",
                )
            else:
                sort_val = y_val * 10000 + m_val * 100
                return NormalizedDate(
                    raw_text=raw,
                    iso_date=f"{y_val:04d}-{m_val:02d}",
                    year=y_val,
                    month=m_val,
                    sort_key=float(sort_val),
                    granularity="month",
                )

    # Pattern 3: Slash / Dot Numeric Formats: A/B/YYYY or A.B.YYYY
    m_num = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", raw)
    if m_num:
        a, b, y_val = int(m_num.group(1)), int(m_num.group(2)), int(m_num.group(3))

        # Case 3a: a > 12 -> Must be Day/Month/Year (UK/Intl format)
        if a > 12 and 1 <= b <= 12 and 1 <= a <= 31:
            sort_val = y_val * 10000 + b * 100 + a
            return NormalizedDate(
                raw_text=raw,
                iso_date=f"{y_val:04d}-{b:02d}-{a:02d}",
                year=y_val,
                month=b,
                day=a,
                sort_key=float(sort_val),
                granularity="day",
                is_ambiguous=False,
            )

        # Case 3b: b > 12 -> Must be Month/Day/Year (US format)
        if b > 12 and 1 <= a <= 12 and 1 <= b <= 31:
            sort_val = y_val * 10000 + a * 100 + b
            return NormalizedDate(
                raw_text=raw,
                iso_date=f"{y_val:04d}-{a:02d}-{b:02d}",
                year=y_val,
                month=a,
                day=b,
                sort_key=float(sort_val),
                granularity="day",
                is_ambiguous=False,
            )

        # Case 3c: Both a <= 12 and b <= 12 and a != b -> Ambiguous date format!
        if 1 <= a <= 12 and 1 <= b <= 12 and a != b:
            # Default to UK (Day/Month/Year) as tentative sort key, but flag ambiguity explicitly
            tentative_sort = y_val * 10000 + b * 100 + a
            return NormalizedDate(
                raw_text=raw,
                iso_date=f"{y_val:04d}-{b:02d}-{a:02d} (ambiguous: or {y_val:04d}-{a:02d}-{b:02d})",
                year=y_val,
                month=b,
                day=a,
                sort_key=float(tentative_sort),
                granularity="day",
                is_ambiguous=True,
                ambiguity_reason=(
                    f"Ambiguous numeric date '{raw}': cannot disambiguate "
                    f"day {a}/month {b} (UK) from month {a}/day {b} (US) without explicit locale metadata."
                ),
            )

        # Case 3d: a == b (e.g. 05/05/1998)
        if 1 <= a <= 12 and a == b:
            sort_val = y_val * 10000 + a * 100 + a
            return NormalizedDate(
                raw_text=raw,
                iso_date=f"{y_val:04d}-{a:02d}-{a:02d}",
                year=y_val,
                month=a,
                day=a,
                sort_key=float(sort_val),
                granularity="day",
                is_ambiguous=False,
            )

    # Pattern 4: Year Only (e.g. "1998", "2002")
    m_year = re.search(r"\b(19\d\d|20\d\d)\b", raw)
    if m_year:
        y_val = int(m_year.group(1))
        sort_val = y_val * 10000
        return NormalizedDate(
            raw_text=raw,
            iso_date=str(y_val),
            year=y_val,
            sort_key=float(sort_val),
            granularity="year",
            is_ambiguous=False,
        )

    # Unknown date format
    return NormalizedDate(
        raw_text=raw,
        granularity="unknown",
        is_missing=False,
        ambiguity_reason=f"Unrecognized date format: '{raw}'",
    )


# ---------------------------------------------------------------------------
# 2. Query Parsing & Concept Identification
# ---------------------------------------------------------------------------

def parse_clinical_query(query: str) -> QuerySpecification:
    """
    Analyze a natural language clinical query to extract:
    1. Target clinical concepts and expected units.
    2. Temporal constraints (specific year, date, interval, or unbounded).
    3. Comparison / longitudinal trend intent.
    """
    q_lower = query.lower()

    # Identify target concepts
    target_concepts: List[str] = []
    target_labels: List[str] = []
    for c_key, c_info in CLINICAL_CONCEPT_MAP.items():
        if any(alias in q_lower for alias in c_info["aliases"]):
            target_concepts.append(c_key)
            target_labels.append(c_info["display_name"])

    # Extract temporal constraints
    temporal_constraints: List[TemporalConstraint] = []

    # Check for 4-digit years (e.g. 1998, 2002)
    year_matches = list(re.finditer(r"\b(19\d\d|20\d\d)\b", query))
    years = [int(m.group(1)) for m in year_matches]

    # Check for full dates (e.g. "15 Jan 1998", "1998-01-15")
    date_patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b",
    ]
    full_dates = []
    for pat in date_patterns:
        full_dates.extend(re.findall(pat, query))

    if full_dates:
        for fd in full_dates:
            norm = normalize_date(fd)
            temporal_constraints.append(
                TemporalConstraint(
                    constraint_type="exact_date",
                    target_year=norm.year,
                    target_date_iso=norm.iso_date,
                    raw_query_segment=fd,
                )
            )
    elif years:
        if len(years) >= 2 and any(k in q_lower for k in ["between", "from", "to", "-"]):
            y_start, y_end = min(years), max(years)
            temporal_constraints.append(
                TemporalConstraint(
                    constraint_type="range",
                    target_year=y_start,
                    start_date_iso=f"{y_start}-01-01",
                    end_date_iso=f"{y_end}-12-31",
                    raw_query_segment=f"{y_start} to {y_end}",
                )
            )
        else:
            for y in years:
                temporal_constraints.append(
                    TemporalConstraint(
                        constraint_type="year",
                        target_year=y,
                        target_date_iso=str(y),
                        raw_query_segment=str(y),
                    )
                )
    else:
        # Unbounded / longitudinal query
        temporal_constraints.append(
            TemporalConstraint(constraint_type="unbounded", raw_query_segment="")
        )

    # Detect comparison intent
    comp_keywords = {"compare", "comparison", "trend", "trends", "vs", "versus", "progression", "over time", "change", "difference"}
    is_comp = any(k in q_lower for k in comp_keywords) or len(temporal_constraints) >= 2 or len(years) >= 2

    return QuerySpecification(
        raw_query=query,
        target_concepts=target_concepts,
        target_concept_labels=target_labels,
        temporal_constraints=temporal_constraints,
        is_comparison=is_comp,
    )


# ---------------------------------------------------------------------------
# 3. Concept Verification & Matching
# ---------------------------------------------------------------------------

def matches_target_concept(obs: Observation, target_concept: str) -> Tuple[bool, Optional[str]]:
    """
    Verify whether an observation actually describes the requested concept.
    Guards against confounding observations that appear on the same page.
    For example: 'Heart Rate: 88 bpm' must NOT match a query for 'blood_pressure'.
    """
    obs_text = (obs.text or "").lower()
    obs_unit = (obs.unit or "").lower()

    concept_info = CLINICAL_CONCEPT_MAP.get(target_concept)
    if not concept_info:
        # Generic match if concept is not in predefined map
        return True, target_concept

    # 1. Negative guard: Check if observation explicitly matches a DIFFERENT concept
    for other_key, other_info in CLINICAL_CONCEPT_MAP.items():
        if other_key == target_concept:
            continue
        # If other concept's primary alias is explicitly present and target is not
        if other_info["aliases"][0] in obs_text and concept_info["aliases"][0] not in obs_text:
            return False, None

    # 2. Positive check: Match target aliases
    for alias in concept_info["aliases"]:
        if alias in obs_text:
            return True, concept_info["display_name"]

    # 3. Unit check: Match standard unit if present
    if obs_unit and any(obs_unit == u for u in concept_info["standard_units"]):
        return True, concept_info["display_name"]

    return False, None


# ---------------------------------------------------------------------------
# 4. Temporal Constraint Verification
# ---------------------------------------------------------------------------

def satisfies_temporal_constraint(
    norm_date: NormalizedDate,
    constraints: List[TemporalConstraint],
) -> Tuple[bool, str]:
    """
    Verify if the normalized observation date satisfies the requested temporal constraint.
    Never invents or assumes missing dates.
    """
    if not constraints or any(c.constraint_type == "unbounded" for c in constraints):
        # Query did not specify a date constraint; all dates or undated are allowed
        return True, "No temporal constraint specified"

    if norm_date.is_missing:
        return False, "Observation lacks a date timestamp; cannot verify against temporal constraint"

    for c in constraints:
        if c.constraint_type == "year":
            if norm_date.year == c.target_year:
                return True, f"Observation matches target year {c.target_year}"
        elif c.constraint_type == "exact_date":
            if norm_date.iso_date and c.target_date_iso in norm_date.iso_date:
                return True, f"Observation matches target date {c.target_date_iso}"
            if norm_date.year == c.target_year:
                return True, f"Observation matches target year {c.target_year}"
        elif c.constraint_type == "range":
            if norm_date.year and c.target_year:
                # Year range
                if c.start_date_iso and c.end_date_iso:
                    y_start = int(c.start_date_iso[:4])
                    y_end = int(c.end_date_iso[:4])
                    if y_start <= norm_date.year <= y_end:
                        return True, f"Observation date {norm_date} falls within range {y_start}–{y_end}"

    target_str = ", ".join(c.raw_query_segment for c in constraints if c.raw_query_segment)
    return False, f"Temporal mismatch: observation date '{norm_date}' does not match requested time period ({target_str})"


# ---------------------------------------------------------------------------
# 5. Temporal Verification Engine
# ---------------------------------------------------------------------------

class TemporalVerificationEngine:
    """
    Deterministic clinical verification and evidence-linking engine.

    Categorizes evidence into:
    - VERIFIED: Valid concept, temporal match, valid unit, unambiguous date, spatial grounding.
    - UNCERTAIN: Low OCR confidence, ambiguous date, missing expected unit.
    - CONFLICTING: Incompatible values on the same clinical encounter.
    - REJECTED: Concept mismatch, temporal mismatch, out-of-scope.
    """

    def __init__(self, ocr_confidence_threshold: float = 0.6):
        self.ocr_confidence_threshold = ocr_confidence_threshold

    def verify(
        self,
        query: str,
        observations: List[Observation],
    ) -> VerificationReport:
        """
        Execute query-conditioned temporal verification across a set of observations.
        Returns a complete structured VerificationReport.
        """
        query_spec = parse_clinical_query(query)

        verified: List[VerifiedEvidence] = []
        uncertain: List[VerifiedEvidence] = []
        conflicting: List[VerifiedEvidence] = []
        rejected: List[VerifiedEvidence] = []
        anomalies: List[str] = []

        # If no specific concept detected from query, allow all candidate observations
        target_concepts = query_spec.target_concepts or list(CLINICAL_CONCEPT_MAP.keys())

        # First pass: concept and temporal matching
        initial_evaluated: List[VerifiedEvidence] = []

        for obs in observations:
            # 1. Date normalization (preserves original text)
            norm_date = normalize_date(obs.date_str or _extract_embedded_date(obs.text))

            if norm_date.is_missing:
                anomalies.append(f"Missing timestamp on observation {obs.obs_id} in {obs.doc_id}")

            if norm_date.is_ambiguous:
                anomalies.append(
                    f"Ambiguous date on observation {obs.obs_id}: {norm_date.ambiguity_reason}"
                )

            # 2. Concept check
            matched_concept_key: Optional[str] = None
            matched_concept_label: Optional[str] = None

            for c_key in target_concepts:
                is_match, label = matches_target_concept(obs, c_key)
                if is_match:
                    matched_concept_key = c_key
                    matched_concept_label = label
                    break

            if not matched_concept_key and query_spec.target_concepts:
                # Rejected due to concept mismatch
                rejected.append(
                    VerifiedEvidence(
                        observation=obs,
                        status=VerificationStatus.REJECTED,
                        reason=f"Concept mismatch: observation '{obs.text}' does not match requested clinical concept(s)",
                        normalized_date=norm_date,
                        concept_matched=None,
                        temporal_match=False,
                    )
                )
                continue

            # 2b. Measurement check: a concept that is a measured quantity (has standard
            # units) cannot be answered by a narrative mention without a recorded value,
            # e.g. "admitted with hypertension" is not a blood-pressure measurement.
            matched_info = CLINICAL_CONCEPT_MAP.get(matched_concept_key or "", {})
            if matched_info.get("standard_units") and obs.value is None:
                rejected.append(
                    VerifiedEvidence(
                        observation=obs,
                        status=VerificationStatus.REJECTED,
                        reason=(
                            f"Related mention only: '{obs.text}' refers to {matched_concept_label} "
                            "terminology but records no measured value"
                        ),
                        normalized_date=norm_date,
                        concept_matched=matched_concept_label,
                        temporal_match=False,
                    )
                )
                continue

            # 3. Temporal constraint check
            temp_match, temp_reason = satisfies_temporal_constraint(
                norm_date, query_spec.temporal_constraints
            )

            if not temp_match:
                # Rejected due to temporal mismatch
                rejected.append(
                    VerifiedEvidence(
                        observation=obs,
                        status=VerificationStatus.REJECTED,
                        reason=temp_reason,
                        normalized_date=norm_date,
                        concept_matched=matched_concept_label,
                        temporal_match=False,
                    )
                )
                continue

            # 4. OCR uncertainty check
            is_ocr_uncertain = (
                obs.ocr_uncertain or
                (obs.ocr_confidence is not None and obs.ocr_confidence < self.ocr_confidence_threshold)
            )

            # 5. Unit check
            concept_info = CLINICAL_CONCEPT_MAP.get(matched_concept_key, {})
            standard_units = concept_info.get("standard_units", [])
            unit_valid = True
            unit_warning = ""

            if standard_units and obs.unit:
                if obs.unit.lower() not in standard_units:
                    unit_valid = False
                    unit_warning = f"Non-standard unit '{obs.unit}' for {matched_concept_label}"

            # Initial status determination
            if is_ocr_uncertain:
                ev = VerifiedEvidence(
                    observation=obs,
                    status=VerificationStatus.UNCERTAIN,
                    reason=f"Uncertain OCR quality (confidence: {obs.ocr_confidence or 0.0:.2f})",
                    normalized_date=norm_date,
                    concept_matched=matched_concept_label,
                    unit_valid=unit_valid,
                    temporal_match=True,
                )
            elif norm_date.is_ambiguous:
                ev = VerifiedEvidence(
                    observation=obs,
                    status=VerificationStatus.UNCERTAIN,
                    reason=norm_date.ambiguity_reason or "Ambiguous date format",
                    normalized_date=norm_date,
                    concept_matched=matched_concept_label,
                    unit_valid=unit_valid,
                    temporal_match=True,
                )
            elif not unit_valid:
                ev = VerifiedEvidence(
                    observation=obs,
                    status=VerificationStatus.UNCERTAIN,
                    reason=unit_warning,
                    normalized_date=norm_date,
                    concept_matched=matched_concept_label,
                    unit_valid=False,
                    temporal_match=True,
                )
            else:
                ev = VerifiedEvidence(
                    observation=obs,
                    status=VerificationStatus.VERIFIED,
                    reason=f"Verified match for {matched_concept_label or 'clinical concept'} at date {norm_date}",
                    normalized_date=norm_date,
                    concept_matched=matched_concept_label,
                    unit_valid=True,
                    temporal_match=True,
                )

            initial_evaluated.append(ev)

        # Second pass: Duplicate detection and Conflict detection across candidates
        # Group by (concept, normalized_date_str)
        grouped_candidates: Dict[str, List[VerifiedEvidence]] = {}
        for ev in initial_evaluated:
            c_key = ev.concept_matched or "general"
            d_key = ev.normalized_date.iso_date or "undated"
            group_key = f"{c_key}::{d_key}"
            grouped_candidates.setdefault(group_key, []).append(ev)

        final_verified: List[VerifiedEvidence] = []
        final_uncertain: List[VerifiedEvidence] = []
        final_conflicting: List[VerifiedEvidence] = []

        for group_key, items in grouped_candidates.items():
            if len(items) == 1:
                ev = items[0]
                if ev.status == VerificationStatus.VERIFIED:
                    final_verified.append(ev)
                else:
                    final_uncertain.append(ev)
                continue

            # Multiple observations on same concept and date: check duplicates vs conflicts
            seen_signatures: Dict[str, VerifiedEvidence] = {}
            conflicting_items: List[VerifiedEvidence] = []

            for item in items:
                # Signature: value + unit
                sig = f"{item.value}::{item.unit or ''}"
                if sig in seen_signatures:
                    # Duplicate record detected!
                    first_item = seen_signatures[sig]
                    item.is_duplicate = True
                    item.duplicate_of_id = first_item.obs_id
                    item.status = VerificationStatus.UNCERTAIN
                    item.reason = f"Duplicate record: identical measurement to observation #{first_item.obs_id}"
                    anomalies.append(
                        f"Duplicate observation detected on {item.normalized_date}: {item.text}"
                    )
                    final_uncertain.append(item)
                else:
                    seen_signatures[sig] = item

            unique_active = [it for it in items if not it.is_duplicate]

            # Only compare observations that have numeric values and were otherwise valid
            valued_active = [
                it for it in unique_active
                if it.value is not None and it.status == VerificationStatus.VERIFIED
            ]
            distinct_values = {u.value for u in valued_active}

            # If there are multiple active observations with differing values on the same date:
            if len(distinct_values) > 1:
                # Conflicting values on identical encounter!
                values_summary = ", ".join(f"{u.value} {u.unit or ''}" for u in valued_active)
                conflict_msg = (
                    f"Conflicting values recorded on same date ({valued_active[0].normalized_date}): {values_summary}"
                )
                anomalies.append(conflict_msg)

                for u in valued_active:
                    u.status = VerificationStatus.CONFLICTING
                    u.conflict_details = conflict_msg
                    u.reason = f"Contradicted by another measurement on {u.normalized_date} ({conflict_msg})"
                    final_conflicting.append(u)

                # Items without numeric values (e.g. general notes) or already uncertain remain as they were
                for u in unique_active:
                    if u not in valued_active:
                        if u.status == VerificationStatus.VERIFIED:
                            final_verified.append(u)
                        elif u.status == VerificationStatus.UNCERTAIN:
                            final_uncertain.append(u)
            else:
                for u in unique_active:
                    if u.status == VerificationStatus.VERIFIED:
                        final_verified.append(u)
                    elif u.status == VerificationStatus.UNCERTAIN:
                        final_uncertain.append(u)

        # Merge initial uncertain items that were not part of conflicts
        for ev in initial_evaluated:
            if ev.status == VerificationStatus.UNCERTAIN and ev not in final_uncertain:
                final_uncertain.append(ev)

        # Chronological sort across all accepted/evaluated non-rejected evidence
        all_accepted = final_verified + final_uncertain + final_conflicting
        chronological_evidence = sorted(
            all_accepted,
            key=lambda x: (x.normalized_date.sort_key, x.doc_id, x.page_number),
        )

        return VerificationReport(
            query_spec=query_spec,
            verified=final_verified,
            uncertain=final_uncertain,
            conflicting=final_conflicting,
            rejected=rejected,
            chronological_evidence=chronological_evidence,
            total_candidates=len(observations),
            detected_anomalies=anomalies,
        )


def _extract_embedded_date(text: str) -> Optional[str]:
    """Helper to detect date string embedded directly in observation text."""
    if not text:
        return None
    m = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", text)
    return m.group(1) if m else None
