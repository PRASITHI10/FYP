"""
tests/test_temporal_verification.py
-----------------------------------
Unit tests for Query-Conditioned Temporal Verification and Evidence Linking:
1. Date normalization (ISO, UK, US, named months, year-only).
2. Preservation of verbatim raw date text.
3. Ambiguous date format detection (e.g. 04/05/1998).
4. Missing timestamp detection and handling.
5. Query specification parsing (concept identification, temporal constraints, comparison).
6. Concept verification (preventing cross-measurement co-occurrence confounding).
7. Evidence categorization: VERIFIED, UNCERTAIN, CONFLICTING, REJECTED with recorded reasons.
8. Conflict detection between contradictory readings on the same date/encounter.
9. Duplicate observation detection.
10. Chronological ordering of evidence across patient history.
11. Source grounding retention (doc_id, page_number, bounding box).
12. Strict restraint (no invented dates, values, or units).
"""

import pytest

from utils.extraction import Observation
from utils.temporal_verification import (
    TemporalVerificationEngine,
    VerificationStatus,
    normalize_date,
    parse_clinical_query,
    matches_target_concept,
    satisfies_temporal_constraint,
)
from utils.synthetic_docs import (
    create_synthetic_retrieval_corpus,
    create_synthetic_verification_corpus,
)


# ---------------------------------------------------------------------------
# 1. Date Normalization & Ambiguity Detection Tests
# ---------------------------------------------------------------------------

class TestDateNormalization:

    def test_normalize_iso_date(self):
        nd = normalize_date("1998-01-15")
        assert nd.iso_date == "1998-01-15"
        assert nd.year == 1998
        assert nd.month == 1
        assert nd.day == 15
        assert not nd.is_ambiguous
        assert not nd.is_missing
        assert nd.raw_text == "1998-01-15"

    def test_normalize_named_month(self):
        nd = normalize_date("15 Jan 1998")
        assert nd.iso_date == "1998-01-15"
        assert nd.year == 1998
        assert nd.month == 1
        assert nd.day == 15
        assert not nd.is_ambiguous
        assert nd.raw_text == "15 Jan 1998"

    def test_normalize_named_month_american(self):
        nd = normalize_date("November 10, 2005")
        assert nd.iso_date == "2005-11-10"
        assert nd.year == 2005
        assert nd.month == 11
        assert nd.day == 10

    def test_normalize_year_only(self):
        nd = normalize_date("2002")
        assert nd.iso_date == "2002"
        assert nd.year == 2002
        assert nd.granularity == "year"
        assert not nd.is_ambiguous

    def test_unambiguous_uk_date(self):
        # 22 > 12 so 22 must be the day
        nd = normalize_date("22/04/2002")
        assert nd.iso_date == "2002-04-22"
        assert nd.year == 2002
        assert nd.month == 4
        assert nd.day == 22
        assert not nd.is_ambiguous

    def test_unambiguous_us_date(self):
        # 25 > 12 so 25 must be the day in month/day/year
        nd = normalize_date("04/25/2002")
        assert nd.iso_date == "2002-04-25"
        assert nd.year == 2002
        assert nd.month == 4
        assert nd.day == 25
        assert not nd.is_ambiguous

    def test_ambiguous_date_detected(self):
        # Both 4 and 5 <= 12: cannot disambiguate 4 May vs 5 April without locale
        nd = normalize_date("04/05/1998")
        assert nd.is_ambiguous
        assert nd.ambiguity_reason is not None
        assert "Ambiguous numeric date" in nd.ambiguity_reason
        # Original text preserved
        assert nd.raw_text == "04/05/1998"

    def test_missing_date_handled(self):
        nd = normalize_date(None)
        assert nd.is_missing
        assert nd.iso_date is None
        assert nd.granularity == "missing"

        nd_empty = normalize_date("")
        assert nd_empty.is_missing
        assert nd_empty.iso_date is None


# ---------------------------------------------------------------------------
# 2. Query Parsing & Specification Tests
# ---------------------------------------------------------------------------

class TestQueryParsing:

    def test_parse_concept_and_year(self):
        q = "What was the blood pressure in 1998?"
        spec = parse_clinical_query(q)
        assert "blood_pressure" in spec.target_concepts
        assert len(spec.temporal_constraints) == 1
        assert spec.temporal_constraints[0].target_year == 1998
        assert not spec.is_comparison

    def test_parse_multi_date_comparison(self):
        q = "Compare blood pressure in 1998 and 2002"
        spec = parse_clinical_query(q)
        assert "blood_pressure" in spec.target_concepts
        assert spec.is_comparison
        years = [c.target_year for c in spec.temporal_constraints if c.target_year]
        assert 1998 in years
        assert 2002 in years

    def test_parse_unbounded_trend(self):
        q = "Show fasting blood glucose trend over time"
        spec = parse_clinical_query(q)
        assert "glucose" in spec.target_concepts
        assert spec.is_comparison
        assert any(c.constraint_type == "unbounded" for c in spec.temporal_constraints)


# ---------------------------------------------------------------------------
# 3. Concept Verification & Co-Occurrence Guard Tests
# ---------------------------------------------------------------------------

class TestConceptVerification:

    def test_positive_concept_match(self):
        obs = Observation(text="Blood Pressure: 140/90 mmHg", unit="mmHg")
        is_match, label = matches_target_concept(obs, "blood_pressure")
        assert is_match
        assert "Blood Pressure" in label

    def test_negative_concept_guard(self):
        """Heart rate on the same page must NOT match blood pressure query."""
        obs = Observation(text="Heart Rate: 88 bpm", unit="bpm")
        is_match, _ = matches_target_concept(obs, "blood_pressure")
        assert not is_match

    def test_glucose_not_matching_cholesterol(self):
        obs = Observation(text="Fasting Blood Glucose: 5.4 mmol/L", unit="mmol/L")
        is_match, _ = matches_target_concept(obs, "cholesterol")
        assert not is_match


# ---------------------------------------------------------------------------
# 4. Temporal Constraint Verification Tests
# ---------------------------------------------------------------------------

class TestTemporalConstraintVerification:

    def test_satisfies_matching_year(self):
        nd = normalize_date("15 Jan 1998")
        spec = parse_clinical_query("What was BP in 1998?")
        match, reason = satisfies_temporal_constraint(nd, spec.temporal_constraints)
        assert match

    def test_rejects_differing_year(self):
        nd = normalize_date("10 Nov 2005")
        spec = parse_clinical_query("What was BP in 1998?")
        match, reason = satisfies_temporal_constraint(nd, spec.temporal_constraints)
        assert not match
        assert "Temporal mismatch" in reason

    def test_rejects_missing_date_when_date_requested(self):
        nd = normalize_date(None)
        spec = parse_clinical_query("What was BP in 1998?")
        match, reason = satisfies_temporal_constraint(nd, spec.temporal_constraints)
        assert not match
        assert "lacks a date timestamp" in reason


# ---------------------------------------------------------------------------
# 5. Core Verification Engine Integration Tests
# ---------------------------------------------------------------------------

class TestTemporalVerificationEngine:

    @pytest.fixture
    def engine(self):
        return TemporalVerificationEngine()

    @pytest.fixture
    def verification_corpus(self):
        return create_synthetic_verification_corpus()

    def test_verified_evidence_separated(self, engine):
        """Query targeting blood pressure on clean corpus verifies valid reading."""
        clean_corpus = create_synthetic_retrieval_corpus()
        report = engine.verify("What was the blood pressure on 15 Jan 1998?", clean_corpus)

        assert len(report.verified) > 0
        # Check verified items match concept and date
        for ev in report.verified:
            assert ev.status == VerificationStatus.VERIFIED
            assert "Blood Pressure" in (ev.concept_matched or "")
            assert ev.normalized_date.year == 1998
            # Source grounding is preserved
            assert ev.doc_id != ""
            assert ev.page_number >= 1
            assert ev.has_valid_coordinates()
            assert ev.reason != ""

    def test_conflict_detection(self, engine, verification_corpus):
        """Contradictory readings (140/90 vs 110/70) on 15 Jan 1998 must be marked CONFLICTING."""
        report = engine.verify("Check blood pressure in 1998", verification_corpus)

        assert len(report.conflicting) >= 2
        conflicting_obs_ids = {c.obs_id for c in report.conflicting}
        assert "obs_conflict_bp" in conflicting_obs_ids

        # Reason should explicitly explain the conflict
        for c in report.conflicting:
            assert c.status == VerificationStatus.CONFLICTING
            assert "Conflicting values" in (c.conflict_details or "")

        # Conflict anomaly recorded in report
        assert any("Conflicting values" in a for a in report.detected_anomalies)

    def test_duplicate_detection(self, engine, verification_corpus):
        """Duplicate entry on 15 Jan 1998 must be detected and flagged."""
        report = engine.verify("Check blood pressure in 1998", verification_corpus)

        # The duplicate observation should be flagged
        dup_items = [ev for ev in report.uncertain if ev.is_duplicate]
        assert len(dup_items) >= 1
        assert dup_items[0].obs_id == "obs_duplicate_bp"
        assert "Duplicate record" in dup_items[0].reason

    def test_ambiguous_date_flagged_uncertain(self, engine, verification_corpus):
        """Observation with '04/05/1998' must be placed in UNCERTAIN with ambiguity reason."""
        report = engine.verify("Show glucose levels in 1998", verification_corpus)

        ambig_items = [ev for ev in report.uncertain if ev.obs_id == "obs_ambiguous_date"]
        assert len(ambig_items) == 1
        ev = ambig_items[0]
        assert ev.status == VerificationStatus.UNCERTAIN
        assert ev.normalized_date.is_ambiguous
        assert "Ambiguous numeric date" in ev.reason

    def test_concept_mismatch_rejected(self, engine, verification_corpus):
        """When query is for blood pressure, Heart Rate and Lisinopril observations are REJECTED."""
        report = engine.verify("What was the blood pressure in 1998?", verification_corpus)

        assert len(report.rejected) > 0
        rejected_texts = [r.text for r in report.rejected]
        assert any("Heart Rate" in t for t in rejected_texts)
        assert any("Lisinopril" in t for t in rejected_texts)

        # Rejection reasons recorded
        for r in report.rejected:
            assert r.status == VerificationStatus.REJECTED
            assert len(r.reason) > 0

    def test_temporal_mismatch_rejected(self, engine, verification_corpus):
        """When query specifies 1998, observations from 2005 are REJECTED with temporal reason."""
        report = engine.verify("What was the blood pressure in 1998?", verification_corpus)

        temporal_rejected = [
            r for r in report.rejected
            if "Temporal mismatch" in r.reason or "2005" in (r.normalized_date.iso_date or "")
        ]
        assert len(temporal_rejected) > 0

    def test_chronological_ordering(self, engine, verification_corpus):
        """Longitudinal query over time arranges verified evidence in ascending temporal order."""
        report = engine.verify("Show fasting blood glucose trend over time", verification_corpus)

        # chronological_evidence should be sorted by normalized date
        chrono = report.chronological_evidence
        assert len(chrono) >= 3

        # Filter to dated entries and check strict non-decreasing sort order
        dated = [e for e in chrono if e.normalized_date.sort_key != float("inf")]
        for i in range(len(dated) - 1):
            assert dated[i].normalized_date.sort_key <= dated[i + 1].normalized_date.sort_key

        # Check that progression from 1998 to 2002 to 2005 is preserved
        years = [e.normalized_date.year for e in dated if e.normalized_date.year]
        assert 1998 in years
        assert 2002 in years
        assert 2005 in years
        # 1998 must precede 2005
        assert years.index(1998) < years.index(2005)

    def test_strict_provenance_retention(self, engine, verification_corpus):
        """All verified evidence retains full source document, page, and coordinate links."""
        report = engine.verify("Show fasting blood glucose trend over time", verification_corpus)
        for ev in report.verified:
            assert ev.doc_id != ""
            assert ev.page_number >= 1
            assert isinstance(ev.bbox, tuple) and len(ev.bbox) == 4
            x, y, w, h = ev.bbox
            assert w > 0 and h > 0
            assert ev.raw_date_str is not None
            # to_dict produces full metadata record
            d = ev.to_dict()
            assert "verification_status" in d
            assert "verification_reason" in d
            assert "normalized_date" in d

    def test_no_hallucination_or_invention(self, engine):
        """Missing fields remain None and are never invented."""
        obs = Observation(
            doc_id="doc_sparse",
            page_number=1,
            text="Patient resting comfortably",
            value=None,
            unit=None,
            date_str=None,
        )
        report = engine.verify("What is the patient status?", [obs])
        assert len(report.chronological_evidence) == 1
        ev = report.chronological_evidence[0]
        assert ev.value is None
        assert ev.unit is None
        assert ev.normalized_date.is_missing
        assert ev.normalized_date.iso_date is None
