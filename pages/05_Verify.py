"""
pages/05_Verify.py
------------------
Page 5: Query-Conditioned Temporal Verification and Evidence Linking

Responsibilities:
1. Accept a user clinical query and parse target concepts and temporal constraints.
2. Normalize dates to ISO formats while preserving original verbatim text.
3. Chronologically sort and present verified observations.
4. Verify concept, date, unit, and source context.
5. Detect missing timestamps, conflicting values on identical encounters,
   duplicate records, and ambiguous numeric date formats (e.g. DD/MM vs MM/DD).
6. Ground every observation with its source document ID, page, and bounding box.
7. Separate evidence into four explicit categories:
   - 🟢 VERIFIED
   - 🟡 UNCERTAIN
   - 🔴 CONFLICTING
   - ⚪ REJECTED
8. Scientific restraint: Never invent missing evidence or dates; avoid causal claims.
"""

from __future__ import annotations

import io
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.temporal_verification import (
    TemporalVerificationEngine,
    VerificationStatus,
    parse_clinical_query,
    normalize_date,
)
from utils.synthetic_docs import (
    create_synthetic_verification_corpus,
    create_synthetic_retrieval_documents,
)
from utils.retrieval import highlight_retrieved_regions, crop_observation_region

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Verify | Temporal-RegionRAG",
    page_icon="⏱️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⏱️ Verification Settings")
    ocr_threshold = st.slider(
        "OCR Confidence Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Observations with OCR confidence below this threshold are categorized as UNCERTAIN.",
    )

    show_bboxes = st.checkbox("Show Spatial Bounding Boxes", value=True)
    show_snippets = st.checkbox("Show Region Snippets", value=True)

    st.markdown("---")
    st.markdown(
        """
        **Verification Categories**
        - 🟢 **VERIFIED**: Concept, date, unit, and grounding verified.
        - 🟡 **UNCERTAIN**: Ambiguous date, low OCR, or duplicate record.
        - 🔴 **CONFLICTING**: Contradictory values on identical encounter.
        - ⚪ **REJECTED**: Concept mismatch or temporal mismatch.
        """
    )
    st.markdown("---")
    st.caption(
        "**Research Notice**\n"
        "• Deterministic rule-based evaluation\n"
        "• No causal inference or medical diagnosis\n"
        "• No fabricated/imputed values or dates"
    )

# ---------------------------------------------------------------------------
# Header & Research Disclaimer
# ---------------------------------------------------------------------------
st.title("⏱️ Temporal Verification & Evidence Linking")
st.subheader("Query-Conditioned Chronological Verification for Historical Medical Records")

st.info(
    "🔬 **Scientific Grounding Notice:** This research prototype verifies whether extracted observations "
    "match requested clinical concepts and temporal constraints without inventing missing values, timestamps, or units. "
    "Similar values are not assumed to be the same measurement, and correlation does not imply causation. "
    "Clinical claims require formal empirical benchmarking."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Corpus & Scenarios
# ---------------------------------------------------------------------------
st.markdown("### 🧪 Pre-Configured Benchmark Scenarios")
st.caption("Click a scenario below to populate a demonstration question and examine the verification engine's behavior:")

sc_col1, sc_col2, sc_col3, sc_col4, sc_col5, sc_col6 = st.columns(6)

scenarios = {
    sc_col1: (
        "📈 Longitudinal Trend",
        "Show fasting blood glucose trend over time",
        "Evaluates chronological progression across 1998, 2002, and 2005 encounters.",
    ),
    sc_col2: (
        "🔴 Conflicting BP",
        "Check blood pressure on 15 Jan 1998",
        "Detects contradictory BP readings (140/90 vs 110/70) on the same date.",
    ),
    sc_col3: (
        "🟡 Ambiguous Date",
        "Show glucose levels in 1998",
        "Flags ambiguous numeric date '04/05/1998' (cannot disambiguate 4 May vs 5 April).",
    ),
    sc_col4: (
        "⏱️ Missing Timestamp",
        "What was the heart rate in 1998?",
        "Identifies observations lacking timestamps when a specific date is requested.",
    ),
    sc_col5: (
        "🚫 Concept Mismatch",
        "What was the blood pressure in 2002?",
        "Rejects co-occurring non-BP readings (e.g. Heart Rate, Lisinopril) from the same page.",
    ),
    sc_col6: (
        "📅 Temporal Mismatch",
        "What was the blood pressure in 1998?",
        "Rejects blood pressure readings taken in 2002 and 2005 due to temporal constraint.",
    ),
}

for col, (label, query_text, desc) in scenarios.items():
    with col:
        if st.button(label, use_container_width=True, help=desc):
            st.session_state["verify_query_input"] = query_text

# Corpus selection
tab_bench, tab_session = st.tabs(["🏥 Verification Benchmark Corpus", "📂 Session Observations"])

corpus = []
corpus_images: dict[str, Image.Image] = {}

with tab_bench:
    corpus = create_synthetic_verification_corpus()
    _, synth_imgs = create_synthetic_retrieval_documents()
    corpus_images = synth_imgs
    st.caption(
        f"Loaded benchmark corpus with **{len(corpus)}** observations "
        f"(includes clean longitudinal visits + injected conflicts, duplicates, ambiguous dates, and faded scans)."
    )

with tab_session:
    session_obs = st.session_state.get("extracted_observations", [])
    if session_obs:
        use_sess = st.checkbox(f"Use session observations ({len(session_obs)} items)", value=False)
        if use_sess:
            corpus = session_obs
            session_imgs = st.session_state.get("extracted_page_images", [])
            doc_id = st.session_state.get("extracted_doc_id", "session_doc")
            corpus_images = {doc_id: session_imgs[0]} if session_imgs else {}
    else:
        st.info("No extracted observations in session. Using benchmark corpus above.")


# ---------------------------------------------------------------------------
# Query Input
# ---------------------------------------------------------------------------
st.markdown("---")
default_q = st.session_state.get(
    "verify_query_input", "Show fasting blood glucose trend over time"
)
user_query = st.text_input(
    "Enter a clinical question to verify against records:",
    value=default_q,
    key="verify_query_input_box",
)

run_verify = st.button("⚡ Run Temporal Verification", type="primary", use_container_width=True)

if user_query and corpus:
    engine = TemporalVerificationEngine(ocr_confidence_threshold=ocr_threshold)
    report = engine.verify(user_query, corpus)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Query Specification Card
    # -----------------------------------------------------------------------
    st.markdown("### 📋 Query Specification & Temporal Parameters")
    q_spec = report.query_spec

    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        st.markdown("**Identified Clinical Concept(s):**")
        if q_spec.target_concept_labels:
            for lbl in q_spec.target_concept_labels:
                st.markdown(f"- 🩺 `{lbl}`")
        else:
            st.markdown("- *General / Unconstrained Concept*")

    with c_col2:
        st.markdown("**Temporal Constraint(s):**")
        for tc in q_spec.temporal_constraints:
            st.markdown(f"- 📅 Type: `{tc.constraint_type}` | Target: `{tc.raw_query_segment or 'All dates / Unbounded'}`")

    with c_col3:
        st.markdown("**Query Intent:**")
        st.markdown(
            f"- {'📈 Multi-Date / Longitudinal Comparison' if q_spec.is_comparison else '🎯 Focused Single-Encounter Query'}"
        )

    # -----------------------------------------------------------------------
    # Verification Category Summary Metrics
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Verification Outcome Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 Verified Evidence", f"{len(report.verified)} items")
    m2.metric("🟡 Uncertain Evidence", f"{len(report.uncertain)} items")
    m3.metric("🔴 Conflicting Evidence", f"{len(report.conflicting)} items")
    m4.metric("⚪ Rejected Candidates", f"{len(report.rejected)} items")

    # -----------------------------------------------------------------------
    # Detected Anomalies Card
    # -----------------------------------------------------------------------
    if report.detected_anomalies:
        with st.expander(f"⚠️ Detected Anomalies & Data Quality Warnings ({len(report.detected_anomalies)})", expanded=True):
            for anom in report.detected_anomalies:
                st.warning(f"• {anom}")

    # -----------------------------------------------------------------------
    # Chronological Evidence Stream (Timeline)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📅 Chronological Evidence Timeline")
    st.caption("All accepted and non-rejected evidence arranged in ascending temporal order with provenance links:")

    if not report.chronological_evidence:
        st.warning("No observations were verified or accepted for this query.")
    else:
        for idx, ev in enumerate(report.chronological_evidence, start=1):
            status_badge = {
                VerificationStatus.VERIFIED: "🟢 VERIFIED",
                VerificationStatus.UNCERTAIN: "🟡 UNCERTAIN",
                VerificationStatus.CONFLICTING: "🔴 CONFLICTING",
            }.get(ev.status, str(ev.status.value))

            with st.container():
                hdr_col1, hdr_col2, hdr_col3 = st.columns([1.5, 2.0, 1.0])
                with hdr_col1:
                    st.markdown(f"**#{idx} | {ev.normalized_date.iso_date or 'Undated'}**")
                    st.caption(f"Raw Date: `{ev.raw_date_str or 'None'}`")
                with hdr_col2:
                    st.markdown(f"**{ev.text}**")
                    val_str = f"{ev.value} {ev.unit or ''}" if ev.value is not None else "No numeric value"
                    st.caption(f"Parsed Value: `{val_str}` | Concept: `{ev.concept_matched or 'Clinical'}`")
                with hdr_col3:
                    st.markdown(f"**Status:** `{status_badge}`")

                # Verification explanation & source links
                exp_col1, exp_col2 = st.columns([2.5, 1.5])
                with exp_col1:
                    st.markdown(f"**Verification Reason:** {ev.reason}")
                    if ev.conflict_details:
                        st.error(f"⚠️ Conflict Details: {ev.conflict_details}")
                with exp_col2:
                    bx = ev.bbox
                    st.markdown(
                        f"**Source Provenance:**\n"
                        f"- Document: `{ev.doc_id}` (Page {ev.page_number})\n"
                        f"- Bounding Box: `x={bx[0]}, y={bx[1]}, w={bx[2]}, h={bx[3]}`"
                    )

                st.divider()

    # -----------------------------------------------------------------------
    # Categorized Evidence Tables & Detailed Audit
    # -----------------------------------------------------------------------
    st.markdown("### 📑 Detailed Evidence Categories")

    tab_v, tab_u, tab_c, tab_r = st.tabs(
        [
            f"🟢 Verified ({len(report.verified)})",
            f"🟡 Uncertain ({len(report.uncertain)})",
            f"🔴 Conflicting ({len(report.conflicting)})",
            f"⚪ Rejected ({len(report.rejected)})",
        ]
    )

    with tab_v:
        if report.verified:
            st.dataframe([e.to_dict() for e in report.verified], use_container_width=True)
        else:
            st.info("No evidence met all verification criteria.")

    with tab_u:
        if report.uncertain:
            st.dataframe([e.to_dict() for e in report.uncertain], use_container_width=True)
        else:
            st.info("No observations flagged as uncertain.")

    with tab_c:
        if report.conflicting:
            st.error("The following observations contradict other readings recorded on the exact same date:")
            st.dataframe([e.to_dict() for e in report.conflicting], use_container_width=True)
        else:
            st.success("No conflicting measurements detected on any encounter.")

    with tab_r:
        if report.rejected:
            st.dataframe([e.to_dict() for e in report.rejected], use_container_width=True)
        else:
            st.info("No candidate observations were rejected.")

    # -----------------------------------------------------------------------
    # CSV Download of Verification Report
    # -----------------------------------------------------------------------
    st.markdown("---")
    all_rows = (
        [e.to_dict() for e in report.verified] +
        [e.to_dict() for e in report.uncertain] +
        [e.to_dict() for e in report.conflicting] +
        [e.to_dict() for e in report.rejected]
    )

    if all_rows:
        import csv
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

        st.download_button(
            "⬇️ Download Verification Audit Report (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name="temporal_verification_audit.csv",
            mime="text/csv",
            use_container_width=True,
        )
