"""
pages/04_Retrieve.py
--------------------
Page 4: Query-Conditioned Temporal Verification and Evidence Linking

Responsibilities:
1. Accept a natural-language clinical question.
2. Retrieve relevant OCR text and structured observations using an offline
   lexical-ranking baseline (BM25 Okapi + clinical domain field matching).
3. Rank matching observations and retain their source document, page, date,
   and coordinates (x, y, w, h).
4. Return multiple relevant results when a question requires comparing different dates.
5. Display retrieved results and visually highlight their source regions in
   the UI where coordinates are available.
6. Display a clear, prominent message when no relevant evidence is found.
"""

from __future__ import annotations

import io
import os
import sys

import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.retrieval import (
    LexicalRetriever,
    RetrievalResult,
    highlight_retrieved_regions,
    crop_observation_region,
    is_comparison_query,
    extract_temporal_anchors,
)
from utils.synthetic_docs import (
    create_synthetic_retrieval_documents,
    create_synthetic_retrieval_corpus,
)
from utils.visual_retrieval import (
    PAGE_LEVEL_LIMITATION_NOTICE,
    DocumentPage,
    VisualRetrievalEngine,
    annotate_connected_page,
    probe_visual_environment,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retrieve | Temporal-RegionRAG",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔍 Retrieval Parameters")

    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=15,
        value=5,
        help="Maximum number of relevant observations to retrieve.",
    )

    score_threshold = st.slider(
        "Minimum Relevance Score",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.02,
        help="Observations below this score are filtered out as irrelevant.",
    )

    st.markdown("---")
    st.markdown("### 🗺 Region Highlighting")
    line_width = st.slider("Highlight Border Width", min_value=1, max_value=6, value=3)
    show_rank_badges = st.checkbox("Show Rank Badges on Image", value=True)
    show_cropped_snippets = st.checkbox("Show Cropped Visual Snippets", value=True)

    st.markdown("---")
    st.markdown("### 🖼 Optional Visual Page Retrieval")
    show_visual_pages = st.checkbox(
        "Include Page-Level Visual Retrieval (ColPali / OCR Fallback)",
        value=True,
        help="Ranks whole pages via ColPali when installed & enabled, or falls back "
             "automatically to OCR page ranking, then connects pages to table regions, "
             "observations, dates, and coordinates.",
    )
    visual_env = probe_visual_environment()
    st.caption(
        f"• ColPali installed: `{visual_env['colpali_installed']}`\n"
        f"• Device: `{visual_env['device']}` | Fallback: `OCR/lexical`"
    )

    st.markdown("---")
    st.caption(
        "**Retriever Info**\n\n"
        "• Baseline: BM25 Okapi + Clinical Field Matching\n"
        "• Visual (optional): ColPali → OCR page fallback\n"
        "• Multi-Date: Automatic temporal grouping & balanced evidence\n"
        "• Grounding: Spatial pixel bounding-boxes retained"
    )


# ---------------------------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------------------------
st.title("🔍 Evidence Retrieval & Temporal Linking")
st.subheader("Query-Conditioned Temporal Verification and Evidence Linking")
st.markdown(
    "Ask a natural-language clinical question to retrieve matching observations. "
    "The system ranks evidence, balances multi-date comparisons, and highlights "
    "exact spatial bounding boxes on source documents."
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Corpus Selection
# ---------------------------------------------------------------------------
tab_synth, tab_session = st.tabs(
    ["🏥 Synthetic Multi-Date Benchmark", "📂 From Uploaded/Extracted Documents"]
)

active_corpus = []
active_images: dict[str, Image.Image] = {}
corpus_label = ""

with tab_synth:
    st.markdown(
        "**Benchmark Patient History (3 Longitudinal Encounters)**\n"
        "- `doc_1998_admission` — 15 Jan 1998 (Admission note, BP 140/90, HR 88, Glucose 5.4, Penicillin allergy)\n"
        "- `doc_2002_clinic` — 22 Apr 2002 (Clinic follow-up, BP 130/80, HR 72, Glucose 6.8, Lisinopril started)\n"
        "- `doc_2005_discharge` — 10 Nov 2005 (Annual review, BP 124/78, HbA1c 7.1%, Glucose 7.4, Metformin adjusted)"
    )
    synth_docs, synth_imgs = create_synthetic_retrieval_documents()
    use_synth = st.checkbox(
        "Use this synthetic benchmark corpus",
        value=True,
        key="use_synthetic_corpus",
    )
    if use_synth:
        active_corpus = synth_docs
        active_images = synth_imgs
        corpus_label = f"Synthetic Benchmark ({len(synth_docs)} observations across 3 dates)"

with tab_session:
    session_obs = st.session_state.get("extracted_observations", [])
    session_imgs = st.session_state.get("extracted_page_images", [])
    session_doc_id = st.session_state.get("extracted_doc_id", "session_doc")

    if not session_obs:
        st.info(
            "No extracted documents found in current session. "
            "Process a PDF or image in the **Extract** page first, or use the synthetic benchmark tab."
        )
    else:
        st.success(
            f"Found **{len(session_obs)}** observations from **{session_doc_id}** in session."
        )
        use_session = st.checkbox("Query session document", value=not use_synth, key="use_session_doc")
        if use_session:
            active_corpus = session_obs
            # Map page images by doc_id or page
            active_images = {}
            for p_idx, p_img in enumerate(session_imgs, start=1):
                active_images[f"{session_doc_id}_p{p_idx}"] = p_img
                if len(session_imgs) == 1:
                    active_images[session_doc_id] = p_img
            corpus_label = f"Session Document: {session_doc_id} ({len(session_obs)} observations)"


st.markdown("---")
st.markdown(f"**Active Corpus:** `{corpus_label}`")

# ---------------------------------------------------------------------------
# Query Input & Example Suggestions
# ---------------------------------------------------------------------------
st.markdown("### 💬 Ask a Clinical Question")

# Sample questions
col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
sample_queries = {
    col_q1: ("Compare BP (1998 vs 2002)", "Compare blood pressure in 1998 and 2002"),
    col_q2: ("Glucose Trend", "Show fasting blood glucose trend over time"),
    col_q3: ("1998 Heart Rate", "What was the heart rate on 15 Jan 1998?"),
    col_q4: ("Check Allergy", "What allergies are documented for penicillin?"),
    col_q5: ("Absent Concept (No Evidence)", "What were the results of the brain MRI scan?"),
}

for col, (label, query_text) in sample_queries.items():
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["query_input_text"] = query_text

default_query = st.session_state.get(
    "query_input_text", "Compare blood pressure in 1998 and 2002"
)

user_query = st.text_input(
    "Natural language question:",
    value=default_query,
    placeholder="e.g. Compare blood pressure in 1998 and 2002, or What was the HbA1c level?",
    key="current_user_query",
)

search_clicked = st.button("🔎 Retrieve Relevant Evidence", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Retrieval Execution & Evidence Rendering
# ---------------------------------------------------------------------------
if user_query:
    retriever = LexicalRetriever(min_score_threshold=score_threshold)
    results: list[RetrievalResult] = retriever.retrieve(
        query=user_query,
        observations=active_corpus,
        top_k=top_k,
        min_score=score_threshold,
    )

    is_comparison = is_comparison_query(user_query)
    temporal_anchors = extract_temporal_anchors(user_query)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Requirement 6: Clear message when no relevant evidence is found
    # -----------------------------------------------------------------------
    if not results:
        st.warning(
            f"⚠️ **No relevant evidence found** for question: *\"{user_query}\"*\n\n"
            f"**Reason:** No observations in `{corpus_label}` matched your query with a relevance "
            f"score above the threshold ({score_threshold:.2f}).\n\n"
            "**Suggestions:**\n"
            "- Try querying clinical terms present in the records (e.g. *blood pressure*, *glucose*, *heart rate*, *cholesterol*, *allergy*).\n"
            "- Lower the **Minimum Relevance Score** in the sidebar if you expect weak lexical matches.\n"
            "- Check that the target document or date is included in your active corpus."
        )
    else:
        # -------------------------------------------------------------------
        # Summary Metrics
        # -------------------------------------------------------------------
        distinct_dates = sorted(list({r.date_str for r in results if r.date_str}))
        distinct_docs = sorted(list({r.doc_id for r in results if r.doc_id}))

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Retrieved Evidence", f"{len(results)} items")
        m_col2.metric("Target Dates", f"{len(distinct_dates)} timepoint(s)")
        m_col3.metric("Source Documents", f"{len(distinct_docs)} file(s)")
        m_col4.metric(
            "Query Mode",
            "📅 Temporal Comparison" if is_comparison else "🎯 Focused Retrieval",
        )

        # -------------------------------------------------------------------
        # Requirement 4: Temporal Comparison Table
        # -------------------------------------------------------------------
        if is_comparison and len(distinct_dates) >= 2:
            st.markdown("### 📅 Temporal Comparison Overview")
            st.info(
                f"Question requires comparing evidence across different dates. "
                f"Retrieved **{len(results)}** evidence points across **{len(distinct_dates)}** dates: "
                f"**{', '.join(distinct_dates)}**."
            )

            comp_rows = []
            for r in results:
                comp_rows.append({
                    "Rank": f"#{r.rank}",
                    "Date": r.date_str or "Undated",
                    "Document ID": r.doc_id,
                    "Observation Text": r.text,
                    "Value": r.value if r.value is not None else "—",
                    "Unit": r.unit or "—",
                    "Relevance Score": f"{r.score:.3f}",
                    "Bounding Box (x, y, w, h)": f"{r.bbox[0]}, {r.bbox[1]}, {r.bbox[2]}, {r.bbox[3]}",
                })
            st.dataframe(comp_rows, use_container_width=True)

        # -------------------------------------------------------------------
        # Requirement 5: Visual Highlighting of Source Regions
        # -------------------------------------------------------------------
        st.markdown("### 🗺 Grounded Evidence & Highlighted Source Regions")
        st.caption(
            "Every retrieved observation retains its exact spatial coordinates. "
            "Source regions are highlighted with color-coded bounding boxes and rank badges."
        )

        # Group results by doc_id to render per-page highlights
        results_by_doc: dict[str, list[RetrievalResult]] = {}
        for r in results:
            results_by_doc.setdefault(r.doc_id, []).append(r)

        for doc_id, doc_results in results_by_doc.items():
            doc_date = doc_results[0].date_str or "Undated"
            with st.container():
                st.markdown(f"#### 📄 Document: `{doc_id}` | Date: **{doc_date}**")

                col_img, col_details = st.columns([1.1, 1.0])

                # Get page image if available
                page_img = active_images.get(doc_id)
                if page_img is None:
                    # Try looking up by doc prefix
                    for k, img in active_images.items():
                        if doc_id in k or k in doc_id:
                            page_img = img
                            break

                with col_img:
                    st.markdown("**Source Document with Highlighted Evidence**")
                    if page_img is not None:
                        annotated_img = highlight_retrieved_regions(
                            page_image=page_img,
                            results=doc_results,
                            line_width=line_width,
                            draw_rank_badges=show_rank_badges,
                        )
                        st.image(annotated_img, use_container_width=True)

                        # Download button for highlighted image
                        img_buf = io.BytesIO()
                        annotated_img.save(img_buf, format="PNG")
                        img_buf.seek(0)
                        st.download_button(
                            f"⬇️ Download Highlighted Image ({doc_id})",
                            data=img_buf,
                            file_name=f"highlighted_{doc_id}.png",
                            mime="image/png",
                            key=f"dl_{doc_id}",
                        )
                    else:
                        st.info("Source page image not available for visual overlay.")

                with col_details:
                    st.markdown("**Retrieved Observation Details**")
                    for res in doc_results:
                        badge_color = "red" if res.rank == 1 else "blue"
                        with st.expander(
                            f"#{res.rank} — {res.text[:50]} (Score: {res.score:.2f})",
                            expanded=(res.rank <= 2),
                        ):
                            st.markdown(f"**Full Text:** `{res.text}`")
                            st.markdown(
                                f"- **Value:** `{res.value}` {res.unit or ''}\n"
                                f"- **Date:** `{res.date_str or 'None'}`\n"
                                f"- **Document:** `{res.doc_id}` (Page {res.page_number})\n"
                                f"- **Coordinates (BBox):** `x={res.bbox[0]}, y={res.bbox[1]}, w={res.bbox[2]}, h={res.bbox[3]}`\n"
                                f"- **Relevance Score:** `{res.score:.4f}`\n"
                                f"- **Matched Query Terms:** `{', '.join(res.matched_terms)}`"
                            )

                            if show_cropped_snippets and page_img is not None and res.has_valid_coordinates():
                                snippet = crop_observation_region(page_img, res.bbox, padding=10)
                                if snippet:
                                    st.markdown("**Cropped Region Snippet:**")
                                    st.image(snippet, caption=f"Region #{res.rank}", use_container_width=False)

                st.markdown("---")

        # -------------------------------------------------------------------
        # Full Evidence Table & CSV Download
        # -------------------------------------------------------------------
        st.markdown("### 📋 Structured Evidence Table")
        table_rows = [r.to_dict() for r in results]
        st.dataframe(table_rows, use_container_width=True)

        # CSV Export
        import csv
        csv_buf = io.StringIO()
        if table_rows:
            writer = csv.DictWriter(csv_buf, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            for row in table_rows:
                writer.writerow(row)

            st.download_button(
                "⬇️ Download Retrieved Evidence (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="retrieved_evidence.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # -------------------------------------------------------------------
        # Optional Page-Level Visual Retrieval & Table/Observation Connection
        # -------------------------------------------------------------------
        if show_visual_pages and active_images:
            st.markdown("---")
            st.markdown("### 🖼 Page-Level Visual Retrieval & Connected Regions")
            st.info(f"ℹ️ **{PAGE_LEVEL_LIMITATION_NOTICE}**")

            doc_pages = [
                DocumentPage(doc_id=doc_key, page_number=1, image=img)
                for doc_key, img in active_images.items()
            ]
            v_engine = VisualRetrievalEngine()
            v_outcome = v_engine.retrieve_and_connect(
                query=user_query,
                pages=doc_pages,
                observations=active_corpus,
                top_k_pages=min(top_k, len(doc_pages)),
                prefer_visual=True,
            )
            st.caption(
                f"**Active Page Retriever Backend:** `{v_outcome.backend_used}`"
                + (f" — _{v_outcome.fallback_reason}_" if v_outcome.used_fallback else "")
            )
            if v_outcome.page_hits:
                st.dataframe(
                    [h.to_dict() for h in v_outcome.page_hits],
                    use_container_width=True,
                )
                for cp in v_outcome.connected_pages:
                    with st.expander(
                        f"📄 Page `{cp.page_id}` (Score: {cp.page_score:.3f}) — "
                        f"{len(cp.table_regions)} table region(s), "
                        f"{len(cp.observations)} observation(s), "
                        f"dates: {', '.join(cp.dates_on_page) or 'none'}"
                    ):
                        p_img = active_images.get(cp.doc_id)
                        if p_img is not None:
                            st.image(
                                annotate_connected_page(p_img, cp),
                                caption=(
                                    f"{cp.page_id}: orange = detected table region(s), "
                                    "green = verified observation(s) for query, "
                                    "blue = other observations on page"
                                ),
                                use_container_width=True,
                            )

        st.markdown("---")
        if st.button("⏱️ Proceed to Temporal Verification & Grounding →", type="primary"):
            st.switch_page("pages/05_Verify.py")

