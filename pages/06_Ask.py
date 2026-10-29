"""
pages/06_Ask.py
---------------
Page 6: Evidence-Grounded Question Answering

question -> retrieval -> temporal verification -> evidence-only answer
(template by default; optional LLM behind a clean interface with template fallback).

All logic lives in utils/qa_pipeline.py — this page only collects input and renders results.
RESEARCH PROTOTYPE — NOT A CLINICAL DIAGNOSTIC TOOL.
"""

from __future__ import annotations

import io
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.qa_pipeline import (
    RESEARCH_DISCLAIMER,
    CallableLLMAnswerGenerator,
    EvidenceQAPipeline,
    TemplateAnswerGenerator,
    evidence_row,
    get_source_page_image,
    highlight_evidence_regions,
)
from utils.retrieval import LexicalRetriever, crop_observation_region
from utils.synthetic_docs import create_synthetic_verification_documents
from utils.temporal_verification import TemporalVerificationEngine, VerificationStatus

st.set_page_config(page_title="Ask | Temporal-RegionRAG", page_icon="💬", layout="wide")

STATUS_ICON = {
    VerificationStatus.VERIFIED: "🟢",
    VerificationStatus.UNCERTAIN: "🟡",
    VerificationStatus.CONFLICTING: "🔴",
    VerificationStatus.REJECTED: "⚪",
}


# ---------------------------------------------------------------------------
# Optional LLM hook. No model is bundled; return a callable(prompt)->str here
# (e.g. a local or hosted model client) to enable it. None => template only.
# ---------------------------------------------------------------------------
def get_optional_llm_fn():
    return None


@st.cache_data(show_spinner=False)
def _benchmark():
    return create_synthetic_verification_documents()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 Q&A Settings")
    candidate_k = st.slider("Retrieved candidates", 5, 50, 25,
                            help="Candidates passed from retrieval to verification.")
    min_score = st.slider("Minimum retrieval score", 0.0, 1.0, 0.05, 0.05)
    ocr_thr = st.slider("OCR confidence threshold", 0.3, 0.9, 0.6, 0.05)
    st.markdown("---")
    gen_choice = st.radio(
        "Answer generator",
        ["Evidence template (default)", "Optional language model"],
        help="The template is deterministic and always available. A language model, if "
             "configured, is used only when its output passes a grounding check.",
    )
    llm_fn = get_optional_llm_fn()
    if gen_choice.startswith("Optional") and llm_fn is None:
        st.info("No language model is configured. The evidence template will be used.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💬 Ask the Documents")
st.error(f"⚠️ **{RESEARCH_DISCLAIMER}**")
st.caption(
    "Each question is answered by retrieving candidate observations, verifying concept, date, "
    "unit and source, and then restating only the verified evidence with its source region."
)

# ---------------------------------------------------------------------------
# Document source
# ---------------------------------------------------------------------------
registry: dict = st.session_state.setdefault("document_registry", {})

src = st.radio(
    "Document source",
    ["Synthetic benchmark (includes conflicts, duplicates, ambiguous dates, faded OCR)",
     "Documents processed in this session"],
    horizontal=False,
)

with st.expander("📤 Upload and process documents here", expanded=not registry and src.startswith("Documents")):
    files = st.file_uploader("PDF or image files", type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
                             accept_multiple_files=True)
    dpi = st.slider("Render DPI", 72, 200, 150, 25)
    unprocessed_pdfs = {k: v for k, v in st.session_state.get("uploaded_pdf_bytes", {}).items() if k not in registry}
    unprocessed_imgs = {k: v for k, v in st.session_state.get("uploaded_images", {}).items() if k not in registry}
    if (unprocessed_pdfs or unprocessed_imgs) and st.button("▶ Process files from Upload page"):
        from utils.extraction import DocumentProcessor
        from utils.pdf_utils import pdf_to_images
        for name, raw in unprocessed_pdfs.items():
            with st.spinner(f"Processing {name}…"):
                proc = DocumentProcessor(doc_id=name, render_dpi=dpi)
                registry[name] = {"observations": proc.process_pdf(raw), "images": pdf_to_images(raw, dpi=dpi)}
        for name, img in unprocessed_imgs.items():
            with st.spinner(f"Processing {name}…"):
                rgb = img.convert("RGB")
                proc = DocumentProcessor(doc_id=name, render_dpi=dpi)
                registry[name] = {"observations": proc.process_image(rgb), "images": [rgb]}
        st.success(f"Processed {len(unprocessed_pdfs) + len(unprocessed_imgs)} file(s) from Upload page.")
    if files and st.button("▶ Process uploaded files"):
        from utils.extraction import DocumentProcessor
        from utils.pdf_utils import pdf_to_images
        for f in files:
            raw = f.read()
            with st.spinner(f"Processing {f.name} (first OCR run loads EasyOCR)…"):
                try:
                    proc = DocumentProcessor(doc_id=f.name, render_dpi=dpi)
                    if f.name.lower().endswith(".pdf"):
                        obs, imgs = proc.process_pdf(raw), pdf_to_images(raw, dpi=dpi)
                    else:
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        obs, imgs = proc.process_image(img), [img]
                    registry[f.name] = {"observations": obs, "images": imgs}
                    errs = sum(1 for o in obs if o.error_message and not o.ocr_uncertain)
                    st.success(f"{f.name}: {len(obs)} observations, {len(imgs)} page(s)"
                               + (f", {errs} processing warning(s)" if errs else ""))
                except Exception as exc:
                    st.error(f"{f.name}: processing failed — {exc}")

if src.startswith("Synthetic"):
    observations, page_images = _benchmark()
else:
    observations = [o for d in registry.values() for o in d["observations"]]
    page_images = {doc: d["images"] for doc, d in registry.items()}
    if not observations:
        st.warning("No processed documents in this session. Upload files above, use the "
                   "**Extract** page, or switch to the synthetic benchmark.")
        st.stop()
    st.caption(f"{len(registry)} document(s), {len(observations)} observations available.")

# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------
examples = {
    "Glucose over time": "Show fasting blood glucose trend over time",
    "BP 1998 vs 2002": "Compare blood pressure in 1998 and 2002",
    "BP on 15 Jan 1998 (conflict)": "Check blood pressure on 15 Jan 1998",
    "Heart rate 1998 (missing date)": "What was the heart rate in 1998?",
    "Brain MRI (no evidence)": "What were the brain MRI results?",
}
cols = st.columns(len(examples))
for col, (label, q) in zip(cols, examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state["qa_question"] = q
        st.session_state["qa_input"] = q

question = st.text_input("Your question", key="qa_input",
                         placeholder="e.g. Compare blood pressure in 1998 and 2002")
if st.button("🔎 Ask", type="primary"):
    st.session_state["qa_question"] = question

question = st.session_state.get("qa_question", "")
if not question:
    st.stop()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
pipeline = EvidenceQAPipeline(
    retriever=LexicalRetriever(),
    verifier=TemporalVerificationEngine(ocr_confidence_threshold=ocr_thr),
    candidate_k=candidate_k,
    min_score=min_score,
)
generator = (CallableLLMAnswerGenerator(llm_fn) if gen_choice.startswith("Optional")
             else TemplateAnswerGenerator())
res = pipeline.answer(question, observations, generator=generator)

st.markdown("---")
st.markdown(f"### Question: _{res.question}_")

# Status banners
if res.insufficient_evidence and not res.evidence:
    st.warning("🚫 **Insufficient evidence** — no observation could be verified for this question.")
elif res.insufficient_evidence:
    st.warning("🚫 **No fully verified evidence** — only conflicting/uncertain evidence was found.")
if res.has_conflicts:
    st.error(f"🔴 **Conflicting evidence** — {len(res.package.conflicting)} observation(s) disagree "
             "on the same date. No value is selected as correct.")
if res.has_uncertain_ocr:
    st.warning("🟡 **Uncertain OCR** — some evidence has low recognition confidence; check the source region.")
elif res.has_uncertain:
    st.warning(f"🟡 **Uncertain evidence** — {len(res.package.uncertain)} item(s) (ambiguous date, "
               "duplicate or non-standard unit).")
if res.missing_timestamp_count:
    st.info(f"⏱️ {res.missing_timestamp_count} matching observation(s) excluded for missing timestamps.")

# Answer
st.markdown("#### Evidence-based answer")
with st.container(border=True):
    st.markdown(res.answer.text)
gen_note = f"Generated by: `{res.answer.generator_name}`"
if res.answer.used_fallback:
    gen_note += f" (fallback: {res.answer.fallback_reason})"
st.caption(gen_note + " · " + RESEARCH_DISCLAIMER)

# Chronological comparison
if res.comparison_rows:
    st.markdown("#### 📅 Chronological comparison")
    st.caption("Values exactly as recorded, ordered by normalized date. No trend, causal or "
               "clinical interpretation is made.")
    st.dataframe(res.comparison_rows, use_container_width=True, hide_index=True)

# Evidence table
if res.evidence:
    st.markdown("#### 📋 Supporting evidence")
    rows = [evidence_row(ev, f"[E{i}]") for i, ev in enumerate(res.evidence, start=1)]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Source inspection
    st.markdown("#### 🗺 Inspect source region")
    labels = [f"[E{i}] {STATUS_ICON[ev.status]} {ev.normalized_date.iso_date or 'undated'} · {ev.text[:60]}"
              for i, ev in enumerate(res.evidence, start=1)]
    sel = st.selectbox("Evidence item", range(len(labels)), format_func=lambda i: labels[i])
    ev = res.evidence[sel]
    show_all = st.checkbox("Also highlight other evidence on the same page", value=False)

    page = get_source_page_image(page_images, ev.doc_id, ev.page_number)
    info_col, img_col = st.columns([1, 2])
    with info_col:
        x, y, w, h = ev.bbox
        st.markdown(
            f"**Status:** {STATUS_ICON[ev.status]} {ev.status.value}\n\n"
            f"**Recorded text:** `{ev.text}`\n\n"
            f"**Value / unit:** `{ev.value if ev.value is not None else '—'}` `{ev.unit or '—'}`\n\n"
            f"**Date:** `{ev.normalized_date.iso_date or 'undated'}` (recorded: `{ev.raw_date_str or 'none'}`)\n\n"
            f"**Document:** `{ev.doc_id}` · **Page:** {ev.page_number}\n\n"
            f"**Region:** x={x}, y={y}, w={w}, h={h}\n\n"
            f"**Reason:** {ev.reason}"
        )
        if page is not None and ev.has_valid_coordinates():
            snip = crop_observation_region(page, ev.bbox, padding=12)
            if snip is not None:
                st.image(snip, caption="Source region (unmodified pixels)")
    with img_col:
        if page is None:
            st.info("The original page image is not available for this document.")
        else:
            if show_all:
                same = [(i, e) for i, e in enumerate(res.evidence)
                        if e.doc_id == ev.doc_id and e.page_number == ev.page_number]
                hl = highlight_evidence_regions(page, [e for _, e in same],
                                                labels=[f"E{i + 1}" for i, _ in same])
            else:
                hl = highlight_evidence_regions(page, [ev], labels=[f"E{sel + 1}"])
            t1, t2 = st.tabs(["Highlighted", "Original page"])
            t1.image(hl, use_container_width=True)
            t2.image(page, use_container_width=True)
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            st.download_button("⬇️ Download original page", buf.getvalue(),
                               file_name=f"{ev.doc_id}_p{ev.page_number}.png", mime="image/png")

# Rejected
if res.package.rejected:
    with st.expander(f"⚪ Rejected candidates ({len(res.package.rejected)})"):
        st.dataframe([evidence_row(r, "—") for r in res.package.rejected],
                     use_container_width=True, hide_index=True)
