"""
pages/03_Extract.py
-------------------
Page 3: Document Processing — Text Extraction, OCR & Region Preview

Features
--------
- Source: Upload a file here OR use documents from the Upload page session.
- Run the DocumentProcessor on PDF or image inputs.
- Show detected regions overlaid on the original page image.
- Display all Observations in a filterable table.
- Flag uncertain OCR results with a warning icon.
- Allow CSV download of the extracted observations.
"""

from __future__ import annotations

import io
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.extraction import DocumentProcessor, annotate_image, Observation
from utils.pdf_utils import pdf_to_images

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Extract | Temporal-RegionRAG",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 Extraction Settings")

    render_dpi = st.slider(
        "PDF Render DPI",
        min_value=72, max_value=200, value=150, step=25,
        help="Higher DPI = more accurate OCR, slower processing.",
    )

    st.markdown("---")
    st.markdown(
        """
        **Legend**
        - 🔵 PDF text block
        - 🟢 OCR result
        - 🟠 Table cell
        - 🔴 Error
        - ⚠️  Uncertain OCR
        """
    )
    st.markdown("---")
    show_uncertain_only = st.checkbox(
        "Show uncertain OCR only",
        value=False,
        help="Filter the table to only show observations with low OCR confidence.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_extraction(
    source_bytes: bytes | None,
    source_img: Image.Image | None,
    doc_id: str,
    dpi: int,
) -> tuple[list[Observation], list[Image.Image]]:
    """
    Run DocumentProcessor on either PDF bytes or a PIL image.
    Returns (observations, rendered_page_images).
    """
    proc = DocumentProcessor(doc_id=doc_id, render_dpi=dpi)

    if source_bytes is not None:
        obs  = proc.process_pdf(source_bytes)
        imgs = pdf_to_images(source_bytes, dpi=dpi)
    elif source_img is not None:
        obs  = proc.process_image(source_img)
        imgs = [source_img]
    else:
        return [], []

    return obs, imgs


def _obs_to_display_rows(obs_list: list[Observation]) -> list[dict]:
    rows = []
    for o in obs_list:
        rows.append({
            "ID":          o.obs_id,
            "Page":        o.page_number if o.page_number > 0 else "img",
            "Type":        o.source_type,
            "Text":        o.text[:80] if o.text else "",
            "Value":       o.value,
            "Unit":        o.unit or "",
            "Date":        o.date_str or "",
            "Conf":        f"{o.ocr_confidence:.2f}" if o.ocr_confidence is not None else "—",
            "⚠ Uncertain": "⚠️" if o.ocr_uncertain else "",
            "Error":       o.error_message or "",
            "BBox":        f"{o.bbox[0]},{o.bbox[1]},{o.bbox[2]},{o.bbox[3]}",
        })
    return rows


def _make_csv(obs_list: list[Observation]) -> str:
    import csv, io as _io
    buf    = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=obs_list[0].to_dict().keys())
    writer.writeheader()
    for o in obs_list:
        writer.writerow(o.to_dict())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧬 Document Processing — Extract & Preview")
st.markdown(
    "Extract text, run OCR, and detect table regions. "
    "Preview annotated results and download as CSV."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
src_tab_session, src_tab_upload = st.tabs(
    ["📂 From Upload Page (session)", "📤 Upload File Here"]
)

source_bytes: bytes | None      = None
source_img:   Image.Image | None = None
doc_id_str:   str               = "unknown"

with src_tab_session:
    available_pdfs = st.session_state.get("uploaded_pdf_pages", {})
    available_imgs = st.session_state.get("uploaded_images", {})
    session_items  = list(available_pdfs.keys()) + list(available_imgs.keys())

    if not session_items:
        st.info(
            "No documents found in session. "
            "Go to the **Upload** page first, or use the tab on the right."
        )
    else:
        selected = st.selectbox("Select document from session", session_items)
        if selected:
            doc_id_str = selected
            if selected in available_pdfs:
                pdf_bytes_map = st.session_state.get("uploaded_pdf_bytes", {})
                if selected in pdf_bytes_map:
                    source_bytes = pdf_bytes_map[selected]
                    st.caption("ℹ️ Ready to extract from uploaded PDF.")
                else:
                    st.caption(
                        "ℹ️ PDF pages are already rendered. "
                        "Re-upload the raw PDF here for full text+OCR extraction."
                    )
            elif selected in available_imgs:
                source_img = available_imgs[selected]

with src_tab_upload:
    up = st.file_uploader(
        "Upload PDF or image for extraction",
        type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )
    if up:
        raw = up.read()
        doc_id_str = up.name
        if up.name.lower().endswith(".pdf"):
            source_bytes = raw
        else:
            source_img = Image.open(io.BytesIO(raw)).convert("RGB")

# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------
run_col, _ = st.columns([1, 5])
with run_col:
    run_clicked = st.button("▶ Run Extraction", type="primary", use_container_width=True)

if not run_clicked:
    st.info(
        "📌 Select a document above and click **Run Extraction** to begin. "
        "Or try the **Sample Documents** below."
    )

    # --- Sample documents shortcut ---
    st.markdown("---")
    st.markdown("#### 🧪 Try with a Synthetic Sample")
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button("📄 Load Synthetic Text PDF", use_container_width=True):
            try:
                from utils.synthetic_docs import make_text_pdf
                st.session_state["_sample_bytes"] = make_text_pdf()
                st.session_state["_sample_type"]  = "pdf"
                st.session_state["_sample_id"]    = "synthetic_text.pdf"
                st.success("Synthetic PDF loaded — click **Run Extraction** now.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not create sample: {e}")
    with sample_col2:
        if st.button("🖼 Load Synthetic Image", use_container_width=True):
            try:
                from utils.synthetic_docs import make_clinical_image
                st.session_state["_sample_img"] = make_clinical_image()
                st.session_state["_sample_type"] = "image"
                st.session_state["_sample_id"]   = "synthetic_image.png"
                st.success("Synthetic image loaded — click **Run Extraction** now.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not create sample: {e}")

# Pull sample from session if set
if "_sample_bytes" in st.session_state and st.session_state.get("_sample_type") == "pdf":
    source_bytes = st.session_state["_sample_bytes"]
    doc_id_str   = st.session_state.get("_sample_id", "sample")
if "_sample_img" in st.session_state and st.session_state.get("_sample_type") == "image":
    source_img   = st.session_state["_sample_img"]
    doc_id_str   = st.session_state.get("_sample_id", "sample")

if not run_clicked:
    st.stop()

if source_bytes is None and source_img is None:
    st.error("No document selected. Use the tabs above or load a sample first.")
    st.stop()

# ---------------------------------------------------------------------------
# Run and display results
# ---------------------------------------------------------------------------
with st.spinner("Processing document… (first run loads EasyOCR model, ~30 s)"):
    try:
        observations, page_images = _run_extraction(
            source_bytes, source_img, doc_id_str, render_dpi
        )
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.stop()

if not observations:
    st.warning("No observations extracted. The document may be empty.")
    st.stop()

# Summary metrics
err_count = sum(1 for o in observations if o.error_message)
unc_count = sum(1 for o in observations if o.ocr_uncertain)
st.success(
    f"✅ Extracted **{len(observations)}** observation(s) from **{len(page_images)}** page(s).  "
    f"⚠️ {unc_count} uncertain OCR  |  ❌ {err_count} errors"
)

# Save into session state for Retrieval module
st.session_state["extracted_observations"] = observations
st.session_state["extracted_doc_id"] = doc_id_str
st.session_state["extracted_page_images"] = page_images

if "document_registry" not in st.session_state:
    st.session_state["document_registry"] = {}
st.session_state["document_registry"][doc_id_str] = {
    "observations": observations,
    "images": page_images,
}

btn_col1, _ = st.columns([1, 2])
with btn_col1:
    if st.button("🔍 Query Evidence in Retrieval Module →", type="primary"):
        st.switch_page("pages/04_Retrieve.py")


# ---------------------------------------------------------------------------
# Annotated page viewer
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🗺 Annotated Page Preview")

if page_images:
    # Page selector
    if len(page_images) > 1:
        page_sel = st.slider("Page", 1, len(page_images), 1)
    else:
        page_sel = 1

    page_img  = page_images[page_sel - 1]
    page_obs  = [
        o for o in observations
        if o.page_number == page_sel or (o.page_number == 0 and len(page_images) == 1)
    ]

    col_orig, col_anno = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.image(page_img, use_container_width=True)

    with col_anno:
        st.markdown("**Annotated Regions**")
        if page_obs:
            annotated = annotate_image(page_img, page_obs, show_text=True)
            st.image(annotated, use_container_width=True)
            # Download annotated image
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "⬇️ Download Annotated Image",
                data=buf,
                file_name=f"annotated_p{page_sel}.png",
                mime="image/png",
            )
        else:
            st.info("No observations detected on this page.")

# ---------------------------------------------------------------------------
# Observation table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Extracted Observations")

display_obs = observations
if show_uncertain_only:
    display_obs = [o for o in observations if o.ocr_uncertain]
    st.caption(f"Showing {len(display_obs)} uncertain OCR observations.")

rows = _obs_to_display_rows(display_obs)
if rows:
    st.dataframe(rows, use_container_width=True, height=400)

    # CSV download
    csv_str = _make_csv(display_obs)
    st.download_button(
        "⬇️ Download All Observations (CSV)",
        data=csv_str.encode(),
        file_name=f"{doc_id_str}_observations.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("No observations to display.")

# ---------------------------------------------------------------------------
# Error log
# ---------------------------------------------------------------------------
errors = [o for o in observations if o.error_message]
if errors:
    with st.expander(f"❌ Processing Errors ({len(errors)})"):
        for e in errors:
            st.warning(
                f"Page {e.page_number} | {e.source_type} | {e.error_message}"
            )
