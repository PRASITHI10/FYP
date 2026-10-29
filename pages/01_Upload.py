"""
pages/01_Upload.py
------------------
Page 1: Document Upload & Preview

Users can upload:
  - PDF files  → each page rendered via PyMuPDF
  - Image files (JPG, PNG, BMP, TIFF) → displayed as-is

Uploaded files are also stored in the session_state so the
Preprocess page can reuse them without re-uploading.
"""

import io
import os
import sys

import streamlit as st
from PIL import Image

# Make utils importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.pdf_utils import pdf_to_images, get_page_count
from utils.image_utils import get_image_info

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Upload | Temporal-RegionRAG",
    page_icon="📤",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
SUPPORTED_PDF_TYPES   = ["pdf"]
ALL_SUPPORTED_TYPES   = SUPPORTED_PDF_TYPES + SUPPORTED_IMAGE_TYPES


def _render_pdf_preview(file_bytes: bytes, filename: str, dpi: int) -> None:
    """Render all pages of a PDF as a scrollable grid."""
    page_count = get_page_count(file_bytes)
    st.info(f"📄 **{filename}** — {page_count} page(s) detected")

    with st.spinner(f"Rendering {page_count} page(s) at {dpi} DPI …"):
        pages = pdf_to_images(file_bytes, dpi=dpi)

    # Show pages in a 3-column grid
    cols_per_row = 3
    for row_start in range(0, len(pages), cols_per_row):
        row_pages = pages[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_pages))
        for col, (page_img, page_idx) in zip(cols, enumerate(row_pages, start=row_start + 1)):
            with col:
                st.image(page_img, caption=f"Page {page_idx}", use_container_width=True)

    # Store for Preprocess / Extract / Ask pages
    if "uploaded_pdf_pages" not in st.session_state:
        st.session_state["uploaded_pdf_pages"] = {}
    st.session_state["uploaded_pdf_pages"][filename] = pages
    st.session_state.setdefault("uploaded_pdf_bytes", {})[filename] = file_bytes


def _render_image_preview(file_bytes: bytes, filename: str) -> None:
    """Display an uploaded image with metadata."""
    img = Image.open(io.BytesIO(file_bytes))
    w, h, mode = get_image_info(img)
    st.image(img, caption=filename, use_container_width=True)
    st.caption(f"Size: {w} × {h} px | Mode: {mode}")

    # Store for Preprocess page
    if "uploaded_images" not in st.session_state:
        st.session_state["uploaded_images"] = {}
    st.session_state["uploaded_images"][filename] = img


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📤 Document Upload & Preview")
st.markdown(
    "Upload one or more **PDF** or **image** files. "
    "PDF pages will be rendered automatically using PyMuPDF."
)
st.markdown("---")

# DPI slider (in sidebar)
with st.sidebar:
    st.markdown("### ⚙️ PDF Render Settings")
    dpi = st.slider(
        "Render DPI",
        min_value=72,
        max_value=300,
        value=150,
        step=25,
        help="Higher DPI = sharper pages but slower rendering.",
    )

# File uploader
uploaded_files = st.file_uploader(
    "Choose PDF or image files",
    type=ALL_SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="Supported formats: PDF, JPG, PNG, BMP, TIFF",
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown(
        """
        <div style='text-align:center; padding:60px; border:2px dashed #555;
                    border-radius:12px; color:#888; margin-top:24px;'>
            <h3>📂 No files uploaded yet</h3>
            <p>Drag & drop files above, or click <strong>Browse files</strong>.</p>
            <p style='font-size:0.85em;'>Supported: PDF, JPG, PNG, BMP, TIFF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Process each uploaded file
# ---------------------------------------------------------------------------
st.success(f"✅ {len(uploaded_files)} file(s) uploaded. Previewing below …")

for uploaded_file in uploaded_files:
    file_bytes = uploaded_file.read()
    filename   = uploaded_file.name
    extension  = filename.rsplit(".", 1)[-1].lower()

    with st.expander(f"📄 {filename}", expanded=True):
        if extension in SUPPORTED_PDF_TYPES:
            _render_pdf_preview(file_bytes, filename, dpi=dpi)
        elif extension in SUPPORTED_IMAGE_TYPES:
            _render_image_preview(file_bytes, filename)
        else:
            st.warning(f"Unsupported file type: .{extension}")

st.markdown("---")
st.info(
    "💡 **Tip:** Go to the **Preprocess** page to apply OpenCV filters "
    "to your uploaded documents."
)
