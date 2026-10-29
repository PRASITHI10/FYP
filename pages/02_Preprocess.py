"""
pages/02_Preprocess.py
----------------------
Page 2: OpenCV Preprocessing Viewer

Shows a side-by-side comparison of:
  - Original image (left)
  - Preprocessed image (right)

Works with:
  1. Images uploaded on the Upload page (from session_state)
  2. A fresh image uploaded directly on this page
"""

import io
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.image_utils import (
    apply_preprocessing,
    PREPROCESSING_METHODS,
    get_image_info,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Preprocess | Temporal-RegionRAG",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Preprocessing Controls")

    method = st.selectbox(
        "Preprocessing method",
        options=list(PREPROCESSING_METHODS.keys()),
        index=2,          # Default: Adaptive Threshold
        help="Select the OpenCV operation to apply.",
    )

    st.markdown("---")
    st.markdown(
        """
        **Method descriptions:**
        - **Greyscale** — Remove colour
        - **Gaussian Blur** — Reduce noise
        - **Adaptive Threshold** — Binarise text (best for docs)
        - **Canny Edge Detection** — Highlight boundaries
        - **Deskew** — Correct rotation skew
        """
    )

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔬 OpenCV Preprocessing Viewer")
st.markdown(
    "Apply computer-vision preprocessing to documents. "
    "Inspect the **original vs. processed** result side by side."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
source_tab, upload_tab = st.tabs(
    ["📂 From Upload Page (session)", "📤 Upload New Image Here"]
)

target_image: Image.Image | None = None
target_label: str = ""

with source_tab:
    # Gather all available images from session_state
    available: dict[str, Image.Image] = {}

    if "uploaded_images" in st.session_state:
        available.update(st.session_state["uploaded_images"])

    if "uploaded_pdf_pages" in st.session_state:
        for pdf_name, pages in st.session_state["uploaded_pdf_pages"].items():
            for i, page_img in enumerate(pages, start=1):
                available[f"{pdf_name} — Page {i}"] = page_img

    if not available:
        st.info(
            "No documents found in this session. "
            "Go to the **Upload** page first, or use the tab on the right "
            "to upload an image directly here."
        )
    else:
        selected = st.selectbox(
            "Choose a document / page to preprocess",
            options=list(available.keys()),
        )
        if selected:
            target_image = available[selected]
            target_label = selected

with upload_tab:
    direct_upload = st.file_uploader(
        "Upload an image to preprocess",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )
    if direct_upload:
        target_image = Image.open(io.BytesIO(direct_upload.read()))
        target_label = direct_upload.name

# ---------------------------------------------------------------------------
# Run preprocessing and show side-by-side
# ---------------------------------------------------------------------------
if target_image is None:
    st.markdown(
        """
        <div style='text-align:center; padding:60px; border:2px dashed #555;
                    border-radius:12px; color:#888; margin-top:24px;'>
            <h3>🖼 No image selected</h3>
            <p>Upload documents on the <strong>Upload</strong> page,
               or use the <em>Upload New Image Here</em> tab above.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Apply preprocessing
with st.spinner(f"Applying {method} …"):
    try:
        processed_image = apply_preprocessing(target_image, method)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        st.stop()

# Side-by-side display
st.subheader(f"📊 Result: {method}")
col_orig, col_proc = st.columns(2)

with col_orig:
    st.markdown("#### 🖼 Original")
    w, h, mode = get_image_info(target_image)
    st.image(target_image, use_container_width=True)
    st.caption(f"`{target_label}` | {w} × {h} px | Mode: {mode}")

with col_proc:
    st.markdown(f"#### ✨ After: {method}")
    pw, ph, pmode = get_image_info(processed_image)
    st.image(processed_image, use_container_width=True)
    st.caption(f"Processed | {pw} × {ph} px | Mode: {pmode}")

st.markdown("---")

# Download button for processed image
buf = io.BytesIO()
processed_image.save(buf, format="PNG")
buf.seek(0)
st.download_button(
    label="⬇️ Download Processed Image",
    data=buf,
    file_name=f"processed_{method.lower().replace(' ', '_')}.png",
    mime="image/png",
    use_container_width=True,
)

st.caption(
    "💡 Switch between methods in the **sidebar** to compare different "
    "preprocessing effects on the same document."
)
