"""
app.py
------
Temporal-RegionRAG  |  Stage 5: Evidence-Grounded Question Answering
Entry point for the Streamlit application.

Run with:
    streamlit run app.py
"""

import streamlit as st

from utils.qa_pipeline import RESEARCH_DISCLAIMER

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Temporal-RegionRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**Temporal-RegionRAG** — Evidence-Grounded Q&A & Temporal Verification\n\n"
            f"{RESEARCH_DISCLAIMER}"
        ),
    },
)

# ---------------------------------------------------------------------------
# Sidebar navigation info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/document.png",
        width=64,
    )
    st.title("Temporal-RegionRAG")
    st.caption("Evidence-Grounded Q&A (Research Prototype)")
    st.markdown("---")
    st.markdown(
        """
        **Navigation**
        - 📤 **Upload** — Upload PDFs & images
        - 🔬 **Preprocess** — Apply OpenCV filters
        - 🧬 **Extract** — OCR, text & table extraction
        - 🔍 **Retrieve** — Query & evidence linking
        - ⏱️ **Verify** — Temporal verification & audit
        - 💬 **Ask** — Evidence-grounded Q&A
        """
    )
    st.markdown("---")
    st.info(
        "👈 Use the **sidebar pages** above to navigate, "
        "or click the links below."
    )

# ---------------------------------------------------------------------------
# Home / landing page content
# ---------------------------------------------------------------------------
st.title("📄 Temporal-RegionRAG")
st.subheader("Query-Conditioned Temporal Verification and Evidence Linking for Historical Medical Documents")
st.warning(f"⚠️ **{RESEARCH_DISCLAIMER}**")
st.markdown("---")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown("### 📤 Upload")
    st.write(
        "Upload **PDF** and **image** files. "
        "Preview each PDF page and image thumbnail."
    )
    if st.button("Go to Upload →", use_container_width=True):
        st.switch_page("pages/01_Upload.py")

with col2:
    st.markdown("### 🔬 Preprocess")
    st.write(
        "Apply **OpenCV** preprocessing — greyscale, blur, "
        "thresholding, and edge detection."
    )
    if st.button("Go to Preprocess →", use_container_width=True):
        st.switch_page("pages/02_Preprocess.py")

with col3:
    st.markdown("### 🧬 Extract")
    st.write(
        "Extract text via **PyMuPDF**, run **EasyOCR** on scanned pages, "
        "and detect **table regions**."
    )
    if st.button("Go to Extract →", use_container_width=True):
        st.switch_page("pages/03_Extract.py")

with col4:
    st.markdown("### 🔍 Retrieve")
    st.write(
        "Natural-language query retrieval, **multi-date comparisons**, "
        "and **spatial region highlighting**."
    )
    if st.button("Go to Retrieve →", use_container_width=True):
        st.switch_page("pages/04_Retrieve.py")

with col5:
    st.markdown("### ⏱️ Verify")
    st.write(
        "**Temporal verification**: normalize dates, detect **conflicts & duplicates**, "
        "order chronologically, and link source regions."
    )
    if st.button("Go to Verify →", use_container_width=True):
        st.switch_page("pages/05_Verify.py")

with col6:
    st.markdown("### 💬 Ask")
    st.write(
        "**Evidence-based Q&A**: retrieve, verify, answer from evidence only, "
        "and inspect highlighted source pages."
    )
    if st.button("Go to Ask →", use_container_width=True):
        st.switch_page("pages/06_Ask.py")

st.markdown("---")
st.markdown("### 🗺 Research Project Roadmap")
st.markdown(
    """
    - ✅ **Stage 1** — Document Ingestion & OpenCV Preprocessing
    - ✅ **Stage 2** — Text Extraction, OCR, and Table Detection
    - ✅ **Stage 3** — Lexical Retrieval Baseline & Spatial Region Overlays
    - ✅ **Stage 4** — Query-Conditioned Temporal Verification & Evidence Linking
    - ✅ **Stage 5** — Evidence-Grounded Q&A Interface & Optional LLM Adapter *(current)*
    - ⏳ **Stage 6** — Dense Temporal-Region Embeddings & Longitudinal Evaluation
    """
)

st.markdown("---")
st.caption(
    "FYP Research Project | Python 3.11 | Streamlit · PyMuPDF · OpenCV · EasyOCR · BM25 · Temporal Verification · Evidence-Grounded Q&A"
)


