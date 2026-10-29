# Temporal-RegionRAG
## Query-Conditioned Temporal Verification and Evidence Linking for Historical Medical Documents

> **Research Prototype Notice:** This application is an academic research prototype for historical document analysis. It is **not** a clinical diagnostic tool, has not been clinically validated, and must not be used for medical diagnosis or treatment decisions.

---

## 📋 Project Overview

This research project develops an evidence-grounded Retrieval-Augmented Generation (RAG) system for **historical medical documents**. Every user question passes through lexical retrieval and deterministic temporal verification before an answer is assembled strictly from verified document evidence, with direct links to the supporting value, unit, date, document ID, page number, and spatial bounding box.

### Implemented Stages
1. **Stage 1 (Ingestion & Preprocessing):** Multi-page PDF rendering via PyMuPDF, image ingestion, and OpenCV preprocessing (greyscale, Gaussian blur, adaptive thresholding, Canny edge detection).
2. **Stage 2 (Extraction & OCR):** Digital text extraction with PyMuPDF, scanned-page OCR via EasyOCR with uncertainty flags (`ocr_confidence`, `ocr_uncertain`) instead of silent correction, contour-based table detection, and structured `Observation` records.
3. **Stage 3 (Lexical Retrieval Baseline):** Offline BM25 Okapi lexical ranking (`utils/retrieval.py`), multi-date comparison balancing, and region highlighting.
4. **Stage 4 (Temporal Verification & Evidence Linking):** Query parsing (`utils/temporal_verification.py`), ISO-8601 date normalization (preserving verbatim date text), chronological ordering, detection of missing timestamps / ambiguous numeric dates / duplicates / same-date conflicts, and four-way evidence separation (`VERIFIED`, `UNCERTAIN`, `CONFLICTING`, `REJECTED`).
5. **Stage 5 (Evidence-Grounded Question Answering):** End-to-end Q&A pipeline (`utils/qa_pipeline.py`) and interactive Streamlit interface (`pages/06_Ask.py`) with a deterministic evidence-template generator by default, an optional language-model adapter behind `AnswerGenerator` with grounding validation and automatic fallback, chronological comparison tables, and interactive original-page / region inspection.
6. **Stage 6 (Optional Visual Document Retrieval):** Page-level visual retrieval interface (`utils/visual_retrieval.py`) with optional ColPali (`vidore/colpali-v1.2`) late-interaction MaxSim scoring, persistent SHA-256 page embedding cache (`.visual_cache/embeddings/`), automatic fallback to OCR-based page retrieval (`OCRPageRetrieverFallback`), and page-to-region linking that connects retrieved pages to table regions, structured observations, normalized dates, and bounding boxes.
7. **Stage 7 (Reproducible Benchmark Evaluation):** Deterministic evaluation suite (`utils/evaluation.py` and CLI runner `evaluate_benchmark.py`) comparing Configuration A (OCR + lexical baseline), Configuration B (page-level visual retrieval), and Configuration C (full `Temporal-RegionRAG` system) on the same labelled synthetic test set, plus digital PDF vs. scanned EasyOCR extraction accuracy, latency, and memory usage.

---

## 🗂 Project Structure

```
FYP/
├── app.py                          # Main Streamlit entry point
├── evaluate_benchmark.py           # Reproducible CLI evaluation runner
├── EVALUATION_REPORT.md            # Generated Markdown evaluation report
├── evaluation_results.json         # Generated JSON evaluation metrics & error examples
├── requirements.txt                # Python 3.11 dependencies
├── README.md                       # Project documentation & setup instructions
│
├── pages/                          # Streamlit multi-page interface
│   ├── 01_Upload.py                # Upload & preview PDFs and images
│   ├── 02_Preprocess.py            # OpenCV preprocessing viewer
│   ├── 03_Extract.py               # PyMuPDF + EasyOCR extraction & table detection
│   ├── 04_Retrieve.py              # Lexical + optional visual page retrieval & highlighting
│   ├── 05_Verify.py                # Temporal verification & four-way evidence audit
│   └── 06_Ask.py                   # Evidence-grounded Q&A + source page inspector
│
├── utils/                          # UI-independent core modules
│   ├── __init__.py
│   ├── pdf_utils.py                # PyMuPDF PDF → image rendering
│   ├── image_utils.py              # OpenCV image filters
│   ├── extraction.py               # DocumentProcessor, Observation model, EasyOCR, tables
│   ├── retrieval.py                # LexicalRetriever (BM25), region highlighting & cropping
│   ├── visual_retrieval.py         # VisualPageRetriever, ColPaliConfig, EmbeddingCache, OCR fallback
│   ├── temporal_verification.py    # TemporalVerificationEngine, date normalization, conflicts
│   ├── qa_pipeline.py              # EvidenceQAPipeline, TemplateAnswerGenerator, LLM adapter
│   ├── evaluation.py               # Reproducible benchmark evaluation across Configs A, B, C
│   └── synthetic_docs.py           # In-memory synthetic PDFs, images, and benchmark corpora
│
└── tests/                          # Automated pytest suite (133 tests)
    ├── __init__.py
    ├── test_extraction.py          # Stage 2 extraction & OCR tests (38)
    ├── test_retrieval.py           # Stage 3 retrieval & highlighting tests (20)
    ├── test_temporal_verification.py # Stage 4 verification & conflict tests (26)
    ├── test_qa_pipeline.py         # Stage 5 Q&A pipeline & Streamlit UI integration tests (28)
    ├── test_visual_retrieval.py    # Stage 6 visual retrieval, caching & fallback tests (11)
    └── test_evaluation.py          # Stage 7 reproducible evaluation & 6-page workflow tests (10)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11
- `pip` 23+

### 1. Create and activate a virtual environment
```bash
cd /Users/prasi/Desktop/FYP
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note on EasyOCR weights:** On the first OCR run, EasyOCR loads detection (`craft_mlt_25k.pth`) and recognition (`english_g2.pth`) weights from `.easyocr_models/` (or downloads them if not yet cached). If macOS Python reports an SSL certificate error during first-time download, export `SSL_CERT_FILE=$(python3 -m certifi)`.

### 3. Run the Streamlit application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser and select **💬 Ask** in the sidebar (or start from **📤 Upload** / **🧬 Extract**).

---

## 💬 Question-and-Answer Pipeline (`utils/qa_pipeline.py`)

1. **Retrieve → Verify → Package → Answer**:
   - `EvidenceQAPipeline.answer(question, observations, generator=...)` retrieves candidate observations with `LexicalRetriever`, runs `TemporalVerificationEngine.verify()`, builds a closed `EvidencePackage`, and generates an answer exclusively from that package.
2. **Evidence-Based Template (Default, Offline)**:
   - `TemplateAnswerGenerator` is deterministic and requires no external model or network access.
   - Lists verified evidence in chronological order with citation tags (`[E1]`, `[E2]`, …), parsed value, unit, normalized and verbatim date, document ID, page number, and bounding-box coordinates `(x, y, w, h)`.
   - Reports conflicting values side by side without selecting either as "correct".
   - Separates uncertain evidence (low OCR confidence, ambiguous date format, duplicate record, non-standard unit) and notes observations excluded for missing timestamps.
   - Returns an explicit **Insufficient evidence** response when no observation survives verification.
3. **Optional Language Model Interface**:
   - Any callable `llm_fn(prompt: str) -> str` can be wrapped in `CallableLLMAnswerGenerator(llm_fn)`.
   - `validate_grounded_answer()` rejects model outputs that introduce numbers not present in the evidence package or omit `[E#]` citations, falling back automatically to `TemplateAnswerGenerator`.
   - When no model is configured (`llm_fn=None`) or if a model call raises an error, the pipeline falls back cleanly to the evidence template.
4. **Source Page & Region Inspection (`pages/06_Ask.py`)**:
   - Users can select any cited evidence item `[E#]` to view the cropped region snippet alongside the full original page (both highlighted with status-coloured bounding boxes and unannotated original).

---

## 🖼 Optional Visual Document Retrieval (`utils/visual_retrieval.py`)

### 1. Interface & Granularity Boundary
- `VisualPageRetriever` defines the page-level retrieval interface returning `List[VisualPageHit]` (`page_id` as `<doc_id>::p<page_number>`, relevance `score`, `rank`, `backend_name`, `cached`, and `granularity_note`).
- **Important Granularity Boundary:** Page-level visual retrieval (including ColPali late-interaction patch scoring) ranks **candidate document pages only**; it does **not** automatically identify or verify the exact table cell or clinical measurement. Candidate pages are connected by `VisualRetrievalEngine` to OpenCV-detected table regions (`detect_table_regions`), extracted `Observation` records, normalized dates, bounding boxes, and `TemporalVerificationEngine` checks.

### 2. Default OCR Page Fallback vs. Optional ColPali Backend
- **Default (`OCRPageRetrieverFallback`):** Aggregates BM25 observation scores per page so page-level retrieval works offline with zero extra model downloads.
- **Optional ColPali (`ColPaliVisualRetriever`):** Uses `vidore/colpali-v1.2` (`colpali-engine` + `google/paligemma-3b-mix-448`, ~6.5 GB weights in `fp16`/`bf16`).
- **Embedding & Model Caching:** Page patch embeddings are cached in memory and on disk under `.visual_cache/embeddings/` keyed by the SHA-256 hash of the page PNG bytes + model name, so pages are never re-encoded across queries. Model weights are cached locally under `.visual_cache/models/` with `local_files_only=True` by default to prevent unexpected multi-GB downloads.

### 3. How to Enable ColPali Later
On a machine with sufficient memory (≥16 GB unified RAM on Apple Silicon MPS or a CUDA GPU with ≥8 GB VRAM) and ~7 GB free disk space:
```bash
source venv/bin/activate
pip install "colpali-engine>=0.3.0,<0.4.0" "transformers>=4.45.0" "peft>=0.12.0" "accelerate>=0.34.0"

# First run: allow Hugging Face weight download into .visual_cache/models/
ENABLE_COLPALI=1 COLPALI_ALLOW_DOWNLOAD=1 streamlit run app.py

# Subsequent runs (uses cached weights offline):
ENABLE_COLPALI=1 streamlit run app.py
```

---

## 📊 Reproducible Benchmark Evaluation (`evaluate_benchmark.py`)

Run the reproducible evaluation script to compare Configurations A, B, and C on the labelled synthetic test set (`N = 15` queries across 8 documents / 25 observations + `N = 12` OCR/PDF extraction items) and regenerate `EVALUATION_REPORT.md` and `evaluation_results.json`:

```bash
source venv/bin/activate
python evaluate_benchmark.py
```

---

## 🧪 Running the Test Suite

Run all unit and integration tests (133 tests):
```bash
source venv/bin/activate
pytest tests/ -v
```

Or run individual test modules:
```bash
pytest tests/test_evaluation.py -v             # 10 reproducible evaluation & 6-page workflow tests
pytest tests/test_visual_retrieval.py -v       # 11 visual retrieval, caching & fallback tests
pytest tests/test_qa_pipeline.py -v            # 28 Q&A pipeline & Streamlit UI integration tests
pytest tests/test_temporal_verification.py -v  # 26 temporal verification unit tests
pytest tests/test_retrieval.py -v              # 20 lexical retrieval unit tests
pytest tests/test_extraction.py -v             # 38 extraction & OCR unit tests
```

---

## ⚠️ Remaining Limitations

1. **Rule-Based Concept Lexicon:** Concept verification (`CLINICAL_CONCEPT_MAP` in `utils/temporal_verification.py`) covers a fixed lexicon of common clinical concepts (blood pressure, heart rate, fasting blood glucose, total cholesterol, HbA1c, body temperature, respiratory rate, allergies). Questions about concepts outside this lexicon fall back to lexical matching without unit/concept-alias verification.
2. **Single-Value Numeric Regex:** `_extract_value_unit()` parses the first number in an observation string (for example, `140` from `140/90 mmHg`); composite ratios such as systolic/diastolic pairs are preserved verbatim in `obs.text` but stored as a single float in `obs.value`.
3. **Line-Level Date Attribution:** Dates are extracted from the same text block/line unless provided at the document level; multi-paragraph inheritance across page sections (e.g. a date header at the top of a scanned page applying to subsequent lines without repeated dates) is not inferred automatically to avoid inventing timestamps.
4. **OCR Degradation on Low-Resolution Scanned Pages:** On simulated low-resolution scans (`make_scanned_pdf`), EasyOCR can misread degree symbols (`°C` → `'C`) or drop decimal points (`12.5 g/dL` → `125gdL`), reducing scanned joint value+unit extraction accuracy (`0.5000` vs. `1.0000` on digital PDFs).
5. **Optional Heavyweight Models Not Bundled by Default:** Both `vidore/colpali-v1.2` (~6.5 GB weights) and external LLMs are disabled by default so the application runs offline on standard hardware without multi-GB downloads.
