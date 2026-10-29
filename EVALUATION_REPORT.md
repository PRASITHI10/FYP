# Temporal-RegionRAG Reproducible Evaluation Report

> **Note:** Metrics are computed on a deterministic synthetic benchmark (N=15 queries, 25 corpus observations, 12 extraction items). Because N is small and synthetic, no claim of statistical significance or clinical generalization is made.

## 1. Benchmark Dataset & Evaluation Procedure
- **Corpus Size:** 8 synthetic historical documents (8 pages, 25 structured observations with ground-truth bounding boxes).
- **Query Test Set (`N = 15`):**
  - Single-date clinical lookups: `6`
  - Multi-date temporal comparisons: `4`
  - Conflict / duplicate / ambiguous-date / missing-timestamp stress cases: `3`
  - Unanswerable negative queries (absent concept / out-of-range year): `2`
- **OCR & Extraction Test Set (`N = 12`):** 8 digital PDF text observations (`make_text_pdf`) + 4 scanned PDF observations (`make_scanned_pdf` via EasyOCR).
- **Hardware & Process Footprint:** `macOS-15.7.2-arm64-arm-64bit`, Python `3.11.4`, Peak Process RSS `5819.41 MiB`.

## 2. Visual Retrieval Operational Status (Configuration B)
- **Target Model:** `vidore/colpali-v1.2`
- **ColPali Installed / Operational:** `False` (`colpali_engine=False`, `transformers=False`)
- **Active Page Retriever Used:** `ocr-lexical-page-fallback`
- **Note:** ColPali (vidore/colpali-v1.2, ~6.5 GB PaliGemma-3B weights) is not installed in this 16 GiB unified-RAM macOS environment. VisualRetrievalEngine ran with its operational OCRPageRetrieverFallback and connected retrieved pages to detected table regions and page observations.

## 3. Configuration Comparison Results Table

| Metric | Config A: OCR + Lexical Baseline | Config B: Page-Level Visual Retrieval (OCR Fallback) | Config C: Full Temporal-RegionRAG |
| :--- | :---: | :---: | :---: |
| **Retrieval Recall@1** (answerable, N=13) | 0.7179 | 0.0000 | 0.7949 |
| **Retrieval Recall@3** (answerable, N=13) | 0.9615 | 0.3462 | 1.0000 |
| **Retrieval Recall@5** (answerable, N=13) | 1.0000 | 0.7692 | 1.0000 |
| **Source-Page Recall@1** (N=13) | 0.7564 | 0.7564 | 0.7564 |
| **Source-Page Recall@3** (N=13) | 0.9615 | 0.9615 | 0.9615 |
| **Region Localization Accuracy (IoU >= 0.5)** | 1.0000 | 0.9474 | 1.0000 |
| **Evidence Linking Precision** (N=15) | 0.3467 | 0.1502 | 1.0000 |
| **Evidence Linking F1** (N=15) | 0.4635 | 0.2163 | 1.0000 |
| **Verification Status Exact-Match** (N=15) | 0.1333 | 0.0667 | 1.0000 |
| **Temporal Date-Set Coverage** (N=4) | 0.5000 | 0.5000 | 1.0000 |
| **Temporal Chronological Order** (N=4) | 0.2500 | 0.2500 | 1.0000 |
| **Temporal Comparison Strict Accuracy** (N=4) | 0.2500 | 0.0000 | 1.0000 |
| **Source-Grounding Completeness** | 1.0000 | 1.0000 | 1.0000 |
| **Mean Query Latency (ms)** | 0.721 | 2.617 | 1.103 |
| **P95 Query Latency (ms)** | 0.830 | 4.826 | 1.310 |
| **Peak Python Heap Delta (KiB)** | 56.81 | 2480.84 | 48.24 |

## 4. OCR & Document Extraction Accuracy

| Modality | Items | Value Accuracy | Unit Accuracy | Joint Value+Unit Accuracy | Date Accuracy | BBox Valid Rate | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Digital PDF (`PyMuPDF`)** | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1598.02 |
| **Scanned Image (`EasyOCR`)** | 4 | 0.7500 | 0.5000 | 0.5000 | 1.0000 | 1.0000 | 4620.81 |
| **Combined Overall** | 12 | 0.9167 | 0.8333 | 0.8333 | 1.0000 | 1.0000 | — |

## 5. Concrete Error Examples from Actual Predictions

### Configuration A (OCR + Lexical Retrieval Baseline) Error Examples
- **`Q01` (What was the blood pressure on 22 Apr 2002?):** Precision=`0.2`, F1=`0.3333` — Included 4 unverified/out-of-scope observation(s): obs_11 (doc_2002_clinic, 'Fasting Blood Glucose: 6.8 mmol/L', date=22 Apr 2002); obs_08 (doc_2002_clinic, 'Follow-up Clinic Note: Cardiovascular ', date=22 Apr 2002); obs_10 (doc_2002_clinic, 'Heart Rate: 72 bpm', date=22 Apr 2002)
- **`Q02` (What was the heart rate on 15 Jan 1998?):** Precision=`0.2`, F1=`0.3333` — Included 4 unverified/out-of-scope observation(s): obs_missing_timestamp (doc_undated_vitals, 'Heart Rate: 84 bpm (routine check)', date=None); obs_02 (doc_1998_admission, 'Blood Pressure: 140/90 mmHg', date=15 Jan 1998); obs_04 (doc_1998_admission, 'Body Temperature: 37.2 °C', date=15 Jan 1998)
- **`Q03` (What was the HbA1c in November 2005?):** Precision=`0.2`, F1=`0.3333` — Included 4 unverified/out-of-scope observation(s): obs_14 (doc_2005_discharge, 'Annual Diabetic & Cardio Review: Compr', date=10 Nov 2005); obs_15 (doc_2005_discharge, 'Blood Pressure: 124/78 mmHg', date=10 Nov 2005); obs_16 (doc_2005_discharge, 'Heart Rate: 70 bpm', date=10 Nov 2005)
- **`Q04` (What was the body temperature on 15 Jan 1998?):** Precision=`0.2`, F1=`0.3333` — Included 4 unverified/out-of-scope observation(s): obs_02 (doc_1998_admission, 'Blood Pressure: 140/90 mmHg', date=15 Jan 1998); obs_03 (doc_1998_admission, 'Heart Rate: 88 bpm', date=15 Jan 1998); obs_07 (doc_1998_admission, 'Documented Allergy: Penicillin causes ', date=15 Jan 1998)

### Configuration B (Page-Level Visual Retrieval) Error Examples
- **`Q01` (What was the blood pressure on 22 Apr 2002?):** Precision=`0.05`, F1=`0.0952` — Included 19 unverified/out-of-scope observation(s): obs_08 (doc_2002_clinic, 'Follow-up Clinic Note: Cardiovascular ', date=22 Apr 2002); obs_10 (doc_2002_clinic, 'Heart Rate: 72 bpm', date=22 Apr 2002); obs_11 (doc_2002_clinic, 'Fasting Blood Glucose: 6.8 mmol/L', date=22 Apr 2002)
- **`Q02` (What was the heart rate on 15 Jan 1998?):** Precision=`0.1111`, F1=`0.2` — Included 8 unverified/out-of-scope observation(s): obs_01 (doc_1998_admission, 'Admission Note: Patient admitted with ', date=15 Jan 1998); obs_02 (doc_1998_admission, 'Blood Pressure: 140/90 mmHg', date=15 Jan 1998); obs_04 (doc_1998_admission, 'Body Temperature: 37.2 °C', date=15 Jan 1998)
- **`Q03` (What was the HbA1c in November 2005?):** Precision=`0.1429`, F1=`0.25` — Included 6 unverified/out-of-scope observation(s): obs_14 (doc_2005_discharge, 'Annual Diabetic & Cardio Review: Compr', date=10 Nov 2005); obs_15 (doc_2005_discharge, 'Blood Pressure: 124/78 mmHg', date=10 Nov 2005); obs_16 (doc_2005_discharge, 'Heart Rate: 70 bpm', date=10 Nov 2005)
- **`Q04` (What was the body temperature on 15 Jan 1998?):** Precision=`0.1111`, F1=`0.2` — Included 8 unverified/out-of-scope observation(s): obs_01 (doc_1998_admission, 'Admission Note: Patient admitted with ', date=15 Jan 1998); obs_02 (doc_1998_admission, 'Blood Pressure: 140/90 mmHg', date=15 Jan 1998); obs_03 (doc_1998_admission, 'Heart Rate: 88 bpm', date=15 Jan 1998)

### OCR / Extraction Error Examples
- **`EXT_OCR_03` (scanned_ocr — `37.2`):** Expected `37.2 °C`, got value=`37.2`, unit=`None`, raw text=`Temperture.37.2'C`.
- **`EXT_OCR_04` (scanned_ocr — `Haem`):** Expected `12.5 g/dL`, got value=`None`, unit=`None`, raw text=`Haemcglobin: 125gdL`.

## 6. Limitations & Threats to Validity
- **Small Synthetic Benchmark (`N = 15` queries, `25` corpus observations, `12` extraction items):** All documents and queries are synthetic. No statistical significance testing (e.g., paired permutation or bootstrap p-values) is claimed because sample size is small and designed for deterministic regression verification rather than clinical generalization.
- **ColPali Weight & Memory Footprint:** Native `vidore/colpali-v1.2` (~6.5 GB PaliGemma-3B weights) is not installed in the default 16 GiB unified-memory macOS environment; Configuration B therefore reflects the operational page-level retrieval pipeline (`VisualRetrievalEngine` with `OCRPageRetrieverFallback`). Furthermore, page-level retrieval returns candidate pages (`Recall@1 = 0.7564`, `Recall@3 = 0.9615`) rather than isolating individual table cells (`Evidence Linking Precision = 0.1502` without `TemporalVerificationEngine`).
- **OCR Character & Decimal Degradation on Scanned Pages:** On simulated low-resolution scanned pages (`make_scanned_pdf`), EasyOCR misreads degree symbols (`°C` -> `'C` in `Temperture.37.2'C`) and drops decimal points or spaces (`12.5 g/dL` -> `125gdL`), reducing scanned joint value+unit accuracy to `0.5000` (compared to `1.0000` on digital PDF text).
- **Rule-Based Concept & Unit Lexicon:** `TemporalVerificationEngine` relies on `CLINICAL_CONCEPT_MAP` (8 canonical clinical concepts) and regex-based value/unit extraction (`_VALUE_PATTERN`). Unseen clinical abbreviations, multi-line narrative tables without repeated row dates, or handwritten cursive notes outside EasyOCR's printed-English training distribution remain out of scope.