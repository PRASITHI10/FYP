"""
tests/test_evaluation.py
------------------------
Unit and integration tests for the reproducible evaluation module
(utils/evaluation.py and evaluate_benchmark.py) and main app startup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.evaluation import (
    bbox_iou,
    build_labelled_evaluation_corpus,
    build_ocr_extraction_benchmark,
    build_query_benchmark,
    compute_page_recall_at_k,
    compute_recall_at_k,
    evaluate_ocr_and_extraction,
    format_markdown_evaluation_report,
    run_reproducible_evaluation,
)


class TestEvaluationMetricsHelpers:

    def test_bbox_iou_identical_and_disjoint(self):
        assert bbox_iou((50, 100, 200, 24), (50, 100, 200, 24)) == pytest.approx(1.0)
        assert bbox_iou((50, 100, 200, 24), (400, 400, 50, 24)) == pytest.approx(0.0)
        assert bbox_iou((0, 0, 0, 0), (50, 100, 200, 24)) == pytest.approx(0.0)

    def test_recall_at_k_computation(self):
        ranked = ["obs_02", "obs_10", "obs_15"]
        gt = ["obs_02", "obs_15"]
        assert compute_recall_at_k(ranked, gt, k=1) == pytest.approx(0.5)
        assert compute_recall_at_k(ranked, gt, k=2) == pytest.approx(0.5)
        assert compute_recall_at_k(ranked, gt, k=3) == pytest.approx(1.0)
        assert compute_recall_at_k(ranked, [], k=3) is None

    def test_page_recall_at_k_deduplicates_pages(self):
        pages = [("doc_a", 1), ("doc_a", 1), ("doc_b", 1)]
        gt = [("doc_a", 1), ("doc_b", 1)]
        assert compute_page_recall_at_k(pages, gt, k=1) == pytest.approx(0.5)
        assert compute_page_recall_at_k(pages, gt, k=2) == pytest.approx(1.0)


class TestBenchmarkDatasetConstruction:

    def test_labelled_corpus_is_deterministic(self):
        c1, pages1, map1 = build_labelled_evaluation_corpus()
        c2, _, map2 = build_labelled_evaluation_corpus()
        assert len(c1) == 25
        assert len(pages1) == 8
        assert [o.obs_id for o in c1] == [o.obs_id for o in c2]
        assert set(map1.keys()) == set(map2.keys())

    def test_query_benchmark_covers_all_categories(self):
        corpus, _, _ = build_labelled_evaluation_corpus()
        queries = build_query_benchmark(corpus)
        assert len(queries) == 15
        cats = {q.category for q in queries}
        assert cats == {
            "single_date",
            "temporal_comparison",
            "conflict_or_uncertainty",
            "unanswerable_negative",
        }

    def test_ocr_extraction_benchmark_size(self):
        items = build_ocr_extraction_benchmark()
        assert len(items) == 12
        assert sum(1 for i in items if i.source_modality == "pdf_text") == 8
        assert sum(1 for i in items if i.source_modality == "scanned_ocr") == 4


class TestReproducibleEvaluationComparison:

    def test_compare_configurations_a_b_c_on_same_test_set(self):
        report = run_reproducible_evaluation(include_ocr_benchmark=False)
        meta = report["benchmark_metadata"]
        assert meta["num_queries"] == 15
        assert meta["num_corpus_observations"] == 25

        vis = report["visual_retrieval_status"]
        assert "colpali_operational" in vis
        assert vis["active_backend_used"] in (
            "ocr-lexical-page-fallback",
            "colpali-late-interaction",
        )

        cfgs = report["configurations"]
        cfg_a = cfgs["A_ocr_lexical_baseline"]
        cfg_b = cfgs["B_visual_page_retrieval"]
        cfg_c = cfgs["C_full_temporal_regionrag"]

        # All 3 configs evaluated on the exact same 15 queries
        assert cfg_a["num_queries_total"] == 15
        assert cfg_b["num_queries_total"] == 15
        assert cfg_c["num_queries_total"] == 15

        # Full system (Config C) outperforms unverified baselines on precision, F1,
        # verification status accuracy, and strict temporal comparison accuracy
        assert cfg_c["evidence_precision"] > cfg_a["evidence_precision"]
        assert cfg_c["evidence_precision"] > cfg_b["evidence_precision"]
        assert cfg_c["evidence_f1"] > cfg_a["evidence_f1"]
        assert cfg_c["verification_status_accuracy"] > cfg_a["verification_status_accuracy"]
        assert cfg_c["temporal_comparison_strict_accuracy"] == pytest.approx(1.0)
        assert cfg_c["source_grounding_accuracy"] == pytest.approx(1.0)

        # Concrete error examples are recorded for baselines A and B
        assert len(cfg_a["error_examples"]) > 0
        assert len(cfg_b["error_examples"]) > 0

        # Markdown report renders all sections cleanly
        md = format_markdown_evaluation_report(report)
        assert "## 3. Configuration Comparison Results Table" in md
        assert "## 5. Concrete Error Examples from Actual Predictions" in md
        assert "## 6. Limitations & Threats to Validity" in md

    def test_ocr_and_extraction_accuracy_evaluation(self):
        ocr_rep = evaluate_ocr_and_extraction()
        assert ocr_rep["total_ground_truth_items"] == 12
        assert ocr_rep["pdf_text"]["joint_value_unit_accuracy"] == pytest.approx(1.0)
        assert 0.0 < ocr_rep["scanned_ocr"]["value_accuracy"] <= 1.0
        assert len(ocr_rep["item_results"]) == 12


class TestAppStartupWithoutHeavyweightModels:

    def test_main_app_starts_without_optional_models(self):
        from streamlit.testing.v1 import AppTest

        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        at = AppTest.from_file(app_path, default_timeout=20)
        at.run()
        assert not at.exception

    def test_end_to_end_workflow_across_all_streamlit_pages(self):
        """
        End-to-end verification of the complete workflow using synthetic documents:
        1. Upload & preview synthetic PDF and image (01_Upload.py)
        2. Preprocess uploaded page with OpenCV (02_Preprocess.py)
        3. Extract observations & tables from synthetic PDF (03_Extract.py)
        4. Retrieve ranked evidence & visual page connections (04_Retrieve.py)
        5. Verify dates, conflicts, and four-way categories (05_Verify.py)
        6. Ask a question on session documents & inspect source references (06_Ask.py)
        """
        from streamlit.testing.v1 import AppTest
        from utils.pdf_utils import pdf_to_images
        from utils.synthetic_docs import make_clinical_image, make_text_pdf

        root = Path(__file__).resolve().parent.parent
        synth_pdf = make_text_pdf(
            observations=[
                {"text": "Blood Pressure", "value": "140/90", "unit": "mmHg", "date": "15 Jan 1998"},
                {"text": "Heart Rate", "value": 88, "unit": "bpm", "date": "15 Jan 1998"},
                {"text": "Blood Glucose", "value": 5.4, "unit": "mmol/L", "date": "15 Jan 1998"},
                {"text": "Blood Pressure", "value": "130/80", "unit": "mmHg", "date": "22 Apr 2002"},
                {"text": "Blood Glucose", "value": 6.8, "unit": "mmol/L", "date": "22 Apr 2002"},
            ]
        )
        synth_pages = pdf_to_images(synth_pdf, dpi=100)
        synth_img = make_clinical_image()

        # 1. Page 01_Upload.py
        at_up = AppTest.from_file(str(root / "pages" / "01_Upload.py"), default_timeout=20)
        at_up.run()
        assert not at_up.exception

        # 2. Page 02_Preprocess.py with uploaded PDF pages & image in session_state
        at_pre = AppTest.from_file(str(root / "pages" / "02_Preprocess.py"), default_timeout=20)
        at_pre.session_state["uploaded_pdf_pages"] = {"synthetic_record.pdf": synth_pages}
        at_pre.session_state["uploaded_images"] = {"synthetic_scan.png": synth_img}
        at_pre.run()
        assert not at_pre.exception

        # 3. Page 03_Extract.py: load synthetic PDF in session and click "▶ Run Extraction"
        at_ext = AppTest.from_file(str(root / "pages" / "03_Extract.py"), default_timeout=30)
        at_ext.session_state["uploaded_pdf_pages"] = {"synthetic_record.pdf": synth_pages}
        at_ext.session_state["uploaded_pdf_bytes"] = {"synthetic_record.pdf": synth_pdf}
        at_ext.run()
        assert not at_ext.exception
        run_ext_btns = [b for b in at_ext.button if "Run Extraction" in b.label]
        assert len(run_ext_btns) == 1
        run_ext_btns[0].click().run()
        assert not at_ext.exception
        extracted_obs = at_ext.session_state["extracted_observations"]
        extracted_imgs = at_ext.session_state["extracted_page_images"]
        doc_registry = at_ext.session_state["document_registry"]
        assert len(extracted_obs) >= 5
        assert len(extracted_imgs) >= 1
        assert "synthetic_record.pdf" in doc_registry

        # 4. Page 04_Retrieve.py: query extracted session observations
        at_ret = AppTest.from_file(str(root / "pages" / "04_Retrieve.py"), default_timeout=20)
        at_ret.session_state["extracted_observations"] = extracted_obs
        at_ret.session_state["extracted_page_images"] = extracted_imgs
        at_ret.session_state["extracted_doc_id"] = "synthetic_record.pdf"
        at_ret.run()
        assert not at_ret.exception

        # 5. Page 05_Verify.py: run temporal verification on session & benchmark observations
        at_ver = AppTest.from_file(str(root / "pages" / "05_Verify.py"), default_timeout=20)
        at_ver.session_state["extracted_observations"] = extracted_obs
        at_ver.session_state["extracted_page_images"] = extracted_imgs
        at_ver.session_state["extracted_doc_id"] = "synthetic_record.pdf"
        at_ver.run()
        assert not at_ver.exception

        # 6. Page 06_Ask.py: switch to "Documents processed in this session" and ask a question
        at_ask = AppTest.from_file(str(root / "pages" / "06_Ask.py"), default_timeout=30)
        at_ask.session_state["document_registry"] = doc_registry
        at_ask.run()
        assert not at_ask.exception
        # Select session documents radio
        doc_src_radios = [r for r in at_ask.radio if r.label == "Document source"]
        assert len(doc_src_radios) == 1
        doc_src_radios[0].set_value("Documents processed in this session").run()
        assert not at_ask.exception
        # Ask a multi-date comparison question on the uploaded synthetic PDF
        at_ask.text_input(key="qa_input").set_value("Compare blood pressure in 1998 and 2002").run()
        ask_btns = [b for b in at_ask.button if "Ask" in b.label]
        assert len(ask_btns) == 1
        ask_btns[0].click().run()
        assert not at_ask.exception
        assert any("1998-01-15" in m.value and "2002-04-22" in m.value for m in at_ask.markdown)
        assert len(at_ask.dataframe) >= 2  # Chronological comparison + Supporting evidence

