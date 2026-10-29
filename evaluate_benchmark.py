#!/usr/bin/env python3
"""
evaluate_benchmark.py
---------------------
Reproducible CLI evaluation script for Temporal-RegionRAG.

Usage:
    source venv/bin/activate
    python evaluate_benchmark.py

Computes all metrics from actual runtime predictions against ground-truth labels
across Configurations A, B, and C, plus OCR/PDF extraction accuracy, latency, and
memory usage. Saves `evaluation_results.json` and `EVALUATION_REPORT.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.evaluation import (
    format_markdown_evaluation_report,
    run_reproducible_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible evaluation for Temporal-RegionRAG (Configs A, B, C)."
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip the EasyOCR scanned-image extraction benchmark for faster execution.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="evaluation_results.json",
        help="Path to write JSON evaluation results (default: evaluation_results.json).",
    )
    parser.add_argument(
        "--md-out",
        type=str,
        default="EVALUATION_REPORT.md",
        help="Path to write Markdown evaluation report (default: EVALUATION_REPORT.md).",
    )
    args = parser.parse_args()

    report = run_reproducible_evaluation(include_ocr_benchmark=not args.skip_ocr)
    md_text = format_markdown_evaluation_report(report)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.md_out:
        md_path = Path(args.md_out)
        md_path.write_text(md_text, encoding="utf-8")

    print(md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
