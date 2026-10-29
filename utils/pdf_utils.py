"""
utils/pdf_utils.py
------------------
Helpers for rendering PDF pages to PIL Images using PyMuPDF (fitz).

Usage:
    from utils.pdf_utils import pdf_to_images
    images = pdf_to_images(pdf_bytes, dpi=150)
"""

import io
from typing import List

try:
    import pymupdf as fitz  # PyMuPDF ≥1.24 canonical import (no deprecation warning)
except ImportError:
    try:
        import fitz  # Fallback for older installs
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        ) from e

from PIL import Image


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    """
    Convert all pages of a PDF (given as raw bytes) into a list of PIL Images.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw bytes of the PDF file (e.g. from st.file_uploader().read()).
    dpi : int
        Rendering resolution. 150 is a good balance between quality and speed.
        Use 200-300 for high-quality preprocessing.

    Returns
    -------
    List[PIL.Image.Image]
        One PIL Image per PDF page, in RGB mode.

    Raises
    ------
    ValueError
        If the bytes do not represent a valid PDF.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes must not be empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    images: List[Image.Image] = []
    zoom = dpi / 72.0          # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pixmap.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(pil_img)

    doc.close()
    return images


def get_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF without rendering them."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


# ---------------------------------------------------------------------------
# Self-test (run directly: python3 utils/pdf_utils.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[pdf_utils] Self-test: creating a minimal in-memory PDF …")
    # Build a 1-page PDF using only fitz (no external file needed)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Temporal-RegionRAG — test page", fontsize=18)
    buf = io.BytesIO()
    doc.save(buf)
    pdf_bytes = buf.getvalue()
    doc.close()

    imgs = pdf_to_images(pdf_bytes, dpi=72)
    assert len(imgs) == 1, f"Expected 1 page, got {len(imgs)}"
    assert imgs[0].mode == "RGB", f"Expected RGB, got {imgs[0].mode}"
    print(f"  ✅  pdf_to_images OK — rendered {len(imgs)} page(s), "
          f"size={imgs[0].size}")
    print("[pdf_utils] All self-tests passed.")
