"""
utils/image_utils.py
--------------------
OpenCV-based image preprocessing helpers.

Each function accepts a PIL Image and returns a PIL Image so they can be
composed and displayed directly in Streamlit without extra conversion steps.

Usage:
    from utils.image_utils import apply_preprocessing
    result = apply_preprocessing(pil_image, method="adaptive_threshold")
"""

from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "opencv-python-headless is not installed. "
        "Run: pip install opencv-python-headless"
    ) from e

from PIL import Image

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV BGR ndarray."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(array: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ndarray to a PIL Image (RGB)."""
    if len(array.shape) == 2:
        # Greyscale — convert to 3-channel RGB for consistent display
        rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------

def to_greyscale(image: Image.Image) -> Image.Image:
    """Convert image to greyscale."""
    bgr = pil_to_bgr(image)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr_to_pil(grey)


def gaussian_blur(image: Image.Image, kernel_size: int = 5) -> Image.Image:
    """
    Apply Gaussian blur.

    Parameters
    ----------
    kernel_size : int
        Must be a positive odd integer (e.g. 3, 5, 7, 9).
    """
    # Ensure kernel_size is positive and odd
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    bgr = pil_to_bgr(image)
    blurred = cv2.GaussianBlur(bgr, (kernel_size, kernel_size), 0)
    return bgr_to_pil(blurred)


def adaptive_threshold(image: Image.Image) -> Image.Image:
    """
    Apply adaptive (local) thresholding — binarises text regions.
    Ideal for uneven lighting in scanned historical documents.
    """
    bgr = pil_to_bgr(image)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Blur first to reduce noise before thresholding
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    return bgr_to_pil(binary)


def canny_edges(
    image: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> Image.Image:
    """
    Apply Canny edge detection.

    Parameters
    ----------
    low_threshold : int
        Lower hysteresis threshold.
    high_threshold : int
        Upper hysteresis threshold.
    """
    bgr = pil_to_bgr(image)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return bgr_to_pil(edges)


def deskew(image: Image.Image) -> Image.Image:
    """
    Attempt to correct slight skew in scanned documents using the
    minimum-area bounding rectangle of foreground pixels.
    """
    bgr = pil_to_bgr(image)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Invert so text is white on black background
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return image  # Not enough foreground pixels to detect skew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    h, w = grey.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return bgr_to_pil(rotated)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PREPROCESSING_METHODS = {
    "Greyscale": to_greyscale,
    "Gaussian Blur": gaussian_blur,
    "Adaptive Threshold": adaptive_threshold,
    "Canny Edge Detection": canny_edges,
    "Deskew": deskew,
}


def apply_preprocessing(image: Image.Image, method: str) -> Image.Image:
    """
    Apply the named preprocessing step to a PIL Image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    method : str
        One of the keys in PREPROCESSING_METHODS.

    Returns
    -------
    PIL.Image.Image
        Processed image.

    Raises
    ------
    ValueError
        If the method name is not recognised.
    """
    if method not in PREPROCESSING_METHODS:
        raise ValueError(
            f"Unknown preprocessing method '{method}'. "
            f"Choose from: {list(PREPROCESSING_METHODS)}"
        )
    return PREPROCESSING_METHODS[method](image)


def get_image_info(image: Image.Image) -> Tuple[int, int, str]:
    """Return (width, height, mode) for an image."""
    return image.width, image.height, image.mode


# ---------------------------------------------------------------------------
# Self-test (run directly: python3 utils/image_utils.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[image_utils] Self-test: generating synthetic test image …")

    # Create a simple synthetic greyscale document-like image
    img_array = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Draw some "text" lines
    for y in range(60, 380, 40):
        cv2.line(img_array, (30, y), (570, y), (40, 40, 40), 2)
    test_img = Image.fromarray(img_array)

    for method_name in PREPROCESSING_METHODS:
        result = apply_preprocessing(test_img, method_name)
        assert isinstance(result, Image.Image), \
            f"{method_name}: result is not a PIL Image"
        assert result.width > 0 and result.height > 0, \
            f"{method_name}: result has zero dimension"
        print(f"  ✅  {method_name:25s} → size={result.size}, mode={result.mode}")

    print("[image_utils] All self-tests passed.")
