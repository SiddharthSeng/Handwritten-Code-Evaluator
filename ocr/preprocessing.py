"""
Image preprocessing pipeline for handwritten code OCR.

Applies a sequence of transformations to raw image bytes to produce
a clean, deskewed, binarized PIL Image suitable for TrOCR inference.

Pipeline steps:
    1. Load image from raw bytes
    2. Convert to grayscale
    3. CLAHE contrast enhancement
    4. Adaptive thresholding (binarization)
    5. Gaussian denoising
    6. Deskew via OpenCV minAreaRect rotation correction
    7. Convert back to RGB (TrOCR expects 3-channel input)
"""

import io
import logging
import math

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Preprocess a raw image for TrOCR handwriting recognition.

    Args:
        image_bytes: Raw image file content (PNG, JPEG, BMP, etc.).

    Returns:
        A clean PIL Image in RGB mode, ready for TrOCR input.

    Raises:
        ValueError: If *image_bytes* is empty or cannot be decoded.
        RuntimeError: If an unexpected failure occurs during any
            preprocessing step.
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty — nothing to preprocess")

    try:
        pil_image = _load_image(image_bytes)
        gray = _to_grayscale(pil_image)
        enhanced = _apply_clahe(gray)
        binarized = _adaptive_threshold(enhanced)
        denoised = _gaussian_denoise(binarized)
        deskewed = _deskew(denoised)
        rgb_image = _to_rgb_pil(deskewed)

        logger.info(
            "Preprocessing complete — output size %s", rgb_image.size
        )
        return rgb_image

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        logger.exception("Unexpected error during image preprocessing")
        raise RuntimeError(
            f"Image preprocessing failed: {exc}"
        ) from exc


# ── Internal helpers ────────────────────────────────────────────────


def _load_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes into a PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # Force full decode so errors surface here
        logger.debug("Loaded image: format=%s, size=%s, mode=%s",
                      image.format, image.size, image.mode)
        return image
    except Exception as exc:
        raise ValueError(
            f"Could not decode image from provided bytes: {exc}"
        ) from exc


def _to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a single-channel grayscale NumPy array."""
    gray = np.array(image.convert("L"), dtype=np.uint8)
    logger.debug("Converted to grayscale — shape %s", gray.shape)
    return gray


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    CLAHE improves local contrast, making faint handwriting more
    visible without over-amplifying noise.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    logger.debug("CLAHE contrast enhancement applied")
    return enhanced


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarize the image using adaptive Gaussian thresholding.

    Adaptive thresholding handles uneven lighting across the page
    far better than a global threshold.
    """
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,
        C=11,
    )
    logger.debug("Adaptive thresholding applied")
    return binary


def _gaussian_denoise(image: np.ndarray) -> np.ndarray:
    """Remove high-frequency noise with a mild Gaussian blur.

    A small 3×3 kernel smooths speckle without destroying thin
    strokes typical of handwriting.
    """
    denoised = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
    logger.debug("Gaussian denoising applied (kernel 3×3)")
    return denoised


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct rotation skew using the minimum-area bounding rectangle.

    Steps:
        1. Find all non-zero (foreground) pixel coordinates.
        2. Compute the minimum-area rotated rectangle via
           ``cv2.minAreaRect``.
        3. Derive the skew angle and rotate the image to level it.

    If the image has no foreground pixels or the detected angle is
    negligibly small (< 0.5°), the image is returned unchanged.
    """
    # For deskew we need dark-on-white text inverted so foreground is
    # non-zero (white pixels).
    inverted = cv2.bitwise_not(image)
    coords = cv2.findNonZero(inverted)

    if coords is None:
        logger.debug("Deskew skipped — no foreground pixels detected")
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]  # rotation angle in degrees

    # cv2.minAreaRect returns angles in [-90, 0).  Normalise so that
    # small clockwise/counter-clockwise tilts are corrected uniformly.
    if angle < -45.0:
        angle = 90.0 + angle
    # Also handle near-90° artefacts
    if angle > 45.0:
        angle = angle - 90.0

    if abs(angle) < 0.5:
        logger.debug("Deskew skipped — angle %.2f° below threshold", angle)
        return image

    logger.info("Deskewing image by %.2f°", angle)

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Compute the rotation matrix and determine the new bounding size
    # so that no content is cropped.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    cos_a = abs(rotation_matrix[0, 0])
    sin_a = abs(rotation_matrix[0, 1])
    new_w = int(math.ceil(h * sin_a + w * cos_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))

    # Adjust the rotation matrix to account for the new canvas size
    rotation_matrix[0, 2] += (new_w - w) / 2.0
    rotation_matrix[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,  # white background fill
    )
    return rotated


def _to_rgb_pil(image: np.ndarray) -> Image.Image:
    """Convert a single-channel NumPy array back to an RGB PIL Image.

    TrOCR's processor expects a 3-channel RGB image.
    """
    rgb_array = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_array)
