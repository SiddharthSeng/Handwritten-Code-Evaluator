"""
Line segmentation for handwritten code images.

Detects individual text lines in a preprocessed handwritten code image
and returns cropped line images with their horizontal offsets.  This
enables TrOCR — which is designed for single-line input — to process
multi-line handwritten code correctly.

Detection strategy:
    1. **Horizontal projection profile** (primary) — sum pixel
       intensities row-by-row to locate text rows vs. inter-line gaps.
    2. **Contour-based grouping** (fallback) — if the projection
       profile yields fewer lines than the image height suggests,
       fall back to ``cv2.findContours`` and cluster bounding boxes
       by Y-coordinate.
    3. Minimum-height filtering removes noise slivers.
    4. Final output is sorted top-to-bottom.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────

_MIN_LINE_HEIGHT_PX = 10
"""Ignore detected regions shorter than this (likely noise)."""

_PROFILE_THRESHOLD_RATIO = 0.02
"""Fraction of the maximum projection value used as the text/gap
threshold.  Rows whose summed intensity falls below
``max_value * ratio`` are treated as gaps."""

_EXPECTED_MIN_LINE_HEIGHT_RATIO = 0.04
"""If the image height divided by the number of detected lines gives
a per-line height smaller than ``image_height * ratio``, the result
is likely an over-segmentation and we keep the projection result
as-is.  Used only in the fallback heuristic."""

_CONTOUR_Y_MERGE_RATIO = 0.5
"""When grouping contours into lines, two bounding boxes whose
vertical overlap exceeds this fraction of the smaller box's height
are considered part of the same text line."""


def segment_lines(
    image: Image.Image,
) -> List[Tuple[Image.Image, int]]:
    """Segment a preprocessed image into individual text lines.

    Args:
        image: A preprocessed PIL Image in RGB mode (already
            grayscale-looking, binarised, and deskewed — i.e. the
            output of :func:`ocr.preprocessing.preprocess_image`).

    Returns:
        A list of ``(cropped_line_image, x_offset)`` tuples sorted
        top-to-bottom.  *cropped_line_image* is an RGB PIL Image
        containing one line of text.  *x_offset* is the horizontal
        pixel offset of the leftmost ink in that line relative to the
        full image, useful for reconstructing indentation later.

        If the image is blank or cannot be segmented, a single-element
        list containing the original image with ``x_offset=0`` is
        returned.

    Raises:
        ValueError: If *image* is ``None``.
    """
    if image is None:
        raise ValueError("Cannot segment lines from a None image")

    width, height = image.size
    logger.debug(
        "segment_lines called — image size %dx%d, mode %s",
        width, height, image.mode,
    )

    # Very small images are unlikely to contain multiple lines.
    if width < 4 or height < 4:
        logger.debug(
            "Image too small (%dx%d) for line segmentation — "
            "returning as single line",
            width, height,
        )
        return [(image, 0)]

    # ── Convert to inverted grayscale (text = white) ────────────
    gray = np.array(image.convert("L"), dtype=np.uint8)
    inverted = cv2.bitwise_not(gray)
    logger.debug("Converted to inverted grayscale — shape %s", inverted.shape)

    # ── Primary: horizontal projection profile ──────────────────
    line_regions = _projection_profile_lines(inverted)
    logger.debug(
        "Projection profile detected %d candidate region(s)",
        len(line_regions),
    )

    # ── Fallback: contour-based detection ───────────────────────
    # If the projection profile found fewer than 2 lines but the
    # image is tall enough to plausibly contain multiple lines,
    # try the contour-based approach.
    if len(line_regions) < 2 and height > _MIN_LINE_HEIGHT_PX * 3:
        logger.debug(
            "Projection profile found <%d lines on a %d-px tall image "
            "— trying contour-based fallback",
            2, height,
        )
        contour_regions = _contour_based_lines(inverted)
        if len(contour_regions) > len(line_regions):
            logger.debug(
                "Contour fallback produced %d lines (better than %d)",
                len(contour_regions), len(line_regions),
            )
            line_regions = contour_regions

    # ── If still nothing useful, return the whole image ─────────
    if not line_regions:
        logger.info(
            "No text lines detected — returning original image as "
            "single line"
        )
        return [(image, 0)]

    # ── Crop and compute x_offsets ──────────────────────────────
    results: List[Tuple[Image.Image, int]] = []
    for y_start, y_end in line_regions:
        cropped, x_offset = _crop_line(image, inverted, y_start, y_end)
        results.append((cropped, x_offset))
        logger.debug(
            "  Line y=[%d, %d] → crop size %s, x_offset=%d",
            y_start, y_end, cropped.size, x_offset,
        )

    logger.info("Segmented image into %d text line(s)", len(results))
    return results


# ── Primary strategy: horizontal projection ─────────────────────────


def _projection_profile_lines(
    inverted: np.ndarray,
) -> List[Tuple[int, int]]:
    """Detect line regions via horizontal projection profile.

    Sums pixel intensities along each row to build a 1-D profile.
    Consecutive rows whose summed value exceeds a threshold are
    grouped into line regions.

    Args:
        inverted: Inverted grayscale image (text pixels are bright).

    Returns:
        List of ``(y_start, y_end)`` tuples for each detected line,
        sorted top-to-bottom.
    """
    h, w = inverted.shape
    profile = inverted.astype(np.int64).sum(axis=1)  # shape (h,)
    max_val = profile.max()

    if max_val == 0:
        logger.debug("Projection profile is all-zero — blank image")
        return []

    threshold = int(max_val * _PROFILE_THRESHOLD_RATIO)
    logger.debug(
        "Projection profile — max=%d, threshold=%d", max_val, threshold
    )

    # Walk through rows and collect contiguous above-threshold spans.
    regions: List[Tuple[int, int]] = []
    in_line = False
    start = 0
    for y in range(h):
        if profile[y] > threshold:
            if not in_line:
                start = y
                in_line = True
        else:
            if in_line:
                regions.append((start, y - 1))
                in_line = False
    # Close a region that runs to the bottom edge
    if in_line:
        regions.append((start, h - 1))

    # Filter out tiny noise slivers
    regions = [
        (s, e) for s, e in regions if (e - s + 1) >= _MIN_LINE_HEIGHT_PX
    ]
    return regions


# ── Fallback strategy: contour-based grouping ───────────────────────


def _contour_based_lines(
    inverted: np.ndarray,
) -> List[Tuple[int, int]]:
    """Detect line regions by grouping contour bounding boxes.

    Finds external contours, extracts their bounding rectangles,
    merges boxes that overlap vertically (i.e. belong to the same
    text line), and returns the merged regions sorted top-to-bottom.

    Args:
        inverted: Inverted grayscale image (text pixels are bright).

    Returns:
        List of ``(y_start, y_end)`` tuples for each detected line.
    """
    # Binarize so findContours works reliably.
    _, binary = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        logger.debug("Contour fallback found no contours")
        return []

    # Extract bounding rects and filter by minimum height.
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= _MIN_LINE_HEIGHT_PX:
            boxes.append((x, y, w, h))

    if not boxes:
        logger.debug("Contour fallback — all bounding boxes below min height")
        return []

    logger.debug(
        "Contour fallback — %d bounding boxes after height filter",
        len(boxes),
    )

    # Sort by Y coordinate then merge overlapping boxes into lines.
    boxes.sort(key=lambda b: b[1])
    merged = _merge_boxes_into_lines(boxes)
    return merged


def _merge_boxes_into_lines(
    boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int]]:
    """Merge bounding boxes that belong to the same text line.

    Two boxes are merged when their vertical extents overlap by more
    than :data:`_CONTOUR_Y_MERGE_RATIO` of the smaller box's height.

    Args:
        boxes: Bounding rectangles ``(x, y, w, h)`` sorted by *y*.

    Returns:
        Merged ``(y_start, y_end)`` line regions sorted top-to-bottom.
    """
    if not boxes:
        return []

    lines: List[Tuple[int, int]] = []
    cur_y_start = boxes[0][1]
    cur_y_end = boxes[0][1] + boxes[0][3] - 1

    for _, y, _, h in boxes[1:]:
        box_y_end = y + h - 1
        overlap = max(0, min(cur_y_end, box_y_end) - max(cur_y_start, y) + 1)
        smaller_h = min(cur_y_end - cur_y_start + 1, h)

        if smaller_h > 0 and overlap / smaller_h >= _CONTOUR_Y_MERGE_RATIO:
            # Belongs to the same line — extend the region.
            cur_y_end = max(cur_y_end, box_y_end)
        else:
            # New line starts.
            lines.append((cur_y_start, cur_y_end))
            cur_y_start = y
            cur_y_end = box_y_end

    lines.append((cur_y_start, cur_y_end))

    # Final height filter
    lines = [
        (s, e) for s, e in lines if (e - s + 1) >= _MIN_LINE_HEIGHT_PX
    ]
    lines.sort(key=lambda r: r[0])
    return lines


# ── Shared helpers ──────────────────────────────────────────────────


def _crop_line(
    rgb_image: Image.Image,
    inverted: np.ndarray,
    y_start: int,
    y_end: int,
) -> Tuple[Image.Image, int]:
    """Crop a single line from the original RGB image.

    Also computes the horizontal offset of the leftmost ink pixel
    in the line region.

    Args:
        rgb_image: The original full RGB PIL Image.
        inverted: Inverted grayscale NumPy array of the same image.
        y_start: Top row of the line region (inclusive).
        y_end:   Bottom row of the line region (inclusive).

    Returns:
        ``(cropped_pil, x_offset)`` where *cropped_pil* is the RGB
        crop and *x_offset* is the pixel column of the leftmost ink.
    """
    width = rgb_image.size[0]

    # Compute x_offset from the inverted (text-bright) slice.
    line_slice = inverted[y_start : y_end + 1, :]
    col_profile = line_slice.sum(axis=0)
    non_zero_cols = np.nonzero(col_profile)[0]

    if len(non_zero_cols) > 0:
        x_offset = int(non_zero_cols[0])
    else:
        x_offset = 0
        logger.debug(
            "Line y=[%d, %d] has no ink pixels — x_offset defaulting to 0",
            y_start, y_end,
        )

    # Crop the full width so TrOCR sees the line in its entirety.
    cropped = rgb_image.crop((0, y_start, width, y_end + 1))
    return cropped, x_offset
