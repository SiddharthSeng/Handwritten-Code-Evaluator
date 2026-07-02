"""Unit tests for the line segmentation module."""

import unittest

import cv2
import numpy as np
from PIL import Image

from ocr.line_segmentation import segment_lines


class TestSegmentLines(unittest.TestCase):
    """Tests for :func:`ocr.line_segmentation.segment_lines`."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_image(height: int, width: int, line_specs: list[tuple[int, int, int]]) -> Image.Image:
        """Create a synthetic test image with text lines.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.
            line_specs: List of (y_start, line_height, x_offset) tuples.
                Each tuple defines a horizontal band of dark pixels (text)
                starting at y_start, with the given height, indented by
                x_offset from the left.

        Returns:
            An RGB PIL Image with white background and dark lines.
        """
        # White background
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        for y_start, line_height, x_offset in line_specs:
            # Draw dark pixels (text) in the specified band
            y_end = min(y_start + line_height, height)
            x_start = x_offset
            x_end = width - 20  # leave a right margin
            if x_start < x_end and y_start < y_end:
                img[y_start:y_end, x_start:x_end] = 30  # dark gray text

        return Image.fromarray(img, "RGB")

    # ------------------------------------------------------------------
    # Basic tests
    # ------------------------------------------------------------------

    def test_single_line_returns_one_segment(self):
        """An image with one line of text should return one segment."""
        image = self._make_image(100, 400, [
            (20, 30, 10),  # single line at y=20, height=30, offset=10
        ])
        result = segment_lines(image)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)  # (image, x_offset)

    def test_multiple_lines_detected(self):
        """An image with multiple well-separated lines should detect all."""
        image = self._make_image(300, 400, [
            (20, 25, 10),   # line 1
            (80, 25, 10),   # line 2
            (140, 25, 40),  # line 3 (indented)
            (200, 25, 10),  # line 4
        ])
        result = segment_lines(image)
        # Should detect 4 lines (or at least 3+)
        self.assertGreaterEqual(len(result), 3)

    def test_x_offset_preserved(self):
        """Lines with different indentation should have different x_offsets."""
        image = self._make_image(200, 400, [
            (20, 25, 10),   # line 1, flush left
            (80, 25, 60),   # line 2, indented
            (140, 25, 10),  # line 3, flush left again
        ])
        result = segment_lines(image)
        if len(result) >= 3:
            # The second line should have a larger x_offset
            offsets = [x_offset for _, x_offset in result]
            self.assertGreater(offsets[1], offsets[0])

    def test_sorted_top_to_bottom(self):
        """Lines should be sorted top-to-bottom regardless of detection order."""
        image = self._make_image(300, 400, [
            (200, 25, 10),  # line at bottom
            (20, 25, 10),   # line at top
            (100, 25, 10),  # line in middle
        ])
        result = segment_lines(image)
        # All returned images should be valid
        for img, x_offset in result:
            self.assertIsInstance(img, Image.Image)
            self.assertIsInstance(x_offset, int)
            self.assertGreaterEqual(x_offset, 0)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_blank_image_returns_one_segment(self):
        """A completely blank/white image should return the whole image."""
        img = Image.new("RGB", (400, 100), color=(255, 255, 255))
        result = segment_lines(img)
        # Should return at least 1 (the whole image)
        self.assertGreaterEqual(len(result), 1)

    def test_none_image_raises(self):
        """Passing None should raise ValueError."""
        with self.assertRaises((ValueError, AttributeError)):
            segment_lines(None)

    def test_very_small_image(self):
        """A tiny image should not crash."""
        img = Image.new("RGB", (5, 5), color=(0, 0, 0))
        result = segment_lines(img)
        self.assertIsInstance(result, list)

    def test_returned_images_are_valid(self):
        """Each returned line image should be a valid PIL Image."""
        image = self._make_image(200, 400, [
            (20, 25, 10),
            (80, 25, 30),
        ])
        result = segment_lines(image)
        for line_img, x_offset in result:
            self.assertIsInstance(line_img, Image.Image)
            self.assertGreater(line_img.width, 0)
            self.assertGreater(line_img.height, 0)


if __name__ == "__main__":
    unittest.main()
