"""
TrOCR inference engine for handwritten code recognition.

Wraps a ``microsoft/trocr-*-handwritten`` model behind a thread-safe
singleton so the heavy model is loaded only once and reused across
requests.

Model selection:
    Set the ``HCE_OCR_MODEL`` environment variable to ``light`` to
    use the smaller ``trocr-small-handwritten`` variant (~130 MB).
    The default is ``trocr-base-handwritten`` (~900 MB).

Multi-line support:
    :meth:`TrOCREngine.recognize_lines` accepts segmented line images
    (as produced by :func:`ocr.line_segmentation.segment_lines`) and
    reconstructs indented, multi-line source code.
"""

import logging
import os
import threading
from typing import List, Tuple

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

# ── Dynamic model selection ─────────────────────────────────────────

_MODEL_VARIANTS = {
    "base": {
        "name": "microsoft/trocr-base-handwritten",
        "size_mb": 900,
    },
    "light": {
        "name": "microsoft/trocr-small-handwritten",
        "size_mb": 130,
    },
}

_model_key = os.environ.get("HCE_OCR_MODEL", "base").strip().lower()
if _model_key not in _MODEL_VARIANTS:
    logger.warning(
        "Unknown HCE_OCR_MODEL=%r — falling back to 'base'", _model_key
    )
    _model_key = "base"

_MODEL_NAME: str = _MODEL_VARIANTS[_model_key]["name"]
_MODEL_SIZE_MB: int = _MODEL_VARIANTS[_model_key]["size_mb"]

logger.info(
    "Selected TrOCR variant: %s (%s, ~%d MB)",
    _model_key, _MODEL_NAME, _MODEL_SIZE_MB,
)


class TrOCREngine:
    """Singleton wrapper around TrOCR for handwritten text recognition.

    Usage::

        engine = TrOCREngine.get_instance()
        text = engine.recognize(pil_image)

    The model and processor are loaded lazily on the first call to
    :meth:`get_instance`.  Subsequent calls return the same object.
    """

    # ── Class-level singleton state ─────────────────────────────────
    _instance: "TrOCREngine | None" = None
    _lock: threading.Lock = threading.Lock()

    # ── Construction ────────────────────────────────────────────────

    def __init__(self) -> None:
        """Load the TrOCR model and processor.

        **Do not call directly** — use :meth:`get_instance` instead so
        the expensive model load happens only once.
        """
        self._device: torch.device = self._select_device()
        self.processor: TrOCRProcessor = self._load_processor()
        self.model: VisionEncoderDecoderModel = self._load_model()

    # ── Public API ──────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "TrOCREngine":
        """Return the singleton :class:`TrOCREngine`, creating it if needed.

        Thread-safe via a class-level lock.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    logger.info("Creating TrOCREngine singleton …")
                    cls._instance = cls()
                    logger.info("TrOCREngine singleton ready")
        return cls._instance

    def recognize(self, image: Image.Image) -> str:
        """Run TrOCR inference on a preprocessed PIL Image.

        Args:
            image: A PIL Image (RGB mode expected) — typically the
                output of :func:`ocr.preprocessing.preprocess_image`.

        Returns:
            The recognised text as a single string.

        Raises:
            ValueError: If *image* is ``None``.
            RuntimeError: If inference fails for any reason.
        """
        if image is None:
            raise ValueError("Cannot recognise text from a None image")

        try:
            logger.debug(
                "Running TrOCR inference — image size %s, mode %s",
                image.size,
                image.mode,
            )

            # Ensure RGB — the processor expects 3-channel input.
            if image.mode != "RGB":
                image = image.convert("RGB")

            pixel_values = self.processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(self._device)

            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            text: str = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            logger.info(
                "TrOCR inference complete — recognised %d characters",
                len(text),
            )
            return text

        except Exception as exc:
            logger.exception("TrOCR inference failed")
            raise RuntimeError(
                f"TrOCR inference error: {exc}"
            ) from exc

    def recognize_lines(
        self,
        line_images: List[Tuple[Image.Image, int]],
    ) -> str:
        """Recognise text from multiple segmented line images.

        Runs :meth:`recognize` on each line image and reconstructs
        multi-line source code, using the per-line *x_offset* values
        to infer indentation.

        Indentation strategy (adaptive):
            * Collect all unique non-zero x_offsets.
            * Compute the smallest positive delta between consecutive
              unique offsets — this becomes the *base indent unit*.
            * Each line's offset is rounded to the nearest multiple
              of that unit (with a ±30 % tolerance band).
            * Every indent level maps to 4 spaces.
            * If all lines share the same offset (or there is only
              one line), no indentation is applied.

        Args:
            line_images: A list of ``(pil_image, x_offset)`` tuples
                as returned by
                :func:`ocr.line_segmentation.segment_lines`.

        Returns:
            Reconstructed multi-line source code as a single string.

        Raises:
            ValueError: If *line_images* is empty.
        """
        if not line_images:
            raise ValueError("line_images list is empty — nothing to recognise")

        logger.info(
            "recognize_lines called with %d line image(s)", len(line_images)
        )

        # ── OCR each line ───────────────────────────────────────
        raw_texts: List[str] = []
        offsets: List[int] = []
        for idx, (img, x_offset) in enumerate(line_images):
            text = self.recognize(img)
            raw_texts.append(text)
            offsets.append(x_offset)
            logger.debug(
                "  Line %d: x_offset=%d, text=%r", idx, x_offset, text
            )

        # ── Compute indentation ─────────────────────────────────
        indent_levels = _compute_indent_levels(offsets)

        # ── Assemble final output ───────────────────────────────
        output_lines: List[str] = []
        for text, level in zip(raw_texts, indent_levels):
            indented = ("    " * level) + text
            output_lines.append(indented)

        result = "\n".join(output_lines)
        logger.info(
            "recognize_lines complete — %d lines, %d total characters",
            len(output_lines), len(result),
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _select_device() -> torch.device:
        """Pick the best available compute device (CUDA → CPU)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("CUDA available — using GPU: %s", gpu_name)
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available — using CPU")
        return device

    def _load_processor(self) -> TrOCRProcessor:
        """Download / cache and instantiate the TrOCR processor."""
        try:
            logger.info(
                "Loading TrOCR processor: %s "
                "(estimated download ~%d MB if not cached)",
                _MODEL_NAME, _MODEL_SIZE_MB // 10,
            )
            processor = TrOCRProcessor.from_pretrained(_MODEL_NAME)
            logger.info("✓ TrOCR processor loaded successfully")
            return processor
        except Exception as exc:
            logger.exception("Failed to load TrOCR processor")
            raise RuntimeError(
                f"Could not load TrOCR processor '{_MODEL_NAME}': {exc}"
            ) from exc

    def _load_model(self) -> VisionEncoderDecoderModel:
        """Download / cache and instantiate the TrOCR model."""
        try:
            logger.info(
                "Loading TrOCR model: %s "
                "(estimated download ~%d MB if not cached)",
                _MODEL_NAME, _MODEL_SIZE_MB,
            )
            model = VisionEncoderDecoderModel.from_pretrained(_MODEL_NAME)
            logger.info("✓ TrOCR model weights downloaded / loaded from cache")
            model.to(self._device)
            model.eval()
            logger.info(
                "✓ TrOCR model ready on %s", self._device
            )
            return model
        except Exception as exc:
            logger.exception("Failed to load TrOCR model")
            raise RuntimeError(
                f"Could not load TrOCR model '{_MODEL_NAME}': {exc}"
            ) from exc


# ── Module-level helpers ────────────────────────────────────────────


def _compute_indent_levels(offsets: List[int]) -> List[int]:
    """Convert raw pixel offsets into integer indent levels.

    Uses an *adaptive* strategy:

    1. Find all unique offsets.
    2. If there is only one unique value, every line is at level 0.
    3. Otherwise, compute the minimum positive delta between
       consecutive unique offsets — this is the *base indent unit*.
    4. Each offset is divided by the base unit (with a ±30 %
       tolerance band for handwriting wobble) and rounded to the
       nearest integer to yield the indent level.

    Args:
        offsets: Per-line x_offset values (pixels).

    Returns:
        A list of non-negative integer indent levels, one per line.
    """
    unique_offsets = sorted(set(offsets))

    # Trivial case: all lines at the same horizontal position.
    if len(unique_offsets) <= 1:
        logger.debug(
            "All lines share the same x_offset (%s) — no indentation",
            unique_offsets,
        )
        return [0] * len(offsets)

    # Compute the smallest positive delta between consecutive unique
    # offsets — this is our best guess for one indent level in pixels.
    deltas = [
        unique_offsets[i + 1] - unique_offsets[i]
        for i in range(len(unique_offsets) - 1)
    ]
    base_unit = min(deltas)
    if base_unit <= 0:
        # Shouldn't happen with sorted unique values, but guard.
        logger.warning(
            "base_unit is %d — falling back to no indentation", base_unit
        )
        return [0] * len(offsets)

    logger.debug(
        "Adaptive indent — unique offsets=%s, deltas=%s, base_unit=%d px",
        unique_offsets, deltas, base_unit,
    )

    # Normalise: subtract the minimum offset so the leftmost line is
    # level 0, then divide by base_unit.
    min_offset = unique_offsets[0]
    tolerance = 0.30  # ±30 % wobble band
    levels: List[int] = []
    for off in offsets:
        relative = off - min_offset
        raw_level = relative / base_unit
        level = round(raw_level)
        # Clamp negative rounding artefacts to 0.
        level = max(level, 0)
        levels.append(level)

    logger.debug("Computed indent levels: %s", levels)
    return levels
