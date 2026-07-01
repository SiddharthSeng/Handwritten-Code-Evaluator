"""
TrOCR inference engine for handwritten code recognition.

Wraps the ``microsoft/trocr-base-handwritten`` model behind a
thread-safe singleton so the heavy model is loaded only once and
reused across requests.
"""

import logging
import threading

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

_MODEL_NAME = "microsoft/trocr-base-handwritten"


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
            logger.info("Loading TrOCR processor: %s", _MODEL_NAME)
            processor = TrOCRProcessor.from_pretrained(_MODEL_NAME)
            logger.info("TrOCR processor loaded successfully")
            return processor
        except Exception as exc:
            logger.exception("Failed to load TrOCR processor")
            raise RuntimeError(
                f"Could not load TrOCR processor '{_MODEL_NAME}': {exc}"
            ) from exc

    def _load_model(self) -> VisionEncoderDecoderModel:
        """Download / cache and instantiate the TrOCR model."""
        try:
            logger.info("Loading TrOCR model: %s", _MODEL_NAME)
            model = VisionEncoderDecoderModel.from_pretrained(_MODEL_NAME)
            model.to(self._device)
            model.eval()
            logger.info(
                "TrOCR model loaded successfully on %s", self._device
            )
            return model
        except Exception as exc:
            logger.exception("Failed to load TrOCR model")
            raise RuntimeError(
                f"Could not load TrOCR model '{_MODEL_NAME}': {exc}"
            ) from exc
