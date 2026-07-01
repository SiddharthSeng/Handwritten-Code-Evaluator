"""
Handwritten Code Evaluator — Flask Application

Main entrypoint for the web application. Provides:
  - POST /evaluate  — image upload → OCR → syntax correction → sandboxed execution
  - GET  /health    — service health check
  - GET  /          — serves the frontend UI
"""

import logging
import os
import time
import uuid

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from ocr.preprocessing import preprocess_image
from ocr.trocr_engine import TrOCREngine
from ocr.postprocessing import correct_syntax
from execution.sandbox import execute_code

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("handwritten_evaluator")

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Max upload size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def _allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Initialize TrOCR Engine (loaded once at startup)
# ---------------------------------------------------------------------------
ocr_engine: TrOCREngine | None = None


def _get_ocr_engine() -> TrOCREngine:
    """Lazy-load the TrOCR engine singleton."""
    global ocr_engine
    if ocr_engine is None:
        logger.info("Loading TrOCR engine (first request)...")
        ocr_engine = TrOCREngine.get_instance()
        logger.info("TrOCR engine loaded successfully.")
    return ocr_engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the frontend UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns service status and whether GPU is active.
    """
    import torch

    engine = None
    model_loaded = False
    try:
        engine = _get_ocr_engine()
        model_loaded = True
    except Exception as e:
        logger.warning(f"Health check: model not loaded — {e}")

    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"

    return jsonify(
        {
            "status": "healthy",
            "gpu_available": gpu_available,
            "device": device,
            "model_loaded": model_loaded,
        }
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Main evaluation endpoint.

    Accepts an image upload of handwritten Python code, runs the full pipeline:
      image → preprocessing → TrOCR OCR → syntax correction → sandboxed execution

    Returns JSON with recognized text, corrected code, execution output, and metadata.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] New evaluation request received.")

    # ---- Validate upload ----
    if "image" not in request.files:
        logger.warning(f"[{request_id}] No image file in request.")
        return jsonify({"error": "No image file uploaded. Please include an 'image' field."}), 400

    file = request.files["image"]

    if file.filename == "":
        logger.warning(f"[{request_id}] Empty filename.")
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        logger.warning(f"[{request_id}] Invalid file type: {file.filename}")
        return (
            jsonify(
                {
                    "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                }
            ),
            400,
        )

    try:
        image_bytes = file.read()
        logger.info(
            f"[{request_id}] Image received: {file.filename} ({len(image_bytes)} bytes)"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Failed to read uploaded file: {e}")
        return jsonify({"error": "Failed to read uploaded file."}), 400

    # ---- Step 1: Preprocessing ----
    try:
        step_start = time.time()
        preprocessed_image = preprocess_image(image_bytes)
        preprocess_time = time.time() - step_start
        logger.info(
            f"[{request_id}] Preprocessing completed in {preprocess_time:.2f}s"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Preprocessing failed: {e}", exc_info=True)
        return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 500

    # ---- Step 2: OCR Inference ----
    try:
        step_start = time.time()
        engine = _get_ocr_engine()
        recognized_text = engine.recognize(preprocessed_image)
        ocr_time = time.time() - step_start
        logger.info(
            f"[{request_id}] OCR completed in {ocr_time:.2f}s — "
            f"recognized {len(recognized_text)} chars"
        )
    except Exception as e:
        logger.error(f"[{request_id}] OCR inference failed: {e}", exc_info=True)
        return jsonify({"error": f"OCR inference failed: {str(e)}"}), 500

    # ---- Step 3: Syntax Correction ----
    try:
        step_start = time.time()
        corrected_text, auto_corrected = correct_syntax(recognized_text)
        correction_time = time.time() - step_start
        logger.info(
            f"[{request_id}] Syntax correction completed in {correction_time:.2f}s — "
            f"auto_corrected={auto_corrected}"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Syntax correction failed: {e}", exc_info=True)
        # Fall back to raw OCR text
        corrected_text = recognized_text
        auto_corrected = False

    # ---- Step 4: Sandboxed Execution ----
    try:
        step_start = time.time()
        exec_result = execute_code(corrected_text)
        exec_time = time.time() - step_start
        logger.info(
            f"[{request_id}] Execution completed in {exec_time:.2f}s — "
            f"status={exec_result['status']}"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Code execution failed: {e}", exc_info=True)
        exec_result = {
            "stdout": "",
            "stderr": f"Execution setup failed: {str(e)}",
            "status": "error",
            "execution_time": 0.0,
        }

    # ---- Build Response ----
    total_time = time.time() - start_time

    response = {
        "request_id": request_id,
        "recognized_text": recognized_text,
        "corrected_text": corrected_text,
        "auto_corrected": auto_corrected,
        "stdout": exec_result["stdout"],
        "stderr": exec_result["stderr"],
        "execution_status": exec_result["status"],
        "processing_time_seconds": round(total_time, 2),
    }

    logger.info(
        f"[{request_id}] Request completed in {total_time:.2f}s — "
        f"status={exec_result['status']}"
    )

    return jsonify(response)


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------


@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeding the max upload limit."""
    logger.warning("Upload rejected: file exceeds 5 MB limit.")
    return (
        jsonify(
            {"error": "File too large. Maximum upload size is 5 MB."}
        ),
        413,
    )


@app.errorhandler(500)
def internal_error(e):
    """Handle unexpected internal errors."""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Handwritten Code Evaluator...")
    logger.info(f"Server will be available at http://localhost:5000")

    # Pre-load the OCR engine on startup (optional — comment out for faster startup)
    # _get_ocr_engine()

    app.run(host="0.0.0.0", port=5000, debug=False)
