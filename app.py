"""
Handwritten Code Evaluator — Flask Application

Main entrypoint for the web application. Provides:
  - POST /evaluate  — image upload → OCR → syntax correction → sandboxed execution
  - GET  /health    — service health check
  - GET  /warmup    — trigger model loading in the background
  - GET  /history   — last 20 evaluation results
  - GET  /history/<request_id> — specific evaluation result
  - GET  /          — serves the frontend UI
"""

import logging
import os
import sqlite3
import threading
import time
import uuid

from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS

from ocr.preprocessing import preprocess_image
from ocr.line_segmentation import segment_lines
from ocr.trocr_engine import TrOCREngine
from ocr.postprocessing import correct_syntax
from execution.sandbox import execute_code, DOCKER_AVAILABLE

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

SUPPORTED_LANGUAGES = {"python", "javascript", "java", "cpp"}

# ---------------------------------------------------------------------------
# Rate Limiting Configuration
# ---------------------------------------------------------------------------
# Max requests per IP per minute on /evaluate.  Adjust as needed.
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

# In-memory sliding window: {ip: [timestamp, ...]}
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()

# ---------------------------------------------------------------------------
# SQLite History Database
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluations.db")


def _get_db() -> sqlite3.Connection:
    """Get a per-request SQLite connection (stored in Flask ``g``)."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def _close_db(exc):
    """Close the database connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _init_db():
    """Create the evaluations table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            request_id       TEXT PRIMARY KEY,
            timestamp        TEXT NOT NULL DEFAULT (datetime('now')),
            execution_status TEXT,
            processing_time  REAL,
            recognized_text  TEXT,
            corrected_text   TEXT,
            stdout           TEXT,
            stderr           TEXT,
            auto_corrected   INTEGER,
            language         TEXT DEFAULT 'python',
            sandbox_mode     TEXT DEFAULT 'subprocess'
        )
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite history database initialized at %s", DATABASE_PATH)


# Initialize the database on import
_init_db()


def _log_evaluation(request_id: str, result: dict):
    """Insert an evaluation result into the SQLite history database."""
    try:
        db = _get_db()
        db.execute(
            """INSERT INTO evaluations
               (request_id, execution_status, processing_time,
                recognized_text, corrected_text, stdout, stderr,
                auto_corrected, language, sandbox_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                result.get("execution_status", ""),
                result.get("processing_time_seconds", 0),
                (result.get("recognized_text", "") or "")[:500],
                (result.get("corrected_text", "") or "")[:500],
                (result.get("stdout", "") or "")[:500],
                (result.get("stderr", "") or "")[:500],
                1 if result.get("auto_corrected") else 0,
                result.get("language", "python"),
                result.get("sandbox_mode", "subprocess"),
            ),
        )
        db.commit()
    except Exception as exc:
        logger.warning("Failed to log evaluation to SQLite: %s", exc)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

def _check_rate_limit(ip: str) -> bool:
    """Return True if the request should be allowed, False if rate-limited."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(ip, [])
        # Prune old entries
        timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_limit_store[ip] = timestamps
            return False

        timestamps.append(now)
        _rate_limit_store[ip] = timestamps
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Initialize TrOCR Engine (lazy-loaded on first request)
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
    Does NOT trigger model loading — just reports current state.
    """
    import torch

    model_loaded = ocr_engine is not None
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"

    response = {
        "status": "healthy",
        "gpu_available": gpu_available,
        "device": device,
        "model_loaded": model_loaded,
        "docker_available": DOCKER_AVAILABLE,
    }
    if not model_loaded:
        response["message"] = "Model will load on first /evaluate request. Use GET /warmup to pre-load."

    return jsonify(response)


@app.route("/warmup", methods=["GET"])
def warmup():
    """
    Trigger model loading in the background.
    Returns immediately and loads the model in a separate thread.
    Useful for pre-warming before a demo.
    """
    if ocr_engine is not None:
        return jsonify({"status": "already_loaded", "message": "Model is already loaded."})

    def _load_in_background():
        try:
            _get_ocr_engine()
            logger.info("Background model warmup complete.")
        except Exception as exc:
            logger.error("Background model warmup failed: %s", exc)

    thread = threading.Thread(target=_load_in_background, daemon=True)
    thread.start()

    return jsonify({
        "status": "warming_up",
        "message": "Model loading started in background. Check /health for status."
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Main evaluation endpoint.

    Accepts an image upload of handwritten code, runs the full pipeline:
      image → preprocessing → line segmentation → TrOCR OCR →
      syntax correction → sandboxed execution

    Returns JSON with recognized text, corrected code, execution output,
    diagnostics, and metadata.
    """
    # ---- Rate limiting ----
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        return jsonify({
            "error": "Rate limit exceeded. Maximum 10 requests per minute. Please try again shortly."
        }), 429

    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] New evaluation request received.")

    # ---- Validate language ----
    language = request.form.get("language", "python").lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": f"Unsupported language: {language}. "
                     f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        }), 400

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

    # ---- Step 2: Line Segmentation ----
    try:
        step_start = time.time()
        line_images = segment_lines(preprocessed_image)
        seg_time = time.time() - step_start
        logger.info(
            f"[{request_id}] Line segmentation completed in {seg_time:.2f}s — "
            f"detected {len(line_images)} line(s)"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Line segmentation failed: {e}", exc_info=True)
        # Fall back to single-image OCR
        line_images = [(preprocessed_image, 0)]
        logger.info(f"[{request_id}] Falling back to single-image OCR")

    # ---- Step 3: OCR Inference ----
    try:
        step_start = time.time()
        engine = _get_ocr_engine()

        if len(line_images) > 1:
            recognized_text = engine.recognize_lines(line_images)
        else:
            recognized_text = engine.recognize(line_images[0][0])

        ocr_time = time.time() - step_start
        logger.info(
            f"[{request_id}] OCR completed in {ocr_time:.2f}s — "
            f"recognized {len(recognized_text)} chars"
        )
    except Exception as e:
        logger.error(f"[{request_id}] OCR inference failed: {e}", exc_info=True)
        return jsonify({"error": f"OCR inference failed: {str(e)}"}), 500

    # ---- Step 4: Syntax Correction ----
    diagnostics = None
    try:
        step_start = time.time()
        corrected_text, auto_corrected, diagnostics = correct_syntax(
            recognized_text, language=language
        )
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

    # ---- Step 5: Sandboxed Execution ----
    try:
        step_start = time.time()
        exec_result = execute_code(corrected_text, language=language)
        exec_time = time.time() - step_start
        logger.info(
            f"[{request_id}] Execution completed in {exec_time:.2f}s — "
            f"status={exec_result['status']}, mode={exec_result.get('sandbox_mode', 'unknown')}"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Code execution failed: {e}", exc_info=True)
        exec_result = {
            "stdout": "",
            "stderr": f"Execution setup failed: {str(e)}",
            "status": "error",
            "execution_time": 0.0,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "original_stdout_length": 0,
            "original_stderr_length": 0,
            "sandbox_mode": "none",
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
        "stdout_truncated": exec_result.get("stdout_truncated", False),
        "stderr_truncated": exec_result.get("stderr_truncated", False),
        "original_stdout_length": exec_result.get("original_stdout_length", len(exec_result["stdout"])),
        "original_stderr_length": exec_result.get("original_stderr_length", len(exec_result["stderr"])),
        "language": language,
        "sandbox_mode": exec_result.get("sandbox_mode", "subprocess"),
        "diagnostics": diagnostics,
    }

    logger.info(
        f"[{request_id}] Request completed in {total_time:.2f}s — "
        f"status={exec_result['status']}"
    )

    # ---- Log to SQLite history ----
    _log_evaluation(request_id, response)

    return jsonify(response)


@app.route("/history", methods=["GET"])
def history():
    """Return the last 20 evaluation results as JSON."""
    try:
        db = _get_db()
        rows = db.execute(
            "SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
        results = [dict(row) for row in rows]
        return jsonify({"evaluations": results, "count": len(results)})
    except Exception as exc:
        logger.error("Failed to fetch history: %s", exc)
        return jsonify({"error": "Failed to fetch history."}), 500


@app.route("/history/<request_id>", methods=["GET"])
def history_detail(request_id):
    """Return a specific evaluation result by request_id."""
    try:
        db = _get_db()
        row = db.execute(
            "SELECT * FROM evaluations WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "Evaluation not found."}), 404
        return jsonify(dict(row))
    except Exception as exc:
        logger.error("Failed to fetch history detail: %s", exc)
        return jsonify({"error": "Failed to fetch history."}), 500


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


@app.errorhandler(429)
def rate_limit_exceeded(e):
    """Handle rate limit exceeded (fallback if raised by other middleware)."""
    return jsonify({"error": "Too many requests. Please try again later."}), 429


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
    logger.info("Server will be available at http://localhost:5000")
    logger.info("Docker sandbox available: %s", DOCKER_AVAILABLE)

    # Pre-load the OCR engine on startup (optional — comment out for faster startup)
    # _get_ocr_engine()

    app.run(host="0.0.0.0", port=5000, debug=False)
