# 📝 Handwritten Code Evaluator

A full-stack web application that takes an uploaded image of handwritten Python code, recognizes the text using Microsoft's TrOCR model (via HuggingFace Transformers), automatically corrects common OCR errors into syntactically valid Python, then executes the code in a sandboxed subprocess with strict resource limits, and returns the results — all through a clean web interface.

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│              │    │                  │    │                  │    │                  │    │                  │
│  Image       │───▶│  Preprocessing   │───▶│  TrOCR OCR       │───▶│  Syntax          │───▶│  Sandboxed       │
│  Upload      │    │  (Grayscale,     │    │  Inference       │    │  Correction      │    │  Execution       │
│  (PNG/JPG)   │    │  Binarize,       │    │  (microsoft/     │    │  (Indentation,   │    │  (subprocess,    │
│              │    │  Denoise,        │    │  trocr-base-     │    │  OCR fixes,      │    │  10s timeout,    │
│              │    │  Deskew)         │    │  handwritten)    │    │  ast.parse)      │    │  temp dir)       │
└──────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                                                                     │
                                                                                                     ▼
                                                                                            ┌──────────────────┐
                                                                                            │  JSON Response   │
                                                                                            │  (recognized,    │
                                                                                            │  corrected,      │
                                                                                            │  stdout/stderr,  │
                                                                                            │  status, time)   │
                                                                                            └──────────────────┘
```

### Flow Summary

1. **Image Upload** — User uploads a PNG/JPG image (max 5 MB) of handwritten Python code.
2. **Preprocessing** — The image is converted to grayscale, contrast-enhanced (CLAHE), binarized (adaptive threshold), denoised, and deskewed using OpenCV/Pillow.
3. **TrOCR Inference** — The preprocessed image is fed to Microsoft's `trocr-base-handwritten` model, which outputs recognized text.
4. **Syntax Correction** — The raw OCR text is cleaned up: indentation is normalized to 4 spaces, common OCR character confusions are fixed (`0`/`O`, `1`/`l`, `;`/`:`), and the result is validated with `ast.parse()`. If parsing fails, targeted regex fixes are retried up to 3 times.
5. **Sandboxed Execution** — The corrected Python code is written to a temporary file and executed via `subprocess.run()` with a 10-second timeout, in an isolated temp directory. stdout/stderr are captured and capped at 10,000 characters.
6. **JSON Response** — The API returns the recognized text, corrected code, execution output, status, and processing time.

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+ (tested with 3.14)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/SiddharthSeng/Handwritten-Code-Evaluator.git
cd Handwritten-Code-Evaluator

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will start on `http://localhost:5000`.

### GPU vs CPU

- **GPU (CUDA):** If you have an NVIDIA GPU with CUDA installed, TrOCR inference will automatically run on GPU. Expect ~0.5–1s per image.
- **CPU:** Without a GPU, inference runs on CPU. Expect ~2–5s per image. The app logs which device is being used on startup.

> **Note:** The TrOCR model checkpoint (`microsoft/trocr-base-handwritten`, ~1.3 GB) is downloaded automatically from HuggingFace on first run and cached locally.

---

## 📡 API Reference

### `POST /evaluate`

Upload an image of handwritten Python code for OCR recognition and execution.

**Request:**
```bash
curl -X POST http://localhost:5000/evaluate \
  -F "image=@handwritten_code.png"
```

**Response (200 OK):**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "recognized_text": "print  (\"Hello World\")",
  "corrected_text": "print(\"Hello World\")",
  "auto_corrected": true,
  "stdout": "Hello World\n",
  "stderr": "",
  "execution_status": "success",
  "processing_time_seconds": 3.42
}
```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `request_id` | string | Unique request identifier for logging |
| `recognized_text` | string | Raw OCR output from TrOCR |
| `corrected_text` | string | Syntax-corrected Python code |
| `auto_corrected` | boolean | `true` if code passed `ast.parse()`, `false` if corrections couldn't fully fix it |
| `stdout` | string | Standard output from code execution |
| `stderr` | string | Standard error from code execution |
| `execution_status` | string | `"success"`, `"error"`, or `"timeout"` |
| `processing_time_seconds` | float | Total processing time in seconds |

**Error Responses:**

| Status | Condition |
|---|---|
| `400` | No image uploaded, invalid file type, or file exceeds 5 MB |
| `500` | Internal server error (model loading failure, etc.) |

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": false,
  "device": "cpu",
  "model_loaded": true
}
```

---

## 🔒 Security Model

> **Honest disclaimer:** This is a portfolio/demo project, not a hardened multi-tenant code execution service. The security model is appropriate for local demonstrations and portfolio showcases, not for deploying on the public internet with untrusted users.

### Protections In Place

| Protection | Implementation |
|---|---|
| **No `eval()`/`exec()`** | User code is NEVER executed via `eval()` or `exec()`. All execution goes through `subprocess.run()`, isolating it from the Flask process. |
| **Hard timeout** | 10-second timeout on subprocess execution. Process is killed if exceeded. |
| **Isolated temp directory** | Each execution runs in a fresh `tempfile.mkdtemp()` directory, cleaned up afterward. |
| **Output size cap** | stdout/stderr are truncated to 10,000 characters to prevent resource exhaustion. |
| **Upload size limit** | 5 MB max upload size enforced by Flask. |

### What Is NOT Protected

| Limitation | Details |
|---|---|
| **No container isolation** | Code does not run inside Docker, gVisor, Firecracker, or any container. It runs as a regular subprocess. |
| **No network restrictions** | The subprocess can make network calls (HTTP requests, DNS lookups, etc.). OS-level network isolation (namespaces, firewall rules) is not implemented. |
| **No filesystem restrictions** | Beyond running from a temp directory, the subprocess can read files accessible to the app's user. |
| **No memory/CPU limits** | Only the timeout limits execution duration. There are no cgroup-based memory or CPU restrictions. |
| **Same user permissions** | The subprocess runs with the same OS user permissions as the Flask app. |

### For Production Use

A production-grade code execution service would require: container isolation (Docker/gVisor), network namespace isolation, cgroup resource limits, seccomp/AppArmor syscall filtering, read-only filesystems, and dedicated execution workers. This project is not that — it's a clean demonstration of the concept.

---

## ⚠️ Known Limitations

- **OCR accuracy depends on handwriting clarity.** TrOCR performs best on clear, reasonably neat handwriting. Very messy or stylized handwriting may produce poor results.
- **Syntax correction is heuristic-based.** The post-processing step uses pattern matching and common OCR confusion rules. It won't fix all errors — complex syntax mistakes or unusual code patterns may not be auto-corrected.
- **Single-line OCR.** TrOCR processes the image as a single text block. Multi-line code with complex indentation may not be perfectly captured.
- **Python only.** The evaluator is designed for Python code. Other languages are not supported.
- **Model download on first run.** The TrOCR model (~1.3 GB) is downloaded from HuggingFace on first launch. Subsequent runs use the cached model.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend** | Python, Flask |
| **OCR Model** | Microsoft TrOCR (`trocr-base-handwritten`) via HuggingFace Transformers |
| **Deep Learning** | PyTorch |
| **Image Processing** | Pillow, OpenCV |
| **Code Execution** | Python `subprocess` module |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **API** | RESTful JSON API with CORS support |

---

## 👤 Author

**Siddharth Senguttuvan**
B.Tech Computer Science (AI & ML)
Hindustan Institute of Technology and Science (HITS)

---

## 📄 License

This project is open source and available for educational and portfolio purposes.
