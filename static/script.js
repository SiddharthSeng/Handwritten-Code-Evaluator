/**
 * Handwritten Code Evaluator — Frontend Logic
 *
 * Handles:
 *   - Drag-and-drop & click-to-browse file selection
 *   - Client-side file validation (type + size)
 *   - Image preview via FileReader
 *   - POST to /evaluate and result rendering
 */

(function () {
    'use strict';

    // ── Constants ──────────────────────────────────────────────
    const ALLOWED_TYPES = ['image/png', 'image/jpeg'];
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB

    // ── DOM References ─────────────────────────────────────────
    const dropZone              = document.getElementById('drop-zone');
    const fileInput             = document.getElementById('file-input');
    const fileNameDisplay       = document.getElementById('file-name-display');
    const fileError             = document.getElementById('file-error');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview          = document.getElementById('image-preview');
    const evaluateBtn           = document.getElementById('evaluate-btn');
    const loadingOverlay        = document.getElementById('loading-overlay');
    const resultsSection        = document.getElementById('results-section');

    // Result elements
    const recognizedText  = document.getElementById('recognized-text');
    const correctedCode   = document.getElementById('corrected-code');
    const correctionBadge = document.getElementById('correction-badge');
    const stdoutPanel     = document.getElementById('stdout-panel');
    const stdoutContent   = document.getElementById('stdout-content');
    const stderrPanel     = document.getElementById('stderr-panel');
    const stderrContent   = document.getElementById('stderr-content');
    const statusBadge     = document.getElementById('status-badge');
    const executionTime   = document.getElementById('execution-time');

    /** The currently selected File object (or null). */
    let selectedFile = null;

    // ── Utilities ──────────────────────────────────────────────

    /**
     * Format bytes into a human-readable string.
     * @param {number} bytes
     * @returns {string}
     */
    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    /** Clear any previous file-validation error message. */
    function clearError() {
        fileError.textContent = '';
    }

    /**
     * Show a file-validation error and reset state.
     * @param {string} message
     */
    function showError(message) {
        fileError.textContent = message;
        selectedFile = null;
        evaluateBtn.disabled = true;
        fileNameDisplay.textContent = '';
        imagePreviewContainer.hidden = true;
    }

    /** Show the loading spinner overlay. */
    function showLoading() {
        loadingOverlay.classList.add('active');
        loadingOverlay.setAttribute('aria-hidden', 'false');
    }

    /** Hide the loading spinner overlay. */
    function hideLoading() {
        loadingOverlay.classList.remove('active');
        loadingOverlay.setAttribute('aria-hidden', 'true');
    }

    // ── File Handling ──────────────────────────────────────────

    /**
     * Validate, store, and preview a chosen file.
     * @param {File} file
     */
    function handleFile(file) {
        clearError();

        // Type check
        if (!ALLOWED_TYPES.includes(file.type)) {
            showError('Invalid file type. Please upload a PNG or JPEG image.');
            return;
        }

        // Size check
        if (file.size > MAX_FILE_SIZE) {
            showError(`File too large (${formatSize(file.size)}). Maximum allowed size is 5 MB.`);
            return;
        }

        selectedFile = file;
        evaluateBtn.disabled = false;
        fileNameDisplay.textContent = `${file.name}  (${formatSize(file.size)})`;

        // Preview with FileReader
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreviewContainer.hidden = false;
        };
        reader.readAsDataURL(file);
    }

    // ── Drag-and-Drop ──────────────────────────────────────────

    /** Prevent default behavior and stop propagation. */
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    dropZone.addEventListener('dragover', function (e) {
        preventDefaults(e);
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragenter', function (e) {
        preventDefaults(e);
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', function (e) {
        preventDefaults(e);
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', function (e) {
        preventDefaults(e);
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // ── Click-to-Browse ────────────────────────────────────────

    dropZone.addEventListener('click', function () {
        fileInput.click();
    });

    dropZone.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // ── Evaluate ───────────────────────────────────────────────

    evaluateBtn.addEventListener('click', async function () {
        clearError();

        if (!selectedFile) {
            showError('Please select an image first.');
            return;
        }

        showLoading();
        resultsSection.hidden = true;

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/evaluate', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                let errorMessage = `Server error (${response.status})`;
                try {
                    const errBody = await response.json();
                    if (errBody && errBody.error) {
                        errorMessage = errBody.error;
                    }
                } catch (_) {
                    // response wasn't JSON — keep the generic message
                }
                showError(errorMessage);
                return;
            }

            const data = await response.json();
            populateResults(data);

        } catch (err) {
            if (err.name === 'TypeError') {
                // Network failure (e.g. server down, CORS)
                showError('Network error — could not reach the server. Please try again.');
            } else {
                showError('An unexpected error occurred. Please try again.');
            }
            console.error('Evaluate error:', err);
        } finally {
            hideLoading();
        }
    });

    // ── Populate Results ───────────────────────────────────────

    /**
     * Fill in the results section from the API response JSON.
     *
     * Expected shape:
     * {
     *   recognized_text: string,
     *   corrected_code:  string,
     *   auto_corrected:  boolean,
     *   stdout:          string,
     *   stderr:          string,
     *   execution_status: 'success' | 'error' | 'timeout',
     *   processing_time_seconds: number
     * }
     */
    function populateResults(data) {
        // Recognized text
        recognizedText.textContent = data.recognized_text || '(no text recognized)';

        // Corrected code
        correctedCode.textContent = data.corrected_code || '';

        // Correction badge
        if (data.auto_corrected === true) {
            correctionBadge.textContent = 'Auto-corrected ✓';
            correctionBadge.className = 'badge badge--success';
        } else {
            correctionBadge.textContent = 'Partially corrected ⚠';
            correctionBadge.className = 'badge badge--warning';
        }

        // Stdout
        stdoutContent.textContent = data.stdout || '(no output)';

        // Stderr — hide panel entirely when empty
        if (data.stderr && data.stderr.trim().length > 0) {
            stderrContent.textContent = data.stderr;
            stderrPanel.hidden = false;
        } else {
            stderrPanel.hidden = true;
        }

        // Status badge
        switch (data.execution_status) {
            case 'success':
                statusBadge.textContent = 'Success ✓';
                statusBadge.className = 'badge badge--success';
                break;
            case 'timeout':
                statusBadge.textContent = 'Timeout ⏱';
                statusBadge.className = 'badge badge--timeout';
                break;
            case 'error':
            default:
                statusBadge.textContent = 'Error ✗';
                statusBadge.className = 'badge badge--error';
                break;
        }

        // Execution time
        if (typeof data.processing_time_seconds === 'number') {
            executionTime.textContent = `Processed in ${data.processing_time_seconds.toFixed(2)} s`;
        } else {
            executionTime.textContent = '';
        }

        // Reveal results with animation
        resultsSection.hidden = false;
        resultsSection.classList.remove('fade-in');
        // Force reflow so the animation replays
        void resultsSection.offsetWidth;
        resultsSection.classList.add('fade-in');
    }
})();
