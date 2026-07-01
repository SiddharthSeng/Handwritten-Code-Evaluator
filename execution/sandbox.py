"""
Sandboxed code execution for the Handwritten Code Evaluator.

Executes user-submitted Python code in an isolated subprocess with
timeout enforcement and output capture.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

# =============================================================================
# SECURITY MODEL — READ THIS
# =============================================================================
# This module provides DEMO-LEVEL sandboxing for executing user-submitted code.
# It is suitable for a portfolio/resume project and local demonstrations.
#
# Protections in place:
#   - Code runs in a subprocess (not eval/exec in the main process)
#   - Hard 10-second timeout with process kill
#   - Execution in an isolated temporary directory
#   - stdout/stderr output size capped to prevent resource exhaustion
#
# Limitations (be honest):
#   - NO OS-level container isolation (no Docker, no cgroups, no namespaces)
#   - NO network access restrictions (the subprocess can make network calls)
#   - NO filesystem access restrictions beyond the working directory
#   - NO memory/CPU limits beyond the timeout
#   - The subprocess runs with the same user permissions as the Flask app
#
# For a production multi-tenant code execution service, you would need:
#   - Container isolation (Docker/gVisor/Firecracker)
#   - Network namespace isolation
#   - cgroup resource limits (memory, CPU, disk I/O)
#   - seccomp/AppArmor syscall filtering
#   - Read-only root filesystem
# =============================================================================

logger = logging.getLogger(__name__)

# Maximum number of characters allowed in stdout/stderr before truncation.
MAX_OUTPUT_LENGTH = 10_000

# Hard timeout in seconds for subprocess execution.
EXECUTION_TIMEOUT = 10


def execute_code(code: str) -> dict:
    """Execute Python code in a sandboxed subprocess.

    The code is written to a temporary file and executed as a completely
    separate Python process.  This function NEVER uses ``eval()`` or
    ``exec()`` — subprocess only.

    Args:
        code: A string containing the Python source code to execute.

    Returns:
        A dictionary with the following keys:

        - ``stdout`` (str): Captured standard output (truncated to
          :data:`MAX_OUTPUT_LENGTH` characters).
        - ``stderr`` (str): Captured standard error (truncated to
          :data:`MAX_OUTPUT_LENGTH` characters).
        - ``status`` (str): One of ``'success'``, ``'error'``, or
          ``'timeout'``.
        - ``execution_time`` (float): Wall-clock execution time in
          seconds.
    """

    temp_dir: str | None = None
    start_time = time.time()

    try:
        # 1. Create a fresh, isolated temporary directory.
        temp_dir = tempfile.mkdtemp(prefix="hce_sandbox_")
        logger.debug("Created temp directory: %s", temp_dir)

        # 2. Write the user code to a .py file inside the temp directory.
        script_path = os.path.join(temp_dir, "user_script.py")
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(code)
        logger.debug("Wrote user script to: %s", script_path)

        # 3. Execute the script in a subprocess — NEVER eval()/exec().
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
            cwd=temp_dir,
        )

        # 4. Capture stdout and stderr.
        stdout = result.stdout
        stderr = result.stderr

        # 5. Truncate output if it exceeds the cap.
        stdout = _truncate_output(stdout)
        stderr = _truncate_output(stderr)

        # Determine status from return code.
        if result.returncode == 0:
            status = "success"
            logger.info("Code executed successfully.")
        else:
            status = "error"
            logger.warning(
                "Code exited with return code %d.", result.returncode
            )

    except subprocess.TimeoutExpired as exc:
        # 6. On timeout: kill the process and report.
        logger.warning("Code execution timed out after %ds.", EXECUTION_TIMEOUT)

        # subprocess.run already kills the process on timeout, but we
        # capture any partial output that was available.
        stdout = _truncate_output(
            exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        )
        stderr = _truncate_output(
            exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        )
        status = "timeout"

    except Exception:
        # 7. On any other exception: report as error.
        logger.exception("Unexpected error during code execution.")
        stdout = ""
        stderr = _truncate_output(
            f"Internal sandbox error: {sys.exc_info()[1]}"
        )
        status = "error"

    finally:
        # 8. Always clean up the temp directory.
        if temp_dir is not None and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temp directory: %s", temp_dir)
            except OSError:
                logger.exception(
                    "Failed to clean up temp directory: %s", temp_dir
                )

    # 9. Record execution time.
    execution_time = time.time() - start_time

    return {
        "stdout": stdout,
        "stderr": stderr,
        "status": status,
        "execution_time": round(execution_time, 4),
    }


def _truncate_output(text: str) -> str:
    """Truncate *text* to :data:`MAX_OUTPUT_LENGTH` characters.

    If the text exceeds the limit, it is cut and the marker
    ``'[...output truncated]'`` is appended so callers know the output
    was incomplete.
    """
    if len(text) > MAX_OUTPUT_LENGTH:
        return text[:MAX_OUTPUT_LENGTH] + "[...output truncated]"
    return text
