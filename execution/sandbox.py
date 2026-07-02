"""
Sandboxed code execution for the Handwritten Code Evaluator.

Executes user-submitted code inside a Docker container when available, with
a subprocess fallback for Python-only execution when Docker is not present.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from execution import sandbox_config

# =============================================================================
# SECURITY MODEL — READ THIS
# =============================================================================
# This module provides TWO tiers of sandboxing:
#
# **Tier 1 — Docker isolation (preferred)**
#   When the Docker daemon is available the submitted code runs inside a
#   disposable container with the following protections:
#     - Network disabled                 (--network none)
#     - Memory capped                    (--memory 256m / 512m for Java)
#     - CPU capped                       (--cpus 0.5)
#     - Read-only root filesystem        (--read-only)
#     - Writable tmpfs at /tmp/sandbox   (--tmpfs /tmp/sandbox:size=64m)
#     - Non-privileged sandbox user      (--user sandbox, UID 65534)
#     - Process limit                    (--pids-limit 64)
#     - All Linux capabilities dropped   (--cap-drop ALL)
#     - Privilege escalation blocked     (--security-opt no-new-privileges)
#     - Auto-removed after execution     (--rm / container.remove)
#     - Code file bind-mounted read-only
#
# **Tier 2 — Subprocess fallback (demo-level)**
#   When Docker is NOT available and ``REQUIRE_DOCKER`` is ``False``, the
#   code is executed in a subprocess.  This path supports **Python only**
#   and carries the same demo-level caveats as the previous implementation:
#     - No OS-level isolation (no cgroups, no namespaces on Win/macOS)
#     - No memory/CPU limits beyond a hard timeout
#     - Subprocess runs with the same user permissions as the Flask app
#     - On Linux, ``unshare --net`` is attempted for network isolation
#
# Set the ``REQUIRE_DOCKER`` environment variable to ``true`` to refuse
# execution when Docker is unavailable.
# =============================================================================

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Docker availability check (runs once at import time)
# ---------------------------------------------------------------------------
DOCKER_AVAILABLE = False
_docker_client = None

try:
    import docker  # type: ignore[import-untyped]

    _docker_client = docker.from_env()
    _docker_client.ping()
    DOCKER_AVAILABLE = True
    logger.info(
        "Docker daemon is reachable — container-based sandbox enabled "
        "(image: %s).",
        sandbox_config.DOCKER_IMAGE_NAME,
    )
except Exception:
    logger.info(
        "Docker is NOT available — %s",
        "execution will be refused (REQUIRE_DOCKER=true)"
        if sandbox_config.REQUIRE_DOCKER
        else "falling back to subprocess sandbox (Python only)",
    )

# ---------------------------------------------------------------------------
# Subprocess-fallback: network isolation on Linux
# ---------------------------------------------------------------------------
_IS_LINUX = platform.system() == "Linux"
_UNSHARE_AVAILABLE = False

if _IS_LINUX:
    try:
        result = subprocess.run(
            ["unshare", "--net", "true"],
            capture_output=True, timeout=5,
        )
        _UNSHARE_AVAILABLE = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _UNSHARE_AVAILABLE = False

NETWORK_ISOLATION_AVAILABLE = _UNSHARE_AVAILABLE


# =========================================================================
# Public API
# =========================================================================


def execute_code(code: str, language: str = "python") -> dict:
    """Execute user-submitted code in an isolated environment.

    When Docker is available the code runs inside a hardened container.
    Otherwise (and only when ``REQUIRE_DOCKER`` is ``False``), Python code
    is executed in a local subprocess.

    Args:
        code: Source code string to execute.
        language: Target language key (must exist in
            :data:`sandbox_config.LANGUAGE_CONFIG`).  Defaults to
            ``'python'``.

    Returns:
        A dictionary with the following keys:

        - ``stdout`` (str): Captured standard output (truncated to
          :data:`sandbox_config.MAX_OUTPUT_LENGTH` characters).
        - ``stderr`` (str): Captured standard error (truncated to
          :data:`sandbox_config.MAX_OUTPUT_LENGTH` characters).
        - ``status`` (str): One of ``'success'``, ``'error'``, or
          ``'timeout'``.
        - ``execution_time`` (float): Wall-clock time in seconds.
        - ``stdout_truncated`` (bool): ``True`` if stdout was truncated.
        - ``stderr_truncated`` (bool): ``True`` if stderr was truncated.
        - ``original_stdout_length`` (int): Length before truncation.
        - ``original_stderr_length`` (int): Length before truncation.
        - ``sandbox_mode`` (str): ``'docker'`` or ``'subprocess'``.
    """

    language = language.lower().strip()
    lang_cfg = sandbox_config.LANGUAGE_CONFIG.get(language)
    if lang_cfg is None:
        return _error_result(
            f"Unsupported language: {language!r}. "
            f"Supported: {', '.join(sandbox_config.LANGUAGE_CONFIG)}",
            sandbox_mode="none",
        )

    # ----- Docker path -----
    if DOCKER_AVAILABLE:
        return _execute_in_docker(code, language, lang_cfg)

    # ----- REQUIRE_DOCKER enforcement -----
    if sandbox_config.REQUIRE_DOCKER:
        logger.error(
            "Docker is required (REQUIRE_DOCKER=true) but not available. "
            "Refusing to execute code."
        )
        return _error_result(
            "Docker sandbox is required but not available. "
            "Set REQUIRE_DOCKER=false or start the Docker daemon.",
            sandbox_mode="none",
        )

    # ----- Subprocess fallback (Python only) -----
    if language != "python":
        return _error_result(
            f"Language {language!r} requires Docker for execution, "
            f"but Docker is not available.",
            sandbox_mode="none",
        )

    return _execute_in_subprocess(code)


# =========================================================================
# Docker execution
# =========================================================================


def _execute_in_docker(code: str, language: str, lang_cfg: dict) -> dict:
    """Execute *code* inside a Docker container with full isolation.

    Args:
        code: Source code to execute.
        language: Language key (e.g. ``'python'``, ``'java'``).
        lang_cfg: Language configuration dict from
            :data:`sandbox_config.LANGUAGE_CONFIG`.

    Returns:
        Execution result dictionary (see :func:`execute_code`).
    """

    temp_dir: str | None = None
    container = None
    start_time = time.time()

    try:
        # 1. Write the code to a temp file on the host.
        temp_dir = tempfile.mkdtemp(prefix="hce_sandbox_")
        code_filename = lang_cfg["filename"]
        host_code_path = os.path.join(temp_dir, code_filename)
        with open(host_code_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.debug(
            "Wrote %s code to %s (%d bytes)",
            language, host_code_path, len(code),
        )

        # 2. Build the command(s) to run inside the container.
        #    For compiled languages we chain compile + run via shell.
        compile_cmd = lang_cfg["compile_cmd"]
        run_cmd = lang_cfg["run_cmd"]

        if compile_cmd is not None:
            # Compile then run, joined with &&
            shell_cmd = " ".join(compile_cmd) + " && " + " ".join(run_cmd)
            container_cmd = f"/bin/sh -c '{shell_cmd}'"
        else:
            container_cmd = " ".join(run_cmd)

        timeout = lang_cfg["timeout"]
        memory_limit = lang_cfg["memory_limit"]
        container_path = f"/tmp/sandbox/{code_filename}"

        # 3. Create and start the container.
        logger.debug(
            "Starting Docker container (image=%s, mem=%s, timeout=%ds)",
            sandbox_config.DOCKER_IMAGE_NAME, memory_limit, timeout,
        )
        container = _docker_client.containers.run(
            image=sandbox_config.DOCKER_IMAGE_NAME,
            command=container_cmd,
            # --- Resource limits ---
            mem_limit=memory_limit,
            nano_cpus=int(sandbox_config.DEFAULT_CPU_LIMIT * 1e9),
            pids_limit=64,
            # --- Security hardening ---
            network_mode="none",
            read_only=True,
            user=sandbox_config.SANDBOX_USER,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            # --- Filesystem ---
            tmpfs={"/tmp/sandbox": f"size={sandbox_config.TMPFS_SIZE}"},
            volumes={
                host_code_path: {
                    "bind": container_path,
                    "mode": "ro",
                }
            },
            # --- Behaviour ---
            detach=True,
            stdout=True,
            stderr=True,
        )

        # 4. Wait for the container to finish (or timeout).
        exit_info = container.wait(timeout=timeout)
        exit_code = exit_info.get("StatusCode", -1)

        # 5. Collect output.
        raw_stdout = container.logs(stdout=True, stderr=False).decode(
            "utf-8", errors="replace"
        )
        raw_stderr = container.logs(stdout=False, stderr=True).decode(
            "utf-8", errors="replace"
        )

        original_stdout_length = len(raw_stdout)
        original_stderr_length = len(raw_stderr)
        stdout, stdout_truncated = _truncate_output(raw_stdout)
        stderr, stderr_truncated = _truncate_output(raw_stderr)

        status = "success" if exit_code == 0 else "error"
        if exit_code == 0:
            logger.info("Docker execution succeeded (%s).", language)
        else:
            logger.warning(
                "Docker execution finished with exit code %d (%s).",
                exit_code, language,
            )

        return _build_result(
            stdout=stdout,
            stderr=stderr,
            status=status,
            execution_time=time.time() - start_time,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            original_stdout_length=original_stdout_length,
            original_stderr_length=original_stderr_length,
            sandbox_mode="docker",
        )

    except Exception as exc:
        # Distinguish timeouts from other errors.
        is_timeout = "timed out" in str(exc).lower() or (
            hasattr(exc, "response")
            and getattr(exc, "response", None) is not None
        )

        if is_timeout:
            logger.warning(
                "Docker container timed out after %ds (%s).",
                lang_cfg["timeout"], language,
            )
            # Try to collect partial output before killing.
            stdout_raw, stderr_raw = _collect_container_logs(container)
            original_stdout_length = len(stdout_raw)
            original_stderr_length = len(stderr_raw)
            stdout, stdout_truncated = _truncate_output(stdout_raw)
            stderr, stderr_truncated = _truncate_output(stderr_raw)
            return _build_result(
                stdout=stdout,
                stderr=stderr,
                status="timeout",
                execution_time=time.time() - start_time,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                original_stdout_length=original_stdout_length,
                original_stderr_length=original_stderr_length,
                sandbox_mode="docker",
            )

        logger.exception(
            "Unexpected error during Docker execution (%s).", language
        )
        error_msg = f"Docker sandbox error: {exc}"
        return _error_result(error_msg, sandbox_mode="docker",
                             execution_time=time.time() - start_time)

    finally:
        # 6. Always clean up the container and temp directory.
        _remove_container(container)
        if temp_dir is not None and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temp directory: %s", temp_dir)
            except OSError:
                logger.exception(
                    "Failed to clean up temp directory: %s", temp_dir
                )


def _collect_container_logs(container) -> tuple[str, str]:
    """Safely collect logs from a container (may be ``None``)."""
    if container is None:
        return "", ""
    try:
        stdout = container.logs(stdout=True, stderr=False).decode(
            "utf-8", errors="replace"
        )
        stderr = container.logs(stdout=False, stderr=True).decode(
            "utf-8", errors="replace"
        )
        return stdout, stderr
    except Exception:
        logger.debug("Could not collect container logs.", exc_info=True)
        return "", ""


def _remove_container(container) -> None:
    """Force-kill and remove the container if it still exists."""
    if container is None:
        return
    try:
        container.kill()
    except Exception:
        pass  # Already stopped — fine.
    try:
        container.remove(force=True)
    except Exception:
        logger.debug("Could not remove container.", exc_info=True)


# =========================================================================
# Subprocess fallback (Python only)
# =========================================================================


def _execute_in_subprocess(code: str) -> dict:
    """Execute Python *code* in a local subprocess (demo-level sandbox).

    This fallback is used only when Docker is unavailable and
    ``REQUIRE_DOCKER`` is ``False``.  A warning is logged on every
    invocation because the subprocess has limited isolation.

    Args:
        code: Python source code to execute.

    Returns:
        Execution result dictionary (see :func:`execute_code`).
    """

    logger.warning(
        "Docker is NOT available — executing Python code in a subprocess "
        "WITHOUT container isolation.  This is suitable for local "
        "development only."
    )

    temp_dir: str | None = None
    start_time = time.time()
    stdout_truncated = False
    stderr_truncated = False
    original_stdout_length = 0
    original_stderr_length = 0

    try:
        # 1. Create a fresh, isolated temporary directory.
        temp_dir = tempfile.mkdtemp(prefix="hce_sandbox_")
        logger.debug("Created temp directory: %s", temp_dir)

        # 2. Write the user code to a .py file inside the temp directory.
        script_path = os.path.join(temp_dir, "user_script.py")
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(code)
        logger.debug("Wrote user script to: %s", script_path)

        # 3. Build the command — use unshare on Linux if available.
        cmd: list[str] = [sys.executable, script_path]
        if NETWORK_ISOLATION_AVAILABLE:
            cmd = ["unshare", "--net"] + cmd

        # 4. Execute the script in a subprocess — NEVER eval()/exec().
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=sandbox_config.DEFAULT_TIMEOUT,
            cwd=temp_dir,
        )

        # 5. Capture stdout and stderr with truncation metadata.
        stdout = result.stdout
        stderr = result.stderr
        original_stdout_length = len(stdout)
        original_stderr_length = len(stderr)
        stdout, stdout_truncated = _truncate_output(stdout)
        stderr, stderr_truncated = _truncate_output(stderr)

        # Determine status from return code.
        if result.returncode == 0:
            status = "success"
            logger.info("Code executed successfully (subprocess).")
        else:
            status = "error"
            logger.warning(
                "Code exited with return code %d (subprocess).",
                result.returncode,
            )

    except subprocess.TimeoutExpired as exc:
        # 6. On timeout: kill the process and report.
        logger.warning(
            "Code execution timed out after %ds (subprocess).",
            sandbox_config.DEFAULT_TIMEOUT,
        )
        raw_stdout = (
            exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        )
        raw_stderr = (
            exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        )
        original_stdout_length = len(raw_stdout)
        original_stderr_length = len(raw_stderr)
        stdout, stdout_truncated = _truncate_output(raw_stdout)
        stderr, stderr_truncated = _truncate_output(raw_stderr)
        status = "timeout"

    except Exception:
        # 7. On any other exception: report as error.
        logger.exception("Unexpected error during subprocess execution.")
        stdout = ""
        error_msg = f"Internal sandbox error: {sys.exc_info()[1]}"
        original_stdout_length = 0
        original_stderr_length = len(error_msg)
        stderr, stderr_truncated = _truncate_output(error_msg)
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

    execution_time = time.time() - start_time

    return _build_result(
        stdout=stdout,
        stderr=stderr,
        status=status,
        execution_time=execution_time,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        original_stdout_length=original_stdout_length,
        original_stderr_length=original_stderr_length,
        sandbox_mode="subprocess",
    )


# =========================================================================
# Helpers
# =========================================================================


def _truncate_output(text: str) -> tuple[str, bool]:
    """Truncate *text* to :data:`sandbox_config.MAX_OUTPUT_LENGTH` characters.

    Returns a tuple of ``(text, was_truncated)``.  If the text exceeds the
    limit, it is cut and the marker ``'[...output truncated]'`` is appended.
    """
    if len(text) > sandbox_config.MAX_OUTPUT_LENGTH:
        return text[: sandbox_config.MAX_OUTPUT_LENGTH] + "[...output truncated]", True
    return text, False


def _build_result(
    *,
    stdout: str,
    stderr: str,
    status: str,
    execution_time: float,
    stdout_truncated: bool,
    stderr_truncated: bool,
    original_stdout_length: int,
    original_stderr_length: int,
    sandbox_mode: str,
) -> dict:
    """Construct the standardised execution-result dictionary."""
    return {
        "stdout": stdout,
        "stderr": stderr,
        "status": status,
        "execution_time": round(execution_time, 4),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "original_stdout_length": original_stdout_length,
        "original_stderr_length": original_stderr_length,
        "sandbox_mode": sandbox_mode,
    }


def _error_result(
    message: str,
    *,
    sandbox_mode: str,
    execution_time: float = 0.0,
) -> dict:
    """Shorthand for returning an error result without execution."""
    return _build_result(
        stdout="",
        stderr=message,
        status="error",
        execution_time=execution_time,
        stdout_truncated=False,
        stderr_truncated=False,
        original_stdout_length=0,
        original_stderr_length=len(message),
        sandbox_mode=sandbox_mode,
    )
