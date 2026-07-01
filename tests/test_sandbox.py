"""Unit tests for the sandboxed code execution module."""

import os
import tempfile
import unittest
from unittest.mock import patch

from execution.sandbox import execute_code


class TestSandboxExecution(unittest.TestCase):
    """Tests for :func:`execution.sandbox.execute_code`."""

    # ------------------------------------------------------------------
    # Basic execution
    # ------------------------------------------------------------------

    def test_simple_execution(self):
        """Simple print statement should succeed and appear in stdout."""
        result = execute_code("print('hello world')")

        self.assertEqual(result["status"], "success")
        self.assertIn("hello world", result["stdout"])
        self.assertIsInstance(result["execution_time"], float)

    # ------------------------------------------------------------------
    # Timeout enforcement
    # ------------------------------------------------------------------

    def test_timeout_enforcement(self):
        """Long-running code must be killed and reported as timeout."""
        result = execute_code("import time; time.sleep(30)")

        self.assertEqual(result["status"], "timeout")
        # Even with generous margin the wall-clock time should be well
        # under the sleep duration.
        self.assertLess(result["execution_time"], 15)

    # ------------------------------------------------------------------
    # stderr capture
    # ------------------------------------------------------------------

    def test_stderr_capture(self):
        """Invalid Python syntax should produce stderr and error status."""
        result = execute_code("if if if")  # clearly invalid syntax

        self.assertEqual(result["status"], "error")
        self.assertTrue(len(result["stderr"]) > 0)

    # ------------------------------------------------------------------
    # Output truncation
    # ------------------------------------------------------------------

    def test_output_truncation(self):
        """Output exceeding MAX_OUTPUT_LENGTH must be truncated."""
        result = execute_code("print('x' * 20000)")

        self.assertEqual(result["status"], "success")
        # 10 000 chars + len('[...output truncated]') = 10 023
        self.assertLessEqual(len(result["stdout"]), 10_100)
        self.assertIn("[...output truncated]", result["stdout"])

    # ------------------------------------------------------------------
    # Temp directory cleanup
    # ------------------------------------------------------------------

    def test_temp_directory_cleanup(self):
        """The temporary directory must be removed after execution."""
        created_dirs: list[str] = []

        original_mkdtemp = tempfile.mkdtemp

        def _tracking_mkdtemp(**kwargs):
            """Wrapper that records which temp dirs are created."""
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        with patch("execution.sandbox.tempfile.mkdtemp", side_effect=_tracking_mkdtemp):
            execute_code("print('cleanup test')")

        # At least one temp directory should have been created.
        self.assertGreater(len(created_dirs), 0)

        # …and all of them should have been cleaned up.
        for d in created_dirs:
            self.assertFalse(
                os.path.exists(d),
                f"Temp directory was not cleaned up: {d}",
            )

    # ------------------------------------------------------------------
    # Runtime error capture
    # ------------------------------------------------------------------

    def test_runtime_error_capture(self):
        """A ZeroDivisionError should be surfaced in stderr."""
        result = execute_code("1/0")

        self.assertEqual(result["status"], "error")
        self.assertIn("ZeroDivisionError", result["stderr"])


if __name__ == "__main__":
    unittest.main()
