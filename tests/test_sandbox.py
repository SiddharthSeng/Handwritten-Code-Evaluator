"""Unit tests for the sandboxed code execution module."""

import os
import tempfile
import unittest
from unittest.mock import patch

from execution.sandbox import execute_code, DOCKER_AVAILABLE


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
        self.assertFalse(result["stdout_truncated"])
        self.assertFalse(result["stderr_truncated"])

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
        self.assertTrue(result["stdout_truncated"])
        self.assertEqual(result["original_stdout_length"], 20_001)  # 20000 x's + newline

    # ------------------------------------------------------------------
    # Temp directory cleanup (subprocess mode)
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

    # ------------------------------------------------------------------
    # Truncation metadata
    # ------------------------------------------------------------------

    def test_no_truncation_metadata(self):
        """Short output should report no truncation."""
        result = execute_code("print('short')")

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["stdout_truncated"])
        self.assertFalse(result["stderr_truncated"])
        self.assertEqual(result["original_stdout_length"], len(result["stdout"]))

    # ------------------------------------------------------------------
    # sandbox_mode field
    # ------------------------------------------------------------------

    def test_sandbox_mode_reported(self):
        """The result should include a 'sandbox_mode' field."""
        result = execute_code("print('mode test')")

        self.assertIn("sandbox_mode", result)
        self.assertIn(result["sandbox_mode"], ("docker", "subprocess"))

    # ------------------------------------------------------------------
    # Language parameter
    # ------------------------------------------------------------------

    def test_default_language_is_python(self):
        """Calling without language should default to Python."""
        result = execute_code("print('default')")
        self.assertEqual(result["status"], "success")
        self.assertIn("default", result["stdout"])

    def test_unsupported_language(self):
        """An unsupported language should return an error."""
        result = execute_code("echo hello", language="bash")
        self.assertEqual(result["status"], "error")
        self.assertIn("Unsupported language", result["stderr"])

    # ------------------------------------------------------------------
    # Multi-language tests (require Docker or skip)
    # ------------------------------------------------------------------

    @unittest.skipUnless(DOCKER_AVAILABLE, "Docker not available")
    def test_javascript_execution(self):
        """JavaScript code should execute via Docker."""
        result = execute_code("console.log('hello from js');", language="javascript")
        self.assertEqual(result["status"], "success")
        self.assertIn("hello from js", result["stdout"])
        self.assertEqual(result["sandbox_mode"], "docker")

    @unittest.skipUnless(DOCKER_AVAILABLE, "Docker not available")
    def test_java_execution(self):
        """Java code should compile and execute via Docker."""
        code = """public class Main {
    public static void main(String[] args) {
        System.out.println("hello from java");
    }
}"""
        result = execute_code(code, language="java")
        self.assertEqual(result["status"], "success")
        self.assertIn("hello from java", result["stdout"])
        self.assertEqual(result["sandbox_mode"], "docker")

    @unittest.skipUnless(DOCKER_AVAILABLE, "Docker not available")
    def test_cpp_execution(self):
        """C++ code should compile and execute via Docker."""
        code = """#include <iostream>
int main() {
    std::cout << "hello from cpp" << std::endl;
    return 0;
}"""
        result = execute_code(code, language="cpp")
        self.assertEqual(result["status"], "success")
        self.assertIn("hello from cpp", result["stdout"])
        self.assertEqual(result["sandbox_mode"], "docker")

    def test_non_python_without_docker_fails_gracefully(self):
        """Non-Python languages without Docker should fail gracefully."""
        if DOCKER_AVAILABLE:
            self.skipTest("Docker is available; testing subprocess fallback not applicable")
        result = execute_code("console.log('test');", language="javascript")
        self.assertEqual(result["status"], "error")
        self.assertIn("Docker", result["stderr"])


if __name__ == "__main__":
    unittest.main()
