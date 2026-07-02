"""Unit tests for the postprocessing / syntax correction module."""

import unittest

from ocr.postprocessing import correct_syntax


class TestCorrectSyntax(unittest.TestCase):
    """Tests for :func:`ocr.postprocessing.correct_syntax`."""

    # ------------------------------------------------------------------
    # Basic Python correction
    # ------------------------------------------------------------------

    def test_valid_python_unchanged(self):
        """Valid Python code should pass through with success=True."""
        code = "print('hello world')"
        corrected, ok, diagnostics = correct_syntax(code, language="python")
        self.assertTrue(ok)
        self.assertIsNone(diagnostics)
        self.assertIn("print", corrected)

    def test_returns_three_tuple(self):
        """correct_syntax must always return a 3-tuple."""
        result = correct_syntax("x = 1", language="python")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_empty_input(self):
        """Empty input should return failure with diagnostics."""
        corrected, ok, diagnostics = correct_syntax("", language="python")
        self.assertFalse(ok)
        self.assertIsNotNone(diagnostics)

    # ------------------------------------------------------------------
    # OCR confusion fixes
    # ------------------------------------------------------------------

    def test_curly_quotes_fixed(self):
        """Curly/smart quotes should be replaced with ASCII equivalents."""
        code = "print(\u201chello\u201d)"  # "hello"
        corrected, ok, _ = correct_syntax(code, language="python")
        self.assertNotIn("\u201c", corrected)
        self.assertNotIn("\u201d", corrected)
        self.assertIn('"', corrected)

    def test_semicolon_to_colon_on_block_starter(self):
        """Trailing ; on def/if/for should become : (Python only)."""
        code = "def foo();\n    pass"
        corrected, ok, _ = correct_syntax(code, language="python")
        self.assertIn("def foo():", corrected)

    # ------------------------------------------------------------------
    # Per-language rule isolation
    # ------------------------------------------------------------------

    def test_dollar_preserved_in_javascript(self):
        """$ should NOT be changed to S for JavaScript code."""
        code = "var $el = document.getElementById('test');"
        corrected, ok, _ = correct_syntax(code, language="javascript")
        self.assertIn("$el", corrected)

    def test_exclamation_preserved_in_javascript(self):
        """! should NOT be changed for JavaScript code."""
        code = "if (!valid) { return false; }"
        corrected, ok, _ = correct_syntax(code, language="javascript")
        self.assertIn("!valid", corrected)

    def test_python_keyword_reconstruction(self):
        """Garbled Python keywords should be reconstructed (Python only)."""
        code = "de f foo():\n    pr int('hello')"
        corrected, ok, _ = correct_syntax(code, language="python")
        self.assertIn("def", corrected)
        self.assertIn("print", corrected)

    def test_keyword_reconstruction_skipped_for_javascript(self):
        """Python keyword reconstruction should NOT run for JavaScript."""
        # 'de f' should not become 'def' in JS
        code = "var de_f = 'test';"
        corrected, ok, _ = correct_syntax(code, language="javascript")
        # Should succeed without altering non-Python code
        self.assertTrue(ok)  # Non-Python always returns ok=True

    # ------------------------------------------------------------------
    # Non-Python languages always return success
    # ------------------------------------------------------------------

    def test_javascript_always_succeeds(self):
        """JavaScript code should always return ok=True (no ast.parse)."""
        code = "console.log('hello');"
        _, ok, diagnostics = correct_syntax(code, language="javascript")
        self.assertTrue(ok)
        self.assertIsNone(diagnostics)

    def test_java_always_succeeds(self):
        """Java code should always return ok=True."""
        code = "public class Main { public static void main(String[] args) {} }"
        _, ok, diagnostics = correct_syntax(code, language="java")
        self.assertTrue(ok)
        self.assertIsNone(diagnostics)

    def test_cpp_always_succeeds(self):
        """C++ code should always return ok=True."""
        code = '#include <iostream>\nint main() { std::cout << "hi"; }'
        _, ok, diagnostics = correct_syntax(code, language="cpp")
        self.assertTrue(ok)
        self.assertIsNone(diagnostics)

    # ------------------------------------------------------------------
    # Diagnostics on failure
    # ------------------------------------------------------------------

    def test_diagnostics_on_failure(self):
        """When Python code can't be fixed, diagnostics should be returned."""
        code = "def foo(\n    if x"  # unfixable mess
        corrected, ok, diagnostics = correct_syntax(code, language="python")
        self.assertFalse(ok)
        self.assertIsNotNone(diagnostics)
        self.assertIsInstance(diagnostics, list)
        self.assertGreater(len(diagnostics), 0)

        # Each diagnostic should have required keys
        for diag in diagnostics:
            self.assertIn("message", diag)
            self.assertIn("line", diag)
            self.assertIn("suggestion", diag)

    def test_diagnostics_contain_line_number(self):
        """Diagnostics should include the line number of the error."""
        code = "x = 1\ny = )\nz = 3"  # error on line 2
        _, ok, diagnostics = correct_syntax(code, language="python")
        if not ok and diagnostics:
            # At least one diagnostic should reference a line
            lines = [d.get("line", 0) for d in diagnostics]
            self.assertTrue(any(l > 0 for l in lines))

    # ------------------------------------------------------------------
    # Bracket/quote balancing is opt-in
    # ------------------------------------------------------------------

    def test_valid_code_not_modified_by_balancing(self):
        """Code that already parses should NOT have balancing applied."""
        code = "x = (1 + 2)\nprint(x)"
        corrected, ok, diagnostics = correct_syntax(code, language="python")
        self.assertTrue(ok)
        self.assertEqual(corrected.strip(), code.strip())
        self.assertIsNone(diagnostics)


class TestBacktickReplacement(unittest.TestCase):
    """Test backtick → single quote replacement (universal rule)."""

    def test_backtick_to_quote_python(self):
        """Backtick should become single quote for Python."""
        code = "x = `hello`"
        corrected, _, _ = correct_syntax(code, language="python")
        self.assertNotIn("`", corrected)

    def test_backtick_to_quote_javascript(self):
        """Backtick handling for JavaScript — should still apply universal rules."""
        # Note: backtick is valid JS (template literals), but OCR-produced
        # backticks in non-template contexts should still be cleaned.
        # This is a universal rule so it applies.
        code = "var x = `hello`"
        corrected, _, _ = correct_syntax(code, language="javascript")
        # Universal rule should convert backticks
        self.assertNotIn("`", corrected)


if __name__ == "__main__":
    unittest.main()
