"""
Post-processing and syntax correction for OCR-recognised Python code.

The OCR engine inevitably introduces character-level errors (``O`` ↔
``0``, ``l`` ↔ ``1``, wrong brackets, curly quotes, etc.).  This
module applies a cascade of rule-based corrections and then validates
the result with :func:`ast.parse`.

Public API
----------
.. autofunction:: correct_syntax
"""

import ast
import logging
import re
import textwrap

logger = logging.getLogger(__name__)

# ── Python block-starter keywords (need a trailing colon) ───────────
_BLOCK_STARTERS = (
    "def", "if", "else", "elif", "for", "while",
    "class", "try", "except", "finally", "with",
)

# Maximum number of AST-guided fix-and-retry cycles
_MAX_RETRIES = 3


# =====================================================================
# Public API
# =====================================================================

def correct_syntax(raw_text: str) -> tuple[str, bool]:
    """Apply heuristic corrections to raw OCR output and validate it.

    Processing pipeline:
        1. Normalise whitespace and indentation.
        2. Fix common OCR character confusions.
        3. Attempt ``ast.parse`` — if it fails, apply targeted
           fixes derived from the error message and retry (up to
           3 times).

    Args:
        raw_text: The raw string produced by TrOCR.

    Returns:
        A ``(corrected_code, success)`` tuple where *success* is
        ``True`` when the corrected code passes ``ast.parse``.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("correct_syntax called with empty input")
        return ("", False)

    text = _normalize_whitespace(raw_text)
    text = _fix_ocr_confusions(text)
    corrected, ok = _validate_and_retry(text)

    if ok:
        logger.info("Syntax correction succeeded — code is valid Python")
    else:
        logger.warning(
            "Syntax correction finished but code still has parse errors"
        )
    return (corrected, ok)


# =====================================================================
# Step 1 — Whitespace normalisation
# =====================================================================

def _normalize_whitespace(text: str) -> str:
    """Normalise indentation, trailing spaces, and line endings.

    * Convert ``\\r\\n`` and ``\\r`` to ``\\n``.
    * Strip trailing whitespace from every line.
    * Replace tab characters with 4 spaces.
    * Attempt to re-indent lines so that the *minimum* non-zero
      indent becomes 4 and larger indents scale proportionally.
    """
    # Uniform line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Tabs → spaces
    text = text.expandtabs(4)

    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]

    # Detect the smallest non-zero leading-space count
    indents = []
    for line in lines:
        stripped = line.lstrip(" ")
        if stripped:  # skip blank lines
            leading = len(line) - len(stripped)
            if leading > 0:
                indents.append(leading)

    if indents:
        min_indent = min(indents)
        if min_indent != 4 and min_indent > 0:
            # Re-scale indentation so the base level is 4 spaces
            scale = 4.0 / min_indent
            new_lines: list[str] = []
            for line in lines:
                stripped = line.lstrip(" ")
                if stripped:
                    leading = len(line) - len(stripped)
                    new_leading = int(round(leading * scale))
                    # Snap to multiples of 4
                    new_leading = round(new_leading / 4) * 4
                    new_lines.append(" " * new_leading + stripped)
                else:
                    new_lines.append("")
            lines = new_lines
            logger.debug(
                "Re-indented: base indent %d → 4 spaces", min_indent
            )

    result = "\n".join(lines)
    logger.debug("Whitespace normalisation complete")
    return result


# =====================================================================
# Step 2 — OCR character-confusion fixes
# =====================================================================

def _fix_ocr_confusions(text: str) -> str:
    """Context-aware substitution of commonly confused characters."""
    original = text
    text = _fix_curly_quotes(text)
    text = _fix_semicolons_on_block_starters(text)
    text = _fix_numeric_confusions(text)
    text = _fix_bracket_confusions(text)
    text = _fix_pipe_in_identifiers(text)

    if text != original:
        logger.info("OCR character confusions corrected")
    return text


def _fix_curly_quotes(text: str) -> str:
    r"""Replace smart / curly quotation marks with ASCII equivalents.

    Handles:
        \u2018 \u2019  →  '
        \u201C \u201D  →  "
        \u00AB \u00BB  →  "
    """
    replacements = {
        "\u2018": "'", "\u2019": "'",  # single curly quotes
        "\u201C": '"', "\u201D": '"',  # double curly quotes
        "\u00AB": '"', "\u00BB": '"',  # guillemets
        "\u2032": "'", "\u2033": '"',  # prime marks
        "\u02BC": "'",                 # modifier letter apostrophe
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fix_semicolons_on_block_starters(text: str) -> str:
    """Replace a trailing ``;`` with ``:`` on Python block-starter lines.

    E.g. ``def foo();`` → ``def foo():``
    """
    pattern = (
        r"^(\s*(?:" + "|".join(_BLOCK_STARTERS) + r")\b.+);\s*$"
    )
    text = re.sub(pattern, r"\1:", text, flags=re.MULTILINE)
    return text


def _fix_numeric_confusions(text: str) -> str:
    """Fix O→0, l→1, I→1 *only* in numeric contexts.

    A "numeric context" is defined as: directly adjacent to at least
    one digit, or inside a clearly numeric literal (e.g. ``0.O5``).
    We process each token within a line, leaving identifiers alone.
    """
    def _fix_token(match: re.Match) -> str:
        token = match.group(0)
        # If the token contains at least one real digit, fix confusables
        if re.search(r"\d", token):
            token = re.sub(r"O", "0", token)
            token = re.sub(r"(?<=[0-9])l", "1", token)
            token = re.sub(r"l(?=[0-9])", "1", token)
            token = re.sub(r"(?<=[0-9])I", "1", token)
            token = re.sub(r"I(?=[0-9])", "1", token)
        return token

    # Match "word-like" tokens that contain at least one digit mixed
    # with letters, or pure number literals with confusable chars.
    # This pattern grabs runs that look like numeric literals or
    # identifiers with digits.
    text = re.sub(
        r"[A-Za-z0-9_.]+",
        _fix_token,
        text,
    )
    return text


def _fix_bracket_confusions(text: str) -> str:
    """Fix ``{`` → ``(`` and ``}`` → ``)`` near function def/call sites.

    OCR may render parentheses as curly braces.  We only substitute
    when the brace is immediately after an identifier (function name)
    or at the matching close position.

    Pattern: ``identifier{…}`` → ``identifier(…)``
    """
    # Opening brace right after a word character (function name)
    text = re.sub(
        r"(\b[A-Za-z_][A-Za-z0-9_]*)\{",
        r"\1(",
        text,
    )
    # Closing brace that follows what looks like an argument list
    # (a sequence ending without a colon/dict pattern)
    # Heuristic: ``}`` preceded by a non-colon, non-brace character
    # on the same "call-like" span.
    text = re.sub(
        r"(?<=\()([^{}]*)\}",
        r"\1)",
        text,
    )
    return text


def _fix_pipe_in_identifiers(text: str) -> str:
    """Replace ``|`` with ``l`` when surrounded by identifier chars.

    E.g. ``va|ue`` → ``value``, ``resu|t`` → ``result``
    But *not* inside bitwise-or expressions like ``a | b``.
    """
    # A pipe directly between two word-characters (no spaces)
    text = re.sub(
        r"(?<=[A-Za-z_])\|(?=[A-Za-z_])",
        "l",
        text,
    )
    return text


# =====================================================================
# Step 3 — AST validation with iterative repair
# =====================================================================

def _validate_and_retry(text: str) -> tuple[str, bool]:
    """Try to parse *text* as Python; on failure, apply targeted fixes.

    Returns ``(final_text, True)`` if parsing eventually succeeds,
    otherwise ``(best_effort_text, False)`` after *_MAX_RETRIES*
    repair attempts.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            ast.parse(text)
            logger.debug("ast.parse succeeded on attempt %d", attempt)
            return (text, True)
        except SyntaxError as err:
            logger.debug(
                "ast.parse attempt %d failed: %s (line %s)",
                attempt, err.msg, err.lineno,
            )
            fixed = _apply_targeted_fix(text, err)
            if fixed == text:
                # No progress — stop retrying
                logger.debug(
                    "No targeted fix available for: %s", err.msg
                )
                break
            text = fixed

    # Final attempt after all retries
    try:
        ast.parse(text)
        return (text, True)
    except SyntaxError:
        return (text, False)


def _apply_targeted_fix(text: str, error: SyntaxError) -> str:
    """Attempt a single targeted fix based on a ``SyntaxError``.

    Each sub-fixer returns the text unchanged if it cannot help, so
    they can be chained safely.
    """
    msg = error.msg.lower() if error.msg else ""
    lineno = error.lineno  # 1-based, may be None

    # ── Missing colon on a block-starter ────────────────────────
    if "expected ':'" in msg or "invalid syntax" in msg:
        text = _fix_missing_colon(text, lineno)

    # ── Unterminated string literal ─────────────────────────────
    if "unterminated string" in msg or "eol while scanning" in msg:
        text = _fix_unterminated_string(text, lineno)

    # ── Unexpected EOF (often a missing closing bracket) ────────
    if "unexpected eof" in msg or "eof while scanning" in msg:
        text = _fix_unexpected_eof(text)

    # ── Unmatched parenthesis / bracket ─────────────────────────
    if "unmatched" in msg or "was never closed" in msg:
        text = _fix_unmatched_brackets(text)

    return text


# ── Targeted fixers ─────────────────────────────────────────────────


def _fix_missing_colon(text: str, lineno: int | None) -> str:
    """Append a colon to a block-starter line that is missing one."""
    if lineno is None:
        return text

    lines = text.split("\n")
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return text

    line = lines[idx]
    stripped = line.rstrip()

    # Only fix lines that actually start a block
    block_re = re.compile(
        r"^\s*(?:" + "|".join(_BLOCK_STARTERS) + r")\b"
    )
    if block_re.match(stripped) and not stripped.endswith(":"):
        lines[idx] = stripped + ":"
        logger.debug("Added missing colon on line %d", lineno)

    return "\n".join(lines)


def _fix_unterminated_string(text: str, lineno: int | None) -> str:
    """Close an unterminated string literal on the offending line."""
    if lineno is None:
        return text

    lines = text.split("\n")
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return text

    line = lines[idx]

    # Count unmatched quotes (single and double)
    for quote in ('"', "'"):
        # Ignore escaped quotes
        unescaped = re.sub(r"\\.", "", line)
        count = unescaped.count(quote)
        if count % 2 != 0:
            lines[idx] = line.rstrip() + quote
            logger.debug(
                "Closed unterminated %s-quote on line %d", quote, lineno
            )
            break

    return "\n".join(lines)


def _fix_unexpected_eof(text: str) -> str:
    """Append missing closing brackets / parentheses at end of file.

    Counts unmatched openers and appends the corresponding closers.
    """
    openers = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []

    # Walk the full text, skipping string literals
    in_string: str | None = None
    prev_char = ""
    for ch in text:
        if in_string:
            if ch == in_string and prev_char != "\\":
                in_string = None
        else:
            if ch in ("'", '"'):
                in_string = ch
            elif ch in openers:
                stack.append(openers[ch])
            elif ch in openers.values():
                if stack and stack[-1] == ch:
                    stack.pop()
        prev_char = ch

    if stack:
        closers = "".join(reversed(stack))
        text = text.rstrip() + closers
        logger.debug(
            "Appended %d missing closer(s): %s", len(stack), closers
        )

    return text


def _fix_unmatched_brackets(text: str) -> str:
    """Remove or balance clearly unmatched closing brackets."""
    # This is the same bracket-balancing logic but invoked when the
    # error specifically says "unmatched".  We attempt the EOF fix
    # first; if that doesn't help, we do a line-by-line scan and
    # remove orphan closers.
    text = _fix_unexpected_eof(text)

    # Second pass: remove stray closing brackets that have no opener
    openers = {"(": ")", "[": "]", "{": "}"}
    closers_map = {v: k for k, v in openers.items()}
    stack: list[str] = []
    remove_indices: set[int] = set()

    in_string: str | None = None
    prev_char = ""
    for i, ch in enumerate(text):
        if in_string:
            if ch == in_string and prev_char != "\\":
                in_string = None
        else:
            if ch in ("'", '"'):
                in_string = ch
            elif ch in openers:
                stack.append(ch)
            elif ch in closers_map:
                expected_opener = closers_map[ch]
                if stack and stack[-1] == expected_opener:
                    stack.pop()
                else:
                    remove_indices.add(i)

    if remove_indices:
        text = "".join(
            ch for i, ch in enumerate(text) if i not in remove_indices
        )
        logger.debug(
            "Removed %d unmatched closing bracket(s)", len(remove_indices)
        )

    return text
