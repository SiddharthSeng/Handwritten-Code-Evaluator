"""
Post-processing and syntax correction for OCR-recognised code.

The OCR engine inevitably introduces character-level errors (``O`` ↔
``0``, ``l`` ↔ ``1``, wrong brackets, curly quotes, etc.).  This
module applies a cascade of rule-based corrections and then validates
the result with :func:`ast.parse` (Python only).

Public API
----------
.. autofunction:: correct_syntax
"""

import ast
import logging
import re

logger = logging.getLogger(__name__)

# ── Python block-starter keywords (need a trailing colon) ───────────
_BLOCK_STARTERS = (
    "def", "if", "else", "elif", "for", "while",
    "class", "try", "except", "finally", "with",
)

# Maximum number of AST-guided fix-and-retry cycles
_MAX_RETRIES = 3

# =====================================================================
# Correction Rules — easy to extend without modifying logic
# =====================================================================
# Each rule is a (pattern, replacement) tuple applied via re.sub.
# Rules are applied in order on each line during the OCR confusion
# fix phase.

# ── Universal rules — safe for ALL languages ────────────────────────
UNIVERSAL_RULES: dict[str, list[tuple[str, str]]] = {
    # ── Arrow operator reconstruction ───────────────────────────
    # OCR may split `->` into `- >` or `— >` etc.
    "arrow_operator": [
        (r"\s*-\s*>\s*", " -> "),    # normalize spacing around ->
        (r"\s*—\s*>\s*", " -> "),    # em-dash variant
        (r"\s*–\s*>\s*", " -> "),    # en-dash variant
    ],

    # ── Common symbol confusions safe for all languages ─────────
    "symbol_fixes": [
        # == vs = context: `if x = y:` → `if x == y:`
        # Handled specially in _fix_equality_confusion below
    ],
}

# ── Python-only rules — NOT safe for JS/Java/C++ ───────────────────
PYTHON_ONLY_RULES: dict[str, list[tuple[str, str]]] = {
    # ── Keyword reconstruction ──────────────────────────────────
    # OCR often splits or garbles Python keywords.
    "keyword_reconstruction": [
        (r"\bde\s*f\b", "def"),
        (r"\bde[£€f]\b", "def"),
        (r"\bpr\s*int\b", "print"),
        (r"\bpr\s*lnt\b", "print"),
        (r"\bretur\s*n\b", "return"),
        (r"\bim\s*port\b", "import"),
        (r"\bfr\s*om\b", "from"),
        (r"\bwh\s*ile\b", "while"),
        (r"\bel\s*if\b", "elif"),
        (r"\bel\s*se\b", "else"),
        (r"\bex\s*cept\b", "except"),
        (r"\bfi\s*nally\b", "finally"),
        (r"\bcl\s*ass\b", "class"),
        (r"\bra\s*nge\b", "range"),
        (r"\ble\s*n\b", "len"),
        (r"\bin\s*put\b", "input"),
        (r"\bTr\s*ue\b", "True"),
        (r"\bFa\s*lse\b", "False"),
        (r"\bNo\s*ne\b", "None"),
        (r"\bas\s*sert\b", "assert"),
        (r"\bla\s*mbda\b", "lambda"),
        (r"\bgl\s*obal\b", "global"),
        (r"\byie\s*ld\b", "yield"),
        (r"\bra\s*ise\b", "raise"),
        (r"\bcon\s*tinue\b", "continue"),
        (r"\bbre\s*ak\b", "break"),
    ],

    # ── rn → m in known Python keywords / builtins ──────────────
    # OCR can render 'm' as 'rn'.  Only applied to known identifiers
    # to avoid false positives.
    "rn_to_m_keywords": [
        (r"\breturm\b", "return"),    # reture is unlikely, but returrn→return
        (r"\bprirnt\b", "print"),     # pri-rn-t → print
        (r"\birnport\b", "import"),   # i-rn-port → import
        (r"\bfrorn\b", "from"),       # fro-rn → from
        (r"\brandon\b", "random"),    # rando-rn → random  (skip, 'randon' is unlikely)
    ],
}

# Legacy alias — kept for backward compatibility with external code
# that may reference CORRECTION_RULES.
CORRECTION_RULES: dict[str, list[tuple[str, str]]] = {
    **{k: v for k, v in UNIVERSAL_RULES.items()},
    **{k: v for k, v in PYTHON_ONLY_RULES.items()},
}


# =====================================================================
# Public API
# =====================================================================

def correct_syntax(
    raw_text: str,
    language: str = 'python',
) -> tuple[str, bool, list[dict] | None]:
    """Apply heuristic corrections to raw OCR output and validate it.

    Processing pipeline:
        1. Normalise whitespace and indentation.
        2. Apply correction rules (universal, and language-specific).
        3. Fix common OCR character confusions.
        4. Structural inference for multi-line code (Python only).
        5. Attempt ``ast.parse`` (Python only) — if it fails, apply
           targeted fixes derived from the error message and retry
           (up to 3 times).  As a last resort, attempt bracket/quote
           balancing.

    Args:
        raw_text: The raw string produced by TrOCR.
        language: Target language identifier (e.g. ``'python'``,
            ``'javascript'``, ``'java'``, ``'cpp'``).  Controls which
            OCR-correction rules are applied and whether ``ast.parse``
            validation runs.

    Returns:
        A ``(corrected_code, success, diagnostics)`` 3-tuple:

        * *corrected_code* — the best-effort corrected source.
        * *success* — ``True`` when the corrected code passes
          ``ast.parse`` (always ``True`` for non-Python languages
          since we cannot validate them).
        * *diagnostics* — ``None`` when correction succeeds; a list
          of diagnostic dicts when correction fails::

              {"line": int, "message": str, "suggestion": str}
    """
    language = language.lower().strip()

    if not raw_text or not raw_text.strip():
        logger.warning("correct_syntax called with empty input")
        return ("", False, [{"line": 0, "message": "Empty input", "suggestion": "Provide non-empty code text"}])

    text = _normalize_whitespace(raw_text)
    text = _apply_correction_rules(text, language)
    text = _fix_ocr_confusions(text, language)

    if language == 'python':
        text = _structural_inference(text)

    corrected, ok, diagnostics = _validate_and_retry(text, language)

    if ok:
        logger.info(
            "Syntax correction succeeded — code is valid %s",
            language.capitalize(),
        )
    else:
        logger.warning(
            "Syntax correction finished but code still has parse errors"
        )
    return (corrected, ok, diagnostics)


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
# Step 1.5 — Apply CORRECTION_RULES
# =====================================================================

def _apply_correction_rules(text: str, language: str = 'python') -> str:
    """Apply correction rules appropriate for the target language.

    Universal rules are always applied.  Python-only rules (keyword
    reconstruction, ``rn`` → ``m`` fixes) are applied only when
    *language* is ``'python'``.
    """
    original = text

    # Always apply universal rules
    for _category, rules in UNIVERSAL_RULES.items():
        for pattern, replacement in rules:
            text = re.sub(pattern, replacement, text)

    # Apply Python-specific rules only for Python
    if language == 'python':
        for _category, rules in PYTHON_ONLY_RULES.items():
            for pattern, replacement in rules:
                text = re.sub(pattern, replacement, text)

    if text != original:
        logger.info("Correction rules applied (language=%s)", language)
    return text


# =====================================================================
# Step 2 — OCR character-confusion fixes
# =====================================================================

def _fix_ocr_confusions(text: str, language: str = 'python') -> str:
    """Context-aware substitution of commonly confused characters.

    Args:
        text: Source code text to fix.
        language: Target language.  Python-specific rules (semicolons
            on block starters, ``$`` → ``S``, ``!`` → ``l`` in
            identifiers) are skipped for non-Python languages.
    """
    original = text

    # ── Universal fixes (safe for all languages) ────────────────
    text = _fix_curly_quotes(text)
    text = _fix_backtick_to_quote(text)
    text = _fix_inverted_exclamation(text)
    text = _fix_numeric_confusions(text)

    # ── Python-only fixes ───────────────────────────────────────
    if language == 'python':
        text = _fix_semicolons_on_block_starters(text)
        text = _fix_bracket_confusions(text)
        text = _fix_pipe_in_identifiers(text)
        text = _fix_dollar_in_identifiers(text)
        text = _fix_bang_in_identifiers(text)
        text = _fix_equality_confusion(text)

    if text != original:
        logger.info("OCR character confusions corrected (language=%s)", language)
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


def _fix_backtick_to_quote(text: str) -> str:
    """Replace backtick characters with single quotes (universal).

    OCR can render single quotes as backticks.  Backticks are not
    valid quote characters in Python, Java, or C++.  In JavaScript
    they denote template literals, but a stray backtick in OCR output
    is more likely to be a misrecognised single quote than a template
    literal — and JS template literals are multi-character (backtick pairs)
    so this substitution is still broadly safe.
    """
    text = text.replace("`", "'")
    return text


def _fix_inverted_exclamation(text: str) -> str:
    """Replace ``¡`` (U+00A1) with ``i`` inside identifiers (universal).

    OCR can confuse the inverted exclamation mark with a lowercase
    ``i``, especially in serif fonts.  Only substituted when
    surrounded by identifier characters.
    """
    text = re.sub(
        r"(?<=[A-Za-z_])\u00A1(?=[A-Za-z_0-9])",
        "i",
        text,
    )
    # Also at the start of an identifier followed by identifier chars
    text = re.sub(
        r"\b\u00A1(?=[A-Za-z_0-9])",
        "i",
        text,
    )
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
    text = re.sub(
        r"(?<=[A-Za-z_])\|(?=[A-Za-z_])",
        "l",
        text,
    )
    return text


def _fix_dollar_in_identifiers(text: str) -> str:
    """Replace ``$`` with ``S`` inside Python identifiers.

    ``$`` is not a valid character in Python identifiers but is
    commonly confused with ``S`` by OCR.  This rule is Python-only
    because ``$`` is idiomatic in JavaScript identifiers (e.g.
    ``$scope``, ``jQuery``).
    """
    text = re.sub(
        r"(?<=[A-Za-z_])\$(?=[A-Za-z_0-9])",
        "S",
        text,
    )
    text = re.sub(
        r"\$(?=[A-Za-z_])",
        "S",
        text,
    )
    return text


def _fix_bang_in_identifiers(text: str) -> str:
    """Replace ``!`` with ``l`` when embedded inside Python identifiers.

    OCR can confuse ``l`` (lowercase L) with ``!`` in serif fonts.
    This is Python-only because ``!`` is load-bearing for ``!=``
    and logical negation in JS, Java, and C++.

    Only substitutes when ``!`` is flanked by identifier characters
    on *both* sides (i.e. it looks like part of a name, not an
    operator).
    """
    text = re.sub(
        r"(?<=[A-Za-z_])!(?=[A-Za-z_0-9])",
        "l",
        text,
    )
    return text


def _fix_equality_confusion(text: str) -> str:
    """Fix ``=`` → ``==`` inside conditional contexts.

    E.g. ``if x = 5:`` → ``if x == 5:``
    Only applies on lines starting with if/elif/while where a single
    ``=`` appears between identifiers/values (not ``==``, ``!=``,
    ``<=``, ``>=``).
    """
    def _fix_line(match: re.Match) -> str:
        line = match.group(0)
        # Only fix single = that isn't part of ==, !=, <=, >=, :=
        # Replace ` = ` with ` == ` but not if already ==
        line = re.sub(r"(?<!=)(?<!!)(?<!<)(?<!>)(?<!:) = (?!=)", " == ", line)
        return line

    # Apply only to conditional lines
    text = re.sub(
        r"^\s*(?:if|elif|while)\b.*$",
        _fix_line,
        text,
        flags=re.MULTILINE,
    )
    return text


# =====================================================================
# Step 3 — Structural inference for multi-line code
# =====================================================================

def _structural_inference(text: str) -> str:
    """Infer and fix indentation structure for multi-line code.

    After per-line OCR, indentation context between lines may be lost.
    This pass:
      1. Ensures lines after block-starters (ending with ``:``) are
         indented by at least 4 spaces more than the block-starter.
      2. Aligns orphaned ``else:``, ``elif:``, ``except:``, ``finally:``
         with the corresponding ``if``/``try`` block above.

    Note:
        This function is Python-specific and is only called when
        ``language == 'python'``.
    """
    lines = text.split("\n")
    if len(lines) <= 1:
        return text

    fixed_lines = list(lines)  # work on a copy

    for i in range(len(fixed_lines)):
        line = fixed_lines[i]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)

        if not stripped:
            continue

        # ── Rule 1: Line after a block-starter must be indented ────
        if i > 0:
            prev_line = fixed_lines[i - 1]
            prev_stripped = prev_line.lstrip()
            prev_indent = len(prev_line) - len(prev_stripped)

            if prev_stripped.rstrip().endswith(":") and _is_block_starter(prev_stripped):
                expected_indent = prev_indent + 4
                # If current line is NOT indented beyond the block-starter,
                # and it's NOT itself a block-closer (else/elif/except/finally),
                # indent it.
                if current_indent <= prev_indent and not _is_block_closer(stripped):
                    fixed_lines[i] = " " * expected_indent + stripped
                    logger.debug(
                        "Structural fix: indented line %d to %d spaces",
                        i + 1, expected_indent,
                    )

        # ── Rule 2: Align orphaned block-closers ──────────────────
        if _is_block_closer(stripped):
            # Find the matching block-opener above
            target_indent = _find_matching_opener_indent(fixed_lines, i)
            if target_indent is not None and current_indent != target_indent:
                fixed_lines[i] = " " * target_indent + stripped
                logger.debug(
                    "Structural fix: aligned '%s' on line %d to indent %d",
                    stripped.split()[0], i + 1, target_indent,
                )

    result = "\n".join(fixed_lines)
    if result != text:
        logger.info("Structural inference applied to fix indentation")
    return result


def _is_block_starter(stripped_line: str) -> bool:
    """Check if a stripped line starts a Python block."""
    for kw in _BLOCK_STARTERS:
        if stripped_line.startswith(kw) and (
            len(stripped_line) == len(kw) or not stripped_line[len(kw)].isalnum()
        ):
            return True
    return False


def _is_block_closer(stripped_line: str) -> bool:
    """Check if a stripped line is an else/elif/except/finally."""
    closers = ("else:", "else :", "elif ", "except:", "except ", "finally:")
    return any(stripped_line.startswith(c) for c in closers)


def _find_matching_opener_indent(lines: list[str], closer_idx: int) -> int | None:
    """Walk backwards to find the indent level of the matching opener.

    For ``else``/``elif`` → match ``if``
    For ``except``/``finally`` → match ``try``
    """
    closer_stripped = lines[closer_idx].lstrip()
    if closer_stripped.startswith(("else", "elif")):
        openers = ("if ",)
    elif closer_stripped.startswith(("except", "finally")):
        openers = ("try:",)
    else:
        return None

    # Walk backwards, respecting nesting depth
    depth = 0
    for j in range(closer_idx - 1, -1, -1):
        line = lines[j]
        s = line.lstrip()
        indent = len(line) - len(s)
        if not s:
            continue

        # Track nesting: if we see another closer at the same or higher
        # level, we need to skip its opener too
        if _is_block_closer(s):
            depth += 1
        elif any(s.startswith(op) for op in openers):
            if depth == 0:
                return indent
            depth -= 1

    return None


# =====================================================================
# Step 4 — Bracket / Quote balancing (last-resort, opt-in)
# =====================================================================

def _balance_quotes(text: str) -> tuple[str, list[dict]]:
    """Balance unmatched quote characters as a last resort.

    Scans each line for unmatched single or double quotes (ignoring
    escaped quotes) and closes them at the end of the line.

    This function must **only** be called after ``ast.parse()`` has
    already failed — never on code that already parses correctly.

    Returns:
        A ``(fixed_text, changes)`` tuple where *changes* is a list
        of diagnostic dicts describing each fix applied.
    """
    lines = text.split("\n")
    changes: list[dict] = []

    for idx, line in enumerate(lines):
        # Strip escaped characters for counting purposes
        unescaped = re.sub(r"\\.", "", line)
        for quote in ('"', "'"):
            count = unescaped.count(quote)
            if count % 2 != 0:
                lines[idx] = line.rstrip() + quote
                change = {
                    "line": idx + 1,
                    "message": f"Unmatched {quote}-quote balanced",
                    "suggestion": (
                        f"Added closing {quote} at end of line {idx + 1}"
                    ),
                }
                changes.append(change)
                logger.debug(
                    "Balance: closed unmatched %s-quote on line %d",
                    quote, idx + 1,
                )
                # Re-read the line and re-strip for the next quote type
                line = lines[idx]
                unescaped = re.sub(r"\\.", "", line)

    return ("\n".join(lines), changes)


def _balance_brackets(text: str) -> tuple[str, list[dict]]:
    """Balance unmatched brackets / parentheses as a last resort.

    Two passes:
      1. **Append missing closers** — count unmatched openers and
         append the corresponding closing characters at end-of-file.
      2. **Remove stray closers** — remove closing brackets that
         have no matching opener.

    This function must **only** be called after ``ast.parse()`` has
    already failed — never on code that already parses correctly.

    Returns:
        A ``(fixed_text, changes)`` tuple where *changes* is a list
        of diagnostic dicts describing each fix applied.
    """
    changes: list[dict] = []

    # ── Pass 1: append missing closers ──────────────────────────
    openers = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []

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
        total_lines = text.count("\n") + 1
        change = {
            "line": total_lines,
            "message": f"Appended {len(stack)} missing closer(s): {closers}",
            "suggestion": "Check that all brackets and parentheses are properly closed",
        }
        changes.append(change)
        logger.debug(
            "Balance: appended %d missing closer(s): %s",
            len(stack), closers,
        )

    # ── Pass 2: remove stray closing brackets ───────────────────
    closers_map = {v: k for k, v in openers.items()}
    stack_2: list[str] = []
    remove_indices: set[int] = set()

    in_string = None
    prev_char = ""
    for i, ch in enumerate(text):
        if in_string:
            if ch == in_string and prev_char != "\\":
                in_string = None
        else:
            if ch in ("'", '"'):
                in_string = ch
            elif ch in openers:
                stack_2.append(ch)
            elif ch in closers_map:
                expected_opener = closers_map[ch]
                if stack_2 and stack_2[-1] == expected_opener:
                    stack_2.pop()
                else:
                    remove_indices.add(i)

    if remove_indices:
        # Determine line numbers of removals for diagnostics
        char_to_line: dict[int, int] = {}
        line_num = 1
        for ci, ch in enumerate(text):
            char_to_line[ci] = line_num
            if ch == "\n":
                line_num += 1

        for ri in sorted(remove_indices):
            change = {
                "line": char_to_line.get(ri, 0),
                "message": f"Removed stray closing bracket '{text[ri]}'",
                "suggestion": "Check bracket pairing near this line",
            }
            changes.append(change)

        text = "".join(
            ch for i, ch in enumerate(text) if i not in remove_indices
        )
        logger.debug(
            "Balance: removed %d unmatched closing bracket(s)",
            len(remove_indices),
        )

    return (text, changes)


# =====================================================================
# Step 5 — AST validation with iterative repair
# =====================================================================

def _validate_and_retry(
    text: str,
    language: str = 'python',
) -> tuple[str, bool, list[dict] | None]:
    """Try to parse *text* as Python; on failure, apply targeted fixes.

    Args:
        text: The code text to validate.
        language: Target language.  ``ast.parse`` validation only runs
            for Python.  For non-Python languages the code is returned
            as-is with ``success=True`` (we cannot validate it).

    Returns:
        A ``(final_text, success, diagnostics)`` 3-tuple:

        * *final_text* — the best-effort corrected source.
        * *success* — ``True`` when parsing succeeds (or for
          non-Python languages where we skip validation).
        * *diagnostics* — ``None`` on success; a list of diagnostic
          dicts on failure.
    """
    # ── Non-Python: skip ast.parse entirely ─────────────────────
    if language != 'python':
        logger.debug(
            "Skipping ast.parse validation for language=%s", language
        )
        return (text, True, None)

    # ── Python: iterative repair loop ───────────────────────────
    last_error: SyntaxError | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            ast.parse(text)
            logger.debug("ast.parse succeeded on attempt %d", attempt)
            return (text, True, None)
        except SyntaxError as err:
            last_error = err
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
        return (text, True, None)
    except SyntaxError as err:
        last_error = err

    # ── Last resort: bracket / quote balancing ──────────────────
    # These ONLY run after ast.parse has failed.
    logger.debug("Attempting last-resort bracket/quote balancing")
    all_diagnostics: list[dict] = []

    text, quote_changes = _balance_quotes(text)
    all_diagnostics.extend(quote_changes)

    text, bracket_changes = _balance_brackets(text)
    all_diagnostics.extend(bracket_changes)

    # Try parsing one more time after balancing
    try:
        ast.parse(text)
        logger.info("ast.parse succeeded after bracket/quote balancing")
        return (text, True, None)
    except SyntaxError as err:
        last_error = err

    # ── Build final diagnostics from the last SyntaxError ───────
    if last_error is not None:
        error_diag: dict[str, object] = {
            "line": last_error.lineno or 0,
            "message": last_error.msg or "Unknown syntax error",
            "suggestion": _suggest_likely_cause(last_error),
        }
        # Prepend the parse error before balancing diagnostics
        all_diagnostics.insert(0, error_diag)

    return (text, False, all_diagnostics if all_diagnostics else None)


def _suggest_likely_cause(error: SyntaxError) -> str:
    """Return a human-readable "likely cause" suggestion for a SyntaxError.

    Maps common ``ast.parse`` error messages to actionable suggestions
    that can help the user or downstream systems understand the root
    cause.
    """
    msg = (error.msg or "").lower()

    if "unterminated string" in msg or "eol while scanning" in msg:
        return (
            "Likely cause: an unclosed string literal.  Check for "
            "missing closing quotes on the indicated line."
        )
    if "unexpected eof" in msg or "eof while scanning" in msg:
        return (
            "Likely cause: the file ends with an unclosed bracket, "
            "parenthesis, or string.  Add the missing closing character."
        )
    if "expected ':'" in msg:
        return (
            "Likely cause: a block-starter (def, if, for, etc.) is "
            "missing its trailing colon."
        )
    if "unmatched" in msg or "was never closed" in msg:
        return (
            "Likely cause: a bracket or parenthesis is opened but "
            "never closed, or a stray closer appears without an opener."
        )
    if "invalid syntax" in msg:
        return (
            "Likely cause: a typo, misrecognised character, or "
            "structurally invalid expression.  Inspect the indicated "
            "line for OCR artefacts."
        )
    if "unexpected indent" in msg:
        return (
            "Likely cause: inconsistent indentation.  Ensure all "
            "indentation uses 4-space increments."
        )
    if "unindent does not match" in msg:
        return (
            "Likely cause: a dedented line does not align with any "
            "enclosing block.  Check indentation levels."
        )

    return (
        "Inspect the indicated line for OCR artefacts or structural "
        "errors."
    )


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
    # error specifically says "unmatched".
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
