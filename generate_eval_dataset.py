"""
generate_eval_dataset.py
------------------------
Generates a small synthetic evaluation dataset for benchmarking the
Handwritten Code Evaluator pipeline.

Each sample consists of:
  - A PNG image rendered in "Inkfree" (Windows' ink-style handwriting font)
  - A .py ground-truth file with the exact expected code

Dataset directory layout:
  eval_dataset/
    sample_001.png
    sample_001.py
    sample_002.png
    sample_002.py
    ...
    manifest.json   <- metadata about every sample

⚠️  SYNTHETIC PROXY NOTICE:
This dataset uses Inkfree.ttf, a digitally clean handwriting-style font.
It is NOT genuine photographed handwriting. Results measured against this
dataset will overestimate real-world accuracy because:
  - No ink bleed, paper texture, or scanning artifacts
  - No tilt, rotation, or pen pressure variation
  - Consistent baseline and glyph spacing
  - TrOCR was trained on IAM (real photographed handwriting)

These benchmark numbers should be labelled:
  "Synthetic proxy dataset (Inkfree font) — not real handwriting"
and NOT presented as evidence of real-world handwriting recognition accuracy.
"""

import json
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("eval_dataset")
FONT_PATH = "C:/Windows/Fonts/Inkfree.ttf"
FONT_SIZE = 26          # px — readable, similar to large student handwriting
LINE_SPACING = 10       # extra px between lines beyond font ascent/descent
PADDING = 20            # px margin around text on all sides
BG_COLOR = (255, 255, 255)
FG_COLOR = (10, 10, 10)
MAX_LINE_WIDTH_PX = 720  # images won't be wider than this

# ---------------------------------------------------------------------------
# Dataset: 20 Python snippets at varying difficulty
#
# Difficulty levels:
#   easy   - single expressions, no syntax ambiguity
#   medium - conditionals, loops, function defs
#   hard   - nested structures, ambiguous chars (l/1, O/0, I/1)
# ---------------------------------------------------------------------------

SAMPLES = [
    # ── Easy: one-liners ────────────────────────────────────────────────
    {
        "id": "001",
        "difficulty": "easy",
        "note": "simple assignment",
        "code": "x = 10",
    },
    {
        "id": "002",
        "difficulty": "easy",
        "note": "arithmetic expression",
        "code": "result = 3 + 4 * 2",
    },
    {
        "id": "003",
        "difficulty": "easy",
        "note": "print statement",
        "code": 'print("Hello, World!")',
    },
    {
        "id": "004",
        "difficulty": "easy",
        "note": "string assignment",
        "code": 'name = "Alice"',
    },
    {
        "id": "005",
        "difficulty": "easy",
        "note": "boolean expression",
        "code": "is_valid = True",
    },
    # ── Medium: conditionals ─────────────────────────────────────────────
    {
        "id": "006",
        "difficulty": "medium",
        "note": "if-else",
        "code": textwrap.dedent("""\
            x = 5
            if x > 0:
                print("positive")
            else:
                print("non-positive")"""),
    },
    {
        "id": "007",
        "difficulty": "medium",
        "note": "for loop with range",
        "code": textwrap.dedent("""\
            total = 0
            for i in range(10):
                total += i
            print(total)"""),
    },
    {
        "id": "008",
        "difficulty": "medium",
        "note": "while loop",
        "code": textwrap.dedent("""\
            n = 1
            while n < 100:
                n = n * 2
            print(n)"""),
    },
    {
        "id": "009",
        "difficulty": "medium",
        "note": "simple function def",
        "code": textwrap.dedent("""\
            def square(n):
                return n * n
            print(square(7))"""),
    },
    {
        "id": "010",
        "difficulty": "medium",
        "note": "list operations",
        "code": textwrap.dedent("""\
            nums = [1, 2, 3, 4, 5]
            nums.append(6)
            print(len(nums))"""),
    },
    # ── Medium: ambiguous characters ─────────────────────────────────────
    {
        "id": "011",
        "difficulty": "medium",
        "note": "l vs 1 ambiguity: variable 'l1' and literal 1",
        "code": textwrap.dedent("""\
            l1 = 1
            l2 = l1 + 10
            print(l2)"""),
    },
    {
        "id": "012",
        "difficulty": "medium",
        "note": "O vs 0 ambiguity: zero in arithmetic",
        "code": textwrap.dedent("""\
            count = 0
            for i in range(10):
                count = count + 1
            print(count)"""),
    },
    {
        "id": "013",
        "difficulty": "medium",
        "note": "I vs 1 ambiguity: index starting at 1",
        "code": textwrap.dedent("""\
            items = [10, 20, 30]
            i = 1
            print(items[i])"""),
    },
    # ── Hard: nested structures ───────────────────────────────────────────
    {
        "id": "014",
        "difficulty": "hard",
        "note": "nested if inside for",
        "code": textwrap.dedent("""\
            for i in range(1, 6):
                if i % 2 == 0:
                    print(i, "even")
                else:
                    print(i, "odd")"""),
    },
    {
        "id": "015",
        "difficulty": "hard",
        "note": "function with loop and conditional",
        "code": textwrap.dedent("""\
            def factorial(n):
                result = 1
                for i in range(1, n + 1):
                    result *= i
                return result
            print(factorial(5))"""),
    },
    {
        "id": "016",
        "difficulty": "hard",
        "note": "list comprehension",
        "code": "squares = [x ** 2 for x in range(10)]\nprint(squares)",
    },
    {
        "id": "017",
        "difficulty": "hard",
        "note": "dictionary usage",
        "code": textwrap.dedent("""\
            scores = {"Alice": 95, "Bob": 87}
            for name, score in scores.items():
                print(name, score)"""),
    },
    {
        "id": "018",
        "difficulty": "hard",
        "note": "string methods — multiple ambiguous chars",
        "code": textwrap.dedent("""\
            text = "Hello World"
            words = text.split()
            print(len(words))"""),
    },
    {
        "id": "019",
        "difficulty": "hard",
        "note": "try-except",
        "code": textwrap.dedent("""\
            try:
                x = int("abc")
            except ValueError:
                print("invalid input")"""),
    },
    {
        "id": "020",
        "difficulty": "hard",
        "note": "class definition",
        "code": textwrap.dedent("""\
            class Counter:
                def __init__(self):
                    self.count = 0
                def increment(self):
                    self.count += 1
            c = Counter()
            c.increment()
            print(c.count)"""),
    },
]


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def render_code_image(code: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render a multi-line code string as a white-background PNG.

    Returns a PIL Image sized to fit the text with PADDING on all sides.
    """
    lines = code.split("\n")
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    # Measure each line
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line if line else " ", font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_h = max(line_heights) if line_heights else FONT_SIZE
    total_h = line_h * len(lines) + LINE_SPACING * (len(lines) - 1) + 2 * PADDING
    total_w = min(max(line_widths) + 2 * PADDING, MAX_LINE_WIDTH_PX) if line_widths else 200 + 2 * PADDING

    img = Image.new("RGB", (total_w, total_h), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    y = PADDING
    for line in lines:
        draw.text((PADDING, y), line, font=font, fill=FG_COLOR)
        y += line_h + LINE_SPACING

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        print(f"Font loaded: {FONT_PATH}")
    except OSError as e:
        print(f"ERROR: Could not load Inkfree font: {e}")
        print("This script requires Inkfree.ttf (available on Windows).")
        return

    manifest = []

    for sample in SAMPLES:
        sid = sample["id"]
        code = sample["code"]

        # Save ground-truth .py
        py_path = OUTPUT_DIR / f"sample_{sid}.py"
        py_path.write_text(code, encoding="utf-8")

        # Render image
        img = render_code_image(code, font)
        img_path = OUTPUT_DIR / f"sample_{sid}.png"
        img.save(img_path)

        manifest.append({
            "id": sid,
            "difficulty": sample["difficulty"],
            "note": sample["note"],
            "image": f"sample_{sid}.png",
            "ground_truth": f"sample_{sid}.py",
            "lines": len(code.split("\n")),
            "chars": len(code),
        })

        print(f"  [{sid}] {sample['difficulty']:6s}  {sample['note']}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDataset generated: {len(SAMPLES)} samples in {OUTPUT_DIR}/")
    print(f"Manifest: {manifest_path}")
    print()
    print("⚠️  SYNTHETIC PROXY DATASET — see file header for limitations.")


if __name__ == "__main__":
    main()
