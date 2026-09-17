"""
generate_training_dataset.py
----------------------------
Generates a large, augmented synthetic dataset for fine-tuning TrOCR
on handwritten Python code.

Features:
  - 1000 Python code snippets across difficulty levels
  - 10 handwriting-style fonts (randomly selected per sample)
  - Realistic augmentation via Augraphy (ink bleed, paper texture,
    rotation, noise, baseline wander, etc.)
  - Heavy emphasis on code operators TrOCR struggles with (=, [], {}, etc.)
  - Train/validation/test split (80/10/10)

Dataset structure:
  finetune_dataset/
    train/
      images/  *.png
      labels.json  {filename: ground_truth_text}
    val/
      images/  *.png
      labels.json
    test/
      images/  *.png
      labels.json
    metadata.json  (dataset stats)

SYNTHETIC DATA NOTICE:
This is NOT real handwritten code. It's font-rendered text with
augmentation designed to simulate handwriting variation. Fine-tuning
results should be labeled as "synthetic fine-tuning" and verified
against real handwriting for honest claims.
"""

import json
import os
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try to import augraphy for realistic augmentation
try:
    import augraphy
    AUGRAPHY_AVAILABLE = True
    print("Augraphy loaded — using realistic document augmentation.")
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("WARNING: Augraphy not available. Using basic numpy augmentation only.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("finetune_dataset")
RANDOM_SEED = 42

# Fonts — all handwriting-style fonts available on this Windows system
FONTS = [
    ("Inkfree", "C:/Windows/Fonts/Inkfree.ttf"),
    ("Lucida_HW", "C:/Windows/Fonts/LHANDW.TTF"),
    ("Comic_Sans", "C:/Windows/Fonts/comic.ttf"),
    ("Comic_Sans_Italic", "C:/Windows/Fonts/comici.ttf"),
    ("Segoe_Print", "C:/Windows/Fonts/segoepr.ttf"),
    ("Segoe_Script", "C:/Windows/Fonts/segoesc.ttf"),
    ("French_Script", "C:/Windows/Fonts/FRSCRIPT.TTF"),
    ("Script_Bold", "C:/Windows/Fonts/SCRIPTBL.TTF"),
    ("Segoe_Print_Bold", "C:/Windows/Fonts/segoeprb.ttf"),
    ("Segoe_Script_Bold", "C:/Windows/Fonts/segoescb.ttf"),
]

# Font sizes to vary
FONT_SIZES = [22, 24, 26, 28, 30, 32]

# Image settings
BG_COLORS = [(255, 255, 255), (250, 248, 240), (245, 245, 245), (240, 235, 225)]
INK_COLORS = [(0, 0, 0), (10, 10, 30), (25, 25, 50), (0, 0, 100), (50, 50, 50)]
PADDING = 20
LINE_SPACING_RANGE = (6, 14)

# Train/val/test split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ---------------------------------------------------------------------------
# Code snippet templates — emphasizing operators TrOCR struggles with
# ---------------------------------------------------------------------------

def _generate_snippets(n_total: int, rng: random.Random) -> list[dict]:
    """Generate n_total Python code snippets with difficulty labels."""
    snippets = []

    # === Category 1: Simple assignments with = (high frequency) ===
    var_names = ["x", "y", "z", "n", "m", "val", "num", "count", "total", "result",
                 "score", "age", "size", "length", "width", "height", "temp", "flag",
                 "index", "limit", "start", "end", "step", "data", "value"]
    for _ in range(n_total // 8):
        v = rng.choice(var_names)
        num = rng.randint(0, 999)
        snippets.append({"code": f"{v} = {num}", "difficulty": "easy", "focus": "assignment"})

    # === Category 2: Arithmetic with operators ===
    ops = ["+", "-", "*", "//", "%", "**"]
    for _ in range(n_total // 8):
        v = rng.choice(var_names)
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        op = rng.choice(ops)
        snippets.append({"code": f"{v} = {a} {op} {b}", "difficulty": "easy", "focus": "arithmetic"})

    # === Category 3: Compound assignment operators (+=, -=, *=, etc.) ===
    comp_ops = ["+=", "-=", "*=", "//=", "%="]
    for _ in range(n_total // 10):
        v = rng.choice(var_names)
        num = rng.randint(1, 50)
        op = rng.choice(comp_ops)
        snippets.append({"code": f"{v} {op} {num}", "difficulty": "easy", "focus": "compound_assign"})

    # === Category 4: Comparison operators (==, !=, <, >, <=, >=) ===
    cmp_ops = ["==", "!=", "<", ">", "<=", ">="]
    for _ in range(n_total // 10):
        v = rng.choice(var_names)
        num = rng.randint(0, 100)
        op = rng.choice(cmp_ops)
        snippets.append({"code": f"if {v} {op} {num}:", "difficulty": "easy", "focus": "comparison"})

    # === Category 5: Print statements with strings ===
    strings = ["Hello", "World", "Error", "Done", "OK", "Yes", "No",
               "result:", "value =", "count:", "total:", "name:", "invalid",
               "positive", "negative", "even", "odd", "found", "not found"]
    for _ in range(n_total // 10):
        s = rng.choice(strings)
        q = rng.choice(['"', "'"])
        snippets.append({"code": f"print({q}{s}{q})", "difficulty": "easy", "focus": "print"})

    # === Category 6: List/dict literals with brackets ===
    for _ in range(n_total // 12):
        nums = [str(rng.randint(0, 99)) for _ in range(rng.randint(2, 6))]
        snippets.append({
            "code": f"data = [{', '.join(nums)}]",
            "difficulty": "medium", "focus": "list_literal"
        })

    for _ in range(n_total // 15):
        keys = rng.sample(["a", "b", "c", "x", "y", "name", "val"], k=rng.randint(2, 4))
        pairs = [f'"{k}": {rng.randint(1, 99)}' for k in keys]
        snippets.append({
            "code": "{" + ", ".join(pairs) + "}",
            "difficulty": "medium", "focus": "dict_literal"
        })

    # === Category 7: Function calls with parentheses ===
    funcs = ["len", "range", "int", "str", "float", "abs", "max", "min", "sum", "sorted", "type"]
    for _ in range(n_total // 12):
        f = rng.choice(funcs)
        arg = rng.choice(var_names)
        snippets.append({"code": f"{f}({arg})", "difficulty": "easy", "focus": "function_call"})

    # === Category 8: Multi-line if/else ===
    for _ in range(n_total // 12):
        v = rng.choice(var_names)
        num = rng.randint(0, 50)
        op = rng.choice(cmp_ops)
        s1 = rng.choice(strings)
        s2 = rng.choice(strings)
        code = f'if {v} {op} {num}:\n    print("{s1}")\nelse:\n    print("{s2}")'
        snippets.append({"code": code, "difficulty": "medium", "focus": "if_else"})

    # === Category 9: For loops with range ===
    for _ in range(n_total // 12):
        v = rng.choice(["i", "j", "k", "x", "n"])
        end = rng.randint(5, 20)
        body_var = rng.choice(["total", "count", "result", "sum_val"])
        body_op = rng.choice(["+=", "-=", "*="])
        code = f"for {v} in range({end}):\n    {body_var} {body_op} {v}"
        snippets.append({"code": code, "difficulty": "medium", "focus": "for_loop"})

    # === Category 10: While loops ===
    for _ in range(n_total // 15):
        v = rng.choice(var_names[:10])
        limit = rng.randint(10, 100)
        code = f"while {v} < {limit}:\n    {v} += 1"
        snippets.append({"code": code, "difficulty": "medium", "focus": "while_loop"})

    # === Category 11: Function definitions ===
    func_names = ["add", "multiply", "square", "double", "negate",
                  "increment", "decrement", "is_even", "is_positive", "factorial"]
    for _ in range(n_total // 12):
        fname = rng.choice(func_names)
        params = rng.choice(["n", "x", "a, b", "val", "n, m"])
        body_op = rng.choice(["+", "-", "*", "//", "**", "%"])
        if "," in params:
            p1, p2 = params.split(", ")
            body = f"return {p1} {body_op} {p2}"
        else:
            body = f"return {params} {body_op} {rng.randint(1, 10)}"
        code = f"def {fname}({params}):\n    {body}"
        snippets.append({"code": code, "difficulty": "medium", "focus": "function_def"})

    # === Category 12: List indexing and slicing ===
    for _ in range(n_total // 15):
        idx = rng.randint(0, 5)
        snippets.append({
            "code": f"data[{idx}]",
            "difficulty": "medium", "focus": "indexing"
        })
    for _ in range(n_total // 15):
        a, b = sorted(rng.sample(range(8), 2))
        snippets.append({
            "code": f"data[{a}:{b}]",
            "difficulty": "medium", "focus": "slicing"
        })

    # === Category 13: Complex expressions — nested brackets ===
    for _ in range(n_total // 15):
        v = rng.choice(var_names)
        nums = [str(rng.randint(1, 20)) for _ in range(3)]
        code = f"{v} = max([{nums[0]}, {nums[1]}, {nums[2]}])"
        snippets.append({"code": code, "difficulty": "hard", "focus": "nested_brackets"})

    # === Category 14: List comprehensions ===
    for _ in range(n_total // 15):
        v = rng.choice(["x", "i", "n"])
        op = rng.choice(["**", "+", "*"])
        n = rng.randint(2, 5)
        end = rng.randint(5, 15)
        code = f"result = [{v} {op} {n} for {v} in range({end})]"
        snippets.append({"code": code, "difficulty": "hard", "focus": "list_comp"})

    # === Category 15: Try-except ===
    for _ in range(n_total // 20):
        err_type = rng.choice(["ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError"])
        code = f'try:\n    result = int(x)\nexcept {err_type}:\n    print("error")'
        snippets.append({"code": code, "difficulty": "hard", "focus": "try_except"})

    # === Category 16: Class definitions ===
    for _ in range(n_total // 20):
        cls_name = rng.choice(["Counter", "Point", "Node", "Stack", "Queue", "Item"])
        attr = rng.choice(["value", "count", "data", "name", "size"])
        code = f"class {cls_name}:\n    def __init__(self):\n        self.{attr} = 0"
        snippets.append({"code": code, "difficulty": "hard", "focus": "class_def"})

    # === Category 17: F-strings and string formatting ===
    for _ in range(n_total // 15):
        v = rng.choice(var_names[:10])
        code = f'print(f"{v} = {{{v}}}")'
        snippets.append({"code": code, "difficulty": "hard", "focus": "fstring"})

    # === Category 18: Multiple assignments / unpacking ===
    for _ in range(n_total // 15):
        a, b = rng.sample(var_names[:8], 2)
        va, vb = rng.randint(1, 50), rng.randint(1, 50)
        code = f"{a}, {b} = {va}, {vb}"
        snippets.append({"code": code, "difficulty": "medium", "focus": "multi_assign"})

    # === Category 19: Boolean logic ===
    for _ in range(n_total // 15):
        v1 = rng.choice(var_names[:8])
        v2 = rng.choice(var_names[:8])
        bool_op = rng.choice(["and", "or", "not"])
        if bool_op == "not":
            code = f"if not {v1}:"
        else:
            code = f"if {v1} {bool_op} {v2}:"
        snippets.append({"code": code, "difficulty": "easy", "focus": "boolean"})

    # === Category 20: Return statements ===
    for _ in range(n_total // 15):
        v = rng.choice(var_names[:10])
        snippets.append({"code": f"return {v}", "difficulty": "easy", "focus": "return"})

    rng.shuffle(snippets)
    return snippets[:n_total]


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def _build_augraphy_pipeline():
    """Build an Augraphy augmentation pipeline for realistic handwriting."""
    if not AUGRAPHY_AVAILABLE:
        return None

    ink_phase = [
        augraphy.InkBleed(
            intensity_range=(0.1, 0.4),
            kernel_size=(3, 5),
            severity=(0.2, 0.4),
            p=0.5,
        ),
    ]

    paper_phase = [
        augraphy.NoiseTexturize(
            sigma_range=(2, 4),
            turbulence_range=(2, 5),
            p=0.4,
        ),
        augraphy.BrightnessTexturize(
            texturize_range=(0.9, 1.0),
            deviation=0.03,
            p=0.4,
        ),
    ]

    post_phase = [
        augraphy.Jpeg(quality_range=(50, 90), p=0.3),
        augraphy.SubtleNoise(p=0.4),
        augraphy.Geometric(
            scale=(0.97, 1.03),
            translation=(-5, 5),
            fliplr=0,
            flipud=0,
            crop=(),
            rotate_range=(-3, 3),
            padding_value=255,
            randomize=True,
            p=0.5,
        ),
    ]

    return augraphy.AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
    )


def _augment_basic(img_np: np.ndarray, rng: random.Random) -> np.ndarray:
    """Basic augmentation when Augraphy is not available.

    Applies: Gaussian noise, slight rotation, brightness variation.
    """
    h, w = img_np.shape[:2]

    # 1. Slight rotation (-3 to +3 degrees)
    angle = rng.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_np = cv2.warpAffine(img_np, M, (w, h), borderValue=(255, 255, 255))

    # 2. Gaussian noise
    if rng.random() < 0.6:
        noise = np.random.normal(0, rng.uniform(3, 12), img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 3. Brightness/contrast variation
    if rng.random() < 0.5:
        alpha = rng.uniform(0.85, 1.15)  # contrast
        beta = rng.randint(-15, 15)  # brightness
        img_np = np.clip(alpha * img_np.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 4. Slight blur
    if rng.random() < 0.3:
        ksize = rng.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

    # 5. Salt-and-pepper noise (simulates paper texture)
    if rng.random() < 0.4:
        num_salt = int(h * w * rng.uniform(0.001, 0.005))
        for _ in range(num_salt):
            y, x = rng.randint(0, h - 1), rng.randint(0, w - 1)
            img_np[y, x] = rng.choice([0, 255])

    return img_np


def augment_image(img_pil: Image.Image, rng: random.Random, aug_pipeline=None) -> Image.Image:
    """Apply augmentation to a rendered code image."""
    img_np = np.array(img_pil)

    if aug_pipeline is not None and AUGRAPHY_AVAILABLE:
        try:
            img_np = aug_pipeline(img_np)
        except Exception:
            # Fallback to basic if augraphy fails on a particular image
            img_np = _augment_basic(img_np, rng)
    else:
        img_np = _augment_basic(img_np, rng)

    return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def render_code_image(
    code: str,
    font: ImageFont.FreeTypeFont,
    bg_color: tuple,
    ink_color: tuple,
    line_spacing: int,
) -> Image.Image:
    """Render a code string as a white-background image."""
    lines = code.split("\n")
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line if line else " ", font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_h = max(line_heights) if line_heights else 28
    total_h = line_h * len(lines) + line_spacing * (len(lines) - 1) + 2 * PADDING
    total_w = max(line_widths) + 2 * PADDING if line_widths else 200 + 2 * PADDING
    total_w = max(total_w, 100)  # minimum width

    img = Image.new("RGB", (total_w, total_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = PADDING
    for line in lines:
        draw.text((PADDING, y), line, font=font, fill=ink_color)
        y += line_h + line_spacing

    return img


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N_TOTAL = 1000
    print(f"Generating {N_TOTAL} Python code snippets...")
    snippets = _generate_snippets(N_TOTAL, rng)
    print(f"  Generated {len(snippets)} snippets")

    # Load fonts
    available_fonts = []
    for name, path in FONTS:
        if os.path.exists(path):
            available_fonts.append((name, path))
    print(f"  Available fonts: {len(available_fonts)}")

    # Build augmentation pipeline
    aug_pipeline = None
    if AUGRAPHY_AVAILABLE:
        try:
            aug_pipeline = _build_augraphy_pipeline()
            print("  Augraphy pipeline built.")
        except Exception as e:
            print(f"  Augraphy pipeline failed: {e}. Using basic augmentation.")
            aug_pipeline = None
    else:
        print("  Using basic augmentation (numpy/cv2).")

    # Split into train/val/test
    rng.shuffle(snippets)
    n_train = int(N_TOTAL * TRAIN_RATIO)
    n_val = int(N_TOTAL * VAL_RATIO)
    n_test = N_TOTAL - n_train - n_val

    splits = {
        "train": snippets[:n_train],
        "val": snippets[n_train:n_train + n_val],
        "test": snippets[n_train + n_val:],
    }

    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")

    # Track stats
    stats = {
        "total": N_TOTAL,
        "splits": {"train": n_train, "val": n_val, "test": n_test},
        "fonts_used": [name for name, _ in available_fonts],
        "augmentation": "augraphy" if aug_pipeline else "basic_numpy_cv2",
        "dataset_type": "SYNTHETIC (font-rendered + augmented, NOT real handwriting)",
        "focus_distribution": {},
        "difficulty_distribution": {},
    }

    # Count distributions
    for s in snippets:
        focus = s.get("focus", "unknown")
        diff = s.get("difficulty", "unknown")
        stats["focus_distribution"][focus] = stats["focus_distribution"].get(focus, 0) + 1
        stats["difficulty_distribution"][diff] = stats["difficulty_distribution"].get(diff, 0) + 1

    # Generate images for each split
    for split_name, split_snippets in splits.items():
        split_dir = OUTPUT_DIR / split_name / "images"
        split_dir.mkdir(parents=True, exist_ok=True)

        labels = {}

        print(f"\n  Rendering {split_name} set ({len(split_snippets)} samples)...")
        for idx, snippet in enumerate(split_snippets):
            code = snippet["code"]
            sid = f"{split_name}_{idx:04d}"

            # Random font, size, colors
            font_name, font_path = rng.choice(available_fonts)
            font_size = rng.choice(FONT_SIZES)
            bg_color = rng.choice(BG_COLORS)
            ink_color = rng.choice(INK_COLORS)
            line_spacing = rng.randint(*LINE_SPACING_RANGE)

            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.truetype(available_fonts[0][1], font_size)

            # Render
            img = render_code_image(code, font, bg_color, ink_color, line_spacing)

            # Augment
            img = augment_image(img, rng, aug_pipeline)

            # Save
            img_path = split_dir / f"{sid}.png"
            img.save(img_path)
            labels[f"{sid}.png"] = code

            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"    [{idx + 1:4d}/{len(split_snippets)}] {sid} ({snippet['difficulty']}, {snippet['focus']})")

        # Save labels
        labels_path = OUTPUT_DIR / split_name / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)

        print(f"    Saved {len(labels)} samples to {split_dir.parent}/")

    # Save metadata
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset generation complete!")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Metadata: {meta_path}")
    print(f"\n  WARNING: SYNTHETIC dataset — NOT real handwriting.")


if __name__ == "__main__":
    main()
