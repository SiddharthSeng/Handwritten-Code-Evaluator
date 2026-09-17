"""
generate_singleline_dataset.py
==============================
Restructures the existing multi-line finetune_dataset/ into per-line
single-line crops suitable for TrOCR fine-tuning.

Architecture rationale:
  TrOCR was pre-trained on IAM single-line handwriting strips.  Its
  encoder (ViT-base) expects a single line of text per image, and
  its decoder (RoBERTa cross-attention) produces one text sequence per
  image.  The previous fine-tuning attempt fed it multi-line images
  with flattened labels, which violated this assumption and produced
  "code-flavored soup" that didn't parse.

  This script renders each individual line of code as its own image
  so the model sees exactly what it was designed for: one clean line.

Data flow:
  1. Re-read the snippet definitions (via labels.json in each split).
  2. For each snippet, split the code string on "\\n" into individual
     lines.  Each line becomes its own training sample.
  3. Render each line individually as a fresh image (random font,
     size, colors) and augment separately.
  4. The label is just that one stripped line's text (leading whitespace
     stripped -- indentation is NOT part of the OCR task; it's
     reconstructed at inference time from pixel x-offsets).
  5. All lines from the same snippet stay in the same split
     (train/val/test) to prevent data leakage.

Output structure:
  singleline_dataset/
    train/
      images/  *.png
      labels.json  {"filename.png": "one line of code"}
    val/
      images/  *.png
      labels.json
    test/
      images/  *.png
      labels.json
    metadata.json

This script is designed to run on both Windows (local) and
Linux (Colab/Kaggle).  Font paths are auto-detected per platform.
"""

import json
import os
import platform
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional Augraphy
try:
    import augraphy
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = Path("finetune_dataset")      # existing multi-line dataset
OUTPUT_DIR = Path("singleline_dataset")   # new per-line dataset
RANDOM_SEED = 123                          # different seed from original

FONT_SIZES = [22, 24, 26, 28, 30, 32]
BG_COLORS = [(255, 255, 255), (250, 248, 240), (245, 245, 245), (240, 235, 225)]
INK_COLORS = [(0, 0, 0), (10, 10, 30), (25, 25, 50), (0, 0, 100), (50, 50, 50)]
PADDING = 15

# ---------------------------------------------------------------------------
# Font detection (cross-platform)
# ---------------------------------------------------------------------------

def _detect_fonts() -> list[tuple[str, str]]:
    """Find usable handwriting-style fonts on this system."""
    candidates = []

    if platform.system() == "Windows":
        candidates = [
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
    else:
        # Linux / Colab -- look for common handwriting-like fonts
        linux_paths = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]
        # On Colab, install fonts with: !apt-get install -y fonts-comic-neue fonts-freefont-ttf
        # We'll also bundle a small fallback font set
        common_linux = [
            ("ComicNeue", "/usr/share/fonts/truetype/comic-neue/ComicNeue-Regular.ttf"),
            ("FreeSans", "/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
            ("FreeSerif", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"),
            ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        candidates = common_linux

        # Also scan for any .ttf files in common dirs
        for font_dir in linux_paths:
            if os.path.isdir(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for f in files:
                        if f.lower().endswith(('.ttf', '.otf')):
                            name = Path(f).stem
                            full = os.path.join(root, f)
                            if (name, full) not in candidates:
                                candidates.append((name, full))

    usable = []
    for name, path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 24)
                bbox = font.getbbox("x = 10")
                if bbox:
                    usable.append((name, path))
            except Exception:
                pass

    if not usable:
        # Absolute fallback: Pillow's built-in bitmap font
        print("WARNING: No TrueType fonts found. Using Pillow default.")
        usable = [("default", None)]

    return usable


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _build_augraphy_pipeline():
    if not AUGRAPHY_AVAILABLE:
        return None
    try:
        return augraphy.AugraphyPipeline(
            ink_phase=[
                augraphy.InkBleed(intensity_range=(0.1, 0.4), kernel_size=(3, 5),
                                  severity=(0.2, 0.4), p=0.5),
            ],
            paper_phase=[
                augraphy.NoiseTexturize(sigma_range=(2, 4), turbulence_range=(2, 5), p=0.4),
                augraphy.BrightnessTexturize(texturize_range=(0.9, 1.0), deviation=0.03, p=0.4),
            ],
            post_phase=[
                augraphy.Jpeg(quality_range=(50, 90), p=0.3),
                augraphy.SubtleNoise(p=0.4),
                augraphy.Geometric(scale=(0.97, 1.03), translation=(-5, 5),
                                   fliplr=0, flipud=0, crop=(),
                                   rotate_range=(-3, 3), padding_value=255,
                                   randomize=True, p=0.5),
            ],
        )
    except Exception as e:
        print(f"Augraphy pipeline error: {e}. Using basic augmentation.")
        return None


def _augment_basic(img_np, rng):
    h, w = img_np.shape[:2]
    # Slight rotation
    angle = rng.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_np = cv2.warpAffine(img_np, M, (w, h), borderValue=(255, 255, 255))
    # Gaussian noise
    if rng.random() < 0.6:
        noise = np.random.normal(0, rng.uniform(3, 12), img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Brightness/contrast
    if rng.random() < 0.5:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.randint(-15, 15)
        img_np = np.clip(alpha * img_np.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return img_np


def augment_image(img_pil, rng, pipeline=None):
    img_np = np.array(img_pil)
    if pipeline is not None:
        try:
            img_np = pipeline(img_np)
        except Exception:
            img_np = _augment_basic(img_np, rng)
    else:
        img_np = _augment_basic(img_np, rng)
    return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# Single-line image rendering
# ---------------------------------------------------------------------------

def render_single_line(text, font, bg_color, ink_color):
    """Render one line of code as an image."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text if text.strip() else " ", font=font)
    w = bbox[2] - bbox[0] + 2 * PADDING
    h = bbox[3] - bbox[1] + 2 * PADDING
    w = max(w, 80)
    h = max(h, 30)

    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((PADDING, PADDING), text.strip(), font=font, fill=ink_color)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    fonts = _detect_fonts()
    print(f"Found {len(fonts)} usable font(s)")

    pipeline = _build_augraphy_pipeline()
    if pipeline:
        print("Augraphy pipeline active.")
    else:
        print("Using basic augmentation.")

    stats = {
        "total_snippets": 0,
        "total_single_lines": 0,
        "splits": {},
        "fonts": [n for n, _ in fonts],
        "augmentation": "augraphy" if pipeline else "basic",
        "dataset_type": "SYNTHETIC SINGLE-LINE CROPS (font-rendered + augmented)",
        "note": "Each line of code rendered as its own image. "
                "Indentation whitespace STRIPPED from labels -- "
                "indentation is reconstructed at inference time from pixel x-offsets.",
    }

    for split in ["train", "val", "test"]:
        src_labels_path = INPUT_DIR / split / "labels.json"
        if not src_labels_path.exists():
            print(f"WARNING: {src_labels_path} not found, skipping {split}")
            continue

        with open(src_labels_path, encoding="utf-8") as f:
            snippet_labels = json.load(f)

        out_dir = OUTPUT_DIR / split / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        line_labels = {}
        line_count = 0

        for snippet_id, code in sorted(snippet_labels.items()):
            # Split into individual lines
            code_lines = code.split("\n")

            for line_idx, line_text in enumerate(code_lines):
                stripped = line_text.strip()
                if not stripped:
                    continue  # skip blank lines

                # Generate image
                font_name, font_path = rng.choice(fonts)
                font_size = rng.choice(FONT_SIZES)
                bg = rng.choice(BG_COLORS)
                ink = rng.choice(INK_COLORS)

                if font_path is not None:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()

                img = render_single_line(stripped, font, bg, ink)
                img = augment_image(img, rng, pipeline)

                # Filename: snippet_id + line index
                base = Path(snippet_id).stem
                fname = f"{base}_L{line_idx:02d}.png"
                img.save(out_dir / fname)

                # Label = stripped text (no leading whitespace)
                line_labels[fname] = stripped
                line_count += 1

        # Save labels
        labels_path = OUTPUT_DIR / split / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(line_labels, f, indent=2, ensure_ascii=False)

        stats["splits"][split] = {
            "snippets": len(snippet_labels),
            "single_lines": line_count,
        }
        stats["total_snippets"] += len(snippet_labels)
        stats["total_single_lines"] += line_count

        print(f"  {split}: {len(snippet_labels)} snippets -> {line_count} single-line images")

    # Save metadata
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! {stats['total_single_lines']} single-line images in {OUTPUT_DIR}/")
    print(f"  Data leakage check: all lines from each snippet stay in the same split.")


if __name__ == "__main__":
    main()
