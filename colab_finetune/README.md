# TrOCR Single-Line Fine-Tuning Package

Self-contained package for fine-tuning TrOCR on single-line handwritten
Python code. Designed for Google Colab (T4 GPU) or Kaggle (P100).

## Why single-line?

TrOCR was pre-trained on IAM — a dataset of single-line handwriting strips.
The previous fine-tuning attempt fed it multi-line code images with flattened
labels, violating this assumption. The fine-tuned model learned code vocabulary
but couldn't produce valid Python syntax (ESR dropped 22% -> 6%).

This approach fixes the architecture mismatch: each training image contains
exactly one line of code, matching what TrOCR expects.

## Quick Start (Colab)

```python
# Cell 1: Upload this folder to Colab (or clone from GitHub)
!git clone https://github.com/SiddharthSeng/Handwritten-Code-Evaluator.git
%cd Handwritten-Code-Evaluator

# Cell 2: Install dependencies
!pip install -r colab_finetune/requirements.txt
!apt-get install -y fonts-comic-neue fonts-freefont-ttf fonts-dejavu 2>/dev/null

# Cell 3: Copy the source labels into the colab_finetune working dir
!cp -r finetune_dataset colab_finetune/finetune_dataset

# Cell 4: Generate single-line dataset
%cd colab_finetune
!python generate_singleline_dataset.py

# Cell 5: Fine-tune (T4 GPU, ~15-25 min for 15 epochs)
!python finetune_singleline.py --epochs 15 --batch-size 8

# Cell 6: Download the model
from google.colab import files
!zip -r finetuned_singleline.zip finetuned_singleline/final/
files.download('finetuned_singleline.zip')
```

## Files

| File | Purpose |
|------|---------|
| `generate_singleline_dataset.py` | Splits multi-line snippets into per-line crops |
| `finetune_singleline.py` | Fine-tuning script with EpochLogger |
| `requirements.txt` | pip dependencies |
| `README.md` | This file |

## Prerequisites

This package reads from `finetune_dataset/` (the existing multi-line dataset).
That directory must exist with `train/labels.json`, `val/labels.json`, and
`test/labels.json` before running `generate_singleline_dataset.py`.

## Expected dataset sizes

| Split | Snippets | Single-line images |
|-------|----------|--------------------|
| Train | 800 | ~1,134 |
| Val   | 100 | ~147 |
| Test  | 100 | ~151 |
| Total | 1,000 | ~1,432 |

## GPU time estimates

| Setting | T4 (Colab free) | P100 (Kaggle) | CPU |
|---------|-----------------|---------------|-----|
| 15 epochs, batch=8 | ~15-25 min | ~10-18 min | ~5-8 hours |
| 5 epochs, batch=8 | ~5-8 min | ~4-6 min | ~2-3 hours |

## Inference-time reconstruction

After fine-tuning, the model recognizes individual lines. To process a
full handwritten code image at inference time:

1. `line_segmentation.segment_lines(image)` -> list of (line_image, x_offset)
2. Run fine-tuned TrOCR on each line_image individually -> list of text strings
3. Use `_compute_indent_levels(offsets)` to convert pixel x-offsets to indent levels
4. Prepend `"    " * indent_level` to each line's text
5. Join with newlines

This is already implemented in `ocr/trocr_engine.py`'s `recognize_lines()`.
To use the fine-tuned model, update `_MODEL_NAME` in `trocr_engine.py`
to point to the downloaded `finetuned_singleline/final/` directory.
