"""
benchmark.py
------------
Benchmarks the Handwritten Code Evaluator pipeline against a labelled
dataset of (image, ground-truth) pairs.

Metrics computed
----------------
CER  — Character Error Rate  = edit_distance(predicted, reference) / len(reference)
WER  — Word Error Rate       = edit_distance(tokens_pred, tokens_ref) / len(tokens_ref)
       (token = split on whitespace)
EM   — Exact Match           = 1 if predicted.strip() == reference.strip() else 0
ESR  — Execution Success Rate = 1 if ast.parse(corrected) succeeds (Python only)

Outputs
-------
  benchmark_results/
    per_sample.csv   -- one row per sample with all metrics
    summary.json     -- aggregate means + metadata
    summary.txt      -- human-readable report

Usage
-----
  python benchmark.py [--dataset eval_dataset] [--output benchmark_results]

WARNING: SYNTHETIC PROXY DATA
This script was written alongside a dataset generated from Inkfree.ttf,
a digitally clean handwriting-style Windows font.  Numbers produced here
MUST be reported as:
  "Synthetic proxy dataset (Inkfree font) -- not real handwriting"
They cannot be used as evidence of real-world handwriting recognition accuracy.
"""

import argparse
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Levenshtein (edit distance) — pure Python, no extra deps
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Use two-row rolling array
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, [0] * (lb + 1)
    return prev[lb]


def cer(predicted: str, reference: str) -> float:
    """Character Error Rate. Returns 0.0–1.0+ (can exceed 1 for short refs)."""
    if not reference:
        return 0.0 if not predicted else 1.0
    return _edit_distance(predicted, reference) / len(reference)


def wer(predicted: str, reference: str) -> float:
    """Word Error Rate (whitespace-tokenised). Returns 0.0–1.0+."""
    ref_tokens = reference.split()
    pred_tokens = predicted.split()
    if not ref_tokens:
        return 0.0 if not pred_tokens else 1.0
    return _edit_distance(pred_tokens, ref_tokens) / len(ref_tokens)


# WER on token lists (needed for list inputs)
def _token_edit_distance(a: list, b: list) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, [0] * (lb + 1)
    return prev[lb]


def wer_tokens(predicted: str, reference: str) -> float:
    ref_toks = reference.split()
    pred_toks = predicted.split()
    if not ref_toks:
        return 0.0 if not pred_toks else 1.0
    return _token_edit_distance(pred_toks, ref_toks) / len(ref_toks)


def exact_match(predicted: str, reference: str) -> bool:
    """Normalised exact match: strip leading/trailing whitespace."""
    return predicted.strip() == reference.strip()


def can_parse(code: str) -> bool:
    """Return True if ast.parse succeeds on the code string."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_pipeline():
    """Import and warm-up the OCR + correction pipeline."""
    from ocr.preprocessing import preprocess_image
    from ocr.line_segmentation import segment_lines
    from ocr.trocr_engine import TrOCREngine
    from ocr.postprocessing import correct_syntax

    print("Loading TrOCR model...", flush=True)
    t0 = time.time()
    engine = TrOCREngine.get_instance()
    print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

    return preprocess_image, segment_lines, engine, correct_syntax


def run_sample(image_path: Path, preprocess_image, segment_lines, engine, correct_syntax):
    """Run one image through the full pipeline.

    Returns a dict with:
      raw_ocr, corrected, parse_ok, pipeline_time_s
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    t0 = time.time()

    # Step 1: preprocess
    preprocessed = preprocess_image(img_bytes)

    # Step 2: line segmentation
    try:
        line_images = segment_lines(preprocessed)
    except Exception:
        line_images = [(preprocessed, 0)]

    # Step 3: OCR
    if len(line_images) > 1:
        raw_ocr = engine.recognize_lines(line_images)
    else:
        raw_ocr = engine.recognize(line_images[0][0])

    # Step 4: syntax correction (Python only dataset)
    corrected, _ok, _diag = correct_syntax(raw_ocr, language="python")

    pipeline_time = time.time() - t0

    parse_ok = can_parse(corrected)

    return {
        "raw_ocr": raw_ocr,
        "corrected": corrected,
        "parse_ok": parse_ok,
        "pipeline_time_s": round(pipeline_time, 3),
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(dataset_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found in {dataset_dir}")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"\nDataset: {dataset_dir}  ({len(manifest)} samples)")
    print("-" * 60)

    # Load pipeline once
    preprocess_image, segment_lines, engine, correct_syntax = load_pipeline()

    rows = []
    total_cer, total_wer, total_em, total_esr = 0.0, 0.0, 0.0, 0.0

    by_difficulty = {"easy": [], "medium": [], "hard": []}

    for i, entry in enumerate(manifest):
        sid = entry["id"]
        img_path = dataset_dir / entry["image"]
        gt_path = dataset_dir / entry["ground_truth"]
        difficulty = entry.get("difficulty", "unknown")

        reference = gt_path.read_text(encoding="utf-8")

        print(f"[{i+1:02d}/{len(manifest)}] sample_{sid} ({difficulty})", end="", flush=True)

        try:
            result = run_sample(img_path, preprocess_image, segment_lines, engine, correct_syntax)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = {
                "raw_ocr": "",
                "corrected": "",
                "parse_ok": False,
                "pipeline_time_s": 0.0,
            }

        predicted_raw = result["raw_ocr"]
        predicted_corrected = result["corrected"]

        # Compute metrics on the *corrected* output (what the user sees)
        c = cer(predicted_corrected, reference)
        w = wer_tokens(predicted_corrected, reference)
        em = 1 if exact_match(predicted_corrected, reference) else 0
        es = 1 if result["parse_ok"] else 0

        total_cer += c
        total_wer += w
        total_em += em
        total_esr += es

        diff_list = by_difficulty.get(difficulty, [])
        diff_list.append({"cer": c, "wer": w, "em": em, "es": es})
        by_difficulty[difficulty] = diff_list

        row = {
            "id": sid,
            "difficulty": difficulty,
            "note": entry.get("note", ""),
            "lines": entry.get("lines", 1),
            "chars_gt": entry.get("chars", len(reference)),
            "raw_ocr": predicted_raw.replace("\n", "\\n"),
            "corrected": predicted_corrected.replace("\n", "\\n"),
            "reference": reference.replace("\n", "\\n"),
            "cer": round(c, 4),
            "wer": round(w, 4),
            "exact_match": em,
            "parse_ok": int(result["parse_ok"]),
            "pipeline_time_s": result["pipeline_time_s"],
        }
        rows.append(row)

        status = "PASS" if em else ("parse_ok" if es else "FAIL")
        print(f"  CER={c:.3f}  WER={w:.3f}  EM={em}  ES={es}  [{status}]  {result['pipeline_time_s']:.1f}s")

    n = len(manifest)
    avg_cer = total_cer / n
    avg_wer = total_wer / n
    avg_em = total_em / n
    avg_esr = total_esr / n

    # Per-difficulty averages
    diff_summary = {}
    for diff, items in by_difficulty.items():
        if items:
            diff_summary[diff] = {
                "n": len(items),
                "avg_cer": round(sum(x["cer"] for x in items) / len(items), 4),
                "avg_wer": round(sum(x["wer"] for x in items) / len(items), 4),
                "exact_match_rate": round(sum(x["em"] for x in items) / len(items), 4),
                "execution_success_rate": round(sum(x["es"] for x in items) / len(items), 4),
            }

    # --- Write per-sample CSV ---
    csv_path = output_dir / "per_sample.csv"
    fieldnames = [
        "id", "difficulty", "note", "lines", "chars_gt",
        "cer", "wer", "exact_match", "parse_ok", "pipeline_time_s",
        "raw_ocr", "corrected", "reference",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Write summary JSON ---
    summary = {
        "dataset": str(dataset_dir),
        "n_samples": n,
        "dataset_type": "SYNTHETIC_PROXY (Inkfree font, NOT real handwriting)",
        "model": "microsoft/trocr-base-handwritten",
        "aggregate": {
            "avg_cer": round(avg_cer, 4),
            "avg_wer": round(avg_wer, 4),
            "exact_match_accuracy": round(avg_em, 4),
            "execution_success_rate": round(avg_esr, 4),
        },
        "by_difficulty": diff_summary,
        "notes": [
            "CER and WER can exceed 1.0 for very short reference strings with many OCR errors.",
            "Execution Success Rate uses ast.parse() -- it checks syntax validity, not runtime correctness.",
            "Exact Match requires the corrected output to exactly equal the ground truth after strip().",
            "SYNTHETIC PROXY: Images rendered with Inkfree.ttf, a clean digital handwriting font.",
            "These numbers likely OVERESTIMATE accuracy vs. real photographed handwriting.",
        ],
    }
    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Write human-readable report ---
    report_lines = [
        "=" * 65,
        "HANDWRITTEN CODE EVALUATOR -- BENCHMARK REPORT",
        "=" * 65,
        "",
        "IMPORTANT: SYNTHETIC PROXY DATASET",
        "  Images were rendered with Inkfree.ttf (a clean digital",
        "  handwriting-style font bundled with Windows), NOT from",
        "  real photographed handwriting samples.",
        "  Results likely OVERESTIMATE real-world accuracy.",
        "  Label these as: 'Synthetic proxy (Inkfree font)'.",
        "",
        f"Model      : microsoft/trocr-base-handwritten",
        f"Dataset    : {dataset_dir}",
        f"Samples    : {n}",
        "",
        "AGGREGATE METRICS (on corrected output vs. ground truth)",
        "-" * 65,
        f"  CER  (Character Error Rate)   : {avg_cer:.4f}  ({avg_cer*100:.1f}%)",
        f"  WER  (Word/Token Error Rate)  : {avg_wer:.4f}  ({avg_wer*100:.1f}%)",
        f"  EM   (Exact Match Accuracy)   : {avg_em:.4f}  ({avg_em*100:.1f}%)",
        f"  ESR  (Execution Success Rate) : {avg_esr:.4f}  ({avg_esr*100:.1f}%)",
        "",
        "BY DIFFICULTY",
        "-" * 65,
    ]
    for diff in ["easy", "medium", "hard"]:
        if diff in diff_summary:
            d = diff_summary[diff]
            report_lines += [
                f"  {diff.upper():6s} (n={d['n']})",
                f"    CER={d['avg_cer']:.4f}  WER={d['avg_wer']:.4f}  "
                f"EM={d['exact_match_rate']:.4f}  ESR={d['execution_success_rate']:.4f}",
            ]

    report_lines += [
        "",
        "PER-SAMPLE RESULTS",
        "-" * 65,
    ]
    for r in rows:
        em_str = "EXACT" if r["exact_match"] else ("parse" if r["parse_ok"] else "FAIL ")
        report_lines.append(
            f"  [{r['id']}] {r['difficulty']:6s} {em_str}  "
            f"CER={r['cer']:.3f}  WER={r['wer']:.3f}  "
            f"-- {r['note']}"
        )

    report_lines += [
        "",
        "METRIC DEFINITIONS",
        "-" * 65,
        "  CER  = Levenshtein(predicted, reference) / len(reference)",
        "  WER  = token-level Levenshtein / len(reference_tokens)",
        "  EM   = predicted.strip() == reference.strip()",
        "  ESR  = ast.parse(corrected_code) succeeds without SyntaxError",
        "",
        "OUTPUT FILES",
        "-" * 65,
        f"  Per-sample CSV : {csv_path}",
        f"  Summary JSON   : {summary_json_path}",
        "=" * 65,
    ]

    report_text = "\n".join(report_lines)
    report_path = output_dir / "summary.txt"
    report_path.write_text(report_text, encoding="utf-8")

    # Print the report to stdout
    print()
    print(report_text)
    print()
    print(f"Results written to: {output_dir}/")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the Handwritten Code Evaluator.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval_dataset"),
        help="Path to dataset directory containing manifest.json and image/gt pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to write benchmark results to.",
    )
    args = parser.parse_args()

    run_benchmark(args.dataset, args.output)
