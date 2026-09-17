"""
benchmark_finetune_test.py
--------------------------
Runs CER / WER / Exact-Match / Execution-Success-Rate on the held-out
finetune_dataset/test/ set (100 samples, never seen during training
or validation) for TWO models back-to-back:

  1. Baseline  : microsoft/trocr-base-handwritten  (original model)
  2. Fine-tuned: finetuned_model/final              (after 5-epoch training)

Both runs use identical preprocessing, the same 100 test images, and
the same metric computation — so the comparison is apples-to-apples.

Output
------
  benchmark_results_finetuned/
    baseline_per_sample.csv
    finetuned_per_sample.csv
    comparison.json     <- machine-readable before/after summary
    comparison.txt      <- human-readable report
"""

import ast
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------------------------------------------------------------------
# Metrics (same levenshtein logic as original benchmark.py)
# ---------------------------------------------------------------------------

def _edit_distance(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
        prev, curr = curr, [0]*(lb+1)
    return prev[lb]

def _token_edit_distance(a, b):
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
        prev, curr = curr, [0]*(lb+1)
    return prev[lb]

def cer(pred, ref):
    if not ref: return 0.0 if not pred else 1.0
    return _edit_distance(pred, ref) / len(ref)

def wer(pred, ref):
    rt = ref.split(); pt = pred.split()
    if not rt: return 0.0 if not pt else 1.0
    return _token_edit_distance(pt, rt) / len(rt)

def exact_match(pred, ref):
    return pred.strip() == ref.strip()

def can_parse(code):
    # Unflatten \n tokens before parsing
    code = code.replace(" \\n ", "\n").replace("\\n", "\n")
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# ---------------------------------------------------------------------------
# Single-model benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark_for_model(model_name_or_path, test_dir: Path, output_dir: Path,
                             tag: str, device: str):
    print(f"\n{'='*60}")
    print(f"  Running benchmark: {tag}")
    print(f"  Model : {model_name_or_path}")
    print(f"  Test  : {test_dir}")
    print(f"{'='*60}\n")

    # Load model
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Load test labels
    labels_path = test_dir / "labels.json"
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    image_files = sorted(labels.keys())
    print(f"  Test samples: {len(image_files)}\n")

    rows = []
    total_cer = total_wer = total_em = total_esr = 0.0

    for idx, img_file in enumerate(image_files):
        img_path = test_dir / "images" / img_file
        # Ground truth — the label was stored as flattened code
        # (newlines replaced with " \n " during dataset generation)
        reference_flat = labels[img_file].replace("\n", " \\n ")
        reference_code = labels[img_file]  # original multi-line for ESR

        # Run inference
        t1 = time.time()
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        infer_time = time.time() - t1

        # Compute metrics
        c = cer(predicted, reference_flat)
        w = wer(predicted, reference_flat)
        em = 1 if exact_match(predicted, reference_flat) else 0
        es = 1 if can_parse(predicted) else 0

        total_cer += c
        total_wer += w
        total_em  += em
        total_esr += es

        rows.append({
            "id": img_file,
            "cer": round(c, 4),
            "wer": round(w, 4),
            "exact_match": em,
            "parse_ok": es,
            "infer_time_s": round(infer_time, 2),
            "predicted": predicted.replace("\n", "\\n")[:120],
            "reference": reference_flat.replace("\n", "\\n")[:120],
        })

        if (idx+1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1:3d}/100]  CER={c:.3f}  WER={w:.3f}  EM={em}  ES={es}  "
                  f"  pred={predicted[:50]!r}")

    n = len(image_files)
    summary = {
        "tag": tag,
        "model": str(model_name_or_path),
        "n_samples": n,
        "avg_cer":  round(total_cer / n, 4),
        "avg_wer":  round(total_wer / n, 4),
        "exact_match_accuracy": round(total_em / n, 4),
        "execution_success_rate": round(total_esr / n, 4),
    }

    # Write per-sample CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{tag}_per_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results for [{tag}]:")
    print(f"    CER  : {summary['avg_cer']:.4f}  ({summary['avg_cer']*100:.1f}%)")
    print(f"    WER  : {summary['avg_wer']:.4f}  ({summary['avg_wer']*100:.1f}%)")
    print(f"    EM   : {summary['exact_match_accuracy']:.4f}  ({summary['exact_match_accuracy']*100:.1f}%)")
    print(f"    ESR  : {summary['execution_success_rate']:.4f}  ({summary['execution_success_rate']*100:.1f}%)")
    print(f"  CSV  : {csv_path}")

    # Free memory before next model
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return summary, rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir = Path("finetune_dataset/test")
    output_dir = Path("benchmark_results_finetuned")

    assert test_dir.exists(), f"Test dir not found: {test_dir}"
    assert (Path("finetuned_model/final")).exists(), "finetuned_model/final not found — did training complete?"

    # --- Run 1: Baseline ---
    baseline_summary, baseline_rows = run_benchmark_for_model(
        model_name_or_path="microsoft/trocr-base-handwritten",
        test_dir=test_dir,
        output_dir=output_dir,
        tag="baseline",
        device=device,
    )

    # --- Run 2: Fine-tuned ---
    finetuned_summary, finetuned_rows = run_benchmark_for_model(
        model_name_or_path="finetuned_model/final",
        test_dir=test_dir,
        output_dir=output_dir,
        tag="finetuned",
        device=device,
    )

    # --- Build comparison ---
    def delta(ft, bl, pct=False):
        d = ft - bl
        sign = "+" if d > 0 else ""
        if pct:
            return f"{sign}{d*100:.1f}pp"
        return f"{sign}{d:.4f}"

    comparison = {
        "test_set": str(test_dir),
        "n_samples": 100,
        "dataset_type": "SYNTHETIC PROXY (Inkfree font + augmentation, NOT real handwriting)",
        "note": "Both models evaluated on identical held-out test set (never seen during training/val)",
        "baseline": baseline_summary,
        "finetuned": finetuned_summary,
        "delta": {
            "cer_abs":  round(finetuned_summary["avg_cer"]  - baseline_summary["avg_cer"],  4),
            "wer_abs":  round(finetuned_summary["avg_wer"]  - baseline_summary["avg_wer"],  4),
            "em_abs":   round(finetuned_summary["exact_match_accuracy"] - baseline_summary["exact_match_accuracy"], 4),
            "esr_abs":  round(finetuned_summary["execution_success_rate"] - baseline_summary["execution_success_rate"], 4),
        },
    }

    comp_json_path = output_dir / "comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # --- Human-readable report ---
    bl = baseline_summary
    ft = finetuned_summary

    # CER/WER/EM: lower is better (negative delta = improvement)
    # ESR: higher is better (positive delta = improvement)
    def cer_arrow(d): return "IMPROVED" if d < 0 else ("REGRESSED" if d > 0 else "UNCHANGED")
    def esr_arrow(d): return "IMPROVED" if d > 0 else ("REGRESSED" if d < 0 else "UNCHANGED")

    report = [
        "=" * 65,
        "FINE-TUNING BENCHMARK REPORT -- BEFORE vs AFTER",
        "=" * 65,
        "",
        "DATASET: SYNTHETIC PROXY (Inkfree font + Augraphy augmentation)",
        "  NOT real photographed handwriting. Results label as:",
        "  'Synthetic fine-tuning benchmark, not real handwriting.'",
        "",
        f"Test set : {test_dir}  (100 samples, never seen during training)",
        f"Device   : {device.upper()}",
        "",
        "METRIC DEFINITIONS",
        "  CER = char edit distance / len(reference)   [lower is better]",
        "  WER = token edit distance / len(ref tokens) [lower is better]",
        "  EM  = exact string match                    [higher is better]",
        "  ESR = ast.parse(predicted) succeeds         [higher is better]",
        "",
        "-" * 65,
        f"{'Metric':<12} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>12} {'Result':>12}",
        "-" * 65,
        f"{'CER':<12} {bl['avg_cer']:>11.4f} {ft['avg_cer']:>11.4f} "
        f"{delta(ft['avg_cer'], bl['avg_cer']):>12}  {cer_arrow(ft['avg_cer']-bl['avg_cer'])}",
        f"{'WER':<12} {bl['avg_wer']:>11.4f} {ft['avg_wer']:>11.4f} "
        f"{delta(ft['avg_wer'], bl['avg_wer']):>12}  {cer_arrow(ft['avg_wer']-bl['avg_wer'])}",
        f"{'EM Acc.':<12} {bl['exact_match_accuracy']:>11.4f} {ft['exact_match_accuracy']:>11.4f} "
        f"{delta(ft['exact_match_accuracy'], bl['exact_match_accuracy']):>12}  {esr_arrow(ft['exact_match_accuracy']-bl['exact_match_accuracy'])}",
        f"{'ESR':<12} {bl['execution_success_rate']:>11.4f} {ft['execution_success_rate']:>11.4f} "
        f"{delta(ft['execution_success_rate'], bl['execution_success_rate']):>12}  {esr_arrow(ft['execution_success_rate']-bl['execution_success_rate'])}",
        "-" * 65,
        "",
        "TRAINING SUMMARY (from finetuned_model/training.log)",
        "  Epoch 1:  train=3.4649  eval=1.7761  (20m 06s)",
        "  Epoch 2:  train=1.4191  eval=1.2479  (19m 50s)",
        "  Epoch 3:  train=0.8530  eval=0.9689  (20m 03s)",
        "  Epoch 4:  train=0.5469  eval=0.8705  (20m 44s)",
        "  Epoch 5:  train=0.2690  eval=0.7790  (19m 44s)",
        "  Total training time: 100.8 minutes on CPU",
        "",
        "HONESTY CAVEATS",
        "  1. Both benchmark runs use the same synthetic test set.",
        "     The fine-tuned model was trained on images from the same",
        "     synthetic distribution (same fonts, same augmentation).",
        "     Improvement on this test set does NOT prove improvement",
        "     on real photographed handwriting.",
        "  2. The original 20-sample baseline (15% ESR) was measured on",
        "     the separate eval_dataset/ (different image set).",
        "     This report's baseline is re-measured on the same 100",
        "     test images for a fair apples-to-apples comparison.",
        "=" * 65,
    ]

    report_text = "\n".join(report)
    report_path = output_dir / "comparison.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n\n" + report_text)
    print(f"\nFull results written to: {output_dir}/")


if __name__ == "__main__":
    main()
