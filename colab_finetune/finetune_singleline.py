"""
finetune_singleline.py
======================
Fine-tune TrOCR on single-line handwritten Python code images.

Designed to run on Google Colab (T4 GPU) or Kaggle (P100).
Also works on CPU (slowly) for testing.

Key difference from the previous finetune_trocr.py:
  - Each training image contains exactly ONE line of code
  - Labels are the stripped text of that line (no indentation)
  - This matches TrOCR's pre-training distribution (IAM single-line)

Usage (Colab):
  !python finetune_singleline.py --epochs 15 --batch-size 8

Usage (CPU, testing):
  python finetune_singleline.py --epochs 1 --batch-size 2 --max-samples 20
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SingleLineCodeDataset(Dataset):
    """Single-line code image dataset for TrOCR fine-tuning."""

    def __init__(self, data_dir: Path, processor: TrOCRProcessor,
                 max_target_length: int = 64):
        self.data_dir = data_dir
        self.processor = processor
        self.max_target_length = max_target_length

        with open(data_dir / "labels.json", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.image_files = sorted(self.labels.keys())
        print(f"  Loaded {len(self.image_files)} single-line samples from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = self.data_dir / "images" / img_file
        text = self.labels[img_file]  # already a single line, no newlines

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class EpochLogger(TrainerCallback):
    """Log train/eval loss to console and CSV file after each epoch."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._accum = 0.0
        self._steps = 0
        self._epoch_t0 = None
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,eval_loss,epoch_time_s\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._accum += logs["loss"]
            self._steps += 1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        eval_loss = metrics.get("eval_loss", float("nan")) if metrics else float("nan")
        train_loss = self._accum / self._steps if self._steps > 0 else float("nan")
        elapsed = time.time() - self._epoch_t0 if self._epoch_t0 else 0
        print(f"[Epoch {epoch:2d}/{args.num_train_epochs}]  "
              f"train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}  "
              f"({elapsed:.0f}s)", flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{eval_loss:.6f},{elapsed:.1f}\n")
        self._accum = 0.0
        self._steps = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dataset", type=Path, default=Path("singleline_dataset"))
    parser.add_argument("--output", type=Path, default=Path("finetuned_singleline"))
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit samples per split (0=all). For quick CPU test.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    print("=" * 60)
    print("TrOCR SINGLE-LINE FINE-TUNING")
    print("=" * 60)
    print(f"  Device      : {device.upper()}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  FP16        : {use_fp16}")
    print(f"  Max samples : {'all' if args.max_samples == 0 else args.max_samples}")
    print()

    # --- Model ---
    model_name = "microsoft/trocr-base-handwritten"
    print(f"Loading {model_name} ...", flush=True)
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_target_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}", flush=True)

    # --- Datasets ---
    print("\nLoading datasets ...", flush=True)
    train_ds = SingleLineCodeDataset(args.dataset / "train", processor, args.max_target_length)
    val_ds = SingleLineCodeDataset(args.dataset / "val", processor, args.max_target_length)

    if args.max_samples > 0:
        train_ds = Subset(train_ds, list(range(min(args.max_samples, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(args.max_samples, len(val_ds)))))
        print(f"  Truncated to {len(train_ds)} train, {len(val_ds)} val")

    # --- Time estimate ---
    n_train = len(train_ds)
    steps_per_epoch = n_train // args.batch_size
    if device == "cuda":
        est_secs_per_step = 0.3  # T4 with fp16
    else:
        est_secs_per_step = 5.5
    est_total = steps_per_epoch * args.epochs * est_secs_per_step / 60
    print(f"\n  Steps/epoch: {steps_per_epoch}")
    print(f"  Est. total time: ~{est_total:.0f} minutes on {device.upper()}")

    # --- Training ---
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "training.log"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        fp16=use_fp16,
        dataloader_num_workers=2 if device == "cuda" else 0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        callbacks=[EpochLogger(log_path)],
    )

    print(f"\nStarting training ...", flush=True)
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"  Final train loss: {result.training_loss:.4f}")

    # --- Save ---
    final_path = args.output / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"  Model saved to: {final_path}")

    # --- Quick validation CER ---
    print("\nQuick validation check ...", flush=True)
    model.eval()
    model.to(device)

    base_ds = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
    n_check = min(20, len(val_ds))

    total_cer = 0.0
    for i in range(n_check):
        actual_idx = val_ds.indices[i] if isinstance(val_ds, Subset) else i
        sample = base_ds[actual_idx]
        px = sample["pixel_values"].unsqueeze(0).to(device)
        with torch.no_grad():
            gen = model.generate(px, max_length=args.max_target_length)
        pred = processor.batch_decode(gen, skip_special_tokens=True)[0]
        ref = base_ds.labels[base_ds.image_files[actual_idx]]

        dist = _edit_distance(pred, ref)
        c = dist / len(ref) if ref else 0.0
        total_cer += c
        if i < 5:
            print(f"  [{i}] CER={c:.3f}  pred={pred[:60]!r}")
            print(f"                ref ={ref[:60]!r}")

    print(f"\n  Val CER (n={n_check}): {total_cer / n_check:.4f}")

    # --- Summary ---
    summary = {
        "model": model_name,
        "dataset_type": "SINGLE-LINE synthetic crops",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": n_train,
        "val_samples": len(val_ds),
        "train_loss": round(float(result.training_loss), 6),
        "val_cer": round(total_cer / n_check, 4),
        "time_minutes": round(elapsed / 60, 1),
        "device": device,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {args.output / 'summary.json'}")
    print("\nDone!")


def _edit_distance(a, b):
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


if __name__ == "__main__":
    main()
