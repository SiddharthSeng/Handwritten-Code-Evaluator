"""
finetune_trocr.py
-----------------
Fine-tune TrOCR (microsoft/trocr-base-handwritten) on synthetic
handwritten Python code images.

Pre-flight guarantees (verified in code below):
  1. save_strategy="epoch" -> checkpoint saved after EVERY epoch
  2. EpochLogger callback writes train_loss + eval_loss to console AND
     to finetuned_model/training.log after each epoch
  3. eval_dataset is passed to Seq2SeqTrainer for monitoring ONLY.
     The Trainer calls model.eval() + torch.no_grad() for all eval
     steps. Gradient updates happen exclusively on train_dataset.

Usage:
  python finetune_trocr.py --epochs 5 --batch-size 4

Monitor progress while running:
  Get-Content finetuned_model/training.log -Wait   (PowerShell, live tail)
"""

import argparse
import json
import logging as pylogging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
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

class HandwrittenCodeDataset(Dataset):
    """PyTorch Dataset for fine-tuning TrOCR on code images."""

    def __init__(self, data_dir: Path, processor: TrOCRProcessor, max_target_length: int = 128):
        self.data_dir = data_dir
        self.processor = processor
        self.max_target_length = max_target_length

        labels_path = data_dir / "labels.json"
        with open(labels_path, encoding="utf-8") as f:
            self.labels = json.load(f)

        self.image_files = sorted(self.labels.keys())
        print(f"  Loaded {len(self.image_files)} samples from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = self.data_dir / "images" / img_file
        text = self.labels[img_file]

        # Flatten multi-line code to a single sequence for TrOCR's
        # single-line decoder. We use " \n " as the newline token.
        text_flat = text.replace("\n", " \\n ")

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text_flat,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # -100 = ignored in loss computation (padding positions)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Epoch-level logging callback
# ---------------------------------------------------------------------------

class EpochLogger(TrainerCallback):
    """Write train_loss and eval_loss to console + CSV log after every epoch.

    VALIDATION ISOLATION GUARANTEE:
    eval_dataset is used here ONLY for forward-pass loss computation.
    The HuggingFace Trainer calls model.eval() and wraps all eval steps
    in torch.no_grad(). No gradients flow through the validation set.
    Gradient updates happen ONLY over train_dataset batches.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._train_loss_accum = 0.0
        self._train_loss_steps = 0
        self._epoch_start_time = None

        # Write CSV header
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,eval_loss,epoch_time_s\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Accumulate step-level training losses as they're emitted."""
        if logs and "loss" in logs:
            self._train_loss_accum += logs["loss"]
            self._train_loss_steps += 1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called immediately after the val-set evaluation each epoch."""
        epoch = int(state.epoch) if state.epoch is not None else 0
        eval_loss = metrics.get("eval_loss", float("nan")) if metrics else float("nan")
        train_loss = (
            self._train_loss_accum / self._train_loss_steps
            if self._train_loss_steps > 0
            else float("nan")
        )
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0.0

        line = (
            f"[Epoch {epoch:2d}/{args.num_train_epochs}]  "
            f"train_loss={train_loss:.4f}  "
            f"eval_loss={eval_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )
        print(line, flush=True)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{eval_loss:.6f},{elapsed:.1f}\n")

        # Reset for next epoch
        self._train_loss_accum = 0.0
        self._train_loss_steps = 0


# ---------------------------------------------------------------------------
# Levenshtein CER for quick post-training check
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
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


def compute_cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, ref) / len(ref)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on handwritten code.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dataset", type=Path, default=Path("finetune_dataset"))
    parser.add_argument("--output", type=Path, default=Path("finetuned_model"))
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    steps_per_epoch = 800 // args.batch_size  # approx
    est_mins_lo = args.epochs * steps_per_epoch * 2 // 60
    est_mins_hi = args.epochs * steps_per_epoch * 5 // 60

    print("=" * 62)
    print("TrOCR FINE-TUNING - PRE-FLIGHT SUMMARY")
    print("=" * 62)
    print(f"  Device          : {device.upper()}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Est. time (CPU) : {est_mins_lo}-{est_mins_hi} minutes")
    print()
    print("  CHECKPOINT GUARANTEE:")
    print("    save_strategy='epoch' -> checkpoint after EVERY epoch")
    print("    save_total_limit=6    -> all 5 epoch checkpoints retained")
    print()
    print("  LOGGING GUARANTEE:")
    print("    EpochLogger -> console + finetuned_model/training.log")
    print("    Monitor live: Get-Content finetuned_model/training.log -Wait")
    print()
    print("  VALIDATION ISOLATION GUARANTEE:")
    print("    val_dataset used only in model.eval()+no_grad() forward pass")
    print("    Zero gradient updates touch the validation set.")
    print("=" * 62)
    print()

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    model_name = "microsoft/trocr-base-handwritten"
    print(f"Loading model: {model_name} ...", flush=True)
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Decoder generation config
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

    # ------------------------------------------------------------------
    # Datasets  (eval_dataset = val only, NEVER used for gradients)
    # ------------------------------------------------------------------
    print("\nLoading datasets ...", flush=True)
    train_dataset = HandwrittenCodeDataset(
        args.dataset / "train", processor, args.max_target_length
    )
    val_dataset = HandwrittenCodeDataset(   # monitoring only
        args.dataset / "val", processor, args.max_target_length
    )

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "training.log"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=min(100, len(train_dataset) // args.batch_size),
        logging_steps=50,
        eval_strategy="epoch",        # eval after each epoch
        save_strategy="epoch",        # <- CHECKPOINT AFTER EVERY EPOCH
        save_total_limit=6,           # retain all 5 + best
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,  # no beam search during training = faster
        fp16=torch.cuda.is_available() and not args.force_cpu,
        dataloader_num_workers=0,     # Windows: must be 0
        report_to="none",
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Trainer - val dataset is eval_dataset (monitoring only, no grads)
    # ------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,     # <- monitoring only
        processing_class=processor,
        callbacks=[EpochLogger(log_path)],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nStarting training ... ({len(train_dataset)} train, {len(val_dataset)} val)")
    print(f"  Epoch log -> {log_path}")
    print(f"  Steps/epoch: {len(train_dataset) // args.batch_size}")
    print()

    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0

    print(f"\nTraining done in {train_time / 60:.1f} minutes")
    print(f"  Final training loss: {train_result.training_loss:.4f}")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_path = args.output / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"  Saved to: {final_path}")

    # ------------------------------------------------------------------
    # Quick val-set CER sanity check (20 samples)
    # ------------------------------------------------------------------
    print("\nQuick validation CER check (20 samples) ...")
    model.eval()
    model.to(device)
    total_cer = 0.0
    n_check = min(20, len(val_dataset))

    for i in range(n_check):
        sample = val_dataset[i]
        px = sample["pixel_values"].unsqueeze(0).to(device)
        with torch.no_grad():
            gen_ids = model.generate(px, max_length=args.max_target_length)
        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        ref = val_dataset.labels[val_dataset.image_files[i]].replace("\n", " \\n ")
        c = compute_cer(pred, ref)
        total_cer += c
        if i < 5:
            print(f"  [{i}] CER={c:.3f}  pred={pred[:70]!r}")
            print(f"          ref ={ref[:70]!r}")

    avg_cer = total_cer / n_check
    print(f"\n  Avg val CER (n={n_check}): {avg_cer:.4f} ({avg_cer * 100:.1f}%)")

    # ------------------------------------------------------------------
    # Save training summary
    # ------------------------------------------------------------------
    summary = {
        "model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_train_loss": round(float(train_result.training_loss), 6),
        "val_cer_quick": round(avg_cer, 4),
        "training_time_minutes": round(train_time / 60, 1),
        "device": device,
        "output_dir": str(final_path),
    }
    summary_path = args.output / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
