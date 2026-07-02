"""Pre-download the TrOCR model weights.

Run this script before starting the app to avoid the model download
on the first /evaluate request:

    python scripts/download_model.py          # base model (~1.3 GB)
    python scripts/download_model.py --light  # small model (~330 MB)

The model will be cached in the default HuggingFace cache directory
(~/.cache/huggingface/hub on Linux/macOS, %USERPROFILE%\\.cache\\huggingface\\hub
on Windows).
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download TrOCR model weights."
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Download the smaller trocr-small-handwritten model (~330 MB) "
             "instead of the base model (~1.3 GB).",
    )
    args = parser.parse_args()

    if args.light:
        model_name = "microsoft/trocr-small-handwritten"
        size_hint = "~330 MB"
    else:
        model_name = "microsoft/trocr-base-handwritten"
        size_hint = "~1.3 GB"

    print(f"Downloading TrOCR model: {model_name}")
    print(f"Estimated download size: {size_hint}")
    print("This may take a few minutes...\n")

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        print("[1/2] Downloading processor...")
        TrOCRProcessor.from_pretrained(model_name)
        print("      ✓ Processor cached.\n")

        print("[2/2] Downloading model weights...")
        VisionEncoderDecoderModel.from_pretrained(model_name)
        print("      ✓ Model weights cached.\n")

        print("Done! The model is now cached locally.")
        print("Start the app with: python app.py")
        if not args.light:
            print(
                "\nTip: Use --light to download the smaller model (~330 MB) "
                "for faster startup. Set HCE_OCR_MODEL=light when running the app."
            )

    except Exception as exc:
        print(f"\n✗ Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
