#!/usr/bin/env python3
"""Download Fish Audio S2 Pro weights from Hugging Face."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from tts_engine import MODEL, MODEL_DIR


def main() -> None:
    if (MODEL_DIR / "codec.pth").exists():
        print(f"Model already present at {MODEL_DIR}")
        return

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL} to {MODEL_DIR} (~10 GB on first run)...")
    snapshot_download(MODEL, local_dir=str(MODEL_DIR))
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
