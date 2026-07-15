#!/usr/bin/env python3
"""Download Fish Audio S2 Pro weights from Hugging Face."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from tts_engine import MODEL, MODEL_DIR, _patch_model_config


def main() -> None:
    if (MODEL_DIR / "codec.pth").exists():
        print(f"Model already present at {MODEL_DIR}")
        _patch_model_config()
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required to download the model. "
            "Copy .env.example to .env and set your Hugging Face token."
        )

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL} to {MODEL_DIR} (~10 GB on first run)...")
    snapshot_download(MODEL, local_dir=str(MODEL_DIR), token=token)
    _patch_model_config()
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
