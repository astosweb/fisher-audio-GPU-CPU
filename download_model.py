#!/usr/bin/env python3
"""Download Fish Audio weights for the active runtime."""

from __future__ import annotations

import os

from runtime import IS_MLX
from tts_common import ensure_all_voice_samples
from tts_engine import MODEL


def download_cuda() -> None:
    from huggingface_hub import snapshot_download

    from tts_engine_cuda import MODEL_DIR, _patch_model_config

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


def download_mlx() -> None:
    from tts_engine_mlx import MODEL_MARKER, MODEL_REPO, get_model

    if MODEL_MARKER.exists():
        print(f"MLX model already present ({MODEL_REPO})")
        return

    print(f"Downloading {MODEL_REPO} (~4.5 GB on first run)...")
    get_model()
    print(f"Saved to Hugging Face cache (marker: {MODEL_MARKER})")


def download_voices() -> None:
    paths = ensure_all_voice_samples()
    print(f"Cached {len(paths)} default voice samples in .cache/voices/")


def main() -> None:
    if IS_MLX:
        download_mlx()
    else:
        download_cuda()
    download_voices()


if __name__ == "__main__":
    main()
