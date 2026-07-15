"""Select Fish Audio runtime: MLX on Apple Silicon, CUDA elsewhere."""

from __future__ import annotations

import os
import platform as _platform


def detect_backend() -> str:
    forced = os.environ.get("FISH_BACKEND", "").strip().lower()
    if forced in {"mlx", "cuda"}:
        return forced
    if _platform.system() == "Darwin" and _platform.machine() == "arm64":
        return "mlx"
    return "cuda"


BACKEND = detect_backend()
IS_MLX = BACKEND == "mlx"
IS_CUDA = BACKEND == "cuda"
