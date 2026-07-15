"""Fish Audio S2 Pro TTS — MLX on Apple Silicon, CUDA on GPU servers."""

from __future__ import annotations

from runtime import BACKEND, IS_CUDA, IS_MLX

from tts_common import (
    DIALOGUE_SPEAKERS,
    MAX_SPEAKERS,
    GenerationResult,
    SpeakerRef,
    effective_max_tokens,
    extract_speaker_ids,
    has_inline_tags,
    has_speaker_tags,
    prepare_ref_text_with_speaker,
    prepare_text_with_speaker,
    save_upload,
    validate_speaker_setup,
    write_mp3,
)

if IS_MLX:
    from tts_engine_mlx import MODEL, MODEL_MARKER, generate_speech, get_model
elif IS_CUDA:
    from tts_engine_cuda import MODEL, MODEL_DIR, generate_speech, get_model
else:
    raise RuntimeError(f"Unknown backend: {BACKEND}")

__all__ = [
    "BACKEND",
    "DIALOGUE_SPEAKERS",
    "GenerationResult",
    "IS_CUDA",
    "IS_MLX",
    "MAX_SPEAKERS",
    "MODEL",
    "SpeakerRef",
    "effective_max_tokens",
    "extract_speaker_ids",
    "generate_speech",
    "get_model",
    "has_inline_tags",
    "has_speaker_tags",
    "prepare_ref_text_with_speaker",
    "prepare_text_with_speaker",
    "save_upload",
    "validate_speaker_setup",
    "write_mp3",
]

if IS_CUDA:
    __all__.append("MODEL_DIR")
if IS_MLX:
    __all__.append("MODEL_MARKER")
