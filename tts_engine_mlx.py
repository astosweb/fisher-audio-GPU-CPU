"""Fish Audio S2 Pro TTS engine (MLX on Apple Silicon)."""

from __future__ import annotations

import os
from pathlib import Path

import mlx_speech
import numpy as np

from tts_common import (
    GenerationResult,
    SpeakerRef,
    effective_max_tokens,
    prepare_ref_text_with_speaker,
    prepare_text_with_speaker,
    validate_speaker_setup,
)

MODEL_REPO = "mlx-community/fishaudio-s2-pro-8bit-mlx"
MODEL = MODEL_REPO
MODEL_MARKER = Path(
    os.environ.get("FISH_MODEL_MARKER", ".cache/mlx-fish-s2-pro-mlx-community.ready")
)

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading {MODEL_REPO} (MLX)...")
        _model = mlx_speech.tts.load(MODEL_REPO)
        MODEL_MARKER.parent.mkdir(parents=True, exist_ok=True)
        MODEL_MARKER.write_text("ok\n", encoding="utf-8")
    return _model


def generate_speech(
    text: str,
    *,
    speaker: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.7,
    max_tokens: int = 1024,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    ref_speaker: int = 0,
    refs: list[SpeakerRef] | None = None,
) -> GenerationResult:
    del temperature, top_p

    if not text.strip():
        raise ValueError("Text is required")

    if refs is None and ref_audio_path:
        if not ref_text or not ref_text.strip():
            raise ValueError("Reference transcript is required with reference audio")
        refs = [SpeakerRef(ref_audio_path, ref_text.strip(), ref_speaker)]

    if refs:
        for ref in refs:
            if not ref.text.strip():
                raise ValueError("Reference transcript is required with reference audio")

    text = prepare_text_with_speaker(text, speaker)
    token_budget = effective_max_tokens(text, max_tokens)

    warning = validate_speaker_setup(text, refs)
    if warning:
        raise ValueError(warning)

    generate_kwargs: dict[str, object] = {"max_new_tokens": token_budget}
    if refs:
        if len(refs) > 1:
            raise ValueError(
                "MLX backend supports one voice clone at a time. "
                "Use <|speaker:0|> and <|speaker:1|> for built-in dialogue voices."
            )
        ref = refs[0]
        generate_kwargs["reference_audio"] = ref.audio_path
        generate_kwargs["reference_text"] = prepare_ref_text_with_speaker(
            ref.text, ref.speaker
        )
    elif ref_audio_path:
        generate_kwargs["reference_audio"] = ref_audio_path
        generate_kwargs["reference_text"] = prepare_ref_text_with_speaker(
            ref_text or "", ref_speaker
        )

    result = get_model().generate(text, **generate_kwargs)
    audio = np.asarray(result.waveform, dtype=np.float32)
    sample_rate = int(result.sample_rate)
    return GenerationResult(
        audio=audio,
        sample_rate=sample_rate,
        duration=len(audio) / sample_rate,
    )
