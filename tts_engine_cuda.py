"""Fish Audio S2 Pro TTS engine (PyTorch / CUDA)."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="Logical operators 'and' and 'or' are deprecated for non-scalar tensors",
    category=UserWarning,
)
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

from tts_common import (
    DIALOGUE_SPEAKERS,
    MAX_SPEAKERS,
    GenerationResult,
    SpeakerRef,
    effective_max_tokens,
    prepare_ref_text_with_speaker,
    prepare_text_with_speaker,
    validate_speaker_setup,
)

MODEL = "fishaudio/s2-pro"
MODEL_DIR = Path(os.environ.get("FISH_MODEL_DIR", "checkpoints/s2-pro"))
DECODER_CONFIG = "modded_dac_vq"
DEFAULT_MAX_SEQ_LEN = 4096

_engine: TTSInferenceEngine | None = None


def _configure_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _patch_model_config() -> None:
    """Shrink KV cache pre-allocation (32768 -> 4096 saves ~5 GB VRAM, ~1.7x faster)."""
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        return

    max_seq_len = int(os.environ.get("FISH_MAX_SEQ_LEN", str(DEFAULT_MAX_SEQ_LEN)))
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    text_cfg = config.get("text_config", {})
    current = text_cfg.get("max_seq_len")
    if current == max_seq_len:
        return

    text_cfg["max_seq_len"] = max_seq_len
    config["text_config"] = text_cfg
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Patched model max_seq_len: {current} -> {max_seq_len}")


def _warmup(engine: TTSInferenceEngine) -> None:
    if os.environ.get("FISH_WARMUP", "1") != "1":
        return
    print("Warming up GPU kernels (first compile may take 1-2 min)...")
    req = ServeTTSRequest(
        text="<|speaker:0|>Hello.",
        references=[],
        temperature=0.7,
        top_p=0.7,
        max_new_tokens=64,
        chunk_length=512,
        format="wav",
    )
    for result in engine.inference(req):
        if result.code == "error":
            raise RuntimeError(str(result.error))
        if result.code == "final":
            break
    print("Warmup complete.")


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_engine() -> TTSInferenceEngine:
    checkpoint = MODEL_DIR
    decoder_path = checkpoint / "codec.pth"
    if not decoder_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {checkpoint}. "
            f"Run: python download_model.py"
        )

    _configure_torch()
    _patch_model_config()

    half = os.environ.get("FISH_HALF", "0") == "1"
    compile = os.environ.get("FISH_COMPILE", "1") == "1"
    device = _resolve_device()
    precision = torch.half if half else torch.bfloat16

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(checkpoint),
        device=device,
        precision=precision,
        compile=compile,
    )
    decoder_model = load_decoder_model(
        config_name=DECODER_CONFIG,
        checkpoint_path=str(decoder_path),
        device=device,
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=compile,
    )
    _warmup(engine)
    return engine


def get_model() -> TTSInferenceEngine:
    global _engine
    if _engine is None:
        _engine = _load_engine()
    return _engine


def _build_references(
    refs: list[SpeakerRef] | None,
    ref_audio_path: str | None,
    ref_text: str | None,
    ref_speaker: int,
) -> list[ServeReferenceAudio]:
    references: list[ServeReferenceAudio] = []
    if refs:
        for ref in refs:
            references.append(
                ServeReferenceAudio(
                    audio=Path(ref.audio_path).read_bytes(),
                    text=prepare_ref_text_with_speaker(ref.text, ref.speaker),
                )
            )
        return references

    if ref_audio_path:
        references.append(
            ServeReferenceAudio(
                audio=Path(ref_audio_path).read_bytes(),
                text=prepare_ref_text_with_speaker(ref_text or "", ref_speaker),
            )
        )
    return references


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

    references = _build_references(refs, ref_audio_path, ref_text, ref_speaker)
    req = ServeTTSRequest(
        text=text,
        references=references,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=token_budget,
        chunk_length=512,
        format="wav",
    )

    engine = get_model()
    for result in engine.inference(req):
        if result.code == "error":
            raise RuntimeError(str(result.error))
        if result.code == "final" and isinstance(result.audio, tuple):
            sample_rate, audio = result.audio
            return GenerationResult(
                audio=audio,
                sample_rate=sample_rate,
                duration=len(audio) / sample_rate,
            )

    raise RuntimeError("No audio generated")
