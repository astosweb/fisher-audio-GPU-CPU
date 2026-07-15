"""Shared Fish Audio text prep and types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

SPEAKER_TAG = re.compile(r"<\|speaker:(\d+)\|>")
INLINE_TAG = re.compile(r"\[[^\]]+\]")
MAX_SPEAKERS = 10
DIALOGUE_SPEAKERS = (0, 1)


def has_speaker_tags(text: str) -> bool:
    return bool(SPEAKER_TAG.search(text))


def has_inline_tags(text: str) -> bool:
    return bool(INLINE_TAG.search(text))


def effective_max_tokens(text: str, max_tokens: int) -> int:
    """Tagged text needs a larger token budget — tags produce audio without text."""
    if not has_inline_tags(text):
        return max_tokens
    tag_count = len(INLINE_TAG.findall(text))
    estimated = int(len(text) * 2.2) + tag_count * 96
    return max(max_tokens, min(estimated, 4096))


def prepare_text_with_speaker(text: str, speaker: int) -> str:
    text = text.strip()
    if has_speaker_tags(text):
        return text
    speaker = max(0, min(speaker, MAX_SPEAKERS - 1))
    return f"<|speaker:{speaker}|>{text}"


def prepare_ref_text_with_speaker(ref_text: str, speaker: int) -> str:
    ref_text = ref_text.strip()
    if has_speaker_tags(ref_text):
        return ref_text
    speaker = max(0, min(speaker, MAX_SPEAKERS - 1))
    return f"<|speaker:{speaker}|>{ref_text}"


def extract_speaker_ids(text: str) -> list[int]:
    return [int(match) for match in SPEAKER_TAG.findall(text)]


def validate_speaker_setup(
    text: str,
    refs: list[SpeakerRef] | None = None,
) -> str | None:
    speaker_ids = extract_speaker_ids(text)
    if len(speaker_ids) < 2:
        return None

    ref_speakers = {ref.speaker for ref in (refs or [])}
    missing_refs = sorted(
        speaker_id
        for speaker_id in set(speaker_ids)
        if speaker_id not in ref_speakers and speaker_id not in DIALOGUE_SPEAKERS
    )
    if missing_refs:
        slots = ", ".join(f"<|speaker:{speaker_id}|>" for speaker_id in missing_refs)
        return (
            f"{slots} has no cloned voice. Upload reference audio for that speaker slot, "
            "or use <|speaker:0|> and <|speaker:1|> for built-in dialogue voices."
        )
    return None


@dataclass
class SpeakerRef:
    audio_path: str
    text: str
    speaker: int = 0


@dataclass
class GenerationResult:
    audio: np.ndarray
    sample_rate: int
    duration: float


def write_mp3(path: Path, result: GenerationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), result.audio, result.sample_rate, format="MP3")


def save_upload(upload: BinaryIO, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(upload.read())
