"""Shared Fish Audio text prep and types."""

from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

SPEAKER_TAG = re.compile(r"<\|speaker:(\d+)\|>")
VOICE_TAG = re.compile(r"<\|voice:([^|>]+)\|>", re.IGNORECASE)
INLINE_TAG = re.compile(r"\[[^\]]+\]")

# Fish Audio library voices. Local S2 Pro has no baked-in timbres — each voice
# is zero-shot cloned from its public sample clip on first use.
DEFAULT_VOICES: tuple[dict[str, str | int], ...] = (
    {
        "id": 0,
        "name": "Egirl",
        "description": "A cute e-girl to chat with you.",
        "reference_id": "98655a12fa944e26b274c535e5e03842",
        "sample_url": "https://platform.r2.fish.audio/task/8093cbd0201449f79d356cc9d2f42156.mp3",
        "sample_text": (
            "Hey why are you sitting here all alone in the corner come on let me sit "
            "next to you what's wrong you're being way too quiet tonight I know you're "
            "not usually like this is it because of what happened earlier don't worry "
            "about that just relax with me"
        ),
    },
    {
        "id": 1,
        "name": "Selene",
        "description": "A meditative female voice.",
        "reference_id": "b347db033a6549378b48d00acb0d06cd",
        "sample_url": "https://platform.r2.fish.audio/task/fcf629b4a0b343558f461338ba1b9da4.mp3",
        "sample_text": (
            "Welcome to this peaceful moment of self-reflection. Today's affirmations "
            "will help you embrace your inner strength. I am worthy of love and respect. "
            "I choose happiness in every moment. I trust my journey. My mind is calm, "
            "my heart is open."
        ),
    },
    {
        "id": 2,
        "name": "Adrian",
        "description": "A steady and reliable narrator.",
        "reference_id": "bf322df2096a46f18c579d0baa36f41d",
        "sample_url": "https://platform.r2.fish.audio/task/d730b09f4a484316a77ecc86137e85fb.mp3",
        "sample_text": (
            "The morning brings little comfort - a handful of dried berries, a cup of "
            "water from yesterday's rain. The wooden bowl is cracked but clean, like "
            "everything else here. Sometimes, watching the sunrise through the window's "
            "dusty pane feels like a feast itself."
        ),
    },
    {
        "id": 3,
        "name": "Sarah",
        "description": "An engaged speaker.",
        "reference_id": "933563129e564b19a115bedd57b7406a",
        "sample_url": "https://platform.r2.fish.audio/task/4e204945670c460c86fdab57512cb308.mp3",
        "sample_text": (
            "When you're creating something new, there's this beautiful mix of wonder "
            "and fear. And it's overwhelming sometimes, and scary, but also incredibly "
            "magical. And even though the process isn't perfect, those moments of "
            "uncertainty make the whole journey more meaningful and real."
        ),
    },
    {
        "id": 4,
        "name": "Ethan",
        "description": "A curious explainer.",
        "reference_id": "536d3a5e000945adb7038665781a4aca",
        "sample_url": "https://platform.r2.fish.audio/task/bdd677ed767744fc91f166468786264b.mp3",
        "sample_text": (
            "Is SpaceX's Starship really revolutionary for space travel? While the "
            "vehicle's fully reusable design and impressive payload capacity are "
            "groundbreaking, there are three major challenges to consider: 1. technical "
            "complexity, 2. operational costs, and 3. regulatory requirements."
        ),
    },
    {
        "id": 5,
        "name": "Laura",
        "description": "A confident female narrator.",
        "reference_id": "e3cd384158934cc9a01029cd7d278634",
        "sample_url": "https://platform.r2.fish.audio/task/e821bef995344d519caa1c93d808db23.mp3",
        "sample_text": (
            "Good communication in a relationship is just as important as trust. It's "
            "about really listening, being honest even when it's uncomfortable, and "
            "making sure your partner knows they can share without fear of judgment. "
            "When you talk openly and with respect, you build the kind of trust that "
            "keeps the relationship strong."
        ),
    },
    {
        "id": 6,
        "name": "Jordan",
        "description": "A motivational speaker.",
        "reference_id": "79d0bd3e4e5444b18f7b6d89b5927bf1",
        "sample_url": "https://platform.r2.fish.audio/task/edecbc0bdd4c46c39eda8520a0f729ef.mp3",
        "sample_text": (
            "Think about this: what's the difference between where you are now and "
            "where you want to be? It's not just knowledge or opportunity - it's the "
            "daily decisions you make. Your future is shaped by the choices you make "
            "in these small moments of truth."
        ),
    },
    {
        "id": 7,
        "name": "Hannah",
        "description": "A conversation specialist.",
        "reference_id": "9a9cf47702da476aa4629e2506d4a857",
        "sample_url": "https://platform.r2.fish.audio/task/08d658835feb427db808e37060b78eb0.mp3",
        "sample_text": (
            "I understand you might have concerns about starting a fitness program, "
            "but let me share information about our personalized training plans. We "
            "offer flexible scheduling, tailored workouts for all fitness levels, and "
            "our certified coaches can help you reach your goals safely and effectively."
        ),
    },
)

MAX_SPEAKERS = len(DEFAULT_VOICES)
DIALOGUE_SPEAKERS = (0, 1)  # preferred pair for multi-speaker presets
VOICE_BY_ID = {int(v["id"]): v for v in DEFAULT_VOICES}
VOICE_BY_NAME = {str(v["name"]).casefold(): v for v in DEFAULT_VOICES}
VOICE_CACHE_DIR = Path(
    os.environ.get("FISH_VOICE_CACHE", Path(__file__).parent / ".cache" / "voices")
)


def public_voices() -> list[dict[str, str | int]]:
    """Voice metadata safe to send to the web UI."""
    return [
        {
            "id": int(v["id"]),
            "name": str(v["name"]),
            "description": str(v["description"]),
        }
        for v in DEFAULT_VOICES
    ]


def voice_name(speaker: int) -> str:
    voice = VOICE_BY_ID.get(speaker)
    return str(voice["name"]) if voice else f"Speaker {speaker}"


def resolve_speaker(value: str | int | None, default: int = 0) -> int:
    """Resolve a voice name or speaker index to a slot id."""
    if value is None or value == "":
        return default
    if isinstance(value, int):
        return max(0, min(value, MAX_SPEAKERS - 1))
    text = str(value).strip()
    if not text:
        return default
    if text.isdigit():
        return max(0, min(int(text), MAX_SPEAKERS - 1))
    voice = VOICE_BY_NAME.get(text.casefold())
    if voice is None:
        names = ", ".join(str(v["name"]) for v in DEFAULT_VOICES)
        raise ValueError(f"Unknown voice {text!r}. Choose one of: {names}")
    return int(voice["id"])


def normalize_voice_tags(text: str) -> str:
    """Rewrite <|voice:Name|> tags to <|speaker:N|> for the model."""

    def replace(match: re.Match[str]) -> str:
        speaker = resolve_speaker(match.group(1))
        return f"<|speaker:{speaker}|>"

    return VOICE_TAG.sub(replace, text)


def has_speaker_tags(text: str) -> bool:
    return bool(SPEAKER_TAG.search(text) or VOICE_TAG.search(text))


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
    text = normalize_voice_tags(text.strip())
    if SPEAKER_TAG.search(text):
        return text
    speaker = max(0, min(speaker, MAX_SPEAKERS - 1))
    return f"<|speaker:{speaker}|>{text}"


def prepare_ref_text_with_speaker(ref_text: str, speaker: int) -> str:
    ref_text = normalize_voice_tags(ref_text.strip())
    if SPEAKER_TAG.search(ref_text):
        return ref_text
    speaker = max(0, min(speaker, MAX_SPEAKERS - 1))
    return f"<|speaker:{speaker}|>{ref_text}"


def extract_speaker_ids(text: str) -> list[int]:
    text = normalize_voice_tags(text)
    return [int(match) for match in SPEAKER_TAG.findall(text)]


def voice_sample_path(speaker: int) -> Path:
    name = voice_name(speaker).casefold().replace(" ", "-")
    return VOICE_CACHE_DIR / f"{speaker:02d}-{name}.mp3"


def ensure_voice_sample(speaker: int) -> Path:
    """Download the Fish Audio sample clip for a default voice if missing."""
    voice = VOICE_BY_ID.get(speaker)
    if voice is None:
        raise ValueError(f"No built-in sample for speaker {speaker}")

    path = voice_sample_path(speaker)
    if path.exists() and path.stat().st_size > 0:
        return path

    url = str(voice["sample_url"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.mp3")
    print(f"Downloading voice sample for {voice['name']}...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def ensure_all_voice_samples() -> list[Path]:
    return [ensure_voice_sample(int(v["id"])) for v in DEFAULT_VOICES]


def builtin_speaker_ref(speaker: int) -> SpeakerRef | None:
    voice = VOICE_BY_ID.get(speaker)
    if voice is None:
        return None
    path = ensure_voice_sample(speaker)
    return SpeakerRef(str(path), str(voice["sample_text"]), speaker)


def resolve_refs_with_builtins(
    text: str,
    speaker: int,
    refs: list[SpeakerRef] | None = None,
    *,
    max_refs: int | None = None,
) -> list[SpeakerRef] | None:
    """Fill missing speakers with built-in Fish Audio samples.

    Without reference audio, local S2 Pro ignores the chosen voice and drifts.
    MLX only supports one clone at a time — for multi-speaker dialogue with no
    user uploads, fall back to speaker tags alone.
    """
    prepared = prepare_text_with_speaker(text, speaker)
    needed = list(dict.fromkeys(extract_speaker_ids(prepared) or [speaker]))
    by_id = {ref.speaker: ref for ref in (refs or [])}
    user_ids = set(by_id)

    if max_refs == 1 and len(needed) > 1 and not user_ids:
        return None

    for speaker_id in needed:
        if speaker_id in by_id:
            continue
        builtin = builtin_speaker_ref(speaker_id)
        if builtin is not None:
            by_id[speaker_id] = builtin

    result = [by_id[sid] for sid in needed if sid in by_id]
    for ref in refs or []:
        if ref.speaker not in {r.speaker for r in result}:
            result.append(ref)

    if max_refs is not None and len(result) > max_refs:
        if user_ids:
            raise ValueError(
                "This backend supports one voice clone at a time. "
                "Upload a single reference, or use one default voice."
            )
        primary = speaker if speaker in by_id else needed[0]
        result = [by_id[primary]] if primary in by_id else result[:max_refs]

    return result or None


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
        if speaker_id not in ref_speakers and speaker_id not in VOICE_BY_ID
    )
    if missing_refs:
        names = ", ".join(voice_name(speaker_id) for speaker_id in missing_refs)
        return (
            f"{names} has no voice sample. Upload reference audio for that voice, "
            f"or pick one of the default voices."
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
