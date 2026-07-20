#!/usr/bin/env python3
"""Generate speech with Fish Audio S2 Pro (MLX on Mac, CUDA on GPU servers)."""

from __future__ import annotations

import argparse
from pathlib import Path

from tts_engine import BACKEND, DEFAULT_VOICES, MODEL, generate_speech, resolve_speaker, write_mp3


def main() -> None:
    voice_names = ", ".join(str(v["name"]) for v in DEFAULT_VOICES)
    parser = argparse.ArgumentParser(description="Fish Audio S2 Pro local TTS")
    parser.add_argument(
        "--text",
        default="[excited] Hello! Fish Audio S2 Pro is running locally on my Mac.",
        help="Text to speak. Use [whisper], [laughing], [pause], etc. inline.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.mp3",
        help="Output MP3 path",
    )
    parser.add_argument(
        "--voice",
        default="Egirl",
        help=f"Default voice name or speaker index ({voice_names})",
    )
    parser.add_argument(
        "--ref-audio",
        help="Reference WAV for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        help="Transcript of the reference audio (required with --ref-audio)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
    )
    args = parser.parse_args()

    if bool(args.ref_audio) != bool(args.ref_text):
        parser.error("--ref-audio and --ref-text must be used together")

    try:
        speaker = resolve_speaker(args.voice)
    except ValueError as e:
        parser.error(str(e))

    print(f"Loading {MODEL} via {BACKEND} (run download_model.py if weights are missing)...")
    print(f"Voice: {args.voice} → speaker {speaker}")
    print(f"Generating: {args.text!r}")
    result = generate_speech(
        args.text,
        speaker=speaker,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
    )
    output = Path(args.output)
    write_mp3(output, result)
    print(f"Saved {output} ({result.duration:.1f}s @ {result.sample_rate} Hz)")


if __name__ == "__main__":
    main()
