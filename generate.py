#!/usr/bin/env python3
"""Generate speech with Fish Audio S2 Pro on GPU (CUDA)."""

from __future__ import annotations

import argparse
from pathlib import Path

from tts_engine import MODEL, generate_speech, write_mp3


def main() -> None:
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

    print(f"Loading {MODEL} (run download_model.py if weights are missing)...")
    print(f"Generating: {args.text!r}")
    result = generate_speech(
        args.text,
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
