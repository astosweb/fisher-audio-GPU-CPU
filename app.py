#!/usr/bin/env python3
"""Fish Audio S2 Pro — GPU web UI."""

from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from tts_engine import SpeakerRef, generate_speech, get_model, write_mp3

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
OUTPUTS = ROOT / "outputs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "7860")
    print("Loading Fish Audio S2 Pro (cached after first run)...")
    get_model()
    OUTPUTS.mkdir(exist_ok=True)
    print(f"Model ready — listening on http://127.0.0.1:{port}")
    if host in ("0.0.0.0", "::"):
        print(f"  Network: http://0.0.0.0:{port} (all interfaces)")
    yield


app = FastAPI(title="Fish Audio S2 Pro", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


@app.post("/api/generate")
async def api_generate(
    text: str = Form(...),
    speaker: int = Form(0),
    ref_speaker: int = Form(0),
    ref_speaker_1: int = Form(1),
    temperature: float = Form(0.7),
    top_p: float = Form(0.7),
    max_tokens: int = Form(2048),
    ref_text: str = Form(""),
    ref_text_1: str = Form(""),
    ref_audio: UploadFile | None = File(None),
    ref_audio_1: UploadFile | None = File(None),
):
    tmp_refs: list[Path] = []

    try:
        refs: list[SpeakerRef] = []

        if ref_audio and ref_audio.filename:
            if not ref_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Reference transcript is required with reference audio",
                )
            suffix = Path(ref_audio.filename).suffix or ".wav"
            tmp_ref = Path(tempfile.gettempdir()) / f"fish-ref-{uuid.uuid4().hex}{suffix}"
            tmp_ref.write_bytes(await ref_audio.read())
            tmp_refs.append(tmp_ref)
            refs.append(SpeakerRef(str(tmp_ref), ref_text.strip(), ref_speaker))

        if ref_audio_1 and ref_audio_1.filename:
            if not ref_text_1.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Reference transcript is required for speaker 1 clone audio",
                )
            suffix = Path(ref_audio_1.filename).suffix or ".wav"
            tmp_ref = Path(tempfile.gettempdir()) / f"fish-ref-{uuid.uuid4().hex}{suffix}"
            tmp_ref.write_bytes(await ref_audio_1.read())
            tmp_refs.append(tmp_ref)
            refs.append(SpeakerRef(str(tmp_ref), ref_text_1.strip(), ref_speaker_1))

        result = generate_speech(
            text,
            speaker=speaker,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            refs=refs or None,
        )

        out = OUTPUTS / f"{uuid.uuid4().hex}.mp3"
        write_mp3(out, result)

        return {
            "audio_url": f"/api/audio/{out.name}",
            "duration": round(result.duration, 2),
            "sample_rate": result.sample_rate,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        for tmp_ref in tmp_refs:
            if tmp_ref.exists():
                tmp_ref.unlink()


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    path = OUTPUTS / filename
    if not path.exists() or path.parent.resolve() != OUTPUTS.resolve():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg")


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
