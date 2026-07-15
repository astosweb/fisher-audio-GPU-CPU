#!/usr/bin/env python3
"""Fish Audio S2 Pro — MLX on Mac, CUDA on GPU servers."""

from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from tts_engine import (
    BACKEND,
    DIALOGUE_SPEAKERS,
    MAX_SPEAKERS,
    MODEL,
    SpeakerRef,
    generate_speech,
    get_model,
    write_mp3,
)

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
OUTPUTS = ROOT / "outputs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "7860")
    print(f"Loading {MODEL} via {BACKEND} backend (cached after first run)...")
    get_model()
    OUTPUTS.mkdir(exist_ok=True)
    print(f"Model ready ({BACKEND}) — listening on http://127.0.0.1:{port}")
    if host in ("0.0.0.0", "::"):
        print(f"  Network: http://0.0.0.0:{port} (all interfaces)")
    yield


app = FastAPI(title="Fish Audio S2 Pro", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


@app.get("/api/config")
def api_config() -> dict[str, object]:
    return {
        "backend": BACKEND,
        "model": MODEL,
        "max_speakers": MAX_SPEAKERS,
        "dialogue_speakers": list(DIALOGUE_SPEAKERS),
    }


@app.post("/api/generate")
async def api_generate(request: Request):
    form = await request.form()
    text = str(form.get("text", ""))
    speaker = int(form.get("speaker", 0))
    temperature = float(form.get("temperature", 0.7))
    top_p = float(form.get("top_p", 0.7))
    max_tokens = int(form.get("max_tokens", 2048))

    tmp_refs: list[Path] = []

    try:
        refs: list[SpeakerRef] = []

        for i in range(MAX_SPEAKERS):
            ref_audio = form.get(f"ref_audio_{i}")
            ref_text = str(form.get(f"ref_text_{i}", "")).strip()
            if not isinstance(ref_audio, UploadFile) or not ref_audio.filename:
                continue
            if not ref_text:
                raise HTTPException(
                    status_code=400,
                    detail=f"Reference transcript is required for speaker {i} clone audio",
                )
            suffix = Path(ref_audio.filename).suffix or ".wav"
            tmp_ref = Path(tempfile.gettempdir()) / f"fish-ref-{uuid.uuid4().hex}{suffix}"
            tmp_ref.write_bytes(await ref_audio.read())
            tmp_refs.append(tmp_ref)
            refs.append(SpeakerRef(str(tmp_ref), ref_text, i))

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
