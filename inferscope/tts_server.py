"""A small speech server for piper voices.

piper ships as a library, not a daemon, and `enginectl` expects an engine to be
a process that answers HTTP on a port and reports `/health`. This is that
adapter: one voice loaded once at boot, then a POST for text and a WAV back.

Synthesis runs on the CPU on purpose. A voice model is tens of megabytes and the
GPU should stay free for the model the user is actually inspecting — which is
the whole premise of this page.
"""
from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response


def find_voice(model: str) -> Optional[Path]:
    """The .onnx to load: a file, or the first voice inside a directory."""
    p = Path(model)
    if p.is_file():
        return p
    if p.is_dir():
        voices = sorted(p.glob("*.onnx"))
        return voices[0] if voices else None
    return None


def create_app(model: str) -> FastAPI:
    app = FastAPI(title="inferscope-tts")
    state: dict[str, object] = {"voice": None, "error": None}

    @app.on_event("startup")
    def load() -> None:
        try:
            from piper.voice import PiperVoice
        except Exception as e:              # noqa: BLE001
            state["error"] = f"piper is not installed: {e}"
            return
        path = find_voice(model)
        if path is None:
            state["error"] = f"no voice found at {model}"
            return
        try:
            state["voice"] = PiperVoice.load(str(path))
            state["voice_name"] = path.name
            state["sample_rate"] = state["voice"].config.sample_rate
        except Exception as e:              # noqa: BLE001
            state["error"] = f"could not load {path.name}: {e}"

    @app.get("/health")
    def health() -> JSONResponse:
        if state["error"]:
            return JSONResponse({"status": "error", "detail": state["error"]}, status_code=503)
        return JSONResponse({"status": "ok", "voice": state.get("voice_name", "?"),
                             "sample_rate": state.get("sample_rate")})

    @app.post("/synthesize")
    def synthesize(payload: dict) -> Response:
        if state["error"] or state["voice"] is None:
            raise HTTPException(503, str(state["error"] or "voice not loaded"))
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "text is required")

        # piper 1.x yields 16-bit chunks; the WAV header is ours to write, so
        # the engine hands back something a browser can just play.
        buf = io.BytesIO()
        rate = int(state["sample_rate"])
        written = 0
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            for chunk in state["voice"].synthesize(text):
                pcm = chunk.audio_int16_array.tobytes()
                w.writeframes(pcm)
                written += len(chunk.audio_int16_array)
        audio = buf.getvalue()
        if written == 0:
            raise HTTPException(502, "the voice produced no audio for that text")
        return Response(content=audio, media_type="audio/wav",
                        headers={"X-Sample-Rate": str(int(state["sample_rate"])),
                                 "Content-Length": str(len(audio))})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="piper as an HTTP engine")
    ap.add_argument("--model", required=True, help="voice .onnx, or a directory of them")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8083)
    a = ap.parse_args()

    import uvicorn
    uvicorn.run(create_app(a.model), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()