"""A small image server for diffusers checkpoints.

Same shape as the speech server: something `enginectl` can start, stop and
health-check on its own port, so "which model is on the GPU" stays a single
question with a single answer.

Distilled checkpoints (SD-Turbo and friends) are the ones worth running on a
shared card — one to four steps, no guidance sweep — so `--steps` is exposed but
defaults to four. Generation happens in float16 on the GPU; loading is the slow
part and happens once, at boot, not per request.
"""
from __future__ import annotations

import argparse
import io
import threading
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response


def find_model(path: str) -> Optional[Path]:
    """A diffusers directory (one with model_index.json), or None."""
    p = Path(path)
    if p.is_dir() and (p / "model_index.json").exists():
        return p
    return None


def create_app(model: str) -> FastAPI:
    app = FastAPI(title="inferscope-image")
    state: dict[str, object] = {"pipe": None, "error": None, "device": "cpu"}
    # diffusers pipelines are not safe to call from two requests at once, and
    # two people pressing "draw" must not corrupt each other's latent.
    lock = threading.Lock()

    @app.on_event("startup")
    def load() -> None:
        try:
            from diffusers import AutoPipelineForText2Image
        except Exception as e:                       # noqa: BLE001
            state["error"] = f"diffusers is not installed: {e}"
            return
        path = find_model(model)
        if path is None:
            state["error"] = (f"no diffusers pipeline at {model} "
                              "(expected a directory with model_index.json)")
            return
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            state["device"] = dev
            half = dev == "cuda"
            try:
                # ask for the fp16 weights explicitly: the repository also
                # carries float32 copies, and taking those would double the
                # download and the VRAM for no benefit on a GPU.
                pipe = AutoPipelineForText2Image.from_pretrained(
                    str(path), torch_dtype=torch.float16 if half else torch.float32,
                    variant="fp16" if half else None)
            except Exception:                       # noqa: BLE001
                pipe = AutoPipelineForText2Image.from_pretrained(
                    str(path), torch_dtype=torch.float16 if half else torch.float32)
            if dev == "cuda":
                pipe = pipe.to(dev)
            # distilled models do not use classifier-free guidance at all
            pipe.set_progress_bar_config(disable=True)
            state["pipe"] = pipe
        except Exception as e:                       # noqa: BLE001
            state["error"] = f"could not load {Path(model).name}: {e}"

    @app.get("/health")
    def health() -> JSONResponse:
        if state["error"]:
            return JSONResponse({"status": "error", "detail": state["error"]}, status_code=503)
        return JSONResponse({"status": "ok", "model": Path(model).name,
                             "device": state["device"]})

    @app.post("/generate")
    def generate(payload: dict) -> Response:
        if state["error"] or state["pipe"] is None:
            raise HTTPException(503, str(state["error"] or "model not loaded"))
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "prompt is required")
        try:
            steps = max(1, min(8, int(payload.get("steps") or 4)))
        except (TypeError, ValueError):
            steps = 4
        pipe = state["pipe"]
        with lock:
            try:
                out = pipe(prompt=prompt, num_inference_steps=steps,
                           guidance_scale=0.0)
            except Exception as e:                   # noqa: BLE001
                raise HTTPException(500, f"generation failed: {e}")
        img = out.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="a diffusers pipeline as an HTTP engine")
    ap.add_argument("--model", required=True, help="a diffusers pipeline directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8084)
    a = ap.parse_args()

    import uvicorn
    uvicorn.run(create_app(a.model), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()