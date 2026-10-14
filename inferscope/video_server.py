"""A small video server for diffusers text-to-video pipelines.

Same contract as the speech and image servers: a process `enginectl` can start,
stop and health-check on its own port.

Video is the reason this one differs in two ways:

* Weights are large, so they live on the non-permanent root filesystem rather
  than the workspace disk. Nothing here is cached across a container restart;
  `bin/fetch-video-model` puts it back.
* A clip takes far longer than an image. Generation holds a lock (one clip at a
  time, however many people ask) and the HTTP call waits for the whole thing,
  because a video you cannot watch yet is not a result.
"""
from __future__ import annotations

import argparse
import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response


def find_model(path: str) -> Optional[Path]:
    p = Path(path)
    return p if p.is_dir() and (p / "model_index.json").exists() else None


def create_app(model: str) -> FastAPI:
    app = FastAPI(title="inferscope-video")
    state: dict[str, object] = {"pipe": None, "error": None, "device": "cpu"}
    # one clip at a time: a second request would either double the VRAM or
    # corrupt the first clip's latents
    lock = threading.Lock()

    @app.on_event("startup")
    def load() -> None:
        path = find_model(model)
        if path is None:
            state["error"] = (f"no text-to-video pipeline at {model} "
                              "(expected a directory with model_index.json)")
            return

        # Take the class from the checkpoint's own index rather than guessing.
        # Guessing is how Wan ended up being loaded with the modular H3 class,
        # which cannot resolve its blocks and fails with an error that says
        # nothing about the real cause.
        import diffusers
        import json as _json
        named = None
        for index in ("modular_model_index.json", "model_index.json"):
            try:
                named = _json.loads((path / index).read_text()).get("_class_name")
            except (OSError, ValueError):
                continue
            if named:
                break
        pipe_cls = getattr(diffusers, str(named), None) if named else None
        if pipe_cls is None:
            # No usable name in the checkpoint: fall back to diffusers' own
            # auto-detection, and only then to the modular class.
            for name in ("AutoPipelineForText2Video", "DiffusionPipeline",
                         "MiniMaxH3ModularPipeline"):
                pipe_cls = getattr(diffusers, name, None)
                if pipe_cls is not None:
                    break
        if pipe_cls is None:
            state["error"] = "no text-to-video pipeline in this diffusers"
            return
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            state["device"] = dev
            pipe = pipe_cls.from_pretrained(
                str(path), torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
            if dev == "cuda":
                try:
                    pipe = pipe.to(dev)
                except torch.cuda.OutOfMemoryError:
                    # A video transformer plus its text encoder is larger than
                    # this card. Offload moves whole modules to the host between
                    # calls: slow per step, but it is the difference between
                    # working at all and refusing to start.
                    pipe.enable_model_cpu_offload()
                    state["offloaded"] = True
            pipe.set_progress_bar_config(disable=True)
            state["pipe"] = pipe
        except Exception as e:                        # noqa: BLE001
            state["error"] = f"could not load {Path(model).name}: {e}"

    @app.get("/health")
    def health() -> JSONResponse:
        if state["error"]:
            return JSONResponse({"status": "error", "detail": state["error"]}, status_code=503)
        return JSONResponse({"status": "ok", "model": Path(model).name,
                             "device": state["device"],
                             "offloaded": bool(state.get("offloaded"))})

    @app.post("/generate")
    def generate(payload: dict) -> Response:
        if state["error"] or state["pipe"] is None:
            raise HTTPException(503, str(state["error"] or "model not loaded"))
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "prompt is required")
        frames = max(1, min(81, int(payload.get("frames") or 33)))
        steps = max(1, min(30, int(payload.get("steps") or 8)))
        height = int(payload.get("height") or 320)
        width = int(payload.get("width") or 480)
        pipe = state["pipe"]
        with lock:
            try:
                kwargs: dict[str, Any] = dict(
                    prompt=prompt, num_frames=frames, num_inference_steps=steps)
                # Wan and the FastH3/MiniMaxH3 line are guidance-distilled and
                # do not accept the knob at all.
                try:
                    out = pipe(**kwargs, height=height, width=width)
                except TypeError:
                    out = pipe(**kwargs)
            except Exception as e:                    # noqa: BLE001
                raise HTTPException(500, f"generation failed: {e}")
        # Pipelines disagree on the shape of `frames`: some return a list per
        # batch item, some a single ndarray of (batch, frame, h, w, c), and the
        # truthiness of an ndarray raises rather than answering.
        raw = getattr(out, "frames", None)
        if raw is None:
            raise HTTPException(500, "the pipeline returned no frames")
        arr = np.asarray(raw[0] if isinstance(raw, (list, tuple)) else raw,
                         dtype=np.float32)
        if arr.ndim == 5:                       # (batch, frames, h, w, c)
            arr = arr[0]
        if arr.ndim != 4 or arr.shape[0] == 0:
            raise HTTPException(500, f"unexpected frames shape {arr.shape}")
        frames_list = arr
        # No ffmpeg on this host, so ship an animated GIF built by Pillow: the
        # browser can play it inline, and it needs no codec on the server.
        try:
            from PIL import Image
            imgs = [Image.fromarray((f * 255).clip(0, 255).astype("uint8"))
                    for f in frames_list]
            buf = io.BytesIO()
            imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                        duration=100, loop=0)
            return Response(content=buf.getvalue(), media_type="image/gif",
                            headers={"X-Frames": str(len(imgs)),
                                     "X-Size": f"{width}x{height}"})
        except Exception as e:                        # noqa: BLE001
            raise HTTPException(500, f"could not encode the clip: {e}")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="a diffusers text-to-video pipeline as an engine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8085)
    a = ap.parse_args()

    import uvicorn
    uvicorn.run(create_app(a.model), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()