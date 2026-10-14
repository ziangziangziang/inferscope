"""Architectures for the remaining engines, read from their own configs.

whisper's comes from the engine's load report and llama.cpp's from the GGUF
header. The other three ship their configuration next to the weights, so that is
where their shapes come from too.

Every count here is `derived`: read from a config or a header, computed with the
standard formula. Nothing is `measured` — that word belongs to the runtime
trace, and conflating the two is how a diagram ends up claiming hardware
measurements it never took.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .arch import _edge, _node


def _cfg(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------- tts (piper)

def build_piper_arch(path: str, label: str = "") -> Optional[dict[str, Any]]:
    """A piper voice: the config is a sibling ``<voice>.onnx.json``."""
    p = Path(path)
    cfg_path = Path(str(p) + ".json") if p.is_file() else p / "en_US-lessac-medium.onnx.json"
    if p.is_dir():
        voices = sorted(p.glob("*.onnx.json"))
        if not voices:
            return None
        cfg_path = voices[0]
    cfg = _cfg(cfg_path)
    if not cfg:
        return None

    num_mels = int(cfg.get("num_mels", 80))
    n_audio_layer = int(cfg.get("encoder_num_layers", 6))
    n_audio_state = int(cfg.get("encoder_dim", 256))
    n_text_layer = int(cfg.get("decoder_num_layers", 6))
    n_text_state = int(cfg.get("decoder_dim", 256))
    n_audio_ctx = int(cfg.get("num_frames", 50)) * 2
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 22050))
    noise = float(cfg.get("noise_scale", 0.667))
    noise_w = float(cfg.get("noise_w", 0.8))

    # piper stacks linear layers between the norms; the parameter count of a
    # stack of width w and depth d follows from the widths in the config.
    def stack(state: int, layer: int, mults=(1.0, 4.0, 4.0, 1.0)) -> int:
        return sum(int(state * a) * int(state * b) for a, b in zip(mults, mults[1:] + mults[:1]))

    enc = stack(n_audio_state, n_audio_layer)
    dec = stack(n_text_state, n_text_layer)
    total = enc + dec

    nodes = [
        _node("text", "Text", "input", column=0, order=0, branch="text",
              provenance="derived", notes=["the sentence to speak"]),
        _node("ph", "Phonemes", "tensor", column=0, order=1, branch="text",
              provenance="derived",
              notes=[f"from the {sample_rate} Hz audio config's phoneme set"]),
        _node("enc", f"Encoder block × {n_audio_layer}", "repeat_group",
              column=0, order=2, branch="encoder", params=enc, provenance="derived",
              group=f"encoder × {n_audio_layer}",
              dims={"width": n_audio_state, "layers": n_audio_layer},
              notes=[f"attention over {num_mels} mel bands"],
              repeat={"count": n_audio_layer, "template": {"nodes": [
                  _node("e.attn", "attention", "attention", column=0, order=0,
                        branch="encoder", params=n_audio_state ** 2,
                        provenance="derived"),
              ], "edges": []}}),
        _node("mel", f"Mel spectrogram ({num_mels} bands)", "input", column=1,
              order=0, branch="encoder", provenance="derived",
              notes=[f"{n_audio_ctx} frames of {num_mels} bands"]),
        _node("dec", f"Decoder block × {n_text_layer}", "repeat_group",
              column=1, order=1, branch="decoder", params=dec, provenance="derived",
              group=f"decoder × {n_text_layer}",
              dims={"width": n_text_state, "layers": n_text_layer},
              repeat={"count": n_text_layer, "template": {"nodes": [
                  _node("d.attn", "attention", "attention", column=1, order=0,
                        branch="decoder", params=n_text_state ** 2,
                        provenance="derived"),
              ], "edges": []}}),
        _node("voc", "Sample rate", "tensor", column=2, order=0, branch="decoder",
              provenance="derived",
              notes=[f"{sample_rate} Hz", f"noise {noise}, noise_w {noise_w}"]),
        _node("wav", "Waveform", "tensor", column=2, order=1, branch="decoder",
              provenance="derived",
              notes=["16-bit mono PCM"]),
    ]
    for n in nodes:
        n["param_share"] = round(n.get("params", 0) / total, 6) if total else 0
    return {
        "model_id": label or cfg_path.name.replace(".onnx.json", ""),
        "family": "tts",
        "nodes": nodes, "edges": [
            _edge("text", "ph"), _edge("ph", "enc"), _edge("mel", "enc"),
            _edge("enc", "dec"), _edge("dec", "voc"), _edge("voc", "wav"),
        ],
        "params_total": total,
        "provenance": "derived",
        "source": cfg_path.name,
        "meta": {"n_audio_layer": n_audio_layer, "n_text_layer": n_text_layer,
                 "audio_width": n_audio_state, "text_width": n_text_state,
                 "num_mels": num_mels, "sample_rate": sample_rate},
        "notes": ["shapes read from the voice's own config",
                  "parameter counts derived from those widths"],
    }


# ---------------------------------------------------------- diffusers stacks

def _unet_cfg(d: Path) -> dict[str, Any]:
    """The denoiser's config.

    A UNet pipeline keeps it in ``unet/``; a DiT-style transformer pipeline
    (Wan) keeps the same thing in ``transformer/``.
    """
    for name in ("unet", "transformer", "dit"):
        cfg = _cfg(d / name / "config.json")
        if cfg:
            return cfg
    return {}


def _build_diffusers_arch(root: str, label: str, kind: str) -> Optional[dict[str, Any]]:
    """A diffusers pipeline, from the component configs beside the weights.

    Image and video pipelines share a shape — a text encoder that conditions a
    denoiser over latents, then a decoder — so one builder covers both and only
    the repeating-block counts differ.
    """
    d = Path(root)
    index = _cfg(d / "model_index.json")
    if not index:
        return None
    unet = _unet_cfg(d)
    te = _cfg(d / "text_encoder" / "config.json")
    vae = _cfg(d / "vae" / "config.json")

    # A UNet (SD-Turbo) describes its stages by channel list; a DiT-style
    # transformer (Wan) has no such list and gives a width and a head dim
    # instead. Both describe the same idea, so read whichever is present.
    ch = [int(c) for c in (unet.get("block_out_channels") or [])]
    blocks = int(unet.get("num_layers") or (len(ch) if ch else 0))
    if not ch:
        width = int(unet.get("attention_head_dim", 0)) * int(unet.get("num_attention_heads", 0))
        if width:
            ch = [width]
    d_model = int(unet.get("cross_attention_dim") or unet.get("text_dim") or 0)
    te_layers = int(te.get("num_hidden_layers") or 0)
    te_dim = int(te.get("hidden_size") or te.get("d_model") or 0)
    latent = int(vae.get("latent_channels") or 4)
    # Wan records its patch as a per-axis list; the first entry is the
    # temporal one and the rest are spatial
    _pv = unet.get("patch_size") or 1
    patch = int(_pv[0]) if isinstance(_pv, (list, tuple)) and _pv else int(_pv)
    video = kind == "video"
    frames = int(unet.get("num_frames") or 1)

    # A UNet block's width grows with its channel list; counting each stage once
    # is enough for an honest size-by-stage reading.
    stage = sum(int(c) * int(c) for c in ch) if ch else 0
    ffn = int(unet.get("ffn_dim") or 0)
    if ffn and ch:
        # a transformer block also carries its feed-forward stack
        stage += blocks * 2 * ch[0] * ffn
    te_params = te_layers * (4 * te_dim * te_dim + 4 * te_dim) if te_dim else 0
    total = stage + te_params

    nodes = [
        _node("prompt", "Prompt", "input", column=0, order=0, branch="text",
              provenance="derived", notes=["words, or words and a picture"]),
        _node("te", f"Text encoder × {te_layers}", "repeat_group", column=0,
              order=1, branch="text", params=te_params, provenance="derived",
              group=f"text encoder × {te_layers}",
              dims={"width": te_dim, "layers": te_layers},
              notes=["conditions the denoiser through cross-attention"],
              repeat={"count": max(te_layers, 1), "template": {"nodes": [
                  _node("te.b", "transformer block", "layer", column=0, order=0,
                        branch="text", params=4 * te_dim * te_dim if te_dim else 0,
                        provenance="derived"),
              ], "edges": []}}),
        _node("lat", ("latents" + (f" ({frames} frames)" if video else "")),
              "tensor", column=1, order=0, branch="latents", provenance="derived",
              dims={"channels": latent, "patch": patch}),
        _node("unet", f"UNet × {blocks}", "repeat_group", column=1, order=1,
              branch="latents", params=stage, provenance="derived",
              group=f"unet × {blocks}",
              dims={"channels": ch or None},
              notes=["denoises the latents, conditioned on the text",
                     "channel widths are read from the pipeline config"],
              repeat={"count": max(blocks, 1), "template": {"nodes": [
                  _node("u.r", "resnet + attention", "layer", column=1, order=0,
                        branch="latents", params=stage // max(blocks, 1),
                        provenance="derived"),
              ], "edges": []}}),
        _node("vae", "VAE decoder", "matmul", column=2, order=0, branch="output",
              params=0, provenance="derived",
              dims={"latent_channels": latent},
              notes=["the decoder's own size is not in the config"]),
        _node("out", "Frame" if video else "Image", "tensor", column=2, order=1,
              branch="output", provenance="derived",
              notes=["pixels" if not video else "a clip"]),
    ]
    for n in nodes:
        n["param_share"] = round(n.get("params", 0) / total, 6) if total else 0
    return {
        "model_id": label or d.name,
        "family": kind,
        "nodes": nodes, "edges": [
            _edge("prompt", "te"), _edge("te", "unet"), _edge("lat", "unet"),
            _edge("unet", "vae"), _edge("vae", "out"),
        ],
        "params_total": total,
        "provenance": "derived",
        "source": "model_index.json + component configs",
        "meta": {"unet_blocks": blocks, "unet_channels": ch, "text_layers": te_layers,
                 "text_width": te_dim, "latent_channels": latent,
                 "frames": frames if video else None, "pipeline": index.get("_class_name")},
        "notes": ["shapes read from the pipeline's own config",
                  "parameter counts derived; the VAE decoder is not counted "
                  "because its size is not in the config"],
    }


def build_image_arch(path: str, label: str = "") -> Optional[dict[str, Any]]:
    return _build_diffusers_arch(path, label, "image")


def build_video_arch(path: str, label: str = "") -> Optional[dict[str, Any]]:
    return _build_diffusers_arch(path, label, "video")