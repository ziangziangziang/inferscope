"""Whisper's architecture, built from what the engine actually loaded.

There is no `config.json` for a whisper.cpp checkpoint — it is a binary blob in
the legacy GGML container, not GGUF. Rather than hardcode the published numbers
for "base", this reads the hyperparameter block that `whisper_model_load`
prints when the model comes up, so the diagram describes the file that is on
this disk and not the one we remember.

Provenance is `derived` throughout: the shapes are read from the engine's own
report and the parameter counts are computed from those shapes with the same
formula for every node. Nothing here is `measured` — that word is reserved for
the runtime trace.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .arch import _edge, _node

_LINE = re.compile(r"whisper_model_load:\s*(n_[a-z_]+)\s*=\s*([0-9]+)")


def read_hparams(log_path: str | Path) -> dict[str, int]:
    """The dimensions whisper.cpp reported for the model it loaded.

    Empty when the engine has not run yet — the caller then says so rather
    than drawing a diagram from numbers it invented.
    """
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return {}
    out: dict[str, int] = {}
    for key, val in _LINE.findall(text):
        out.setdefault(key, int(val))          # first block = the model, not a reload
    return out


def _transformer(h: dict[str, int], prefix: str, side: str,
                 column: int = 1) -> tuple[list[dict], list[dict], int]:
    """The stack of blocks on one side, with per-block parameter counts.

    Whisper's blocks are the standard pre-norm attention + MLP pair, so the
    counts follow from n_state and n_layer rather than from a table.
    """
    state = h[f"{prefix}_state"]
    layer = h[f"{prefix}_layer"]
    head = h[f"{prefix}_head"]

    # per block: q/k/v/o projections, then the MLP's two layers
    attn = 4 * state * state + 4 * state
    mlp = 2 * state * 4 * state + 4 * state + state
    per_block = attn + mlp
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for i in range(layer):
        bid = f"{side}.block{i}"
        nodes.append(_node(
            bid, f"block {i + 1}", "layer", column=column, order=i,
            group=f"{side} stack × {layer}", branch=side,
            dims={"n_state": state, "n_head": head},
            params=per_block,
            provenance="derived",
            notes=[f"attention over {head} heads of {state // head}",
                   "the same shape as every other block in the stack"],
            repeat={"count": layer,
                    "template": {"nodes": [_node(
                        f"{side}.t.attn", "self-attention", "attention", column=0,
                        order=0, params=attn, provenance="derived", branch=side,
                        notes=[f"{head} heads x {state // head} dims"])],
                        "edges": []}},
        ))
    return nodes, edges, per_block * layer


def build_whisper_arch(h: dict[str, int], model_label: str = "") -> Optional[dict[str, Any]]:
    """A graph for the loaded whisper model, or None if we do not know its shape."""
    need = ("n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
            "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer", "n_mels")
    if any(k not in h for k in need):
        return None

    vocab = h["n_vocab"]
    n_audio_ctx = h["n_audio_ctx"]
    n_audio_state = h["n_audio_state"]
    n_text_state = h["n_text_state"]
    n_text_ctx = h["n_text_ctx"]
    mels = h["n_mels"]

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # --- the front end: 30s of audio becomes mel frames, no weights ---
    nodes.append(_node(
        "mel", "mel spectrogram", "input", column=0, order=0, branch="front end",
        dims={"frames": n_audio_ctx, "bands": mels}, params=0,
        provenance="derived",
        notes=["no weights — this is the fixed front end",
               f"{n_audio_ctx} frames of {mels} mel bands"]))
    nodes.append(_node(
        "conv", "convolutional stem", "encoder", column=0, order=1, branch="front end",
        dims={"out": n_audio_state}, params=0,
        provenance="estimated",
        notes=["two stride-2 convolutions; shapes are the standard ones for "
               "this family, not read from the file"]))

    enc_nodes, enc_edges, enc_params = _transformer(h, "n_audio", "encoder", column=1)
    dec_nodes, dec_edges, dec_params = _transformer(h, "n_text", "decoder", column=2)
    nodes += enc_nodes + dec_nodes
    edges += enc_edges + dec_edges

    nodes.append(_node(
        "embed", "token embedding", "embed", column=3, order=0, branch="output",
        dims={"vocab": vocab, "d": n_text_state}, params=vocab * n_text_state,
        provenance="derived"))
    # Whisper ties the output projection to the token embedding: they are the
    # same weights, so counting both would inflate the total by 26M. The
    # projection gets zero parameters and says why.
    nodes.append(_node(
        "out", "output projection", "linear", column=3, order=1, branch="output",
        dims={"vocab": vocab}, params=0,
        provenance="derived",
        notes=["tied to the token embedding — the same weights, not a second copy"]))
    nodes.append(_node(
        "pos", "position embedding", "embed", column=0, order=2, branch="front end",
        dims={"positions": n_text_ctx, "d": n_text_state},
        params=n_text_ctx * n_text_state, provenance="derived"))

    total = enc_params + dec_params + (vocab * n_text_state) + n_text_ctx * n_text_state
    for n in nodes:
        n["param_share"] = round(n.get("params", 0) / total, 6) if total else 0

    edges += [
        _edge("mel", "conv", shape=[n_audio_ctx, mels], dtype="float32"),
        _edge("conv", "encoder.block0", shape=[n_audio_ctx, n_audio_state]),
        _edge("pos", "decoder.block0", shape=[n_text_ctx, n_text_state]),
        _edge("embed", "decoder.block0", shape=[n_text_ctx, n_text_state]),
        _edge("embed", "out", shape=[n_text_ctx, n_text_state]),
        _edge("out", "decoder.block0", shape=[n_text_ctx, n_text_state]),
    ]
    for side, n in (("encoder", h["n_audio_layer"]), ("decoder", h["n_text_layer"])):
        for i in range(n - 1):
            edges.append(_edge(f"{side}.block{i}", f"{side}.block{i + 1}"))

    return {
        "model_id": model_label or "whisper",
        "family": "asr",
        "nodes": nodes,
        "edges": edges,
        "params_total": total,
        "notes": [
            "shapes read from whisper.cpp's own load report for the file on disk",
            "parameter counts are derived from those shapes, not measured",
        ],
    }