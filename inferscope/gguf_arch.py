"""Architectures for llama.cpp checkpoints, read from the GGUF header.

The registry path builds a graph from a full-precision checkpoint's
``config.json``. The checkpoints this catalogue actually offers are quantised
GGUF files, which carry their own metadata instead — and which are frequently
*different models* from anything in the registry. Borrowing a registry entry
because the name looked close describes a checkpoint that is not on the GPU.

So the shapes here come from the file: block count, embedding width, feed
forward width, head counts, context length. Parameter counts are derived from
those widths by the standard formulas, and are labelled `derived` for that
reason.

What a GGUF header does *not* carry is a vision tower's dimensions. For a
vision-language checkpoint the language stack is described and the omission is
stated on the node, rather than filled in with a plausible-looking guess.
"""
from __future__ import annotations

from typing import Any, Optional

from . import gguf
from .arch import _edge, _node


def build_gguf_arch(path: str, label: str = "") -> Optional[dict[str, Any]]:
    kv = gguf.read_kv(path)
    arch = gguf.arch_name(kv)
    if not arch:
        return None
    d_model = gguf.g(kv, f"{arch}.embedding_length")
    n_layer = gguf.g(kv, f"{arch}.block_count")
    if not d_model or not n_layer:
        return None
    d_ff = gguf.g(kv, f"{arch}.feed_forward_length") or 4 * d_model
    n_head = gguf.g(kv, f"{arch}.attention.head_count")
    n_kv = gguf.g(kv, f"{arch}.attention.head_count_kv") or n_head
    d_head = gguf.g(kv, f"{arch}.attention.key_length") or (
        d_model // n_head if n_head else 0)
    ctx = gguf.g(kv, f"{arch}.context_length")
    deep = gguf.g(kv, f"{arch}.n_deepstack_layers")
    vision = gguf.is_vision(kv)

    # Standard pre-norm decoder block, with grouped-query attention: the K and V
    # projections are sized by kv heads, the rest by the full width.
    q = d_model * n_head * d_head
    kv_proj = 2 * d_model * n_kv * d_head
    attn = q + kv_proj + d_model * n_head * d_head
    mlp = 3 * d_model * d_ff                      # gate, up, down (swiglu)
    per_block = attn + mlp + 2 * d_model
    embed = d_model * d_model                      # lm_head, untied by default

    nodes: list[dict[str, Any]] = [
        _node("ids", "Token IDs", "input", column=0, order=0, branch="text",
              provenance="derived", notes=["the ids this stack reads"]),
        _node("embed", "Token embedding", "embed", column=0, order=1,
              branch="text", params=embed, provenance="derived",
              dims={"d_model": d_model},
              notes=["one learned vector per embedding width"]),
        _node("blocks", f"Transformer block × {n_layer}", "repeat_group",
              column=0, order=2, branch="text", params=per_block * n_layer,
              provenance="derived", group=f"decoder stack × {n_layer}",
              dims={"n_layer": n_layer, "d_model": d_model, "d_ff": d_ff},
              notes=[f"{n_head} query heads over {n_kv} key/value heads",
                     "grouped-query attention" if n_kv != n_head else "multi-head attention",
                     "shape read from the checkpoint's own header"],
              repeat={"count": n_layer, "template": {"nodes": [
                  _node("b.n1", "RMSNorm", "norm", column=0, order=0, branch="text",
                        params=d_model, provenance="derived"),
                  _node("b.attn", "attention", "attention", column=0, order=1,
                        branch="text", params=attn, provenance="derived",
                        dims={"n_head": n_head, "n_kv": n_kv, "d_head": d_head}),
                  _node("b.n2", "RMSNorm", "norm", column=0, order=2, branch="text",
                        params=d_model, provenance="derived"),
                  _node("b.mlp", "MLP (swiglu)", "matmul", column=0, order=3,
                        branch="text", params=mlp, provenance="derived",
                        dims={"d_ff": d_ff}),
              ], "edges": [_edge("b.n1", "b.attn"), _edge("b.attn", "b.n2"),
                           _edge("b.n2", "b.mlp")]}}),
        _node("fnorm", "Final RMSNorm", "norm", column=0, order=3, branch="text",
              params=d_model, provenance="derived"),
        _node("head", "LM head", "matmul", column=0, order=4, branch="text",
              params=embed, provenance="derived",
              notes=["vocabulary size is not in the GGUF header, so the width "
                     "is known but the row count is not"]),
        _node("sample", "Sampling (temp / top-p)", "sample", column=0, order=5,
              branch="text", provenance="derived"),
    ]

    if vision:
        # The header tells us this model takes pictures and how many deepstack
        # layers it has. It does not carry the tower's width or depth, so say so
        # rather than drawing one.
        nodes.insert(0, _node(
            "vimg", "Image input", "input", column=1, order=0, branch="vision",
            provenance="derived",
            notes=["this checkpoint is a vision-language model",
                   "the GGUF header carries no vision-tower dimensions, so the "
                   "tower is described by the engine, not drawn here"]))
        if deep:
            nodes.insert(1, _node(
                "deep", f"Deepstack layers × {deep}", "repeat_group", column=1,
                order=1, branch="vision", params=0, provenance="derived",
                group=f"deepstack × {deep}",
                notes=[f"{deep} layers fused back into the language stack — "
                       "count is known, width is not"]))

    total = per_block * n_layer + 2 * embed + d_model
    for n in nodes:
        n["param_share"] = round(n.get("params", 0) / total, 6) if total else 0

    edges = [
        _edge("ids", "embed"),
        _edge("embed", "blocks"),
        _edge("blocks", "fnorm"),
        _edge("fnorm", "head"),
        _edge("head", "sample"),
    ]
    if vision:
        edges.append(_edge("vimg", "deep"))
        edges.append(_edge("deep", "blocks"))

    return {
        "model_id": label or kv.get("general.name", "") or "gguf",
        "family": gguf.family_of(kv),
        "nodes": nodes,
        "edges": edges,
        "params_total": total,
        "meta": {
            "n_layers": n_layer, "hidden": d_model, "intermediate": d_ff,
            "n_heads": n_head, "kv_heads": n_kv, "head_dim": d_head,
            "max_pos": ctx, "architecture": arch,
            "deepstack_layers": deep or None, "vision": vision,
        },
        "provenance": "derived",
        "source": "gguf header",
        "notes": [
            "shapes read from the checkpoint's own GGUF header",
            "parameter counts derived from those widths, not measured",
            "vocabulary size is not stored in GGUF, so the LM head is described "
            "by width only",
        ],
    }