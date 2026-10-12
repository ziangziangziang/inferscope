"""Analytic parameter and FLOP accounting for a checkpoint config.

Everything here is *derived* from the checkpoint config — never measured at
runtime. It answers "where does the capacity sit", which is a different
question from "where did the time go" (that is the Timeline/Bottlenecks side).

Conventions
-----------
* A multiply-accumulate counts as 2 FLOPs, so a ``[in] × [out]`` matmul
  costs ``2 * in * out`` per token.
* ``flops`` is reported **per token** at a reference context length, because
  attention cost is context dependent. The context used is returned alongside
  so the UI can label it honestly.
* Embedding is a gather, not a matmul: it carries parameters but ~no FLOPs.
"""
from __future__ import annotations

from typing import Any

# Default reference context for per-token attention cost.
DEFAULT_CTX = 512


def _lin(i: int, o: int, bias: bool = False) -> int:
    return i * o + (o if bias else 0)


def _matmul_flops(i: int, o: int, n: int = 1, bias: bool = False) -> int:
    return 2 * n * i * o + (n * o if bias else 0)


def text_costs(t: dict[str, Any], *, ctx: int = DEFAULT_CTX) -> dict[str, Any]:
    """Per-node parameter counts and per-token FLOPs for the language model.

    Returns ``{"nodes": {id: {...}}, "per_layer": {...}, "totals": {...}}``.
    ``flops`` is per token at ``ctx`` tokens of context.
    """
    H = t.get("hidden") or 4096
    V = t.get("vocab") or 100000
    N = t.get("n_layers") or 32
    nh = t.get("n_heads") or 32
    kvh = t.get("kv_heads") or nh
    hd = t.get("head_dim") or (H // nh)
    inter = t.get("intermediate") or int(2.7 * H)
    qdim, kvdim = nh * hd, kvh * hd
    qkv_bias = bool(t.get("qkv_bias"))
    tied = bool(t.get("tie"))

    # ---- one transformer block -------------------------------------------
    qkv_p = _lin(H, qdim, qkv_bias) + 2 * _lin(H, kvdim, qkv_bias)
    aproj_p = _lin(qdim, H, qkv_bias)
    mlp_p = 3 * _lin(H, inter)          # SwiGLU: gate, up, down
    norm_p = H                          # RMSNorm weight only

    qkv_f = _matmul_flops(H, qdim + 2 * kvdim, 1, qkv_bias)
    aproj_f = _matmul_flops(qdim, H, 1, qkv_bias)
    mlp_f = 3 * _matmul_flops(H, inter)
    # Full attention: QK^T then (P)V, both [ctx, qdim] per token.
    attn_full_f = 4 * ctx * qdim

    linear = bool(t.get("full_interval"))
    # Linear/recurrent attention cost depends on the recurrent state size, which
    # is not described by this config. Report it as unavailable rather than
    # inventing a number.
    attn_f = None if linear else attn_full_f

    block = {
        "norm1": {"params": norm_p, "flops": 0},
        "qkv": {"params": qkv_p, "flops": qkv_f},
        "rope": {"params": 0, "flops": 0},
        "attn": {"params": 0, "flops": attn_f},
        "aproj": {"params": aproj_p, "flops": aproj_f},
        "mlp": {"params": mlp_p, "flops": mlp_f},
        "norm2": {"params": norm_p, "flops": 0},
    }
    layer_params = sum(b["params"] for b in block.values())
    layer_flops = sum(b["flops"] or 0 for b in block.values())

    nodes = {
        "embed": {"params": _lin(V, H), "flops": 0},
        "blocks": {"params": layer_params * N, "flops": layer_flops * N},
        "head": {"params": 0 if tied else _lin(H, V), "flops": _matmul_flops(H, V)},
        "final_norm": {"params": norm_p, "flops": 0},
    }
    total_params = sum(n["params"] for n in nodes.values())
    total_flops = sum(n["flops"] or 0 for n in nodes.values())

    return {
        "nodes": nodes,
        "block": block,
        "per_layer": {"params": layer_params, "flops": layer_flops},
        "totals": {
            "params": total_params,
            "flops_per_token": total_flops,
            "params_per_layer": layer_params,
            "layers": N,
            "ctx": ctx,
            "attention": "linear" if linear else "full",
            "attention_flops": "unavailable" if linear else "estimated",
        },
    }


def vision_costs(v: dict[str, Any], *, ctx: int = DEFAULT_CTX) -> dict[str, Any]:
    """Per-node parameter counts for the vision tower."""
    p = v.get("patch") or 16
    tp = v.get("temporal_patch") or 2
    cv = v.get("in_channels") or 3
    vh = v.get("hidden") or 1152
    depth = v.get("depth") or 27
    inter = v.get("intermediate") or int(3.7 * vh)
    merge = v.get("merge") or 2
    out_h = v.get("out_hidden") or vh

    patch_embed_p = _lin(cv * tp * p * p, vh)
    vblock_p = 2 * vh                       # two LayerNorms
    vblock_p += _lin(vh, vh) + 2 * _lin(vh, vh)   # qkv + out proj
    vblock_p += 2 * _lin(vh, inter)                # MLP up + down
    merger_p = _lin(vh * merge * merge, out_h)

    return {
        "nodes": {
            "patch": {"params": patch_embed_p, "flops": 0},
            "vblocks": {"params": vblock_p * depth, "flops": 0},
            "merger": {"params": merger_p, "flops": _matmul_flops(vh * merge * merge, out_h)},
        },
        "per_block": {"params": vblock_p},
        "totals": {
            "params": patch_embed_p + vblock_p * depth + merger_p,
            "flops_per_token": 0,
            "depth": depth,
            "ctx": ctx,
        },
    }


def compute_costs(text: dict[str, Any], vision: dict[str, Any] | None,
                  *, ctx: int = DEFAULT_CTX) -> dict[str, Any]:
    """Merge language + vision costs into one graph-level cost report."""
    tc = text_costs(text, ctx=ctx)
    out: dict[str, Any] = {
        "nodes": dict(tc["nodes"]),
        "block": tc["block"],
        "per_layer": tc["per_layer"],
        "totals": dict(tc["totals"]),
        "ctx": ctx,
    }
    if vision:
        vc = vision_costs(vision, ctx=ctx)
        out["nodes"].update(vc["nodes"])
        out["vision_per_block"] = vc["per_block"]
        vt = vc["totals"]
        out["totals"]["params"] += vt["params"]
        out["totals"]["vision_params"] = vt["params"]
        out["totals"]["params_per_token"] = out["totals"]["params"]
    else:
        out["totals"]["params_per_token"] = out["totals"]["params"]
    return out