"""Build TikZ-style computational graphs from real model configs.

The graph is data, not a picture: nodes carry dimensions, dtype and a provenance
tag, edges carry tensor metadata, and repetition is expressed as a collapsed
group with an expandable single-instance template so the frontend can render
"Transformer block × 64" by default and expand it on demand — never 64 identical
boxes.

Provenance discipline is the point: a shape read from a checkpoint's config.json
is `derived` (from config), a shape from a template with no config is `estimated`,
and nothing here is ever labelled `measured` — that word is reserved for the
runtime trace, which is correlated to these nodes elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .kv import dtype_bytes
from .cost import compute_costs, text_costs, vision_costs


def _kw(c: dict[str, Any]) -> dict[str, Any]:
    """Spread a cost entry into _node kwargs.

    A missing/None flops value is passed through as None so _node omits the key
    entirely and the UI shows the node as "compute unavailable" rather than 0.
    """
    return {"params": c.get("params", 0), "flops": c.get("flops")}

# --------------------------------------------------------------------------- config


def load_config(model_dir: str | Path) -> dict[str, Any]:
    p = Path(model_dir) / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(errors="replace"))
    except Exception:  # noqa: BLE001
        return {}


def detect_family(cfg: dict[str, Any]) -> str:
    """Classify a checkpoint into a template family from its real config."""
    archs = " ".join(cfg.get("architectures") or []).lower()
    mt = str(cfg.get("model_type", "")).lower()
    has_vis = ("vision" in cfg or "visual" in str(cfg.get("quantization_config", {}))[:8000]
               or "image_token_id" in cfg or "Vision" in " ".join(cfg.get("architectures") or []))
    if any(k in archs or k in mt for k in ("diffusion", "unet", "stablediffusion", "sdxl", "flux")):
        return "image" if "video" not in mt else "video"
    if any(k in archs or k in mt for k in ("video", "wan", "cogvideo", "mochi", "opensora")):
        return "video"
    if has_vis or "conditionalgeneration" in archs or "vlm" in mt or "vision" in mt:
        return "vlm"
    return "text"


def _norm_text(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg.get("text_config") or cfg
    layer_types = tc.get("layer_types") or []
    return {
        "n_layers": int(tc.get("num_hidden_layers") or len(layer_types) or 0),
        "hidden": tc.get("hidden_size"),
        "intermediate": tc.get("intermediate_size"),
        "n_heads": tc.get("num_attention_heads"),
        "kv_heads": tc.get("num_key_value_heads") or tc.get("num_attention_heads"),
        "head_dim": tc.get("head_dim"),
        "vocab": tc.get("vocab_size"),
        "max_pos": tc.get("max_position_embeddings"),
        "act": tc.get("hidden_act", "silu"),
        "rope": tc.get("rope_theta"),
        "partial_rotary": tc.get("partial_rotary_factor"),
        "full_interval": tc.get("full_attention_interval"),
        "attn_gate": tc.get("attn_output_gate"),
        "layer_types": layer_types,
        "linear_k_heads": tc.get("linear_num_key_heads"),
        "linear_v_heads": tc.get("linear_num_value_heads"),
        "linear_k_dim": tc.get("linear_key_head_dim"),
        "linear_v_dim": tc.get("linear_value_head_dim"),
        "tie": tc.get("tie_word_embeddings", False),
    }


def _norm_vision(cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    vc = cfg.get("vision_config") or cfg.get("visual") or {}
    if not vc:
        return None
    return {
        "depth": vc.get("depth"),
        "hidden": vc.get("hidden_size"),
        "heads": vc.get("num_heads"),
        "intermediate": vc.get("intermediate_size"),
        "patch": vc.get("patch_size"),
        "temporal_patch": vc.get("temporal_patch_size"),
        "merge": vc.get("spatial_merge_size"),
        "out_hidden": vc.get("out_hidden_size"),
        "in_channels": vc.get("in_channels", 3),
    }


# --------------------------------------------------------------------------- node helpers

def _node(nid, label, kind, *, column=0, order=0, group=None, dims=None,
          provenance="derived", notes=None, repeat=None, op=None,
          params=None, flops=None, branch=None):
    n: dict[str, Any] = {
        "id": nid, "label": label, "kind": kind, "column": column, "order": order,
        "provenance": provenance,
    }
    if group:
        n["group"] = group
    if branch:
        n["branch"] = branch
    if dims:
        n["dims"] = dims
    if notes:
        n["notes"] = notes
    if repeat:
        n["repeat"] = repeat
    if op:
        n["op"] = op
    if params is not None:
        n["params"] = int(params)
    if flops is not None:
        n["flops"] = int(flops)
    return n


def _edge(src, dst, *, shape=None, dtype="bfloat16", provenance="derived", label=""):
    e: dict[str, Any] = {"from": src, "to": dst, "provenance": provenance}
    if label:
        e["label"] = label
    if shape:
        nbytes = _prod(shape)
        t: dict[str, Any] = {"shape": shape, "dtype": dtype, "provenance": provenance}
        # Only compute a byte size when every dim is concrete; symbolic dims
        # like "B·T" are real in the diagram but have no numeric size yet.
        if nbytes is not None:
            t["bytes"] = int(nbytes * dtype_bytes(dtype))
        e["tensor"] = t
    return e


def _prod(xs):
    r = 1
    for x in xs:
        if not isinstance(x, (int, float)):
            return None          # symbolic dimension -> no numeric size
        r *= x
    return r


# --------------------------------------------------------------------------- text block template

def _block_template(t: dict[str, Any], *, idx_label: str = "i") -> dict[str, Any]:
    """One transformer block, expanded, with real dims from the config."""
    H = t["hidden"] or 4096
    nh = t["n_heads"] or 32
    kvh = t["kv_heads"] or nh
    hd = t["head_dim"] or (H // nh)
    inter = t["intermediate"] or int(2.7 * H)
    qdim = nh * hd
    kvdim = kvh * hd
    hybrid = bool(t.get("full_interval"))
    costs = text_costs(t)["block"]
    nodes = [
        _node("in", "block input", "tensor", order=0, dims={"shape": ["B·T", H]},
              provenance="derived"),
        _node("norm1", "RMSNorm", "norm", order=1, **_kw(costs["norm1"])),
        _node("qkv", "Q/K/V projection", "matmul", order=2, **_kw(costs["qkv"]),
              dims={"q": [qdim], "kv": [kvdim], "kv_heads": kvh, "heads": nh, "head_dim": hd},
              notes=[f"GQA: {nh} query heads share {kvh} KV heads ({nh//max(1,kvh)}:1)"]
              if kvh < nh else None),
        _node("rope", "RoPE", "positional", order=3, **_kw(costs["rope"]),
              notes=([f"partial rotary: {t['partial_rotary']} of head_dim"]
                     if t.get("partial_rotary") else None)),
        _node("attn", "Attention", "attention", order=4, **_kw(costs["attn"]),
              notes=(["hybrid: linear (gated deltanet) on most layers, full every "
                      f"{t['full_interval']}"] if hybrid else None)),
        _node("aproj", "Attention out projection", "matmul", order=5, **_kw(costs["aproj"]),
              dims={"out": [H]},
              notes=["attn output gate" if t.get("attn_gate") else None]),
        _node("add1", "Residual add", "add", order=6),
        _node("norm2", "RMSNorm", "norm", order=7, **_kw(costs["norm2"])),
        _node("mlp", "MLP / SwiGLU", "matmul", order=8, **_kw(costs["mlp"]),
              dims={"up/gate": [inter], "down": [H]},
              notes=[f"activation {t['act']}"]),
        _node("add2", "Residual add", "add", order=9),
        _node("out", "block output", "tensor", order=10, dims={"shape": ["B·T", H]}),
    ]
    edges = [
        _edge("in", "norm1", shape=["B·T", H]),
        _edge("norm1", "qkv", shape=["B·T", H]),
        _edge("qkv", "rope", shape=["B·T", qdim], label="Q,K,V"),
        _edge("rope", "attn", shape=["B·T", qdim], label="Q,K,V"),
        _edge("attn", "aproj", shape=["B·T", qdim]),
        _edge("aproj", "add1", shape=["B·T", H]),
        _edge("in", "add1", shape=["B·T", H], label="residual"),
        _edge("add1", "norm2", shape=["B·T", H]),
        _edge("norm2", "mlp", shape=["B·T", H]),
        _edge("mlp", "add2", shape=["B·T", H]),
        _edge("add1", "add2", shape=["B·T", H], label="residual"),
        _edge("add2", "out", shape=["B·T", H]),
    ]
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- family graphs

def _text_graph(model_id, t, prov="derived") -> dict[str, Any]:
    H = t["hidden"] or 4096
    V = t["vocab"] or 100000
    N = t["n_layers"] or 32
    c = text_costs(t)
    cn = c["nodes"]
    nodes = [
        _node("ids", "Token IDs", "input", order=0, dims={"shape": ["B", "T"]},
              provenance=prov),
        _node("embed", "Token embedding", "embed", order=1,
              dims={"table": [V, H]}, provenance=prov, **_kw(cn["embed"])),
        _node("blocks", f"Transformer block × {N}", "repeat_group", order=2,
              provenance=prov, **_kw(cn["blocks"]),
              repeat={"count": N, "template": _block_template(t)},
              notes=(["hybrid attention: linear + full interleaved"]
                     if t.get("full_interval") else None)),
        _node("fnorm", "Final RMSNorm", "norm", order=3, provenance=prov,
              **_kw(cn["final_norm"])),
        _node("head", "LM head (vocab proj)", "matmul", order=4,
              dims={"out": [V]}, provenance=prov, **_kw(cn["head"]),
              notes=["weights tied to embedding" if t.get("tie") else None]),
        _node("logits", "Logits", "tensor", order=5, dims={"shape": ["B", "T", V]},
              provenance=prov),
        _node("sample", "Sampling (temp/top-p)", "sample", order=6, provenance=prov),
    ]
    edges = [
        _edge("ids", "embed", shape=["B", "T"], provenance=prov),
        _edge("embed", "blocks", shape=["B·T", H], provenance=prov),
        _edge("blocks", "fnorm", shape=["B·T", H], provenance=prov),
        _edge("fnorm", "head", shape=["B·T", H], provenance=prov),
        _edge("head", "logits", shape=["B", "T", V], provenance=prov),
        _edge("logits", "sample", shape=["B", "T", V], provenance=prov),
    ]
    return _finish(model_id, "text", nodes, edges, t, prov, costs=c)


def _vlm_graph(model_id, t, v, prov="derived") -> dict[str, Any]:
    g = _text_graph(model_id, t, prov)
    # Prepend the vision branch (column 1) and a merger feeding the LM embed stage.
    p = v["patch"] or 16
    tp = v["temporal_patch"] or 2
    cv = v["in_channels"] or 3
    vh = v["hidden"] or 1152
    merge = v["merge"] or 2
    out_h = v["out_hidden"] or t["hidden"] or 4096
    vis_nodes = [
        _node("img", "Image / video input", "input", column=1, order=0,
              dims={"shape": ["B", cv, "T·frame", "H", "W"]}, provenance=prov),
        _node("patch", "Patch embed", "patchify", column=1, order=1,
              dims={"patch": [tp, p, p], "out": [vh]}, provenance=prov,
              **_kw(vision_costs(v)["nodes"]["patch"]),
              notes=[f"temporal_patch {tp}, spatial patch {p}×{p}"]),
        _node("vblocks", f"Vision block × {v['depth'] or 27}", "repeat_group",
              column=1, order=2, provenance=prov,
              **_kw(vision_costs(v)["nodes"]["vblocks"]),
              repeat={"count": v["depth"] or 27, "template": _vision_block(v)}),
        _node("merger", "Patch merger / projector", "projector", column=1, order=3,
              dims={"in": [vh * merge * merge], "out": [out_h]}, provenance=prov,
              **_kw(vision_costs(v)["nodes"]["merger"]),
              notes=[f"spatial merge {merge}×{merge} → project to {out_h}"]),
        _node("vtok", "Visual tokens", "tensor", column=1, order=4,
              dims={"shape": ["B", "Nv", out_h]}, provenance=prov),
    ]
    vis_edges = [
        _edge("img", "patch", provenance=prov),
        _edge("patch", "vblocks", shape=["B", "P", vh], provenance=prov),
        _edge("vblocks", "merger", shape=["B", "P", vh], provenance=prov),
        _edge("merger", "vtok", shape=["B", "Nv", out_h], provenance=prov),
    ]
    # splice: visual tokens join the text stream at the embedding stage.
    splice = _edge("vtok", "embed", shape=["B", "Nv", out_h], provenance=prov,
                   label="image_token_id")
    g["nodes"] = vis_nodes + g["nodes"]
    g["edges"] = vis_edges + [splice] + g["edges"]
    g["family"] = "vlm"
    g["meta"]["vision"] = v
    g["meta"]["image_token_id"] = None
    full = compute_costs(t, v)
    g["costs"] = full
    # Recompute shares now that vision parameters are included in the total.
    _apply_shares(g["nodes"], full)
    return g


def _vision_block(v) -> dict[str, Any]:
    vh = v["hidden"] or 1152
    inter = v["intermediate"] or int(3.7 * vh)
    nodes = [
        _node("vin", "vision block in", "tensor", order=0, dims={"shape": ["B", "P", vh]}),
        _node("vn1", "LayerNorm", "norm", order=1),
        _node("vqkv", "QKV", "matmul", order=2, dims={"out": [vh]}),
        _node("vattn", "Attention", "attention", order=3),
        _node("vproj", "out proj", "matmul", order=4, dims={"out": [vh]}),
        _node("vadd1", "Residual", "add", order=5),
        _node("vn2", "LayerNorm", "norm", order=6),
        _node("vmlp", "MLP", "matmul", order=7, dims={"hidden": [inter], "out": [vh]}),
        _node("vadd2", "Residual", "add", order=8),
        _node("vout", "vision block out", "tensor", order=9, dims={"shape": ["B", "P", vh]}),
    ]
    edges = [
        _edge("vin", "vn1", shape=["B", "P", vh]), _edge("vn1", "vqkv", shape=["B", "P", vh]),
        _edge("vqkv", "vattn", shape=["B", "P", vh]), _edge("vattn", "vproj", shape=["B", "P", vh]),
        _edge("vproj", "vadd1", shape=["B", "P", vh]), _edge("vin", "vadd1", shape=["B", "P", vh]),
        _edge("vadd1", "vn2", shape=["B", "P", vh]), _edge("vn2", "vmlp", shape=["B", "P", vh]),
        _edge("vmlp", "vadd2", shape=["B", "P", vh]), _edge("vadd1", "vadd2", shape=["B", "P", vh]),
        _edge("vadd2", "vout", shape=["B", "P", vh]),
    ]
    return {"nodes": nodes, "edges": edges}


def _image_graph(model_id, spec: dict[str, Any]) -> dict[str, Any]:
    """Diffusion image graph. spec may carry real dims; else estimated."""
    p = spec.get("provenance", "estimated")
    latent = spec.get("latent_channels", 4)
    dim = spec.get("model_dim", 1024)
    S = spec.get("steps", 20)
    nodes = [
        _node("prompt", "Text prompt", "input", order=0, provenance=p),
        _node("textenc", "Text encoder (CLIP/T5)", "encoder", order=1, provenance=p,
              dims={"out": [spec.get("text_dim", 768)]}),
        _node("noise", "Initial noise", "tensor", order=2,
              dims={"shape": ["B", latent, "H/8", "W/8"]}, provenance=p),
        _node("denoise", f"Denoising step × {S}", "repeat_group", order=3, provenance=p,
              repeat={"count": S, "template": _denoise_template(dim, latent, p)}),
        _node("sched", "Scheduler update", "scheduler", order=4, provenance=p),
        _node("vae", "VAE decode", "vae", order=5, provenance=p),
        _node("img", "Output image", "output", order=6,
              dims={"shape": ["B", 3, "H", "W"]}, provenance=p),
    ]
    edges = [
        _edge("prompt", "textenc", provenance=p),
        _edge("textenc", "denoise", label="conditioning", provenance=p),
        _edge("noise", "denoise", provenance=p),
        _edge("denoise", "sched", provenance=p),
        _edge("sched", "denoise", label="x_{t-1}", provenance=p),
        _edge("sched", "vae", provenance=p),
        _edge("vae", "img", provenance=p),
    ]
    return _finish(model_id, "image", nodes, edges, spec, prov=p)


def _denoise_template(dim, latent, p) -> dict[str, Any]:
    nodes = [
        _node("xin", "noisy latent x_t", "tensor", order=0, dims={"shape": ["B", latent, "h", "w"]}),
        _node("embed", "Latent patchify / input proj", "patchify", order=1, dims={"out": [dim]}),
        _node("blocks", "DiT / U-Net block × L", "repeat_group", order=2,
              repeat={"count": 28, "template": {"nodes": [
                  _node("n1", "Norm", "norm", order=0),
                  _node("self", "Self-attention", "attention", order=1),
                  _node("cross", "Cross-attention (text)", "attention", order=2),
                  _node("ff", "Feed-forward", "matmul", order=3)],
                  "edges": []}}),
        _node("ada", "Timestep / AdaLN modulation", "condition", order=3),
        _node("eps", "Noise / velocity prediction", "head", order=4),
    ]
    edges = [
        _edge("xin", "embed"), _edge("embed", "blocks"), _edge("blocks", "eps"),
        _edge("ada", "blocks", label="scale/shift"),
    ]
    return {"nodes": nodes, "edges": edges}


def _video_graph(model_id, spec: dict[str, Any]) -> dict[str, Any]:
    p = spec.get("provenance", "estimated")
    dim = spec.get("model_dim", 3072)
    S = spec.get("steps", 30)
    F = spec.get("frames", 81)
    nodes = [
        _node("prompt", "Text prompt", "input", order=0, provenance=p),
        _node("textenc", "Text encoder", "encoder", order=1, provenance=p),
        _node("noise", "Spatiotemporal noise", "tensor", order=2,
              dims={"shape": ["B", "C", F, "h", "w"]}, provenance=p,
              notes=[f"{F} frames"]),
        _node("denoise", f"Denoising step × {S}", "repeat_group", order=3, provenance=p,
              repeat={"count": S, "template": _video_denoise_template(dim, p, F)}),
        _node("sched", "Scheduler update", "scheduler", order=4, provenance=p),
        _node("vae", "3D VAE decode", "vae", order=5, provenance=p),
        _node("vid", "Output video", "output", order=6,
              dims={"shape": ["B", 3, F, "H", "W"]}, provenance=p),
    ]
    edges = [
        _edge("prompt", "textenc", provenance=p),
        _edge("textenc", "denoise", label="conditioning", provenance=p),
        _edge("noise", "denoise", provenance=p),
        _edge("denoise", "sched", provenance=p),
        _edge("sched", "denoise", label="x_{t-1}", provenance=p),
        _edge("sched", "vae", provenance=p),
        _edge("vae", "vid", provenance=p),
    ]
    return _finish(model_id, "video", nodes, edges, spec, prov=p)


def _video_denoise_template(dim, p, F) -> dict[str, Any]:
    nodes = [
        _node("xin", "video latent x_t", "tensor", order=0,
              dims={"shape": ["B", "C", F, "h", "w"]}),
        _node("patch", "Spatiotemporal patchify", "patchify", order=1, dims={"out": [dim]}),
        _node("spatial", "Spatial attention × L", "repeat_group", order=2,
              repeat={"count": 30, "template": {"nodes": [
                  _node("sn", "Norm", "norm", order=0),
                  _node("sattn", "Spatial self-attention", "attention", order=1),
                  _node("sff", "Feed-forward", "matmul", order=2)], "edges": []}}),
        _node("temporal", "Temporal attention × L", "repeat_group", order=3,
              repeat={"count": 30, "template": {"nodes": [
                  _node("tn", "Norm", "norm", order=0),
                  _node("tattn", "Temporal self-attention", "attention", order=1),
                  _node("tff", "Feed-forward", "matmul", order=2)], "edges": []}}),
        _node("eps", "Noise prediction", "head", order=4),
    ]
    edges = [
        _edge("xin", "patch"), _edge("patch", "spatial"), _edge("spatial", "temporal"),
        _edge("temporal", "eps"),
    ]
    return {"nodes": nodes, "edges": edges}


def _apply_shares(nodes, costs: dict[str, Any]) -> None:
    """Annotate each node with its share of total parameters / per-token FLOPs."""
    share_total = costs["totals"].get("params") or 0
    flops_total = costs["totals"].get("flops_per_token") or 0
    for n in nodes:
        p = n.get("params")
        if p and share_total:
            n["param_share"] = round(100.0 * p / share_total, 4)
        f = n.get("flops")
        if f and flops_total:
            n["flop_share"] = round(100.0 * f / flops_total, 4)


def _finish(model_id, family, nodes, edges, meta, prov="derived",
           costs: dict[str, Any] | None = None) -> dict[str, Any]:
    if costs:
        _apply_shares(nodes, costs)
    return {
        "model_id": model_id, "family": family, "provenance": prov,
        "source": "model_config" if prov == "derived" else "template",
        "meta": meta, "nodes": nodes, "edges": edges,
        "costs": costs or {},
    }


def build_graph(model_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Produce the graph for a checkpoint config, or an estimated template."""
    if not cfg:
        # No config at all: fall back to a generic text-LLM template, clearly
        # marked estimated (not derived from any real checkpoint).
        g = _text_graph(model_id, _norm_text({}), prov="estimated")
        g["meta"]["note"] = "generic template; no checkpoint config supplied"
        return g
    fam = detect_family(cfg)
    t = _norm_text(cfg)
    v = _norm_vision(cfg)
    if fam == "vlm" and v:
        g = _vlm_graph(model_id, t, v)
    elif fam == "text":
        g = _text_graph(model_id, t)
    elif fam == "image":
        g = _image_graph(model_id, {**t, "provenance": "derived"})
    else:
        g = _video_graph(model_id, {**t, "provenance": "derived"})
    g["meta"]["model_type"] = cfg.get("model_type")
    g["meta"]["architectures"] = cfg.get("architectures")
    return g


def list_models(model_dirs: str) -> list[dict[str, Any]]:
    out = []
    for base in [d.strip() for d in model_dirs.split(",") if d.strip()]:
        p = Path(base)
        if not p.exists():
            continue
        for child in sorted(p.iterdir()):
            cfg = load_config(child)
            if not cfg:
                continue
            out.append({
                "model_id": child.name, "path": str(child),
                "family": detect_family(cfg),
                "architectures": cfg.get("architectures"),
            })
    return out
