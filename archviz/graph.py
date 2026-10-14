"""Fetch a model's declared architecture from a Hugging Face repo.

Only files a repository publishes — `config.json`, `generation_config.json`,
`preprocessor_config.json`, `model.safetensors.index.json` — are read. No
weights are downloaded, because the architecture is declared rather than
learned: every dimension a diagram needs is written down in the config, and
fetching gigabytes of tensors to read a few hundred integers is slow,
expensive, and produces a diagram that is wrong whenever the fetch is partial.

Every field keeps where it came from. ``origin`` on each value is either
``"config"`` (read from a file) or ``"derived"`` (computed from other config
values), which is what lets the front end label provenance honestly instead of
presenting arithmetic as if it were a measurement.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "model.safetensors.index.json",
    "config.sub_configs.json",
)

ENDPOINTS = (
    "https://hf-mirror.com/{repo}/resolve/main/{name}",
    "https://huggingface.co/{repo}/resolve/main/{name}",
)


class FetchError(RuntimeError):
    """The repo could not be read, with the reason rather than a stack trace."""


@dataclass
class Value:
    """A number the diagram will draw, and where it came from."""
    value: Any
    origin: str = "config"          # "config" | "derived" | "assumed"

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "origin": self.origin}


@dataclass
class Tensor:
    shape: list[Any]
    dtype: str = "float16"


@dataclass
class Node:
    id: str
    label: str
    kind: str
    column: int = 0
    order: int = 0
    params: int = 0
    group: Optional[str] = None
    branch: Optional[str] = None
    repeat: Optional[dict[str, Any]] = None
    dims: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "label": self.label, "kind": self.kind,
                             "column": self.column, "order": self.order,
                             "params": self.params}
        for k in ("group", "branch", "repeat", "dims"):
            v = getattr(self, k)
            if v:
                d[k] = v
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    tensor: Optional[Tensor] = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"from": self.src, "to": self.dst}
        if self.label:
            d["label"] = self.label
        if self.tensor:
            d["tensor"] = {"shape": self.tensor.shape, "dtype": self.tensor.dtype}
        return d


@dataclass
class Graph:
    model_id: str
    family: str
    nodes: list[Node]
    edges: list[Edge]
    meta: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)
    params_total: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "family": self.family,
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
            "meta": self.meta, "sources": self.sources,
            "params_total": self.params_total,
            "provenance": "derived",
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------- fetch

#: The mirror's CDN answers urllib's default "Python-urllib/3" with 403 while
#: serving curl happily, so the client has to identify itself like anything else.
USER_AGENT = "archviz/1.0 (+model metadata reader)"


def _get(url: str, timeout: float) -> bytes:
    # Plain urlopen, deliberately: it honours https_proxy itself and follows
    # the mirror's redirect to its CDN. A hand-built ProxyHandler looked
    # reasonable and silently broke exactly that.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_configs(repo: str, timeout: float = 30.0) -> dict[str, Any]:
    """Every config file a repo publishes, keyed by name.

    A missing optional file is not an error; a missing ``config.json`` is,
    because without it there is nothing to draw.
    """
    out: dict[str, Any] = {}
    errors: list[str] = []
    for name in CONFIG_FILES:
        first: Optional[Exception] = None
        for tpl in ENDPOINTS:
            try:
                raw = _get(tpl.format(repo=repo, name=name), timeout)
                out[name] = json.loads(raw)
                first = None
                break
            except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
                # Report the mirror's reason, not the last endpoint's: the
                # fallback host is usually unreachable and its timeout would
                # hide a perfectly clear 403 or 404.
                first = first or e
        if first is not None and name == "config.json":
            errors.append(f"{repo}: {name} not readable ({first})")
    if errors:
        raise FetchError("; ".join(errors))
    return out


# ------------------------------------------------------------------- shapes

def _first(cfg: dict[str, Any], *paths, default=None):
    for path in paths:
        node: Any = cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return default


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def detect_family(cfg: dict[str, Any]) -> str:
    """Which shape of model this is, from what the config declares.

    Vision is detected from concrete config keys rather than from the model
    name: a repo called "Llama-3.2-11B-Vision" that has no vision tower in its
    config should be drawn as a language model, and one called "text-only" that
    does have `vision_config` should not be drawn without it.
    """
    if cfg.get("vision_config"):
        return "vlm"
    arch = _first(cfg, "architectures", default=[]) or []
    arch = arch[0] if isinstance(arch, list) and arch else ""
    if "CLIP" in arch or "VisionEncoder" in arch:
        return "vlm"
    if _first(cfg, "audio_config", "speech_config") is not None:
        return "omni"
    if _first(cfg, "model_type", default="") in ("llama", "mistral", "qwen2", "gemma"):
        return "llm"
    return "llm"


def _linear(d_in: int, d_out: int, bias: bool = True) -> int:
    return d_in * d_out + (d_out if bias else 0)


def attention_params(cfg: dict[str, Any], d_model: int) -> tuple[int, dict[str, Any]]:
    """Attention weight count and shape, honouring multi-query / grouped-query.

    The key and value projections are sized by *kv* heads, not query heads; get
    that wrong and the count is wrong for every model that shares weights
    between them, which is most of the recent ones.
    """
    n_head = _int(_first(cfg, "num_attention_heads", "n_head", "num_heads"))
    n_kv = _int(_first(cfg, "num_key_value_heads", "num_kv_heads", "n_kv_heads")) or n_head
    hidden = _int(_first(cfg, "hidden_size", "n_embd", "d_model", "dim")) or d_model
    head_dim = _int(_first(cfg, "head_dim")) or (hidden // n_head if n_head else 0)
    if not head_dim and n_head:
        head_dim = hidden // n_head

    bias = bool(_first(cfg, "attention_bias", default=False))
    q = _linear(d_model, n_head * head_dim, bias)
    k = _linear(d_model, n_kv * head_dim, bias)
    v = _linear(d_model, n_kv * head_dim, bias)
    o = _linear(n_head * head_dim, d_model, bias)
    return q + k + v + o, {"n_head": n_head, "n_kv": n_kv, "head_dim": head_dim}


def mlp_params(cfg: dict[str, Any], d_model: int) -> tuple[int, dict[str, Any]]:
    """Feed-forward weight count and shape.

    The activation decides the gate count: SwiGLU-family models have three
    matrices for the same hidden width instead of two.
    """
    act = str(_first(cfg, "hidden_act", "activation_function", default="") or "").lower()
    d_ff = _int(_first(cfg, "intermediate_size", "ffn_dim", "n_inner", "d_ff"))
    gated = act in ("silu", "swish", "geglu", "gated") or \
        _first(cfg, "intermediate_size", "ffn_dim", "n_inner") is None and \
        _first(cfg, "gate_proj", default=None) is not None
    if not d_ff:
        mult = 4
        if act in ("gelu", "relu", "silu"):
            mult = 8 if gated else 4
        d_ff = d_model * mult
    bias = bool(_first(cfg, "mlp_bias", default=False))
    if gated:
        # gate and up share the input, then down maps back
        total = (_linear(d_model, d_ff, bias) * 2
                 + _linear(d_ff, d_model, bias))
    else:
        total = _linear(d_model, d_ff, bias) + _linear(d_ff, d_model, bias)
    return total, {"d_ff": d_ff, "gated": gated}


def build_graph(repo: str, cfg: dict[str, Any],
                sources: dict[str, Any] | None = None) -> Graph:
    """A data-flow graph for a decoder-only transformer, from its config.

    Nodes are laid out in reading order: each column is a stage, and the edges
    between them are the tensors that actually flow. Repetition is a single
    node labelled `× N` carrying an expandable template, so a 60-layer model is
    drawn as sixty blocks but *shown* as one.
    """
    sources = sources or {}
    family = detect_family(cfg)
    # A vision-language config keeps the language stack under text_config, so
    # reading only the top level would find no layers at all.
    text = _first(cfg, "text_config", default=None) or cfg
    d_model = _int(_first(text, "hidden_size", "n_embd", "d_model", "dim"))
    n_layer = _int(_first(text, "num_hidden_layers", "n_layer", "num_layers"))
    vocab = _int(_first(text, "vocab_size"))
    if not d_model or not n_layer:
        raise FetchError(f"{repo}: config declares no hidden size or layer count")

    attn, attn_shape = attention_params(text, d_model)
    mlp, mlp_shape = mlp_params(text, d_model)
    # The epsilon is a float; casting it to int made every norm vanish
    # and left blocks as bare attention-then-MLP.
    rms = (d_model
           if _first(text, "rms_norm_eps", "layer_norm_eps",
                     "layer_norm_epsilon") is not None else 0)
    per_block = attn + mlp + 2 * (rms or d_model)
    embed = vocab * d_model if vocab else 0
    # A tied embedding and output head are one matrix, not two. Counting both
    # inflates the total by the whole embedding — 389M parameters on a 4B model,
    # which is exactly the kind of error that looks plausible and is not.
    tied = bool(_first(text, "tie_word_embeddings", default=False))

    nodes: list[Node] = []
    edges: list[Edge] = []

    nodes.append(Node("ids", "Token IDs", "input", column=1, order=0,
                      branch="text", notes=["the ids this stack reads"]))
    nodes.append(Node("embed", "Token embedding", "embed", column=1, order=1,
                      branch="text", params=embed,
                      dims={"vocab": vocab, "d_model": d_model},
                      notes=["also used as the output head" if tied
                             else "separate from the output head"]))

    inner: list[Node] = []
    inner_edges: list[Edge] = []
    if rms:
        inner.append(Node("n1", "RMSNorm", "norm", column=0, order=0,
                          branch="text", params=rms))
    inner.append(Node("attn", "Attention", "attention", column=0, order=1,
                      branch="text", params=attn, dims=attn_shape))
    if rms:
        inner.append(Node("n2", "RMSNorm", "norm", column=0, order=2,
                          branch="text", params=rms))
    inner.append(Node("mlp", "MLP", "matmul", column=0, order=3,
                      branch="text", params=mlp, dims=mlp_shape))
    for a, b in zip(inner, inner[1:]):
        inner_edges.append(Edge(a.id, b.id))

    nodes.append(Node("blocks", f"Transformer block × {n_layer}", "repeat_group",
                      column=2, order=0, branch="text",
                      params=per_block * n_layer,
                      group=f"decoder stack × {n_layer}",
                      dims={"n_layer": n_layer, "d_model": d_model, **mlp_shape},
                      notes=[
                          f"{attn_shape['n_head']} query heads over "
                          f"{attn_shape['n_kv']} key/value heads"
                          + ("  (grouped-query)" if attn_shape["n_kv"] != attn_shape["n_head"] else ""),
                          "every block has the same shape, so one is shown and the count is stated",
                      ],
                      repeat={"count": n_layer,
                              "template": {"nodes": [n.as_dict() for n in inner],
                                           "edges": [e.as_dict() for e in inner_edges]}}))

    nodes.append(Node("fnorm", "Final RMSNorm", "norm", column=3, order=0,
                      branch="text", params=d_model))
    nodes.append(Node("head", "LM head", "matmul", column=3, order=1,
                      branch="text", params=0 if tied else embed,
                      dims={"vocab": vocab, "d_model": d_model},
                      notes=["tied to the token embedding — the same weights, "
                             "not a second copy" if tied
                             else "a separate matrix"]))
    nodes.append(Node("logits", "Logits", "tensor", column=3, order=2,
                      branch="text",
                      dims={"vocab": vocab}))
    nodes.append(Node("sample", "Sampling", "sample", column=3, order=3,
                      branch="text",
                      notes=[f"top-p/top-k/temperature from "
                             f"generation_config" if sources.get("generation_config.json")
                             else "no generation_config published"]))

    # --- data flow, in reading order ---
    edges += [
        Edge("ids", "embed", "int64[B,T]", Tensor(["B", "T"], "int64")),
        Edge("embed", "blocks", "[B,T,C]", Tensor(["B", "T", d_model])),
        Edge("blocks", "fnorm", "[B,T,C]", Tensor(["B", "T", d_model])),
        Edge("fnorm", "head", "[B,T,C]", Tensor(["B", "T", d_model])),
        Edge("head", "logits", "[B,T,V]", Tensor(["B", "T", vocab or "V"])),
        Edge("logits", "sample"),
    ]

    total = per_block * n_layer + (embed if tied else 2 * embed) + d_model

    # --- vision branch, when the config declares a tower ---
    if family == "vlm" and isinstance(cfg.get("vision_config"), dict):
        vis = cfg["vision_config"]
        v_hidden = _int(_first(vis, "hidden_size", "hidden_dim", "d_model"))
        v_layer = _int(_first(vis, "num_hidden_layers", "num_layers", "depth"))
        v_patch = _int(_first(vis, "patch_size", "patch_dim"), default=16)
        v_image = _int(_first(vis, "image_size"), default=0)
        v_attn, v_attn_shape = attention_params(vis, v_hidden)
        v_mlp, v_mlp_shape = mlp_params(vis, v_hidden)
        v_block = (v_attn + v_mlp + 2 * v_hidden) if (v_hidden and v_layer) else 0
        warn = [n for n in nodes if n.id == "blocks"][0]
        merge = _int(_first(vis, "projector_hidden_act") is not None)
        merge_params = (v_hidden * v_hidden) if v_hidden else 0

        # placed on the back row (column 1), read behind the text stack
        vis_nodes = [
            Node("img", "Image", "input", column=0, order=0, branch="vision",
                 dims={"size": v_image or None, "patch": v_patch or None},
                 notes=[f"split into {v_patch}px patches" if v_patch else
                        "patch size is absent from the config"]),
            Node("patch", "Patch embedding", "patchify", column=0, order=1,
                 branch="vision", params=merge_params,
                 dims={"d_model": v_hidden},
                 notes=["linear projection of each flattened patch"]),
            Node("vblocks", f"Vision block × {v_layer}", "repeat_group",
                 column=1, order=2, branch="vision", params=v_block * v_layer,
                 group=f"vision stack × {v_layer}",
                 dims={"n_layer": v_layer, "d_model": v_hidden, **v_attn_shape},
                 notes=["same shape in every block, so one is shown"],
                 repeat={"count": v_layer or 1, "template": {"nodes": [
                     Node("v.attn", "Attention", "attention", column=0, order=0,
                          branch="vision", params=v_attn, dims=v_attn_shape).as_dict(),
                     Node("v.mlp", "MLP", "matmul", column=1, order=1,
                          branch="vision", params=v_mlp, dims=v_mlp_shape).as_dict(),
                 ], "edges": [Edge("v.attn", "v.mlp").as_dict()]}}),
            Node("merger", "Merger / projector", "projector", column=1, order=3,
                 branch="vision", params=merge_params,
                 dims={"d_model": v_hidden},
                 notes=["maps vision width to the language width"
                        if v_hidden != d_model else
                        "already the language width"]),
            Node("vtok", "Visual tokens", "tensor", column=1, order=4,
                 branch="vision",
                 notes=[f"interleaved with the text tokens"
                        if _first(vis, "spatial_merge_size") else
                        "concatenated with the text tokens"]),
        ]
        # insert before the text stack so the vision branch reads behind it
        at = next(i for i, n in enumerate(nodes) if n.id == "blocks")
        nodes[at:at] = vis_nodes
        edges += [Edge("img", "patch", "[B,T,P]", Tensor(["B", "T", "patch"], "int32")),
                  Edge("patch", "vblocks", "[B,S,Cv]", Tensor(["B", "S", v_hidden or "Cv"])),
                  Edge("vblocks", "merger"),
                  Edge("merger", "vtok", "[B,S,C]", Tensor(["B", "S", d_model])),
                  Edge("vtok", "blocks", "vision into the language stack")]
        if v_block:
            total += v_block * v_layer + merge_params * 2
        warn.notes.append("vision tokens join the token stream here")

    warnings: list[str] = []
    for n in nodes:
        n.params = min(n.params, total) if total else n.params
    if not vocab:
        warnings.append("vocab_size is absent, so the embedding and head are "
                        "not counted and the logits width is unknown")
    missing = [k for k, v in (("num_attention_heads", attn_shape["n_head"]),
                              ("intermediate_size", mlp_shape["d_ff"]))
               if not _first(cfg, k) and not v]
    for k in missing:
        warnings.append(f"{k} is absent from the config; the shape was inferred")

    return Graph(
        model_id=repo, family=family, nodes=nodes, edges=edges,
        params_total=total, sources=sources, warnings=warnings,
        meta={
            "hidden_size": d_model, "num_hidden_layers": n_layer,
            "vocab_size": vocab or None,
            "num_attention_heads": attn_shape["n_head"],
            "num_key_value_heads": attn_shape["n_kv"],
            "head_dim": attn_shape["head_dim"],
            "intermediate_size": mlp_shape["d_ff"],
            "max_position_embeddings": _first(cfg, "max_position_embeddings"),
            "architectures": _first(cfg, "architectures"),
            "torch_dtype": _first(cfg, "torch_dtype"),
            "params_per_block": per_block,
        },
    )

# ------------------------------------------------------------------------ cli

def _cli() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="architecture graph from a HF repo id")
    ap.add_argument("repo")
    ap.add_argument("--json", action="store_true", help="emit the graph as JSON")
    ap.add_argument("--md", action="store_true", help="emit a summary table")
    a = ap.parse_args()

    try:
        cfgs = fetch_configs(a.repo)
        graph = build_graph(a.repo, cfgs["config.json"], cfgs)
    except FetchError as e:
        print(str(e), file=sys.stderr)
        return 1

    if a.json:
        print(json.dumps(graph.as_dict(), indent=2))
        return 0
    if a.md:
        m = graph.meta
        print(f"### {a.repo}\n")
        print(f"- family: **{graph.family}**")
        print(f"- layers: **{m['num_hidden_layers']}**, width **{m['hidden_size']}**, "
              f"heads **{m['num_attention_heads']}**"
              f" (kv {m['num_key_value_heads']}), head_dim **{m['head_dim']}**")
        print(f"- ffn: **{m['intermediate_size']}**, vocab **{m['vocab_size']}**")
        print(f"- parameters (derived): **{graph.params_total:,}**\n")
        print("| # | node | kind | params |")
        print("|---|------|------|--------|")
        for i, n in enumerate(graph.nodes, 1):
            print(f"| {i} | {n.label} | {n.kind} | {n.params:,} |")
        for w in graph.warnings:
            print(f"\n> {w}")
        return 0

    for n in graph.nodes:
        extra = f" ×{n.repeat['count']}" if n.repeat else ""
        print(f"  {n.column}  {n.label}{extra:>18}  {n.params:>14,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
