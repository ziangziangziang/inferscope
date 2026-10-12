import pytest

from inferscope.cost import compute_costs, text_costs, vision_costs
from inferscope.arch import build_graph


CFG = {
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 32000,
}


def norm(cfg):
    return {
        "hidden": cfg.get("hidden_size"),
        "intermediate": cfg.get("intermediate_size"),
        "n_layers": cfg.get("num_hidden_layers"),
        "n_heads": cfg.get("num_attention_heads"),
        "kv_heads": cfg.get("num_key_value_heads"),
        "head_dim": cfg.get("head_dim"),
        "vocab": cfg.get("vocab_size"),
        "qkv_bias": cfg.get("qkv_bias", True),
        "tie": cfg.get("tie_word_embeddings", False),
        "full_interval": cfg.get("full_interval"),
    }


def test_linear_param_count_is_exact():
    c = text_costs(norm(CFG))
    H, I, V, nh, kvh, hd = 4096, 11008, 32000, 32, 8, 128
    qkv = H * (nh * hd) + 2 * (H * kvh * hd) + (nh * hd + 2 * kvh * hd)
    mlp = 3 * (H * I)
    aproj = (nh * hd) * H + H
    # two RMSNorms (weight only) + qkv + attn out proj + SwiGLU MLP
    per_block = 2 * H + qkv + aproj + mlp
    # embedding + blocks + untied LM head + final RMSNorm
    expected = V * H + 32 * per_block + H * V + H
    assert c["totals"]["params"] == expected


def test_flops_are_dominated_by_mlp():
    c = text_costs(norm(CFG))
    mlp = c["block"]["mlp"]["flops"]
    qkv = c["block"]["qkv"]["flops"]
    assert mlp > qkv
    assert c["totals"]["flops_per_token"] == sum(
        n["flops"] for n in c["nodes"].values())


def test_attention_flops_scale_with_context():
    a = text_costs(norm(CFG), ctx=1)["block"]["attn"]["flops"]
    b = text_costs(norm(CFG), ctx=1024)["block"]["attn"]["flops"]
    assert b > a * 100


def test_linear_attention_flops_reported_unavailable():
    """A recurrent state's per-token cost is not implied by the config, so the
    cost model must mark it unavailable rather than invent a number."""
    cfg = dict(CFG, full_interval=6)
    c = text_costs(norm(cfg), ctx=4096)
    assert c["block"]["attn"]["flops"] is None
    assert c["totals"]["attention"] == "linear"
    assert c["totals"]["attention_flops"] == "unavailable"
    # Parameters are still exact even when compute is unknown.
    assert c["block"]["attn"]["params"] == 0


def test_full_attention_flops_grow_with_context():
    a = text_costs(norm(CFG), ctx=1)["block"]["attn"]["flops"]
    b = text_costs(norm(CFG), ctx=1024)["block"]["attn"]["flops"]
    assert b > a * 100


def test_tied_embeddings_do_not_double_count_head():
    untied = text_costs(norm(dict(CFG, tie_word_embeddings=False)))
    tied = text_costs(norm(dict(CFG, tie_word_embeddings=True)))
    assert untied["totals"]["params"] > tied["totals"]["params"]
    assert tied["nodes"]["head"]["params"] == 0


def test_vision_costs_positive():
    v = {"hidden": 1152, "depth": 27, "patch": 16, "temporal_patch": 2,
         "in_channels": 3, "merge": 2, "out_hidden": 4096}
    vc = vision_costs(v)
    assert vc["totals"]["params"] > 0
    assert vc["nodes"]["vblocks"]["params"] > vc["nodes"]["patch"]["params"]


def test_graph_exposes_params_and_shares():
    g = build_graph("t", CFG)
    tot = g["costs"]["totals"]["params"]
    assert tot > 0
    blocks = [n for n in g["nodes"] if n["id"] == "blocks"][0]
    assert blocks["params"] > 0
    assert 0 < blocks["param_share"] <= 100
    share_sum = sum(n.get("param_share", 0) for n in g["nodes"])
    assert 40 < share_sum <= 101   # top-level nodes only; blocks count once


def test_vlm_graph_includes_vision_params():
    cfg = dict(CFG, model_type="qwen3_vlm",
               vision_config={"depth": 27, "hidden_size": 1152,
                              "patch_size": 16, "spatial_merge_size": 2,
                              "temporal_patch_size": 2})
    g = build_graph("vlm", cfg)
    t = g["costs"]["totals"]
    assert "vision_params" in t
    assert t["params"] == t["params_per_token"]
    assert t["vision_params"] > 0


def test_estimated_template_has_costs():
    g = build_graph("unknown", {})
    assert g["costs"]["totals"]["params"] > 0