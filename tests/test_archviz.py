"""archviz: an architecture diagram must describe the checkpoint it names.

The failure these guard against is a plausible-looking wrong answer: a diagram
with the right shape and the wrong numbers is worse than no diagram, because it
is believed.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "archviz"))

from graph import (FetchError, Node, build_graph, detect_family,       # noqa: E402
                   attention_params, mlp_params)


# --------------------------------------------------------------- attention

def test_key_value_width_uses_kv_heads_not_query_heads():
    """Grouped-query attention shares K/V across heads; sizing them by the
    query count overstates every such model."""
    cfg = {"hidden_size": 4096, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128}
    total, shape = attention_params(cfg, 4096)
    assert shape == {"n_head": 32, "n_kv": 8, "head_dim": 128}
    q = 4096 * 32 * 128
    kv = 2 * 4096 * 8 * 128
    o = 32 * 128 * 4096
    assert total == q + kv + o


def test_multi_query_attention_with_one_kv_head():
    _, shape = attention_params(
        {"hidden_size": 2048, "num_attention_heads": 16,
         "num_key_value_heads": 1}, 2048)
    assert shape["n_kv"] == 1


def test_head_dim_defaults_to_width_over_heads():
    _, shape = attention_params({"hidden_size": 4096, "num_attention_heads": 32}, 4096)
    assert shape["head_dim"] == 128


# --------------------------------------------------------------------- mlp

def test_swiglu_has_three_matrices_where_gelu_has_two():
    _, plain = mlp_params({"hidden_act": "gelu", "intermediate_size": 5632}, 4096)
    _, gated = mlp_params({"hidden_act": "silu", "intermediate_size": 5632}, 4096)
    assert plain["gated"] is False and gated["gated"] is True
    # gate+up from the model, then down: one extra d_model x d_ff matrix
    assert gated["d_ff"] == plain["d_ff"] == 5632


def test_missing_intermediate_size_falls_back_rather_than_crashing():
    total, shape = mlp_params({"hidden_act": "gelu"}, 4096)
    assert shape["d_ff"] == 4096 * 4 and total > 0


# ------------------------------------------------------------------ family

def test_vision_is_detected_from_the_config_not_the_name():
    assert detect_family({"vision_config": {"hidden_size": 1152}}) == "vlm"
    assert detect_family({"model_type": "llama"}) == "llm"
    # A name mentioning vision, with no tower in the config, is a language model
    assert detect_family({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}) == "llm"


# ------------------------------------------------------------------- graph

QWEN3_4B = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "hidden_act": "silu", "vocab_size": 151936, "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


def test_node_order_is_the_data_flow():
    g = build_graph("Qwen/Qwen3-4B", QWEN3_4B)
    assert [n.id for n in g.nodes] == [
        "ids", "embed", "blocks", "fnorm", "head", "logits", "sample"]
    assert [(e.src, e.dst) for e in g.edges][:4] == [
        ("ids", "embed"), ("embed", "blocks"),
        ("blocks", "fnorm"), ("fnorm", "head")]


def test_repetition_is_one_node_with_a_count_and_a_template():
    g = build_graph("Qwen/Qwen3-4B", QWEN3_4B)
    block = next(n for n in g.nodes if n.id == "blocks")
    assert len([n for n in g.nodes if n.id == "blocks"]) == 1, "36 blocks must not be 36 nodes"
    assert block.repeat["count"] == 36
    assert "× 36" in block.label
    # the template is one block, and it is the real thing
    tpl = block.repeat["template"]
    assert [n["id"] for n in tpl["nodes"]] == ["n1", "attn", "n2", "mlp"]
    assert tpl["edges"]


def test_a_block_is_pre_norm_attention_then_mlp():
    g = build_graph("x", QWEN3_4B)
    tpl = next(n for n in g.nodes if n.id == "blocks").repeat["template"]
    kinds = [n["kind"] for n in tpl["nodes"]]
    assert kinds == ["norm", "attention", "norm", "matmul"]
    assert [(e["from"], e["to"]) for e in tpl["edges"]] == [
        ("n1", "attn"), ("attn", "n2"), ("n2", "mlp")]


def test_tied_embedding_is_counted_once():
    """Qwen3-4B is ~4.02B; counting the tied head twice gives 4.41B."""
    g = build_graph("x", dict(QWEN3_4B))
    assert 3.9e9 < g.params_total < 4.1e9, g.params_total
    head = next(n for n in g.nodes if n.id == "head")
    assert head.params == 0 and any("tied" in n for n in head.notes)


def test_untied_embedding_is_counted_twice():
    g = build_graph("x", dict(QWEN3_4B, tie_word_embeddings=False))
    head = next(n for n in g.nodes if n.id == "head")
    assert head.params == 151936 * 2560


def test_vision_model_gets_a_second_branch_that_feeds_the_stack():
    cfg = dict(QWEN3_4B)
    cfg["vision_config"] = {"hidden_size": 1152, "depth": 27, "num_heads": 16,
                            "intermediate_size": 4304, "patch_size": 16,
                            "image_size": 384}
    g = build_graph("x", cfg)
    assert g.family == "vlm"
    ids = [n.id for n in g.nodes]
    assert {"img", "patch", "vblocks", "merger", "vtok"} <= set(ids)
    assert g.meta is not None
    # the picture must arrive at the language stack, not sit beside it
    flow = {(e.src, e.dst) for e in g.edges}
    assert ("merger", "vtok") in flow and ("vtok", "blocks") in flow


def test_vision_repeats_are_counted_from_the_config():
    cfg = dict(QWEN3_4B, vision_config={"hidden_size": 1152, "depth": 27,
                                         "num_heads": 16, "intermediate_size": 4304})
    g = build_graph("x", cfg)
    assert next(n for n in g.nodes if n.id == "vblocks").repeat["count"] == 27


def test_text_config_is_read_when_the_language_stack_is_nested():
    cfg = {"vision_config": {"hidden_size": 1152, "depth": 27, "num_heads": 16},
           "text_config": {"hidden_size": 4096, "intermediate_size": 12288,
                           "num_hidden_layers": 36, "num_attention_heads": 32,
                           "num_key_value_heads": 8, "vocab_size": 151936}}
    g = build_graph("x", cfg)
    assert g.meta["hidden_size"] == 4096 and g.meta["num_hidden_layers"] == 36


def test_missing_vocab_is_reported_not_guessed():
    cfg = {k: v for k, v in QWEN3_4B.items() if k != "vocab_size"}
    g = build_graph("x", cfg)
    assert any("vocab" in w for w in g.warnings)
    assert next(n for n in g.nodes if n.id == "embed").params == 0


def test_a_config_with_no_shape_cannot_be_drawn():
    with pytest.raises(FetchError):
        build_graph("x", {"model_type": "llama"})


def test_param_shares_sum_to_about_one():
    g = build_graph("x", QWEN3_4B)
    total = sum(n.params for n in g.nodes)
    assert total == pytest.approx(g.params_total, rel=0.01)


def test_graph_is_json_serialisable_in_the_apps_shape():
    import json
    g = build_graph("Qwen/Qwen3-4B", QWEN3_4B)
    d = json.loads(json.dumps(g.as_dict()))
    assert d["provenance"] == "derived"
    assert {"nodes", "edges", "meta", "params_total"} <= set(d)