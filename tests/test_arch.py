from inferscope.arch import (build_graph, detect_family, list_models, load_config,
                             _norm_text, _norm_vision)


def test_detect_family_vlm(qwen_config):
    assert detect_family(qwen_config) == "vlm"


def test_detect_family_text():
    assert detect_family({"architectures": ["LlamaForCausalLM"],
                          "num_hidden_layers": 32}) == "text"


def test_text_graph_collapses_layers():
    g = build_graph("m", {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 32,
                         "hidden_size": 4096, "num_attention_heads": 32,
                         "num_key_value_heads": 8, "head_dim": 128,
                         "intermediate_size": 11008, "vocab_size": 32000})
    assert g["family"] == "text"
    repeats = [n for n in g["nodes"] if n.get("repeat")]
    assert repeats and repeats[0]["repeat"]["count"] == 32
    assert "× 32" in repeats[0]["label"]
    # template has the inner ops but is NOT drawn 32 times at top level
    assert len(g["nodes"]) < 12
    assert repeats[0]["repeat"]["template"]["nodes"]


def test_vlm_graph_has_both_branches(qwen_config):
    g = build_graph("Qwen3.8-27B-INT4", qwen_config)
    assert g["family"] == "vlm"
    cols = {n["column"] for n in g["nodes"]}
    assert 0 in cols and 1 in cols  # text stream + vision branch
    vrep = next(n for n in g["nodes"] if "Vision block" in n["label"])
    lrep = next(n for n in g["nodes"] if "Transformer block" in n["label"])
    assert vrep["repeat"]["count"] == 27 and lrep["repeat"]["count"] == 64
    # visual tokens splice into the embedding via image_token_id
    assert any(e.get("label") == "image_token_id" for e in g["edges"])


def test_symbolic_dims_do_not_crash_and_omit_bytes(qwen_config):
    g = build_graph("m", qwen_config)
    for e in g["edges"]:
        t = e.get("tensor")
        if t and any(isinstance(s, str) for s in t.get("shape", [])):
            assert "bytes" not in t  # symbolic -> no fake numeric size


def test_image_and_video_templates():
    img = build_graph("i", {"model_type": "stable_diffusion"})
    assert img["family"] == "image"
    assert any(n.get("repeat") and "Denoising" in n["label"] for n in img["nodes"])
    vid = build_graph("v", {"model_type": "wan_video"})
    assert vid["family"] == "video"
    # the denoise template must contain a temporal-attention repeat group
    denoise = next(n for n in vid["nodes"] if n.get("repeat") and "Denoising" in n["label"])
    tmpl_labels = [n["label"] for n in denoise["repeat"]["template"]["nodes"]]
    assert any("Temporal attention" in l for l in tmpl_labels)


def test_empty_config_is_text_template_not_image():
    g = build_graph("m", {})
    assert g["family"] == "text"
    assert g["provenance"] == "estimated"


def test_norm_extracts_real_numbers(qwen_config):
    t = _norm_text(qwen_config)
    v = _norm_vision(qwen_config)
    assert t["n_layers"] == 64 and t["kv_heads"] == 4 and t["head_dim"] == 256
    assert v["depth"] == 27 and v["patch"] == 16 and v["merge"] == 2
