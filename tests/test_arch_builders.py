"""Architecture builders: every engine's diagram must describe its own model.

The failure these guard against is quiet: a checkpoint with no architecture
falls back to some other model's, and the diagram looks entirely plausible while
being about a different set of weights.
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferscope import gguf                                     # noqa: E402
from inferscope.gguf_arch import build_gguf_arch                # noqa: E402
from inferscope.diffusers_arch import (build_image_arch,        # noqa: E402
                                       build_piper_arch,
                                       build_video_arch)


# --------------------------------------------------------------- gguf writer

def _gguf(path: pathlib.Path, kv: dict) -> pathlib.Path:
    """A GGUF v3 header with just a metadata block."""
    def s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    body = b""
    for k, v in kv.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            body += s(k) + struct.pack("<I", 4) + struct.pack("<i", v)
        elif isinstance(v, str):
            body += s(k) + struct.pack("<I", 8) + s(v)
        elif isinstance(v, (list, tuple)):
            body += (s(k) + struct.pack("<I", 9) + struct.pack("<I", 8)
                     + struct.pack("<Q", len(v)) + b"".join(s(str(x)) for x in v))
        elif isinstance(v, float):
            body += s(k) + struct.pack("<I", 6) + struct.pack("<f", v)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + body)
    return path


@pytest.fixture
def vl_gguf(tmp_path):
    return _gguf(tmp_path / "vl.gguf", {
        "general.architecture": "qwen3vl",
        "general.type": "model",
        "general.tags": ["image-text-to-text"],
        "qwen3vl.block_count": 36,
        "qwen3vl.embedding_length": 4096,
        "qwen3vl.feed_forward_length": 12288,
        "qwen3vl.attention.head_count": 32,
        "qwen3vl.attention.head_count_kv": 8,
        "qwen3vl.attention.key_length": 128,
        "qwen3vl.context_length": 262144,
        "qwen3vl.n_deepstack_layers": 3,
    })


# --------------------------------------------------------------- gguf reading

def test_reads_the_header(vl_gguf):
    kv = gguf.read_kv(vl_gguf)
    assert gguf.arch_name(kv) == "qwen3vl"
    assert gguf.g(kv, "qwen3vl.block_count") == 36
    assert gguf.is_vision(kv)


def test_a_non_gguf_file_is_not_a_model(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    assert gguf.read_kv(p) == {}
    assert build_gguf_arch(str(p), "x") is None


def test_a_truncated_header_does_not_explode(tmp_path):
    p = tmp_path / "cut.gguf"
    p.write_bytes(b"GGUF" + b"\x03\x00")
    assert gguf.read_kv(p) == {}


# --------------------------------------------------------------- gguf graphs

def test_vision_model_gets_a_vision_branch_of_its_own(vl_gguf):
    g = build_gguf_arch(str(vl_gguf), "vl.gguf")
    assert g["meta"]["vision"] is True
    assert g["meta"]["n_layers"] == 36
    ids = {n["id"] for n in g["nodes"]}
    assert "vimg" in ids, "a vision checkpoint must not be drawn as text-only"
    assert {n.get("branch") for n in g["nodes"]} >= {"text", "vision"}
    assert g["family"] == "image-text-to-text"


def test_block_count_comes_from_the_file_not_a_constant(vl_gguf):
    g = build_gguf_arch(str(vl_gguf), "x")
    block = next(n for n in g["nodes"] if n["id"] == "blocks")
    assert block["repeat"]["count"] == 36
    assert "36" in block["label"]


def test_grouped_query_attention_is_read_from_both_head_counts(vl_gguf):
    g = build_gguf_arch(str(vl_gguf), "x")
    assert g["meta"]["n_heads"] == 32 and g["meta"]["kv_heads"] == 8


def test_a_model_without_dims_has_no_graph(tmp_path):
    p = _gguf(tmp_path / "bare.gguf", {"general.architecture": "mystery"})
    assert build_gguf_arch(str(p), "x") is None


def test_vision_omission_is_stated_rather_than_invented(vl_gguf):
    g = build_gguf_arch(str(vl_gguf), "x")
    note = " ".join(t for n in g["nodes"] for t in (n.get("notes") or []))
    assert "no vision-tower dimensions" in note


# --------------------------------------------------------------- piper

@pytest.fixture
def voice(tmp_path):
    (tmp_path / "v.onnx").write_bytes(b"\0")
    (tmp_path / "v.onnx.json").write_text(json.dumps({
        "num_mels": 80, "encoder_num_layers": 6, "encoder_dim": 256,
        "decoder_num_layers": 6, "decoder_dim": 256,
        "audio": {"sample_rate": 22050},
    }))
    return tmp_path


def test_piper_reads_its_own_config(voice):
    g = build_piper_arch(str(voice))
    assert g["meta"]["sample_rate"] == 22050
    assert g["meta"]["num_mels"] == 80
    assert {n.get("branch") for n in g["nodes"]} >= {"encoder", "decoder"}


def test_piper_without_a_config_has_no_graph(tmp_path):
    assert build_piper_arch(str(tmp_path)) is None


# --------------------------------------------------------------- diffusers

def _pipeline(root: pathlib.Path, unet: dict, te: dict, vae: dict, name="unet"):
    (root / name).mkdir(parents=True)
    (root / name / "config.json").write_text(json.dumps(unet))
    (root / "text_encoder").mkdir()
    (root / "text_encoder" / "config.json").write_text(json.dumps(te))
    (root / "vae").mkdir()
    (root / "vae" / "config.json").write_text(json.dumps(vae))
    (root / "model_index.json").write_text(json.dumps({"_class_name": "P"}))
    return str(root)


def test_unet_pipeline(tmp_path):
    root = _pipeline(tmp_path / "sd",
                     {"num_layers": 4, "block_out_channels": [320, 640, 1280, 1280],
                      "cross_attention_dim": 1024},
                     {"num_hidden_layers": 23, "hidden_size": 1024},
                     {"latent_channels": 4})
    g = build_image_arch(root)
    assert g["meta"]["unet_blocks"] == 4
    assert g["meta"]["unet_channels"][0] == 320
    assert g["family"] == "image"


def test_transformer_pipeline_reads_the_transformer_dir(tmp_path):
    """Wan keeps its denoiser in transformer/, not unet/, and has no channel list."""
    root = _pipeline(tmp_path / "wan",
                     {"num_layers": 30, "attention_head_dim": 128,
                      "num_attention_heads": 12, "ffn_dim": 8960,
                      "text_dim": 4096, "patch_size": [1, 2, 2]},
                     {}, {"latent_channels": 16},
                     name="transformer")
    g = build_video_arch(root)
    assert g["meta"]["unet_blocks"] == 30
    assert g["meta"]["unet_channels"] == [1536]
    assert g["params_total"] > 0
    assert g["family"] == "video"


def test_a_directory_without_a_pipeline_is_not_an_architecture(tmp_path):
    assert build_image_arch(str(tmp_path)) is None