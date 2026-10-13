"""Vision-language support: projector discovery and image message plumbing."""
from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferscope import units as U          # noqa: E402
from inferscope.chat import ManagedServer  # noqa: E402


def _big_file(path: pathlib.Path, size: int = 2 << 20) -> pathlib.Path:
    path.write_bytes(b"\0" * size)
    return path


# ---------------------------------------------------------------- projector

def test_no_projector_means_text_only(tmp_path):
    _big_file(tmp_path / "model.gguf")
    assert U.mmproj_for(str(tmp_path / "model.gguf")) is None


def test_projector_beside_weights_is_found(tmp_path):
    _big_file(tmp_path / "model.gguf")
    proj = _big_file(tmp_path / "mmproj-F16.gguf")
    assert U.mmproj_for(str(tmp_path / "model.gguf")) == str(proj)


def test_half_downloaded_projector_is_not_claimed(tmp_path):
    """A truncated download must not read as a working projector."""
    _big_file(tmp_path / "model.gguf")
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"\0" * 16)
    assert U.mmproj_for(str(tmp_path / "model.gguf")) is None


def test_missing_directory_is_not_an_error(tmp_path):
    assert U.mmproj_for(str(tmp_path / "nope" / "model.gguf")) is None


# ---------------------------------------------------------------- catalogue

def _cfg(chat_model: str, binary: pathlib.Path):
    return types.SimpleNamespace(
        chat_model_path=chat_model, chat_server_bin=str(binary), chat_port=8081,
    )


def _exe(tmp_path) -> pathlib.Path:
    b = tmp_path / "llama-server"
    b.write_text("#!/bin/sh\n")
    b.chmod(0o755)
    return b


def test_vision_model_advertises_image_input(tmp_path):
    binary = _exe(tmp_path)
    weights = _big_file(tmp_path / "vl.gguf")
    _big_file(tmp_path / "mmproj-F16.gguf")

    # The search root is the model's directory *and its parent*, so a shared
    # tmp tree can contain other tests' checkpoints. Assert on this one.
    found = U._llm_units(_cfg(str(weights), binary))
    unit = next(u for u in found if u.model_path == str(weights))
    assert unit.accepts == ("text", "image")
    assert unit.mmproj and unit.mmproj.endswith("mmproj-F16.gguf")
    d = unit.as_dict()
    assert d["accepts"] == ["text", "image"]
    assert "mmproj" in d


def test_text_model_advertises_text_only(tmp_path):
    binary = _exe(tmp_path)
    weights = _big_file(tmp_path / "plain.gguf")
    found = U._llm_units(_cfg(str(weights), binary))
    unit = next(u for u in found if u.model_path == str(weights))
    assert unit.accepts == ("text",)
    assert unit.mmproj is None


def test_projector_is_never_listed_as_a_model(tmp_path):
    """mmproj*.gguf is a projector, not a checkpoint."""
    binary = _exe(tmp_path)
    weights = _big_file(tmp_path / "vl.gguf")
    _big_file(tmp_path / "mmproj-F16.gguf")
    found = U._llm_units(_cfg(str(weights), binary))
    assert not any("mmproj" in u.model_label for u in found)


def test_no_executable_means_no_language_units(tmp_path):
    binary = tmp_path / "llama-server"          # not executable
    binary.write_text("#!/bin/sh\n")
    weights = _big_file(tmp_path / "vl.gguf")
    assert U._llm_units(_cfg(str(weights), binary)) == []


def test_configured_model_is_listed_first(tmp_path):
    binary = _exe(tmp_path)
    root = tmp_path / "models"
    root.mkdir()
    a = _big_file(root / "aaa.gguf", 1024)
    b = _big_file(root / "zzz.gguf", 1024)
    found = U._llm_units(_cfg(str(b), binary))
    assert [u.model_path for u in found][0] == str(b)
    assert str(a) in [u.model_path for u in found]


# ---------------------------------------------------------------- argv

def test_mmproj_reaches_the_server_argv(tmp_path):
    weights = tmp_path / "m.gguf"
    proj = tmp_path / "mmproj-F16.gguf"
    s = ManagedServer("/bin/true", str(weights), 8081, mmproj=str(proj))
    argv = s.server_argv()
    assert "--mmproj" in argv
    assert argv[argv.index("--mmproj") + 1] == str(proj)


def test_text_only_server_has_no_mmproj_flag(tmp_path):
    weights = tmp_path / "m.gguf"
    argv = ManagedServer("/bin/true", str(weights), 8081).server_argv()
    assert "--mmproj" not in argv


# ---------------------------------------------------------------- api

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from inferscope import api as A
    monkeypatch.setenv("INFERSCOPE_DATA", str(tmp_path / "data"))
    return TestClient(A.create_app())


def test_send_requires_text_or_an_image(client):
    r = client.post("/api/chat/send", json={"message": ""})
    assert r.status_code == 400
    assert "image" in r.json()["detail"].lower()


def test_send_with_no_server_is_a_conflict_not_a_crash(client):
    r = client.post("/api/chat/send", json={"message": "hi"})
    assert r.status_code == 409


def test_start_rejects_a_model_it_does_not_know(client):
    r = client.post("/api/chat/server/start", json={"model": "llm/not-a-real-model.gguf"})
    assert r.status_code == 400, r.text

# ---------------------------------------------------------------- asr arch

def test_hparams_read_from_the_engines_own_report(tmp_path):
    log = tmp_path / "engine-asr.log"
    log.write_text(
        "whisper_init_state: loading\n"
        "whisper_model_load: n_vocab       = 51864\n"
        "whisper_model_load: n_audio_ctx   = 1500\n"
        "whisper_model_load: n_audio_state = 512\n"
        "whisper_model_load: n_audio_head  = 8\n"
        "whisper_model_load: n_audio_layer = 6\n"
        "whisper_model_load: n_text_ctx    = 448\n"
        "whisper_model_load: n_text_state  = 512\n"
        "whisper_model_load: n_text_head   = 8\n"
        "whisper_model_load: n_text_layer  = 6\n"
        "whisper_model_load: n_mels        = 80\n")
    from inferscope import asr_arch
    hp = asr_arch.read_hparams(log)
    assert hp["n_vocab"] == 51864 and hp["n_audio_layer"] == 6


def test_no_log_means_no_architecture_rather_than_a_guess(tmp_path):
    from inferscope import asr_arch
    assert asr_arch.read_hparams(tmp_path / "nothing.log") == {}
    assert asr_arch.build_whisper_arch({}, "x") is None


def test_whisper_graph_has_both_stacks_and_real_totals(tmp_path):
    from inferscope import asr_arch
    hp = {"n_vocab": 51864, "n_audio_ctx": 1500, "n_audio_state": 512,
          "n_audio_head": 8, "n_audio_layer": 6, "n_text_ctx": 448,
          "n_text_state": 512, "n_text_head": 8, "n_text_layer": 6, "n_mels": 80}
    g = asr_arch.build_whisper_arch(hp, "ggml-base.en.bin")
    assert g["family"] == "asr"
    ids = {n["id"] for n in g["nodes"]}
    assert "encoder.block0" in ids and "decoder.block0" in ids
    assert "mel" in ids and "out" in ids
    # 12 blocks total, six a side
    assert sum(1 for i in ids if i.startswith("encoder.block")) == 6
    assert sum(1 for i in ids if i.startswith("decoder.block")) == 6
    assert g["params_total"] > 0
    # the weightless front end must not claim a share of the weights
    mel = next(n for n in g["nodes"] if n["id"] == "mel")
    assert mel["params"] == 0 and mel["param_share"] == 0
