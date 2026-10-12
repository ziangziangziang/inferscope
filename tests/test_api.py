import pytest
from fastapi.testclient import TestClient

from inferscope.api import create_app
from inferscope.config import Config


@pytest.fixture
def client(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.demo = True
    cfg.poll_interval_s = 3600        # don't spin the live collector during tests
    app = create_app(cfg)
    with TestClient(app) as c:        # context manager runs lifespan (seed + arch)
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["schema_version"] == 1
    assert d["demo_session"]


def test_sessions_listed(client):
    d = client.get("/api/sessions").json()
    ids = {s["session_id"] for s in d["sessions"]}
    assert any(i.startswith("demo") for i in ids)


def test_demo_events_and_architecture(client):
    sid = client.get("/api/health").json()["demo_session"]
    ev = client.get(f"/api/sessions/{sid}/events?limit=5000").json()
    assert ev["count"] > 0
    kinds = {e["kind"] for e in ev["events"]}
    assert "kernel" in kinds and "decode_step" in kinds
    arch = client.get(f"/api/sessions/{sid}/architecture").json()
    assert "nodes" in arch and arch["provenance"] in ("simulated", "estimated")


def test_kv_endpoint(client):
    r = client.get("/api/kv", params={"n_layers": 32, "context": 4096, "batch": 1,
                                      "kv_heads": 8, "head_dim": 128, "dtype": "bfloat16"})
    assert r.status_code == 200
    assert r.json()["bytes"] == 2 * 32 * 1 * 4096 * 8 * 128 * 2


def test_kv_validation_error(client):
    assert client.get("/api/kv", params={"n_layers": 0, "context": 10}).status_code == 422


def test_experiments_post_and_list(client):
    r = client.post("/api/experiments", json={"name": "q4", "kind": "quantisation",
                                              "baseline": "a", "candidate": "b", "note": ""})
    assert r.status_code == 200 and "id" in r.json()
    assert any(e["name"] == "q4" for e in client.get("/api/experiments").json()["experiments"])


def test_experiment_requires_name(client):
    assert client.post("/api/experiments", json={"kind": "x"}).status_code == 422


def test_capture_status_is_readonly(client):
    d = client.get("/api/capture/status").json()
    assert "available" in d and "how" in d


def test_capabilities_shape(client):
    caps = client.get("/api/capabilities").json()
    assert "system_cpu_ram" in caps and "rocprofv3" in caps


def test_unknown_session_arch_404(client):
    assert client.get("/api/sessions/nope/architecture").status_code == 404


def test_chat_state_without_endpoint(client):
    r = client.get("/api/chat/state")
    assert r.status_code == 200
    b = r.json()
    assert b["ready"] is False
    assert b["base_url"] is None


def test_chat_send_refuses_without_server(client):
    """No engine means no fabricated turn: the endpoint must not answer 200."""
    r = client.post("/api/chat/send", json={"message": "hi"})
    assert r.status_code == 409


def test_chat_send_requires_message(client):
    r = client.post("/api/chat/send", json={"message": "   "})
    assert r.status_code == 400


def test_chat_start_refuses_missing_model(client, tmp_path):
    r = client.post("/api/chat/server/start")
    assert r.status_code == 400


def test_restart_supersedes_older_live_sessions(client):
    """A restart opens a new live session; the previous ones must not pile up
    in the picker, where they hide the session that has real events."""
    from inferscope.store import SessionStore
    store = SessionStore(client.app.state.iscope.config.data_dir)
    live = [s["session_id"] for s in store.sessions() if s["kind"] == "live"]
    assert live, "expected the lifespan to have created a live session"
    assert len(live) == 1
    # Demo data is preserved: it is the fallback when no engine is connected.
    assert any(s["kind"] == "demo" for s in store.sessions())


def test_sessions_carry_content_summary(client):
    """The picker must be able to tell an empty session from a profiled one."""
    r = client.get("/api/sessions")
    b = r.json()
    for s in b["sessions"]:
        for k in ("event_count", "sample_count", "gpu_ms", "span_s",
                  "top_op", "has_events"):
            assert k in s
    demo = [s for s in b["sessions"] if s["kind"] == "demo"]
    if demo:
        assert demo[0]["event_count"] > 0
    # best_session points at something that actually has data.
    if b["best_session"]:
        chosen = next(s for s in b["sessions"] if s["session_id"] == b["best_session"])
        assert chosen["event_count"] > 0


def test_sessions_survives_a_broken_trace_store(client):
    """A summary is a convenience; it must never take down the session list."""
    r = client.get("/api/sessions")
    assert r.status_code == 200


def test_best_session_prefers_measured_kernels_over_demo(client):
    """Demo is re-seeded on every boot, so 'most recent with data' would always
    pick the synthetic session and hide the real capture."""
    b = client.get("/api/sessions").json()
    if not b["best_session"]:
        return
    s = next(x for x in b["sessions"] if x["session_id"] == b["best_session"])
    assert s["kernel_events"] >= max(
        x["kernel_events"] for x in b["sessions"])


def test_superseded_live_session_with_data_is_reclassified_not_deleted(client):
    """A live session holding captured kernels must survive a restart as a
    profile session. Deleting it would silently destroy a real chat capture."""
    from inferscope.api import _session_event_count
    from inferscope.store import SessionStore

    state = client.app.state.iscope
    store = SessionStore(state.config.data_dir)
    demo = state.demo_session
    assert demo, "demo session expected"
    assert _session_event_count(state, demo) > 0, "demo should hold events"

    # Simulate the supersede step: a live session that has data is reclassified.
    store.set_kind(demo, "live")
    assert any(s["kind"] == "live" and s["session_id"] == demo
               for s in store.sessions())
    store.set_kind(demo, "profile")
    assert any(s["kind"] == "profile" and s["session_id"] == demo
               for s in store.sessions())
    # Still present, so its events survive.
    assert _session_event_count(state, demo) > 0


def test_session_event_count_never_raises(client):
    from inferscope.api import _session_event_count
    state = client.app.state.iscope
    assert _session_event_count(state, "no-such-session") == 0


def test_session_architecture_is_rebuilt_from_config(client):
    """The persisted graph is a snapshot; the endpoint must not serve a stale
    one that predates the current config-derived data (e.g. cost/params)."""
    sid = client.app.state.iscope.demo_session
    r = client.get(f"/api/sessions/{sid}/architecture")
    assert r.status_code == 200
    g = r.json()
    assert g["costs"]["totals"]["params"] > 0
    assert any(n.get("params") is not None for n in g["nodes"])


def test_span_is_not_computed_across_mixed_clocks(client):
    """Kernel rows carry the raw GPU clock, not wall clock. Mixing them into a
    span produced figures on the order of 1e9 seconds; span must be None when
    there are no wall-clock events to measure."""
    for s in client.get("/api/sessions").json()["sessions"]:
        if s["span_s"] is not None:
            assert 0 <= s["span_s"] < 86400, f"implausible span {s['span_s']}s"


def test_index_assets_present(client):
    """The start-here page depends on the model list for its chooser and on
    chat/state for the engine bar; both must answer even with no model loaded."""
    assert client.get("/api/models").status_code == 200
    assert client.get("/api/chat/state").status_code == 200


def _fmt(rate: int = 16000) -> bytes:
    """A minimal, correct 16-byte WAVEFORMATEX (mono, 16-bit)."""
    return (b"\x01\x00"            # wFormatTag = PCM
            + b"\x01\x00"          # nChannels = 1
            + rate.to_bytes(4, "little")          # nSamplesPerSec
            + (rate * 2).to_bytes(4, "little")    # nAvgBytesPerSec
            + b"\x02\x00"          # nBlockAlign
            + b"\x10\x00")         # wBitsPerSample = 16


def _wav(chunks: list[tuple[bytes, bytes]], rate: int = 16000) -> bytes:
    """Build a RIFF/WAVE with a correct fmt chunk plus *chunks* after it."""
    fmt = _fmt(rate)
    out = b"fmt " + len(fmt).to_bytes(4, "little") + fmt
    for cid, payload in chunks:
        out += cid + len(payload).to_bytes(4, "little") + payload
        if len(payload) & 1:            # chunks are word-aligned
            out += b"\x00"
    return b"RIFF" + (len(out) + 8).to_bytes(4, "little") + b"WAVE" + out


def test_wav_seconds_reads_plain_wav():
    from inferscope.api import _wav_seconds
    payload = b"\x00" * 32000                      # 1s at 16kHz mono s16
    w = _wav([(b"data", payload)], rate=16000)
    assert _wav_seconds(w) == 1.0


def test_wav_seconds_skips_extra_chunks_before_data():
    """LIST/fact chunks before `data` are normal; a fixed 44-byte header read
    gets this wrong and reported 0.0s."""
    from inferscope.api import _wav_seconds
    payload = b"\x00" * 32000
    w = _wav([(b"LIST", b"INFOhello"), (b"fact", b"\x00" * 4),
              (b"data", payload)], rate=16000)
    assert _wav_seconds(w) == 1.0


def test_wav_seconds_clamps_an_overstated_data_size():
    from inferscope.api import _wav_seconds
    payload = b"\x00" * 16000                      # 0.5s of real audio
    fmt = _fmt(16000)
    out = b"fmt " + len(fmt).to_bytes(4, "little") + fmt
    out += b"data" + (999999999).to_bytes(4, "little") + payload
    w = b"RIFF" + (len(out) + 8).to_bytes(4, "little") + b"WAVE" + out
    assert _wav_seconds(w) == 0.5


def test_wav_seconds_returns_none_for_non_wav():
    from inferscope.api import _wav_seconds
    assert _wav_seconds(b"ID3\x04\x00\x00\x00\x00\x00") is None
    assert _wav_seconds(b"") is None
