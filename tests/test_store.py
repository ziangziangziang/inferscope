from inferscope.store import SessionStore, TraceStore
from inferscope.schema import Event, EventKind, Provenance, Sample, Source


def _ev(sid, i):
    return Event(event_id=f"e{i}", session_id=sid, kind=EventKind.KERNEL,
                 source=Source.ROCPROF, provenance=Provenance.MEASURED,
                 ts_wall_ns=1000 + i, ts_mono_ns=i, duration_us=10 + i,
                 component="gemm_x", extra={"op_class": "gemm"})


def _smp(sid, i):
    return Sample(session_id=sid, ts_wall_ns=2000 + i, ts_mono_ns=i,
                  source=Source.AMD_SMI, provenance=Provenance.MEASURED,
                  vram_used_bytes=1000 + i, temp_c=40 + i)


def test_trace_append_and_query(tmp_data):
    t = TraceStore(tmp_data, read_only=False)
    assert t.append_events([_ev("s1", i) for i in range(5)]) == 5
    assert t.append_samples([_smp("s1", i) for i in range(3)]) == 3
    rows = t.query("SELECT count(*) n FROM pq_events WHERE session_id='s1'")
    assert rows[0]["n"] == 5
    # extra dict survived as JSON
    r0 = t.query("SELECT extra_json FROM pq_events WHERE event_id='e0'")[0]
    assert "gemm" in r0["extra_json"]


def test_reader_sees_data_written_after_open(tmp_data):
    w = TraceStore(tmp_data, read_only=False)
    r = TraceStore(tmp_data, read_only=True)
    # reader opened before any data -> empty
    assert r.query("SELECT count(*) n FROM pq_events")[0]["n"] == 0
    w.append_events([_ev("s2", i) for i in range(4)])
    # after refresh the reader sees the newly written parquet
    r.refresh()
    assert r.query("SELECT count(*) n FROM pq_events WHERE session_id='s2'")[0]["n"] == 4


def test_reset_session_clears(tmp_data):
    t = TraceStore(tmp_data, read_only=False)
    t.append_events([_ev("s3", i) for i in range(3)])
    t.reset_session("s3")
    assert t.query("SELECT count(*) n FROM pq_events WHERE session_id='s3'")[0]["n"] == 0


def test_architecture_roundtrip(tmp_data):
    t = TraceStore(tmp_data, read_only=False)
    t.set_architecture("s4", "model-x", {"nodes": [], "edges": []}, "derived", "model_config")
    row = t.query("SELECT graph_json, provenance FROM pq_architecture WHERE session_id='s4'")[0]
    assert row["provenance"] == "derived" and "nodes" in row["graph_json"]


def test_session_registry(tmp_data):
    s = SessionStore(tmp_data)
    s.create_session("a", "live", model_id="m")
    s.create_session("b", "demo", model_id="m")
    assert {x["session_id"] for x in s.sessions()} == {"a", "b"}
    s.end_session("a")
    assert s.session("a")["ended_ns"] is not None
    eid = s.add_experiment("exp", "quantisation", "a", "b", {"x": 1}, {"status": "recorded"})
    assert s.experiments()[0]["name"] == "exp"
