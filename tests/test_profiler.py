from inferscope.profiler import classify, parse_kernel_csv, validate
from inferscope.schema import EventKind


def test_classify():
    assert classify("gemm_q4_wmma_kernel") == "gemm"
    assert classify("attn_fwd") == "attn_full"
    assert classify("chunk_scaled_dot_kkt_fwd_kernel") == "attn_linear"
    assert classify("vectorized_elementwise_kernel") == "elementwise"
    assert classify("some_random_thing") == "other"


def test_parse_kernel_csv_durations(kernel_csv):
    ev = parse_kernel_csv(kernel_csv, "s", "model")
    assert len(ev) == 5
    gemm = next(e for e in ev if "gemm" in e.component)
    assert gemm.duration_us == 450.0        # 1000450000-1000000000 ns = 450us
    assert gemm.provenance.value == "measured"
    assert gemm.extra["op_class"] == "gemm"
    # the copy row is a memcpy, not a kernel
    cp = next(e for e in ev if e.kind == EventKind.MEMCPY)
    assert cp.duration_us == 30.0


def test_parse_missing_file_is_empty(tmp_path):
    assert parse_kernel_csv(str(tmp_path / "nope.csv"), "s") == []


def test_parse_bad_rows_skipped(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Name,Start_Timestamp,End_Timestamp\n"
                 "gemm_a,notanumber,alsobad\n"
                 "gemm_b,1000,2000\n")
    ev = parse_kernel_csv(str(p), "s")
    assert len(ev) == 1 and ev[0].duration_us == 1.0


def test_validate_reports_unavailable_without_rocprof():
    ok, msg = validate({})
    assert not ok and "not available" in msg
