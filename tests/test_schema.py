from inferscope.schema import (Event, EventKind, Provenance, Sample, Source,
                               now_mono_ns, now_wall_ns)
import pytest
from pydantic import ValidationError


def test_event_requires_core_fields():
    with pytest.raises(ValidationError):
        Event(session_id="s", kind=EventKind.KERNEL)  # missing event_id/source/provenance/ts


def test_event_roundtrip_and_provenance():
    e = Event(event_id="a", session_id="s", kind=EventKind.KERNEL,
              source=Source.ROCPROF, provenance=Provenance.MEASURED,
              ts_wall_ns=now_wall_ns(), ts_mono_ns=now_mono_ns(), duration_us=12.5)
    d = e.model_dump(mode="json")
    assert d["provenance"] == "measured"
    assert d["duration_us"] == 12.5
    # optional fields are None, not zero
    assert e.request_id is None and e.bytes_moved is None


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Event(event_id="a", session_id="s", kind=EventKind.KERNEL, source=Source.DERIVED,
              provenance=Provenance.DERIVED, ts_wall_ns=1, ts_mono_ns=1, confidence=1.5)


def test_sample_unknown_is_none_not_zero():
    s = Sample(session_id="s", ts_wall_ns=1, ts_mono_ns=1,
               source=Source.AMD_SMI, provenance=Provenance.UNAVAILABLE)
    assert s.gpu_util_pct is None
    assert s.vram_used_bytes is None


def test_timestamps_are_monotonic_nondecreasing():
    a = now_mono_ns()
    b = now_mono_ns()
    assert b >= a
    assert now_wall_ns() > 1_600_000_000_000_000_000  # sane epoch-scale ns
