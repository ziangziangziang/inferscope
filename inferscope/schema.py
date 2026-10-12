"""The common event/metric schema every adapter feeds and the frontend reads.

Two principles the whole product hangs on:

1. A value's *origin* is part of the value. `measured`, `derived`, `estimated`,
   `simulated` and `unavailable` are never collapsed into a bare number, so the UI
   can never present an architecture-only estimate as if it were a GPU reading.
2. Unknown is unknown, not zero. Optional fields stay ``None`` and a missing
   metric carries ``provenance=unavailable`` with a reason, rather than a 0 that
   reads as a real measurement.

This module is dependency-light at import time: pydantic is the only hard
requirement, so `inferscope check` runs on a machine with no GPU and no engines.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


def now_wall_ns() -> int:
    """Wall-clock ns, for session inspection and cross-process correlation."""
    return time.time_ns()


def now_mono_ns() -> int:
    """Monotonic ns, the only safe basis for durations/intervals."""
    return time.monotonic_ns()


class Provenance(str, Enum):
    """How a value came to be. Drives every label and colour in the UI."""

    MEASURED = "measured"      # read directly from hardware/engine telemetry
    DERIVED = "derived"        # computed from measured inputs (e.g. rate over a window)
    ESTIMATED = "estimated"    # model/config arithmetic, NOT observed at runtime
    SIMULATED = "simulated"    # synthetic demo data — never presented as live
    UNAVAILABLE = "unavailable"  # this platform/engine does not expose it

    @property
    def trustworthy(self) -> bool:
        return self in (Provenance.MEASURED, Provenance.DERIVED)


class Source(str, Enum):
    AMD_SMI = "amdsmi"
    ROCM_SMI = "rocm-smi"
    SYSFS = "sysfs"
    PROC = "proc"
    PSUTIL = "psutil"
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"
    DIFFUSERS = "diffusers"
    TORCH_PROFILER = "torch-profiler"
    ROCPROF = "rocprofv3"
    MODEL_CONFIG = "model_config"
    SYNTHETIC = "synthetic"
    DERIVED = "derived"


class EventKind(str, Enum):
    # lifecycle / work
    REQUEST = "request"
    PREFILL = "prefill"
    DECODE_STEP = "decode_step"
    TOKEN = "token"
    PIPELINE_STAGE = "pipeline_stage"
    DENOISE_STEP = "denoise_step"
    COMPONENT = "component"
    # device execution
    KERNEL = "kernel"
    MEMCPY = "memcpy"
    SYNC = "sync"
    ALLOC = "alloc"
    FREE = "free"
    # periodic telemetry
    GPU_SAMPLE = "gpu_sample"
    CPU_SAMPLE = "cpu_sample"
    RAM_SAMPLE = "ram_sample"
    # meta
    HEALTH = "health"


# Which kinds are periodic time-series samples (stored in the samples table).
SAMPLE_KINDS = {
    EventKind.GPU_SAMPLE,
    EventKind.CPU_SAMPLE,
    EventKind.RAM_SAMPLE,
}


class TensorMeta(BaseModel):
    """Tensor shape/dtype/size where known. Absent fields stay absent."""

    shape: Optional[list[int]] = None
    dtype: Optional[str] = None
    bytes: Optional[int] = None
    layout: Optional[str] = None
    device: Optional[str] = None
    provenance: Provenance = Provenance.ESTIMATED


class Event(BaseModel):
    """A single observation. Every adapter produces these; nothing else is stored.

    Timestamps are dual: ``ts_wall_ns`` for humans and cross-process alignment,
    ``ts_mono_ns`` for arithmetic. They come from different clocks and are NOT
    interchangeable — correlation across sources documents its uncertainty rather
    than assuming a shared origin (see ``clock_note``).
    """

    schema_version: int = SCHEMA_VERSION
    event_id: str
    session_id: str
    kind: EventKind
    source: Source
    provenance: Provenance

    ts_wall_ns: int
    ts_mono_ns: int

    engine: Optional[str] = None
    model_id: Optional[str] = None

    request_id: Optional[str] = None
    step_idx: Optional[int] = None
    token_idx: Optional[int] = None

    component: Optional[str] = None
    layer_idx: Optional[int] = None
    op_id: Optional[str] = None

    start_ns: Optional[int] = None
    end_ns: Optional[int] = None
    duration_us: Optional[float] = None
    unit: Optional[str] = None

    device: Optional[str] = None
    pid: Optional[int] = None

    bytes_moved: Optional[int] = None
    bytes_alloc: Optional[int] = None

    tensor: Optional[TensorMeta] = None

    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    native_ref: Optional[str] = None      # pointer into a Perfetto/rocprofv3 trace
    clock_note: Optional[str] = None      # how this clock relates to others
    reason: Optional[str] = None          # why a value is unavailable, if so

    # Free-form numeric/string payload for adapter-specific fields that do not
    # warrant a first-class column (kept JSON so the schema stays queryable).
    value: Optional[float] = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def model_dump_jsonable(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        return d


class Sample(BaseModel):
    """One periodic telemetry row. Wide numeric columns; None == not reported.

    Split from Event because the overview renders these as time series and wants
    stable numeric columns to plot, not a JSON blob per poll.
    """

    schema_version: int = SCHEMA_VERSION
    session_id: str
    ts_wall_ns: int
    ts_mono_ns: int
    source: Source
    provenance: Provenance

    # GPU
    gpu_util_pct: Optional[float] = None
    mem_used_pct: Optional[float] = None
    vram_used_bytes: Optional[int] = None
    vram_total_bytes: Optional[int] = None
    power_w: Optional[float] = None
    temp_c: Optional[float] = None
    sclk_mhz: Optional[float] = None
    mclk_mhz: Optional[float] = None
    encode_util_pct: Optional[float] = None
    # CPU / RAM
    cpu_util_pct: Optional[float] = None
    load1: Optional[float] = None
    load5: Optional[float] = None
    load15: Optional[float] = None
    ram_used_bytes: Optional[int] = None
    ram_total_bytes: Optional[int] = None
    proc_rss_bytes: Optional[int] = None
    # engine-derived
    active_requests: Optional[int] = None
    queue_depth: Optional[int] = None
    tokens_per_s: Optional[float] = None

    reason: Optional[str] = None          # why a field is unavailable, if so
    per_core: Optional[list[float]] = None
    extra: dict[str, Any] = Field(default_factory=dict)
