"""Bounded rocprofv3 capture -> InferScope events.

Profiling is opt-in and time-boxed, and never attaches to a foreign process
(rocprofv3 cannot ptrace a running one here anyway). We validate the CLI
arguments against the *installed* version before running, execute the target
under a hard ``timeout``, then parse the CSV trace into kernel/memcpy Events with
provenance=measured. Perfetto-compatible output is emitted when the installed
rocprofv3 supports it.

A kernel event is NOT attributed to a request or token unless the capture also
carried that correlation — we never invent one.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import os
import subprocess
from typing import Optional

from .schema import Event, EventKind, Provenance, Source, now_wall_ns


def _eid(*p) -> str:
    return "rp-" + hashlib.sha1("|".join(map(str, p)).encode()).hexdigest()[:16]


def classify(name: str) -> str:
    low = name.lower()
    # GEMMs: llama.cpp/GGML uses mul_mat*/mmv/mmq; also rocBLAS/Tensile (Cijk) + hipBLAS.
    if any(k in low for k in ("mul_mat", "mul_mat_vec", "mmv", "mmq", "matmul",
                              "gemm", "wmma", "cijk", "sgemm", "hgemm", "hipblas", "sgem")):
        return "gemm"
    if any(k in low for k in ("flash_attn", "attn_fwd", "attention", "sdpa", "flash", "fmha")):
        return "attn_full"
    if any(k in low for k in ("chunk_", "gdn", "delta", "recurrent", "mamba", "ssm")):
        return "attn_linear"
    if any(k in low for k in ("rms_norm", "layernorm", "layer_norm", "elementwise",
                              "silu", "softmax", "quantize", "rope", "set_rows", "cpy",
                              "cast", "gelu", "swiglu", "scale", "add_", "bin_", "unary")):
        return "elementwise"
    return "other"


def is_copy_kernel(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ("memcpy", "memorycopy", "copybuffer", "agent_ex",
                                  "sdma", "fillbuffer"))


def validate(caps: dict) -> tuple[bool, str]:
    rp = caps.get("rocprofv3")
    if not (rp and rp.available):
        return False, "rocprofv3 not available on this host"
    if not rp.supports.get("kernel_trace"):
        return False, "installed rocprofv3 lacks --kernel-trace"
    return True, rp.detail or "rocprofv3 available"


def parse_kernel_csv(path: str, session_id: str, model_id: str = "") -> list[Event]:
    """Parse a rocprofv3 kernel-trace CSV. Header-name based, so column order
    changes across versions don't break it. Missing timestamps -> row skipped."""
    events: list[Event] = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        # Match columns by substring on a normalised header, so the many
        # rocprofv3 layouts (Start_Timestamp / "Start Timestamp" / start) all work.
        def find(*needles):
            for c in fieldnames:
                norm = c.lower().replace("_", " ")
                if any(n in norm for n in needles):
                    return c
            return None
        name_c = find("name")
        start_c = find("start", "begin")
        end_c = find("end", "stop", "finish")
        if not (name_c and start_c and end_c):
            return events
        with open(path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                try:
                    start = int(float(row[start_c]))
                    end = int(float(row[end_c]))
                except (ValueError, KeyError, TypeError):
                    continue
                nm = row[name_c]
                dur_us = max(0.0, (end - start) / 1000.0)
                copy = is_copy_kernel(nm)
                events.append(Event(
                    event_id=_eid(session_id, path, i), session_id=session_id,
                    kind=EventKind.MEMCPY if copy else EventKind.KERNEL,
                    source=Source.ROCPROF, provenance=Provenance.MEASURED,
                    ts_wall_ns=start, ts_mono_ns=start, engine="rocprofv3", model_id=model_id,
                    component=nm[:120], op_id=nm[:60],
                    start_ns=start, end_ns=end, duration_us=dur_us, unit="us",
                    extra={"op_class": "memcpy" if copy else classify(nm),
                           "native_ref": f"{os.path.basename(path)}#{i}"},
                ))
    except Exception:  # noqa: BLE001 — a malformed trace must not crash collection
        return events
    return events


def run_capture_parse_only(out_dir: str, session_id: str, caps: dict,
                           model_id: str = "") -> dict:
    """Parse an existing rocprofv3 output directory into events.

    Used when a server ran *under* the profiler (wrap mode): the trace already
    exists on disk and only needs ingesting, so no new capture is started.
    """
    ok, msg = validate(caps)
    if not ok:
        return {"ok": False, "error": msg}
    files = sorted(glob.glob(os.path.join(out_dir, "*kernel_trace.csv"))) or \
        sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    if not files:
        return {"ok": False, "error": f"no trace files under {out_dir}"}
    events: list[Event] = []
    for fp in files:
        events.extend(parse_kernel_csv(fp, session_id, model_id))
    return {"ok": True, "out_dir": out_dir, "trace_files": files,
            "events": len(events), "_events": events}


def run_capture(cmd: list[str], session_id: str, seconds: int, out_dir: str,
                caps: dict, perfetto: bool = False,
                model_id: str = "") -> dict:
    """Run a bounded capture of an explicit command. Returns a summary dict.

    The caller must have already confirmed intent (CLI --yes / API opt-in); this
    function enforces the time bound and refuses if rocprofv3 is unsupported.
    """
    ok, msg = validate(caps)
    if not ok:
        return {"ok": False, "error": msg}
    os.makedirs(out_dir, exist_ok=True)
    # rocprofv3 needs an explicit output base (-o); without it the trace files
    # land under a default name/location we would not find.
    fmt_name = "pftrace" if (perfetto and caps["rocprofv3"].supports.get("perfetto_output")) else "csv"
    base = os.path.join(out_dir, "trace")
    # Bound via the outer subprocess timeout only. Wrapping the target in
    # `timeout` makes rocprofv3 trace the wrapper process, not the child, so no
    # kernel dispatches are recorded. stdin is closed so interactive targets exit.
    full = ["rocprofv3", "--kernel-trace", "-d", out_dir, "-o", base,
            "-f", fmt_name, "--", *cmd]
    try:
        proc = subprocess.run(full, capture_output=True, text=True,
                              timeout=max(1, min(seconds, 60)) + 20,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "capture exceeded hard timeout", "out_dir": out_dir}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    files = sorted(glob.glob(os.path.join(out_dir, "*kernel_trace.csv"))) or \
        sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    events: list[Event] = []
    for fp in files:
        events.extend(parse_kernel_csv(fp, session_id, model_id))
    return {
        "ok": True, "rc": proc.returncode, "out_dir": out_dir,
        "trace_files": files, "events": len(events),
        "perfetto": bool(perfetto and any(f.endswith((".pftrace", ".pp")) for f in
                                          glob.glob(out_dir + "/*"))),
        "_events": events,
    }
