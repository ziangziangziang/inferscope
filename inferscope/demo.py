"""A clearly-labelled synthetic session so the UI is fully explorable with no
GPU, no engine, and no profiling. Every record carries provenance=simulated and
source=synthetic; the frontend must never render these as live readings.
"""
from __future__ import annotations

import hashlib
import random
import time

from .schema import (Event, EventKind, Provenance, Sample, Source, TensorMeta,
                     now_mono_ns, now_wall_ns)

SIM = (Source.SYNTHETIC, Provenance.SIMULATED)


def _eid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def build_demo_events(session_id: str, model_id: str = "demo/TextLLM-7B") -> list[Event]:
    rng = random.Random(1234)
    ev: list[Event] = []
    base_wall = now_wall_ns()
    base_mono = now_mono_ns()
    n_steps = 24
    # two concurrent requests
    for r in range(2):
        req = f"req-{r}"
        arrival = base_wall + r * 40_000_000
        ev.append(Event(
            event_id=_eid(session_id, req, "req"), session_id=session_id,
            kind=EventKind.REQUEST, source=SIM[0], provenance=SIM[1],
            ts_wall_ns=arrival, ts_mono_ns=base_mono + r * 40_000_000,
            engine="demo", model_id=model_id, request_id=req,
            extra={"n_prompt": 512, "n_completion": n_steps},
        ))
        # prefill
        p_start = arrival + 5_000_000
        p_dur = rng.uniform(180, 260) * 1000  # us
        ev.append(Event(
            event_id=_eid(session_id, req, "prefill"), session_id=session_id,
            kind=EventKind.PREFILL, source=SIM[0], provenance=SIM[1],
            ts_wall_ns=p_start, ts_mono_ns=base_mono + (p_start - base_wall),
            request_id=req, duration_us=p_dur, unit="us",
            tensor=TensorMeta(shape=[1, 512, 4096], dtype="bfloat16",
                              provenance=Provenance.SIMULATED),
        ))
        for step in range(n_steps):
            ts = p_start + int(p_dur * 1000) + step * 28_000_000
            mono = base_mono + (ts - base_wall)
            ev.append(Event(
                event_id=_eid(session_id, req, step, "step"), session_id=session_id,
                kind=EventKind.DECODE_STEP, source=SIM[0], provenance=SIM[1],
                ts_wall_ns=ts, ts_mono_ns=mono, request_id=req, step_idx=step,
                duration_us=rng.uniform(24, 33) * 1000, unit="us",
                extra={"n_running": 2 - (step > 12 and r), "kv_used_frac": 0.1 + 0.02 * step},
            ))
            # a few kernels per step
            for k, (name, op) in enumerate([
                ("gemm_q4_wmma", "gemm"), ("attn_fwd", "attn_full"),
                ("rmsnorm", "elementwise"), ("silu_mul", "elementwise"),
            ]):
                kdur = rng.uniform(3, 60) * 1000 if op == "gemm" else rng.uniform(1, 8) * 1000
                ev.append(Event(
                    event_id=_eid(session_id, req, step, k, name), session_id=session_id,
                    kind=EventKind.KERNEL, source=SIM[0], provenance=SIM[1],
                    ts_wall_ns=ts + k * 3_000_000, ts_mono_ns=mono + k * 3_000_000,
                    request_id=req, step_idx=step, component=name, op_id=name,
                    duration_us=kdur, unit="us", device="gfx1100",
                    extra={"op_class": op},
                ))
            # one H2D memcpy occasionally
            if step % 6 == 0:
                ev.append(Event(
                    event_id=_eid(session_id, req, step, "memcpy"), session_id=session_id,
                    kind=EventKind.MEMCPY, source=SIM[0], provenance=SIM[1],
                    ts_wall_ns=ts, ts_mono_ns=mono, request_id=req, step_idx=step,
                    duration_us=rng.uniform(20, 120) * 1000, unit="us",
                    bytes_moved=rng.choice([1 << 20, 4 << 20, 16 << 20]),
                    extra={"direction": "H2D"},
                ))
    return ev


def build_demo_samples(session_id: str, n: int = 120) -> list[Sample]:
    rng = random.Random(99)
    out: list[Sample] = []
    base_wall = now_wall_ns()
    base_mono = now_mono_ns()
    for i in range(n):
        wall = base_wall - (n - i) * 1_000_000_000
        mono = base_mono - (n - i) * 1_000_000_000
        load = 0.5 + 0.4 * (1 if i > n // 3 else 0)
        out.append(Sample(
            session_id=session_id, ts_wall_ns=wall, ts_mono_ns=mono,
            source=SIM[0], provenance=SIM[1],
            gpu_util_pct=max(0.0, min(100.0, rng.gauss(72 * load, 9))),
            mem_used_pct=40 + 30 * load + rng.uniform(-3, 3),
            vram_used_bytes=int((20 + 14 * load) * 2**30),
            vram_total_bytes=48 * 2**30,
            power_w=rng.gauss(180 * load, 12),
            temp_c=rng.gauss(58 + 10 * load, 2),
            sclk_mhz=rng.gauss(1900, 40),
            cpu_util_pct=rng.gauss(35 * load, 6),
            load1=2.0 * load, ram_used_bytes=int((24 + 8 * load) * 2**30),
            ram_total_bytes=128 * 2**30, proc_rss_bytes=int(18 * 2**30),
            active_requests=int(2 if i > n // 3 else 1),
            queue_depth=max(0, int(rng.gauss(1, 1))),
            tokens_per_s=max(0.0, rng.gauss(120 * load, 15)),
        ))
    return out


def seed_demo(sessions, trace, model_id: str = "demo/TextLLM-7B") -> str:
    """Write a full synthetic session into the stores. Returns the session_id.

    Uses a stable id so restarting the server replaces the demo session instead
    of accumulating a new timestamped one each time (which cluttered the picker).
    """
    sid = "demo"
    trace.reset_session(sid)
    sessions.create_session(
        sid, kind="demo", engine="demo", model_id=model_id,
        capability={}, source="synthetic",
        notes="Synthetic data for UI development. Not live readings.")
    trace.append_events(build_demo_events(sid, model_id))
    trace.append_samples(build_demo_samples(sid))
    from .arch import build_graph
    graph = build_graph(model_id, {})  # estimated text template
    # force simulated provenance across the demo graph
    graph["provenance"] = "simulated"
    graph["source"] = "synthetic"
    trace.set_architecture(sid, model_id, graph, "simulated", "synthetic")
    return sid
