"""FastAPI backend: REST + SSE, a bounded live collector, and static frontend.

Design notes:
* The live collector buffers recent samples in memory (fast SSE + overview) and
  flushes to Parquet in batches, so a long session does not rewrite the whole
  file every poll.
* Every adapter reports its own health; the top-level /api/health aggregates them
  without pretending a missing adapter is fine.
* The app binds loopback by default and serves only read APIs — no shell, no
  arbitrary paths, no engine control.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import asr_arch, chat, profiler, tunnel
from .enginectl import APP_ROOT
from .arch import build_graph, list_models, load_config
from .capability import probe_all
from .collectors import GpuCollector, SystemCollector
from .config import Config
from .demo import seed_demo
from .engines import LlamaCppAdapter, VLLMAdapter
from .enginectl import EngineManager, build_engine_specs
from .units import discover, mmproj_for
from .kv import KVInputs, kv_cache_bytes
from .schema import Event, EventKind, Provenance, Sample, Source
from .store import SessionStore, TraceStore

log = logging.getLogger("inferscope.api")

LIVE_BUFFER = 4000


class AppState:
    def __init__(self, config: Config):
        self.config = config
        self.sessions = SessionStore(config.data_dir)
        self.trace = TraceStore(config.data_dir, read_only=False)
        self.reader = TraceStore(config.data_dir, read_only=True)
        self.caps = probe_all(config)
        self.system = SystemCollector()
        self.gpu = GpuCollector(self.caps)
        self.engines = {
            "llamacpp": LlamaCppAdapter(config.llamacpp_metrics_url),
            "vllm": VLLMAdapter(config.vllm_metrics_url),
        }
        self.live_session: Optional[str] = None
        self.demo_session: Optional[str] = None
        self.samples: dict[str, deque] = {}
        self.events: dict[str, deque] = {}
        self.adapter_health: dict[str, dict] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._flush_count = 0
        self._pending_samples: list[Sample] = []
        self._pending_events: list[Event] = []
        self.chat_server = None      # ManagedServer, created lazily
        self.chat_lock = asyncio.Lock()
        # Load/offload control for the non-language engines (speech, image,
        # video). One heavy engine at a time on one GPU.
        self.engine_ctl = EngineManager(build_engine_specs(config))
        self.engine_lock = asyncio.Lock()
        # A redeploy restarts this process but not the engines it started, and
        # the manager's memory of them went with it. Clear the ones we can
        # prove are ours before anything tries to claim their ports.
        try:
            self.engine_ctl.sweep_orphans(str(APP_ROOT))
        except Exception:                                       # noqa: BLE001
            logging.getLogger("inferscope").warning("orphan sweep failed",
                                                    exc_info=True)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, msg: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # slow consumer: drop, never block the collector

    def remember(self, sample: Sample) -> None:
        dq = self.samples.setdefault(sample.session_id, deque(maxlen=LIVE_BUFFER))
        dq.append(sample.model_dump(mode="json"))

    def remember_event(self, ev: Event) -> None:
        dq = self.events.setdefault(ev.session_id, deque(maxlen=LIVE_BUFFER))
        dq.append(ev.model_dump(mode="json"))

    async def collect_once(self) -> None:
        sid = self.live_session
        if not sid:
            return
        merged: dict[str, Any] = {}
        for coll in (self.system, self.gpu):
            try:
                s = await asyncio.to_thread(coll.poll, sid)
            except Exception as e:  # noqa: BLE001
                self.adapter_health[coll.__class__.__name__] = {
                    "available": False, "detail": str(e), "state": "error"}
                continue
            if s is None:
                continue
            # merge system + gpu into one row per poll for the overview
            for f in ("gpu_util_pct", "mem_used_pct", "vram_used_bytes",
                      "vram_total_bytes", "power_w", "temp_c", "sclk_mhz",
                      "mclk_mhz", "encode_util_pct", "reason"):
                if getattr(s, f) is not None:
                    merged[f] = getattr(s, f)
            for f in ("cpu_util_pct", "load1", "load5", "load15", "ram_used_bytes",
                      "ram_total_bytes", "proc_rss_bytes", "per_core"):
                if getattr(s, f) is not None:
                    merged[f] = getattr(s, f)
            self._pending_samples.append(s)
        # engine scrapes
        for name, ad in self.engines.items():
            res = await asyncio.to_thread(ad.scrape)
            self.adapter_health[name] = res.as_dict()
            if res.available:
                smp = ad.to_sample(res, sid)
                if smp:
                    for f in ("active_requests", "queue_depth", "tokens_per_s"):
                        if getattr(smp, f) is not None:
                            merged[f] = getattr(smp, f)
                    merged.setdefault("engine_metrics", {})
                    merged["engine_metrics"][name] = res.metrics
                    self._pending_samples.append(smp)
        if merged:
            base = self.samples.get(sid, deque())
            row = {"session_id": sid, "ts_wall_ns": int(time.time_ns()),
                   "ts_mono_ns": time.monotonic_ns(), "provenance": "measured",
                   "source": "collector", **merged}
            self.samples.setdefault(sid, deque(maxlen=LIVE_BUFFER)).append(row)
            self.publish({"type": "sample", "session_id": sid, "data": row})
        # periodic flush to parquet (bounded write amplification)
        self._flush_count += 1
        if self._flush_count >= 30:
            self._flush_count = 0
            ps, self._pending_samples = self._pending_samples, []
            if ps:
                await asyncio.to_thread(self.trace.append_samples, ps)


async def _collector_loop(state: AppState):
    interval = state.config.poll_interval_s
    while True:
        try:
            await state.collect_once()
        except Exception as e:  # noqa: BLE001 — never let one poll kill the loop
            state.adapter_health["collector"] = {"available": False,
                                                 "detail": str(e), "state": "error"}
        await asyncio.sleep(interval)


def _session_event_count(state, sid: str) -> int:
    """How many events a session holds. Never raises — used to decide deletion."""
    try:
        state.reader.refresh()
        return int(state.reader.query(
            "SELECT count(*) n FROM pq_events WHERE session_id=?", [sid])[0]["n"])
    except Exception:  # noqa: BLE001
        return 0


def _multipart(fields: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body. Returns (body, content_type)."""
    boundary = "----inferscope" + hashlib.sha1(os.urandom(16)).hexdigest()[:16]
    out = bytearray()
    for key, val in fields.items():
        out += f"--{boundary}\r\n".encode()
        if isinstance(val, tuple):
            filename, blob, ctype = val
            out += (f'Content-Disposition: form-data; name="{key}"; '
                    f'filename="{filename}"\r\n').encode()
            out += f"Content-Type: {ctype}\r\n\r\n".encode()
            out += blob + b"\r\n"
        else:
            out += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            out += str(val).encode() + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def raw_rate(audio: bytes) -> str:
    """Sample rate out of a WAV header, without parsing the whole file."""
    if len(audio) >= 28 and audio[:4] == b"RIFF":
        try:
            return str(int.from_bytes(audio[24:28], "little"))
        except Exception:                                   # noqa: BLE001
            pass
    return ""


def _http_post_raw(url: str, body: bytes, content_type: str, timeout: float):
    """POST bytes and return (status, raw-bytes, content-type)."""
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read(), r.headers.get("content-type", "")


def _http_post(url: str, body: bytes, content_type: str, timeout: float):
    """POST bytes and return (status, text, content-type)."""
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace"), \
            r.headers.get("content-type", "")


def _wav_seconds(data: bytes) -> Optional[float]:
    """Duration of a RIFF/WAVE file in seconds, or None if it is not WAV.

    Walks the chunk list rather than assuming a 44-byte header: real files
    carry LIST/fact chunks before ``data``, and a header-size assumption reads
    the wrong bytes and reports 0.0s for every clip.
    """
    import struct
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos = 12
    byte_rate: Optional[int] = None
    data_size = 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body = pos + 8
        if cid == b"fmt " and size >= 16 and body + 12 <= len(data):
            # WAVEFORMATEX: wFormatTag(0) nChannels(2) nSamplesPerSec(4)
            # nAvgBytesPerSec(8) nBlockAlign(12) wBitsPerSample(14). The byte
            # rate is therefore at body+8.
            (byte_rate,) = struct.unpack_from("<I", data, body + 8)
        elif cid == b"data":
            # trust the bytes we actually have: some encoders write a padded or
            # streaming size that overruns the file
            data_size = min(size, len(data) - body)
        pos = body + size + (size & 1)      # chunks are word-aligned
    if not byte_rate or data_size <= 0:
        return None
    return round(data_size / byte_rate, 2)


def create_app(config: Optional[Config] = None) -> FastAPI:
    cfg = config or Config.from_env()
    state = AppState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Demo session always available, even with no GPU.
        if cfg.demo:
            state.demo_session = await asyncio.to_thread(
                seed_demo, state.sessions, state.trace)
        # Live session if any collector works.
        if state.system.available or state.gpu.backend != "none":
            sid = "live-" + time.strftime("%Y%m%d-%H%M%S")
            await asyncio.to_thread(state.sessions.create_session, sid, "live",
                                    "", "", {},
                                    {k: c.as_dict() for k, c in state.caps.items()},
                                    "live", "")
            state.live_session = sid
            # Architecture is a property of the model. With no engine connected we
            # do not know what is *running*, so attach the first checkpoint found
            # on disk as the host's available architecture — clearly labelled as
            # config-derived, not as "the live model".
            found = await asyncio.to_thread(list_models, cfg.model_dirs)
            if found:
                g = build_graph(found[0]["model_id"], load_config(found[0]["path"]))
                await asyncio.to_thread(state.trace.set_architecture, sid,
                                        found[0]["model_id"], g, "derived", "model_config")
            else:
                g = build_graph("unknown", {})
                await asyncio.to_thread(state.trace.set_architecture, sid,
                                        "unknown", g, "estimated", "template")
        task = asyncio.create_task(_collector_loop(state))
        # A restart opens a new live session, which piles up in the picker. But a
        # live session may hold captured kernels from a chat run — those must
        # never be discarded. A superseded session that has data is *reclassified*
        # as a profile session instead; only empty ones are deleted. Pruning is
        # per session so one bad file cannot leave a half-deleted orphan that
        # later poisons the Parquet catalog.
        stale = await asyncio.to_thread(
            state.sessions.superseded_sessions, "live", state.live_session or "")
        for old in stale:
            if await asyncio.to_thread(_session_event_count, state, old) > 0:
                await asyncio.to_thread(state.sessions.set_kind, old, "profile")
                continue
            try:
                await asyncio.to_thread(state.trace.reset_session, old)
            except Exception:  # noqa: BLE001
                log.warning("prune: could not clear trace data for %s", old, exc_info=True)
            try:
                await asyncio.to_thread(state.sessions.delete_session, old)
            except Exception:  # noqa: BLE001
                log.warning("prune: could not delete registry row for %s", old, exc_info=True)

        for old in state.sessions.prune_sessions(cfg.retention_sessions):
            try:
                await asyncio.to_thread(state.trace.reset_session, old)
            except Exception:  # noqa: BLE001
                log.warning("prune: could not clear trace data for %s", old, exc_info=True)
            try:
                await asyncio.to_thread(state.sessions.delete_session, old)
            except Exception:  # noqa: BLE001
                log.warning("prune: could not delete registry row for %s", old, exc_info=True)
        try:
            yield
        finally:
            task.cancel()
            if state._pending_samples:
                state.trace.append_samples(state._pending_samples)
            # Never leave an orphaned llama-server holding the GPU.
            try:
                await chat.stop_managed(state.chat_server)
            except Exception:  # noqa: BLE001
                log.warning("chat: managed server did not stop cleanly", exc_info=True)
            state.gpu.close()

    app = FastAPI(title="InferScope", version=__version__, lifespan=lifespan)
    app.state.iscope = state

    # ---------------- access / tunnel ----------------
    @app.get("/api/access")
    def access():
        return tunnel.status()

    # ---------------- health / caps ----------------
    @app.get("/api/health")
    def health():
        adapters = dict(state.adapter_health)
        caps = {k: c.as_dict() for k, c in state.caps.items()}
        return {
            "status": "ok",
            "version": __version__,
            "schema_version": 1,
            "bind": cfg.bind_url,
            "public_url": tunnel.current_url(),
            "live_session": state.live_session,
            "demo_session": state.demo_session,
            "collectors": {
                "system": {"available": state.system.available},
                "gpu": {"backend": state.gpu.backend,
                        "available": state.gpu.backend != "none"},
            },
            "adapters": adapters,
            "capabilities": caps,
        }

    @app.get("/api/capabilities")
    def capabilities():
        return {k: c.as_dict() for k, c in state.caps.items()}

    # ---------------- sessions ----------------
    @app.get("/api/sessions")
    def sessions():
        """Sessions plus what is actually *in* each one.

        The picker used to list bare ids, so a session holding half a million
        kernel events looked identical to an empty one. Each row now carries its
        event count, GPU time and top op class, and `best` names the newest
        session that has data — which is what a user opening the app wants
        selected.
        """
        rows = state.sessions.sessions()
        try:
            state.reader.refresh()
            agg = state.reader.query(
                "SELECT session_id, count(*) AS n, "
                "sum(coalesce(duration_us,0)) AS us "
                "FROM pq_events GROUP BY session_id")
            # Wall-clock span is only meaningful over events that actually carry a
            # wall clock. Kernel rows carry the raw GPU clock instead, so mixing
            # them yields an absurd figure (order 1e9 seconds) — report no span
            # rather than a wrong one.
            span_rows = state.reader.query(
                "SELECT session_id, min(ts_wall_ns) AS t0, max(ts_wall_ns) AS t1 "
                "FROM pq_events WHERE kind<>'kernel' AND kind<>'memcpy' "
                "GROUP BY session_id")
            top = state.reader.query(
                "SELECT session_id, json_extract(extra_json,'$.op_class') AS c, "
                "sum(coalesce(duration_us,0)) AS us FROM pq_events "
                "WHERE kind='kernel' GROUP BY session_id, c")
            kern = state.reader.query(
                "SELECT session_id, count(*) AS n FROM pq_events "
                "WHERE kind='kernel' GROUP BY session_id")
            samples = state.reader.query(
                "SELECT session_id, count(*) AS n FROM pq_samples GROUP BY session_id")
        except Exception:  # noqa: BLE001 — a summary must never break the list
            agg, top, kern, samples, span_rows = [], [], [], [], []

        by_sid = {a["session_id"]: a for a in agg}
        spans = {r["session_id"]: r for r in span_rows}
        samples_by = {s["session_id"]: s["n"] for s in samples}
        kern_by = {k["session_id"]: int(k["n"]) for k in kern}
        best_op: dict[str, str] = {}
        for r in sorted(top, key=lambda r: -(r["us"] or 0)):
            best_op.setdefault(r["session_id"], (r["c"] or "other").strip('"'))

        out = []
        for s in rows:
            a = by_sid.get(s["session_id"], {})
            n = int(a.get("n") or 0)
            s = dict(s)
            s["event_count"] = n
            s["kernel_events"] = kern_by.get(s["session_id"], 0)
            s["sample_count"] = int(samples_by.get(s["session_id"], 0))
            s["gpu_ms"] = round((a.get("us") or 0) / 1000.0, 1)
            sp = spans.get(s["session_id"])
            span = ((sp["t1"] - sp["t0"]) / 1e9) if sp and sp.get("t1") else None
            s["span_s"] = round(span, 2) if span and span >= 0 else None
            s["top_op"] = best_op.get(s["session_id"])
            s["has_events"] = n > 0
            out.append(s)

        # "Best" = the session holding the most *measured kernel* evidence. Demo
        # is re-seeded on every boot, so "most recent with data" would always
        # select the synthetic session and hide the real capture.
        best = max(out, key=lambda s: (s["kernel_events"], s["event_count"])) \
            if any(s["kernel_events"] for s in out) else None
        return {"sessions": out, "live_session": state.live_session,
                "demo_session": state.demo_session,
                "best_session": best["session_id"] if best else None}

    def _sid_or_default(sid: Optional[str]) -> str:
        if sid:
            return sid
        return state.live_session or state.demo_session or ""

    @app.get("/api/sessions/{sid}/overview")
    def overview(sid: str):
        buf = state.samples.get(sid)
        latest = buf[-1] if buf else None
        sess = state.sessions.session(sid)
        if not sess and sid != (state.live_session or ""):
            raise HTTPException(404, "unknown session")
        return {"session_id": sid, "session": sess, "latest": latest,
                "sample_count": len(buf) if buf else 0,
                "adapter_health": state.adapter_health}

    @app.get("/api/sessions/{sid}/samples")
    def samples(sid: str, limit: int = Query(600, ge=1, le=5000)):
        buf = state.samples.get(sid)
        if buf:
            return {"samples": list(buf)[-limit:]}
        state.reader.refresh()
        rows = state.reader.query(
            "SELECT * FROM (SELECT * FROM pq_samples WHERE session_id=? "
            "ORDER BY ts_wall_ns DESC LIMIT ?) ORDER BY ts_wall_ns", [sid, limit])
        return {"samples": rows}

    @app.get("/api/sessions/{sid}/events")
    def events(sid: str, kind: Optional[str] = None, limit: int = Query(2000, ge=1, le=20000)):
        state.reader.refresh()
        rows = state.reader.query(
            "SELECT * FROM pq_events WHERE session_id=?"
            + (" AND kind=?" if kind else "") + " ORDER BY ts_wall_ns LIMIT ?",
            [sid] + ([kind] if kind else []) + [limit])
        for r in rows:
            for jc in ("tensor_json", "extra_json"):
                if r.get(jc):
                    try:
                        r[jc[:-5]] = json.loads(r[jc])
                    except Exception:  # noqa: BLE001
                        pass
                    r.pop(jc, None)
        return {"events": rows, "count": len(rows)}

    @app.get("/api/sessions/{sid}/architecture")
    def architecture(sid: str):
        state.reader.refresh()
        rows = state.reader.query(
            "SELECT graph_json, provenance, source FROM pq_architecture "
            "WHERE session_id=? ORDER BY updated_ns DESC LIMIT 1", [sid])
        if not rows:
            raise HTTPException(404, "no architecture for session")
        stored = json.loads(rows[0]["graph_json"])
        # The stored graph is a snapshot. Rebuild it from the checkpoint config
        # so the view reflects the current code and config rather than whatever
        # was persisted when the session was created.
        model_id = stored.get("model_id")
        if model_id:
            for m in list_models(cfg.model_dirs):
                if m["model_id"] == model_id:
                    fresh = build_graph(model_id, load_config(m["path"]))
                    if fresh.get("nodes") and len(fresh["nodes"]) == len(stored.get("nodes", [])):
                        return {**fresh, "_provenance": rows[0]["provenance"],
                                "_source": rows[0]["source"]}
                    break
        return {**stored, "_provenance": rows[0]["provenance"],
                "_source": rows[0]["source"]}

    @app.get("/api/sessions/{sid}/timeline")
    def timeline(sid: str):
        state.reader.refresh()
        total = state.reader.query(
            "SELECT count(*) n FROM pq_events WHERE session_id=?", [sid])[0]["n"]
        ev = state.reader.query(
            "SELECT kind, source, provenance, request_id, step_idx, component, "
            "op_id, ts_wall_ns, duration_us, bytes_moved, extra_json FROM pq_events "
            "WHERE session_id=? ORDER BY ts_wall_ns LIMIT 8000", [sid])
        lanes: dict[str, list] = {}
        for e in ev:
            lanes.setdefault(e["kind"], []).append(e)
        # Downsample each lane to a renderable count so a 100k+ capture doesn't
        # flood the DOM or the (bandwidth-limited) tunnel. Positions stay real;
        # we drop events, never invent them.
        cap = 500
        for k in lanes:
            if len(lanes[k]) > cap:
                step = -(-len(lanes[k]) // cap)
                lanes[k] = lanes[k][::step]
        returned = sum(len(v) for v in lanes.values())
        return {"lanes": lanes, "total": total, "returned": returned,
                "downsampled": returned < total}

    @app.get("/api/sessions/{sid}/opclasses")
    def opclasses(sid: str):
        """Server-side kernel-time aggregation by op class, so the Bottlenecks
        view reads measured shares without shipping thousands of raw events."""
        state.reader.refresh()
        rows = state.reader.query(
            "SELECT json_extract(extra_json, '$.op_class') AS op_class, "
            "count(*) launches, sum(duration_us) total_us FROM pq_events "
            "WHERE session_id=? AND kind='kernel' GROUP BY op_class ORDER BY total_us DESC",
            [sid])
        total = sum(r["total_us"] or 0 for r in rows)
        return {"by_class": [{"op_class": (r["op_class"] or "other").strip('"'),
                              "launches": r["launches"], "total_us": r["total_us"] or 0,
                              "pct": round(100 * (r["total_us"] or 0) / total, 1) if total else 0}
                             for r in rows],
                "total_us": total, "provenance": "measured"}

    # ---------------- models / architecture from config ----------------
    # ---- engine load / offload ------------------------------------------
    # One heavy engine at a time on one GPU: loading one offloads whatever held
    # it, and the response says so rather than leaving VRAM unexplained.

    @app.get("/api/units")
    async def units():
        """Every (engine, model) pair this machine can load.

        The scene and the picker are both driven from this: a "block" in the
        diagram is a unit, so choosing a different checkpoint or a different
        engine is the same swap.
        """
        found = await asyncio.to_thread(discover, cfg, state.engine_ctl.specs())
        # whisper.cpp has no config to read, so its architecture comes from the
        # hyperparameter block the engine prints when it loads the file.
        hp = asr_arch.read_hparams(cfg.data_dir / "logs" / "engine-asr.log")
        for u in found:
            if u.engine == "asr" and hp:
                u.arch = asr_arch.build_whisper_arch(hp, u.model_label)
        for u in found:
            if u.engine == "llm":
                continue
            spec = state.engine_ctl.spec(u.engine)
            if spec is None:
                u.problem = "no such engine"
            elif not u.binary or not os.path.exists(u.binary):
                u.problem = f"binary not found: {u.binary}"
            elif not u.configured:
                u.problem = u.problem or "binary is not executable"
        return {"units": [u.as_dict() for u in found],
                "resident": state.engine_ctl.loaded}

    @app.get("/api/engines")
    async def engines_status():
        return await asyncio.to_thread(state.engine_ctl.status)

    @app.post("/api/engines/{key}/load")
    async def engine_load(key: str):
        async with state.engine_lock:
            return await asyncio.to_thread(
                state.engine_ctl.load, key, str(cfg.data_dir / "logs"))

    @app.post("/api/engines/offload")
    async def engine_offload():
        async with state.engine_lock:
            return await asyncio.to_thread(state.engine_ctl.offload)

    @app.post("/api/tts/synthesize")
    async def tts_synthesize(req: Request):
        """Turn text into speech with whatever engine is resident.

        Refuses when nothing is loaded, for the same reason the ASR endpoint
        does: starting an engine behind the user's back would evict whatever
        they were already measuring.
        """
        if state.engine_ctl.loaded != "tts":
            raise HTTPException(409, "the TTS engine is not loaded — load it first")
        body = await req.json()
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "text is required")
        spec = state.engine_ctl.spec("tts")
        t0 = time.perf_counter()
        try:
            st, raw, _ = await asyncio.to_thread(_http_post_raw,
                                                 spec.base_url + "/synthesize",
                                                 json.dumps({"text": text}).encode(),
                                                 "application/json", 300.0)
        except Exception as e:                              # noqa: BLE001
            raise HTTPException(502, f"TTS engine failed: {e}")
        if st >= 400:
            raise HTTPException(502, f"TTS engine returned {st}: {raw[:200]}")
        audio_s = time.perf_counter() - t0
        return Response(content=raw, media_type="audio/wav", headers={
            "X-Engine-Seconds": f"{audio_s:.4f}",
            "X-Sample-Rate": raw_rate(raw),
        })

    @app.post("/api/image/generate")
    async def image_generate(req: Request):
        """Render a picture from a prompt with the resident image engine."""
        if state.engine_ctl.loaded != "image":
            raise HTTPException(409, "the image engine is not loaded — load it first")
        body = await req.json()
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "prompt is required")
        spec = state.engine_ctl.spec("image")
        steps = int(body.get("steps") or 4)
        t0 = time.perf_counter()
        try:
            st, raw, _ = await asyncio.to_thread(_http_post_raw,
                                                 spec.base_url + "/generate",
                                                 json.dumps({"prompt": prompt,
                                                             "steps": steps}).encode(),
                                                 "application/json", 900.0)
        except Exception as e:                              # noqa: BLE001
            raise HTTPException(502, f"image engine failed: {e}")
        if st >= 400:
            raise HTTPException(502, f"image engine returned {st}: {raw[:200]}")
        engine_s = time.perf_counter() - t0
        return JSONResponse({"image": "data:image/png;base64," + base64.b64encode(raw).decode(),
                             "engine_s": engine_s, "steps": steps, "prompt": prompt})

    @app.post("/api/asr/transcribe")
    async def asr_transcribe(req: Request):
        """Transcribe an uploaded audio file with the ASR engine.

        Refuses when the engine is not resident: InferScope does not reach out
        and start an engine behind the user's back, because loading one evicts
        whatever else was using the GPU.
        """
        form = await req.form()
        up = form.get("file")
        if up is None or not hasattr(up, "read"):
            raise HTTPException(400, "an audio file is required")
        data = await up.read()
        if not data:
            raise HTTPException(400, "the uploaded file is empty")
        name = getattr(up, "filename", "") or "audio"
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(413, "audio file is larger than 200 MB")
        if state.engine_ctl.loaded != "asr":
            raise HTTPException(409, "the ASR engine is not loaded — load it first")

        spec = state.engine_ctl.spec("asr")
        t0 = time.perf_counter()
        try:
            body, ctype = _multipart(fields={"file": (name, data, "application/octet-stream"),
                                             "temperature": "0",
                                             "response_format": "text"})
            st, text, _ = await asyncio.to_thread(_http_post, spec.base_url + "/inference",
                                                   body, ctype, 600.0)
        except Exception as e:                              # noqa: BLE001
            raise HTTPException(502, f"ASR engine failed: {e}")
        if st != 200:
            raise HTTPException(502, f"ASR engine returned {st}: {text[:200]}")
        dt = time.perf_counter() - t0
        return {
            "text": (text or "").strip(),
            "filename": name,
            "bytes": len(data),
            "seconds": round(dt, 3),
            "engine": spec.label,
            "model": spec.model,
            "audio_s": _wav_seconds(data),
            "provenance": "measured",
        }

    @app.get("/api/models")
    def models():
        return {"models": list_models(cfg.model_dirs)}

    @app.get("/api/models/{model_id}/architecture")
    def model_arch(model_id: str):
        for m in list_models(cfg.model_dirs):
            if m["model_id"] == model_id:
                cfgj = load_config(m["path"])
                return build_graph(model_id, cfgj)
        raise HTTPException(404, "model not found")

    # ---------------- KV calculator ----------------
    @app.get("/api/kv")
    def kv(n_layers: int, context: int, batch: int = 1, kv_heads: int = 8,
           head_dim: int = 128, dtype: str = "bfloat16",
           sliding_window: Optional[int] = None,
           swa_fraction: float = 1.0, shared_prefix: Optional[int] = None):
        inp = KVInputs(n_layers=n_layers, context=context, batch=batch,
                       kv_heads=kv_heads, head_dim=head_dim, dtype=dtype,
                       sliding_window=sliding_window, swa_layer_fraction=swa_fraction,
                       shared_prefix=shared_prefix)
        errs = inp.validate()
        if errs:
            raise HTTPException(422, "; ".join(errs))
        r = kv_cache_bytes(inp)
        return {
            "bytes": r.bytes, "gib": round(r.gib, 4), "baseline_bytes": r.baseline_bytes,
            "breakdown": r.breakdown, "notes": r.notes, "formula": r.formula,
            "provenance": "estimated",
        }

    # ---------------- experiments ----------------
    @app.get("/api/experiments")
    def experiments():
        return {"experiments": state.sessions.experiments()}

    @app.post("/api/experiments")
    def add_experiment(body: dict):
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(422, "experiment needs a name")
        # A result is only computed when both sessions exist and comparable data
        # is present; recording stores the intent, not a fabricated improvement.
        eid = state.sessions.add_experiment(
            name, str(body.get("kind", "")), str(body.get("baseline", "")),
            str(body.get("candidate", "")), {"note": str(body.get("note", ""))},
            {"status": "recorded", "improvement": None})
        return {"id": eid}

    # ---------------- chat: talk to a model, profile each turn --------------
    def _chat_base_url() -> str:
        if state.chat_server is not None and state.chat_server.running:
            return state.chat_server.base_url
        return (cfg.chat_base_url or "").rstrip("/")

    def _chat_session(sid: str | None) -> str:
        """Resolve the session a chat turn is recorded into."""
        if sid:
            return sid
        return state.live_session or state.demo_session or "demo"

    @app.get("/api/chat/state")
    async def chat_state():
        srv = state.chat_server
        managed = srv.status() if srv is not None else {"managed": cfg.chat_manage,
                                                         "running": False}
        base = _chat_base_url()
        ready = False
        detail = ""
        if base:
            h = await chat.health(base)
            ready = bool(h.get("ready"))
            detail = h.get("error", "")
        else:
            detail = ("no chat endpoint configured and no managed server running"
                      if not cfg.chat_base_url else "")
        attach_ok, attach_why = chat.ptrace_capable()
        return {
            "managed_mode": cfg.chat_manage and bool(cfg.chat_model_path),
            "base_url": base or None,
            "ready": ready,
            "detail": detail,
            "server": managed,
            "profiling_available": bool(
                state.caps.get("rocprofv3", {}).available),
            "per_turn_trace": False,
            "per_turn_trace_reason": None if attach_ok else attach_why,
            "profile_default": cfg.chat_profile_default,
            "max_tokens": cfg.chat_max_tokens,
            "model_path": cfg.chat_model_path or None,
            "server_bin": cfg.chat_server_bin if cfg.chat_manage else None,
            "note": (
                "Managed mode: InferScope starts llama-server. With 'kernel trace' "
                "the server runs *under* rocprofv3, so the trace covers the whole "
                "session (per-turn attach needs CAP_SYS_PTRACE, which this container "
                "does not have). Every turn is timed and counted regardless."
            ),
        }

    @app.post("/api/chat/server/start")
    async def chat_start(body: dict | None = None):
        body = body or {}
        if not cfg.chat_manage:
            raise HTTPException(400, "managed mode is disabled (INFERSCOPE_CHAT_MANAGE=0)")
        # A request may name the checkpoint it wants, in which case the
        # configured default is irrelevant — checking it first would refuse a
        # perfectly good model just because the env var was never set.
        want = str(body.get("model") or "").strip()
        if not want and (not cfg.chat_model_path or not os.path.exists(cfg.chat_model_path)):
            raise HTTPException(400, "no model file configured at INFERSCOPE_CHAT_MODEL")
        if not os.path.exists(cfg.chat_server_bin):
            raise HTTPException(400, f"llama-server not found at {cfg.chat_server_bin}")
        if state.chat_server is not None and state.chat_server.running:
            return state.chat_server.status()
        profile = bool(body.get("profile", cfg.chat_profile_default))
        out_dir = ""
        if profile:
            ok, msg = profiler.validate(state.caps)
            if not ok:
                raise HTTPException(409, f"kernel tracing unavailable: {msg}")
            out_dir = str(cfg.data_dir / "captures" / f"chat-{int(time.time())}")
        # Which checkpoint to load. Default to the configured one, but let the
        # picker choose any llama.cpp checkpoint on the box, taking its vision
        # projector with it when it has one.
        model_path = cfg.chat_model_path
        mmproj = mmproj_for(model_path) if model_path else None
        if want:
            found = discover(cfg, state.engine_ctl.specs())
            match = next((u for u in found if u.engine == "llm" and u.key == want), None)
            if match is None:
                raise HTTPException(400, f"unknown model: {want}")
            if not os.path.exists(match.model_path):
                raise HTTPException(400, f"model file is gone: {match.model_path}")
            model_path, mmproj = match.model_path, match.mmproj
        state.chat_server = chat.ManagedServer(
            cfg.chat_server_bin, model_path, cfg.chat_port,
            ctx=cfg.chat_ctx, parallel=cfg.chat_parallel, mmproj=mmproj,
            profile=profile, out_dir=out_dir, max_seconds=cfg.capture_max_s * 60)
        await asyncio.to_thread(state.chat_server.start,
                                str(cfg.data_dir / "logs" / "llama-server.log"))
        if state.chat_server.error and not state.chat_server.running:
            raise HTTPException(409, state.chat_server.error)
        ok = await asyncio.to_thread(state.chat_server.wait_ready)
        st = state.chat_server.status()
        st["ready"] = ok
        if not ok:
            raise HTTPException(504, st.get("error") or "server did not become ready")
        return st

    @app.post("/api/chat/server/stop")
    async def chat_stop():
        """Stop the server and, if it ran under the profiler, ingest its trace."""
        if state.chat_server is None:
            return {"stopped": False, "detail": "no managed server"}
        srv = state.chat_server
        r = await asyncio.to_thread(srv.stop)
        ingested: dict[str, Any] = {"events": 0}
        if srv.profile and srv.out_dir:
            ok, msg = profiler.validate(state.caps)
            if ok:
                res = await asyncio.to_thread(
                    profiler.run_capture_parse_only, srv.out_dir,
                    state.live_session or "chat", state.caps,
                    model_id=os.path.basename(srv.model_path))
                evs = res.get("_events") or []
                if evs:
                    await asyncio.to_thread(state.trace.append_events, evs)
                ingested = {"events": len(evs), "ok": res.get("ok"),
                            "error": res.get("error")}
            else:
                ingested = {"events": 0, "error": msg}
        return {**r, **srv.status(), "ingested": ingested}

    @app.post("/api/chat/send")
    async def chat_send(req: Request):
        """Stream one chat turn as SSE, then record the turn as an event.

        Profile flag requests a bounded kernel trace for this turn; it is
        refused (not silently ignored) when the server is not ours.
        """
        body = await req.json()
        message = (body.get("message") or "").strip()
        images = [i for i in (body.get("images") or [])
                  if isinstance(i, str) and i.startswith("data:image/")]
        if not message and not images:
            raise HTTPException(400, "message or an image is required")
        base = _chat_base_url()
        if not base:
            raise HTTPException(409, "no chat endpoint: start a managed server first")
        history = body.get("history") or []
        messages = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in history if m.get("content")]
        if images:
            # OpenAI-shaped content parts: llama.cpp's /v1/chat/completions takes
            # pictures this way, and only if the server was started with the
            # matching projector.
            parts: list[dict[str, Any]] = []
            if message:
                parts.append({"type": "text", "text": message})
            parts += [{"type": "image_url", "image_url": {"url": u}} for u in images]
            messages.append({"role": "user", "content": parts})
        else:
            messages.append({"role": "user", "content": message})
        want_profile = bool(body.get("profile", cfg.chat_profile_default))
        sid = _chat_session(body.get("session_id"))
        max_tokens = int(body.get("max_tokens") or cfg.chat_max_tokens)
        attach_ok, attach_why = chat.ptrace_capable()

        async def gen():
            request_id = f"chat-{int(time.time()*1000)}"
            turn = None
            started = time.time_ns()

            def frame(kind: str, data: Any) -> str:
                return f"event: {kind}\ndata: {json.dumps(data)}\n\n"

            if want_profile:
                # Per-turn kernel tracing needs rocprofv3's attach helper to
                # ptrace the server. Say so plainly instead of failing later
                # with an opaque error.
                yield frame("notice", {
                    "message": (f"per-turn kernel trace unavailable: {attach_why}. "
                                f"This turn is still timed and counted. To get a "
                                f"kernel trace, start the server with profiling on "
                                f"— the trace then covers the whole session."),
                    "severity": "warn",
                })

            try:
                async for kind, payload in chat.stream_turn(
                        base, messages, max_tokens=max_tokens,
                        temperature=float(body.get("temperature") or 0.7)):
                    if kind == "delta":
                        yield frame("delta", payload)
                    elif kind == "reasoning":
                        yield frame("reasoning", payload)
                    else:
                        turn = payload
            except Exception as e:  # noqa: BLE001
                yield frame("error", {"message": str(e)})

            if turn is not None:
                now_wall = time.time_ns()
                now_mono = time.monotonic_ns()
                ev = Event(
                    event_id=f"{request_id}-turn",
                    session_id=sid, kind=EventKind.REQUEST, component="chat",
                    source=Source.LLAMA_CPP, provenance=Provenance.MEASURED,
                    request_id=request_id, ts_wall_ns=started,
                    ts_mono_ns=now_mono,
                    engine="llama.cpp",
                    model_id=turn.model or None,
                    start_ns=started, end_ns=now_wall,
                    duration_us=turn.total_ms * 1000.0, unit="us",
                    extra={
                        "prompt_tokens": turn.prompt_tokens,
                        "completion_tokens": turn.completion_tokens,
                        "ttft_ms": turn.ttft_ms,
                        "tok_per_s": turn.tok_per_s,
                        "model": turn.model,
                        "finish_reason": turn.finish_reason,
                        "per_turn_kernel_trace": False,
                        "preview": (turn.text or turn.reasoning or "")[:160],
                        "reasoning_only": turn.reasoning_only,
                    })
                await asyncio.to_thread(state.trace.append_events, [ev])
                state.remember_event(ev)
                state.publish({"type": "event", "session_id": sid, "kind": "request",
                               "request_id": request_id})
                yield frame("done", {**turn.as_dict(), "request_id": request_id,
                                    "session_id": sid})
            else:
                yield frame("done", {"error": "no response", "session_id": sid})

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------- profiling status (capability only; execution is CLI-only) --
    @app.get("/api/capture/status")
    def capture_status():
        from .profiler import validate
        ok, msg = validate(state.caps)
        return {
            "available": ok, "detail": msg,
            "perfetto_supported": bool(state.caps.get("rocprofv3")
                                       and state.caps["rocprofv3"].supports.get("perfetto_output")),
            "how": "run on the host: inferscope capture --yes --seconds 10 -- <cmd>",
            "note": "The API never executes browser-supplied commands; captures are "
                    "operator-initiated, bounded, and self-launched.",
        }

    # ---------------- SSE ----------------
    @app.get("/api/stream")
    async def stream(request: Request):
        q = state.subscribe()

        async def gen():
            try:
                yield f"event: hello\ndata: {json.dumps({'live': state.live_session, 'demo': state.demo_session})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"event: {msg['type']}\ndata: {json.dumps(msg)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                state.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # ---------------- static frontend ----------------
    dist = cfg.data_dir.parent / "ui" / "dist"
    if dist.exists():
        class _AssetCache(StaticFiles):
            """Long cache for hashed bundles, no cache for the shell.

            Without this a browser can pin an old index.html and keep loading a
            stale bundle after a deploy, so UI fixes appear not to apply.
            Hashed filenames change on every build, so caching them forever is
            safe.
            """

            async def get_response(self, path: str, scope) -> Any:
                resp = await super().get_response(path, scope)
                if path.rsplit("/", 1)[-1].startswith("index-"):
                    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                return resp

        # "/" is served explicitly: StaticFiles resolves the directory case
        # internally, so a cache header set in get_response never reaches it.
        @app.get("/", include_in_schema=False)
        def index():
            return HTMLResponse(
                (dist / "index.html").read_text(),
                headers={"Cache-Control": "no-cache, must-revalidate"})

        app.mount("/", _AssetCache(directory=str(dist), html=True), name="ui")
    else:
        @app.get("/")
        def index():
            return HTMLResponse(
                "<h1>InferScope API</h1><p>Frontend not built. "
                "Run <code>cd ui &amp;&amp; npm install &amp;&amp; npm run build</code>. "
                "API is live at <a href='/api/health'>/api/health</a>.</p>")

    return app
