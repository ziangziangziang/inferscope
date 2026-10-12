"""InferScope command line.

    inferscope check      probe the environment and print capabilities
    inferscope serve      run the API + collector (loopback by default)
    inferscope demo       seed a synthetic session and report counts
    inferscope models     list discoverable model configs and their families
    inferscope capture    bounded, opt-in rocprofv3 capture (never by default)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from .capability import probe_all, _run
from .config import Config


def _fmt_cap(c) -> str:
    mark = "yes" if c.available else " no"
    line = f"  [{mark}] {c.name:16} {c.detail}"
    if c.supports:
        sup = " ".join(f"{k}={'y' if v else 'n'}" for k, v in c.supports.items())
        line += f"   ({sup})"
    return line


def cmd_check(args) -> int:
    cfg = Config.from_env()
    caps = probe_all(cfg)
    print("InferScope environment self-check")
    print("=" * 64)
    for c in caps.values():
        print(_fmt_cap(c))
        for n in c.notes:
            print(f"        note: {n}")
    print("=" * 64)
    from .tunnel import current_url
    url = current_url()
    gpu = caps.get("amdsmi") or caps.get("sysfs_gpu")
    live = (caps["system_cpu_ram"].available or
            (caps.get("amdsmi", None) and caps["amdsmi"].available) or
            (caps.get("sysfs_gpu") and caps["sysfs_gpu"].available))
    print(f"bind (loopback): {cfg.bind_url}")
    if url:
        print(f"access URL    : {url}   <- open this (tunnel host)")
    else:
        print("access URL    : (no active tunnel — start it: tunnel.sh expose --port "
              f"{cfg.port}; fallback: {cfg.ssh_tunnel_hint()})")
    print(f"demo mode: {'on' if cfg.demo else 'off'}   "
          f"live telemetry: {'available' if live else 'NONE (UI will run on demo data)'}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from .api import create_app
    cfg = Config.from_env()
    if args.port:
        cfg.port = args.port
    if args.host:
        cfg.host = args.host
    if args.no_demo:
        cfg.demo = False
    app = create_app(cfg)
    print(f"InferScope serving on {cfg.bind_url}")
    print(f"Access from your workstation:  {cfg.ssh_tunnel_hint()}")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
    return 0


def cmd_demo(args) -> int:
    from .demo import seed_demo
    from .store import SessionStore, TraceStore
    cfg = Config.from_env()
    sessions = SessionStore(cfg.data_dir)
    trace = TraceStore(cfg.data_dir, read_only=False)
    sid = seed_demo(sessions, trace)
    ev = trace.query("SELECT count(*) n FROM pq_events WHERE session_id=?", [sid])[0]["n"]
    sm = trace.query("SELECT count(*) n FROM pq_samples WHERE session_id=?", [sid])[0]["n"]
    print(f"seeded demo session {sid}: {ev} events, {sm} samples (provenance=simulated)")
    return 0


def cmd_models(args) -> int:
    from .arch import list_models
    cfg = Config.from_env()
    ms = list_models(cfg.model_dirs)
    if not ms:
        print(f"no model configs found under {cfg.model_dirs}")
        return 0
    for m in ms:
        print(f"  {m['model_id']:40} family={m['family']:6} {m['architectures']}")
    return 0


def cmd_capture(args) -> int:
    """Validate rocprofv3 args, run a bounded capture of an explicit command, and
    persist the resulting kernel/memcpy events (plus telemetry sampled during the
    capture) into a new 'profile' session the UI can open.

    Refuses without --yes and an explicit command, so it can never profile a
    foreign/long-running process by accident.
    """
    cfg = Config.from_env()
    caps = probe_all(cfg)
    ok, msg = __import__("inferscope.profiler", fromlist=["validate"]).validate(caps)
    if not ok:
        print("rocprofv3 capture unsupported:", msg)
        return 2
    if not args.capture_cmd:
        print("no command given; nothing to capture. Usage: "
              "inferscope capture --yes --seconds 10 -- <cmd...>")
        return 2
    if not args.yes:
        print("refusing to start a profiling capture without --yes "
              "(bounded, opt-in only).")
        return 2

    import threading
    import time as _t
    from .profiler import run_capture
    from .store import SessionStore, TraceStore
    from .collectors import GpuCollector, SystemCollector

    sid = "prof-" + _t.strftime("%Y%m%d-%H%M%S")
    out_dir = f"/tmp/inferscope-capture-{int(_t.time())}"
    sec = max(1, min(args.seconds, cfg.capture_max_s))

    stores = SessionStore(cfg.data_dir)
    trace = TraceStore(cfg.data_dir, read_only=False)
    gpu = GpuCollector(caps)
    sysc = SystemCollector()

    stop = threading.Event()
    collected = []

    def sampler():
        while not stop.is_set():
            for c in (sysc, gpu):
                try:
                    s = c.poll(sid)
                    if s:
                        collected.append(s)
                except Exception:  # noqa: BLE001
                    pass
            _t.sleep(cfg.poll_interval_s)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    print(f"bounded capture ({sec}s) -> {out_dir}; sampling telemetry meanwhile")
    res = run_capture(args.capture_cmd, sid, sec, out_dir, caps,
                      perfetto=getattr(args, "perfetto", False), model_id="captured")
    stop.set(); th.join(timeout=5)
    gpu.close()

    if not res.get("ok"):
        print("capture failed:", res.get("error"))
        return 1
    events = res.pop("_events", [])
    stores.create_session(sid, kind="profile", engine="rocprofv3",
                          model_id="captured", source="profile",
                          notes=f"bounded capture of {' '.join(args.capture_cmd[:3])}… for {sec}s")
    n_ev = trace.append_events(events)
    n_sm = trace.append_samples(collected)
    # attach an architecture so the session is coherent in the UI
    from .arch import build_graph, list_models, load_config
    found = list_models(cfg.model_dirs)
    g = build_graph(found[0]["model_id"], load_config(found[0]["path"])) if found else build_graph("captured", {})
    trace.set_architecture(sid, g.get("model_id", "captured"), g, "derived" if found else "estimated", "model_config")
    print(f"session {sid}: {n_ev} kernel events, {n_sm} telemetry samples, trace in {out_dir}")
    print("open it in the UI (session selector) — Timeline + Bottlenecks show measured op-class shares.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="inferscope")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    sp = sub.add_parser("serve")
    sp.add_argument("--host"); sp.add_argument("--port", type=int)
    sp.add_argument("--no-demo", action="store_true")
    sp.set_defaults(func=cmd_serve)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    sub.add_parser("models").set_defaults(func=cmd_models)
    cp = sub.add_parser("capture")
    cp.add_argument("--yes", action="store_true")
    cp.add_argument("--seconds", type=int, default=10)
    cp.add_argument("--perfetto", action="store_true")
    # named capture_cmd, not cmd: "cmd" is the subparsers dest and would collide
    cp.add_argument("capture_cmd", nargs=argparse.REMAINDER)
    cp.set_defaults(func=cmd_capture)
    args = ap.parse_args(argv)
    # strip the "--" separator before the target command
    if getattr(args, "capture_cmd", None) and args.capture_cmd[:1] == ["--"]:
        args.capture_cmd = args.capture_cmd[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
