"""Discover the current public tunnel URL dynamically.

The rc-tunnel hands out an ephemeral URL that changes whenever the tunnel is
re-established, so nothing here is hard-coded. Sources, in order:

1. ``INFERSCOPE_PUBLIC_URL`` — explicit override.
2. ``/root/.local/state/rc-tunnel/state.json`` — the tunnel tool's own state
   file, written on every (re)start. This is the authoritative source.
3. ``rc-tunnel status`` — with the broker environment inherited from PID 1,
   because the broker credentials live in the pod environment and are not
   exported into non-interactive shells.
4. The rc-tunnel log.

The app binds loopback and is reached *through* this tunnel; no URL is ever
baked into the source or the UI.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

_URL_RE = re.compile(r"https?://[^\s\"']+")
_STATE = Path("/root/.local/state/rc-tunnel/state.json")
_LOGS = ("/root/.local/state/rc-tunnel/frpc.log",
         "/tmp/tunnel8090.log", "/tmp/tunnel.log",
         "/tmp/inferscope-tunnel.log")
_BROKER_VARS = ("FRP_BROKER_URL", "FRP_BROKER_SERVICE_HOST",
                "FRP_BROKER_SERVICE_PORT", "FRP_BROKER_TLS_SERVER_NAME",
                "RC_TUNNEL_CA_FILE")

_cache: tuple[float, str | None] = (0.0, None)
_TTL = 20.0


def _pod_env() -> dict[str, str]:
    """Recover the pod-level broker environment from PID 1.

    rc-tunnel refuses to run without FRP_BROKER_* , and a non-interactive SSH
    shell does not inherit them. Reading them from the init process lets the
    status subcommand work instead of silently failing.
    """
    env = {}
    try:
        raw = Path("/proc/1/environ").read_bytes().split(b"\0")
    except OSError:
        return env
    for item in raw:
        if b"=" not in item:
            continue
        k, _, v = item.decode("utf-8", "replace").partition("=")
        if k in _BROKER_VARS:
            env[k] = v
    return env


def _from_state() -> str | None:
    try:
        st = json.loads(_STATE.read_text())
    except (OSError, ValueError):
        return None
    fqdn = (st.get("fqdn") or "").strip()
    if not fqdn:
        return None
    return fqdn if fqdn.startswith("http") else f"https://{fqdn}"


def _from_status() -> str | None:
    env = {**os.environ, **_pod_env()}
    for cmd in (["/root/.local/bin/rc-tunnel", "status"],
                ["rc-tunnel", "status"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8, env=env)
        except Exception:  # noqa: BLE001
            continue
        for m in re.finditer(r"(https?://\S+)", (r.stdout or "") + (r.stderr or "")):
            u = m.group(1).strip().rstrip(".,")
            if "127.0.0.1" not in u and "localhost" not in u and "0.0.0.0" not in u:
                return u
    return None


def _from_log() -> str | None:
    for log in _LOGS:
        try:
            with open(log) as f:
                urls = _URL_RE.findall(f.read())
        except OSError:
            continue
        public = [u for u in urls
                  if "127.0.0.1" not in u and "localhost" not in u and "0.0.0.0" not in u]
        if public:
            return public[-1]
    return None


def current_url(force: bool = False) -> str | None:
    global _cache
    now = time.monotonic()
    if not force and now - _cache[0] < _TTL:
        return _cache[1]
    url = (os.environ.get("INFERSCOPE_PUBLIC_URL") or "").strip() or None
    if not url:
        url = _from_state() or _from_status() or _from_log()
    _cache = (now, url)
    return url


_FRPC_LOG = "/root/.local/state/rc-tunnel/frpc.log"
_AUTH_FAIL = re.compile(
    r"invalid_client|invalid client credentials|login to server failed"
    r"|does not match control reservation")
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")


def _frpc_health() -> tuple[bool | None, str]:
    """Is the tunnel client actually connected?

    Deliberately *not* an HTTP request to the public URL: the pod has no
    hairpin route to its own public host ("Network is unreachable"), so such a
    probe would always fail and wrongly report a healthy tunnel as dead. The
    frpc client process and its log are the reliable local signals.

    Only log lines *after* the tunnel's own start time count — the log is
    appended across restarts, so older auth failures from a previous tunnel
    would otherwise mark a freshly re-issued tunnel as broken.
    """
    started = None
    try:
        st = json.loads(_STATE.read_text())
        pid = st.get("pid")
        raw = st.get("started_at")
        if raw:
            started = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if started.tzinfo is not None:
                started = started.astimezone().replace(tzinfo=None)
    except (OSError, ValueError):
        pid = None
    if pid:
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError):
            return False, f"tunnel client process {pid} is not running"
    try:
        with open(_FRPC_LOG) as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            tail = f.read()
    except OSError:
        return None, "tunnel log not readable; liveness unverified"

    # The log is chronological, so the *last* decisive line is the current
    # state. Preferring "connected" over a later auth failure would report a
    # dead tunnel as healthy.
    state: bool | None = None
    for line in tail.splitlines():
        ts = _TS_RE.match(line)
        if started and ts:
            try:
                when = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue
            if when < started:
                continue          # belongs to an earlier tunnel
        if _AUTH_FAIL.search(line):
            state = False
        elif "start proxy success" in line or "login to server success" in line:
            state = True
    if state is True:
        return True, "tunnel client connected"
    if state is False:
        return False, ("tunnel client's broker credentials were rejected "
                       "(invalid_client) or its reservation is stale; re-issue "
                       "with `rc-tunnel expose --port 8090`")
    return None, "tunnel client is running; connected state not directly observable"


def status() -> dict:
    url = current_url()
    healthy, detail = _frpc_health()
    how = ("the app binds loopback on 127.0.0.1:8090 and is served through "
           "the rc-tunnel. The URL is ephemeral and rotates whenever the tunnel "
           "is re-established; the app reads the current one from rc-tunnel "
           "state, so reload after `rc-tunnel expose`.")
    if healthy is False:
        how += f" {detail}."
    return {
        "public_url": url,
        "reachable": healthy,
        "detail": detail,
        "how": how,
        "source": "INFERSCOPE_PUBLIC_URL" if os.environ.get("INFERSCOPE_PUBLIC_URL")
                  else ("rc-tunnel state.json" if _from_state()
                        else ("rc-tunnel status" if _from_status() else "rc-tunnel log")),
    }