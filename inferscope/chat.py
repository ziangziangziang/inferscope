"""Chat against a llama.cpp server, with per-message profiling.

This is deliberately narrow:

* It only speaks to an **OpenAI-compatible** ``/v1/chat/completions`` endpoint
  on loopback. It never executes a command supplied by the browser.
* Turn-level numbers (TTFT, wall time) are **measured** client-side.
* Token counts are read from the engine's own ``/metrics`` counters by taking a
  before/after delta — those are **measured** counter increments, attributed to
  the turn by bracketing it. They are not tokenized locally.
* A kernel trace for the turn is only possible in **managed** mode, where
  InferScope itself started the server and therefore owns its pid (rocprofv3
  attach needs ptrace, and ptrace_scope=1 only permits a direct parent). In
  external mode the turn is still profiled at request level, and the API says
  so rather than implying a kernel trace exists.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

log = logging.getLogger("inferscope.chat")

import httpx

# llama.cpp metric names this module relies on. If an engine renames one we
# report the turn without it rather than substituting a guess.
M_PROMPT = "llamacpp:prompt_tokens_total"
M_PREDICTED = "llamacpp:tokens_predicted_total"

_LOAD_RE = re.compile(r"(model loading|loading model|llama_model_loader)", re.I)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse *unlabelled* counters from Prometheus text format.

    Labelled series are deliberately skipped: two label sets can share a metric
    name, and collapsing them into one key would silently report the last one as
    if it were the total. The counters this module needs are unlabelled; if an
    engine version labels them, the turn reports no token counts rather than a
    wrong one.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "{" in line:
            continue
        m = re.match(r"^([A-Za-z_:][\w:]*)\s+([-\d.eE+]+)$", line)
        if not m:
            continue
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    return out


def token_counters(text: str) -> dict[str, int]:
    m = parse_prometheus(text)
    return {
        "prompt_tokens": int(m.get(M_PROMPT, 0)),
        "completion_tokens": int(m.get(M_PREDICTED, 0)),
    }


def _props(base_url: str, timeout: float = 4.0) -> dict[str, Any]:
    """Server identity: which model this endpoint has actually loaded."""
    try:
        with urllib.request.urlopen(f"{base_url}/props", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return {}


CAP_SYS_PTRACE = 19


def ptrace_capable() -> tuple[bool, str]:
    """Whether this process may ptrace another one.

    rocprofv3's attach mode is implemented by a *separate* helper process
    (rocprofv3-attach), not by us, so yama ptrace_scope=1 blocks it: the helper
    is not the target's parent. Only CAP_SYS_PTRACE in our own effective set can
    lift that. We check rather than discover it by failing a capture.
    """
    try:
        for line in open("/proc/self/status"):
            if line.startswith("CapEff:"):
                eff = int(line.split()[1], 16)
                if eff & (1 << CAP_SYS_PTRACE):
                    return True, "CAP_SYS_PTRACE present"
                return False, (
                    "this container lacks CAP_SYS_PTRACE and yama ptrace_scope=1 "
                    "only permits a direct parent, but rocprofv3 attaches from a "
                    "separate helper process")
    except OSError as e:
        return False, f"cannot read /proc/self/status: {e}"
    return False, "CapEff not found in /proc/self/status"


def port_in_use(port: int) -> int | None:
    """Return the pid listening on a loopback TCP port, if any."""
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                         (socket.AF_INET6, ("::1", port))):
        with contextlib.suppress(OSError):
            inodes: set[str] = set()
            with open("/proc/net/tcp" if family == socket.AF_INET else "/proc/net/tcp6") as f:
                next(f, None)
                for line in f:
                    p = line.split()
                    if len(p) < 10 or p[3] != "0A":       # 0A == LISTEN
                        continue
                    if int(p[1].split(":")[1], 16) == port:
                        inodes.add(p[9])
            if not inodes:
                continue
            for pid_dir in os.listdir("/proc"):
                if not pid_dir.isdigit():
                    continue
                try:
                    for fd in os.listdir(f"/proc/{pid_dir}/fd"):
                        link = os.readlink(f"/proc/{pid_dir}/fd/{fd}")
                        if link.startswith("socket:[") and link[8:-1] in inodes:
                            return int(pid_dir)
                except OSError:
                    continue
    return None


@dataclass
class Turn:
    text: str = ""
    reasoning: str = ""
    ttft_ms: float | None = None
    total_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str = ""
    finish_reason: str = ""
    error: str | None = None
    kernel_events: int = 0
    capture: dict[str, Any] = field(default_factory=dict)

    @property
    def tok_per_s(self) -> float | None:
        # Measured generation rate: engine-reported completion tokens over the
        # measured post-first-token window.
        if not self.completion_tokens or not self.total_ms or self.total_ms <= 0:
            return None
        return self.completion_tokens / (self.total_ms / 1000.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reasoning": self.reasoning,
            "reasoning_only": bool(self.reasoning) and not self.text,
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "total_ms": round(self.total_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tok_per_s": round(self.tok_per_s, 3) if self.tok_per_s else None,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "kernel_events": self.kernel_events,
            "capture": self.capture,
        }


async def _metrics(base_url: str, timeout: float = 5.0) -> tuple[dict[str, int], str | None]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/metrics")
            r.raise_for_status()
            return token_counters(r.text), None
    except Exception as e:  # noqa: BLE001
        return {}, str(e)


async def health(base_url: str, timeout: float = 4.0) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/health")
            r.raise_for_status()
            return {"ready": True, "status": r.json().get("status", "ok")}
    except Exception as e:  # noqa: BLE001
        return {"ready": False, "error": str(e)}


async def stream_turn(base_url: str, messages: list[dict[str, str]],
                      *, max_tokens: int = 256, temperature: float = 0.7,
                      model: str = "local") -> AsyncIterator[tuple[str, Any]]:
    """Yield ("delta", str) / ("done", Turn) while streaming a chat turn.

    The token counters are sampled immediately before the request and again
    after it completes, so the delta is attributable to this turn even though
    the engine increments global counters.
    """
    before, _ = await _metrics(base_url)
    t0 = time.perf_counter()
    ttft: float | None = None
    turn = Turn(model=model)
    parts: list[str] = []
    think: list[str] = []
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "stream": True}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as c:
            async with c.stream("POST", f"{base_url}/v1/chat/completions",
                                json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        continue
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    for ch in obj.get("choices") or []:
                        delta = (ch.get("delta") or {})
                        # Reasoning models emit `reasoning_content` before any
                        # `content`. Ignoring it made a short turn look like
                        # the model said nothing at all, even though it had
                        # generated tokens. Both streams are forwarded.
                        reasoning = delta.get("reasoning_content")
                        if reasoning:
                            yield ("reasoning", reasoning)
                        piece = delta.get("content")
                        if piece:
                            parts.append(piece)
                            yield ("delta", piece)
                        if ch.get("finish_reason"):
                            turn.finish_reason = ch["finish_reason"]
                    if obj.get("model"):
                        turn.model = obj["model"]
    except Exception as e:  # noqa: BLE001
        turn.error = str(e)
        turn.total_ms = (time.perf_counter() - t0) * 1000.0
        turn.reasoning = "".join(think)
        after, _ = await _metrics(base_url)
        turn.prompt_tokens = _delta(before.get("prompt_tokens"), after.get("prompt_tokens"))
        turn.completion_tokens = _delta(before.get("completion_tokens"), after.get("completion_tokens"))
        turn.ttft_ms = ttft
        yield ("done", turn)
        return

    turn.total_ms = (time.perf_counter() - t0) * 1000.0
    turn.text = "".join(parts)
    turn.reasoning = "".join(think)
    turn.ttft_ms = ttft
    after, _ = await _metrics(base_url)
    turn.prompt_tokens = _delta(before.get("prompt_tokens"), after.get("prompt_tokens"))
    turn.completion_tokens = _delta(before.get("completion_tokens"), after.get("completion_tokens"))
    yield ("done", turn)


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return max(0, after - before)


def _proc_cmdline(pid: int) -> str:
    with contextlib.suppress(OSError):
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    return ""


def _proc_cwd(pid: int) -> str:
    with contextlib.suppress(OSError):
        return os.readlink(f"/proc/{pid}/cwd")
    return ""


def _APP_ROOT() -> str:
    """Where this checkout lives; used only to recognise our own leftovers."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reap_own_orphan(port: int, app_root: str, model_path: str = "",
                    expect: str = "") -> int | None:
    """Kill a *provably ours* leftover server on *port*, and return its pid.

    InferScope refuses to adopt a server it did not spawn, because kernel
    tracing has to attach to a pid it owns. That refusal is correct, but it used
    to strand the user: the managed server is started with
    ``start_new_session=True``, so if InferScope is killed hard (``pkill -9``,
    an OOM, a redeploy) the ``llama-server`` child survives, is reparented to
    init, and every later "start engine" fails with a 409 about a pid that
    InferScope can neither trace nor stop.

    So the owner check still stands — this only fires when the listener is
    demonstrably an InferScope orphan, never for a foreign process:

    * its command line names the configured model, or
    * its working directory is this application;

    *and* it is the engine binary the caller asked about. `expect` is that
    basename. It has to come from the caller rather than being hardcoded here:
    this function is shared by every engine, and an earlier version only
    recognised ``llama-server``, so a perfectly good ``whisper-server``
    orphan sitting on the ASR port was reported as "not an inference server at
    all" and left running.
    """
    pid = port_in_use(port)
    if pid is None:
        return None
    cmd = _proc_cmdline(pid)
    by_model = bool(model_path) and model_path in cmd
    by_cwd = bool(app_root) and _proc_cwd(pid).rstrip("/") == app_root.rstrip("/")
    if not (by_model or by_cwd):
        return None
    if expect:
        if expect not in cmd:
            return None
    elif "server" not in cmd.lower():
        return None
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if port_in_use(port) is None:
            return pid
        time.sleep(0.25)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    return pid


class ManagedServer:
    """A llama-server process owned by InferScope.

    ``profile=True`` starts the server *under* rocprofv3 (wrap mode), which is
    the mode that works without CAP_SYS_PTRACE: the profiled process is then a
    child of the profiler and needs no attach. The cost is that the trace covers
    the whole server lifetime, so it is session-level, not per-turn.

    Per-turn attach needs rocprofv3's attach helper to ptrace the server, which
    this container cannot do; see :func:`ptrace_capable`.
    """

    def __init__(self, binary: str, model_path: str, port: int, *,
                 ctx: int = 8192, parallel: int = 1,
                 extra_args: list[str] | None = None,
                 profile: bool = False, out_dir: str = "", max_seconds: int = 900):
        self.binary = binary
        self.model_path = model_path
        self.port = port
        self.ctx = ctx
        self.parallel = parallel
        self.extra_args = extra_args or []
        self.profile = profile
        self.out_dir = out_dir
        self.max_seconds = max_seconds
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = ""
        self.started_at: float | None = None
        self.error: str | None = None
        self._stderr: list[str] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def server_argv(self) -> list[str]:
        return [self.binary, "-m", self.model_path, "--port", str(self.port),
                "--host", "127.0.0.1", "--metrics",
                "-c", str(self.ctx), "-np", str(self.parallel),
                *self.extra_args]

    def argv(self) -> list[str]:
        """The full command, wrapped in rocprofv3 when profiling."""
        inner = self.server_argv()
        if not self.profile:
            return inner
        if not self.out_dir:
            raise ValueError("profile=True requires out_dir")
        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.join(self.out_dir, "trace")
        return ["rocprofv3", "--kernel-trace", "-d", self.out_dir, "-o", base,
                "-f", "csv", "--", *inner]

    def start(self, log_path: str) -> None:
        if self.running:
            return
        # A previous InferScope may have been killed hard and left its server
        # behind; that one is ours to clear.
        reaped = reap_own_orphan(self.port, app_root=_APP_ROOT(),
                                 model_path=self.model_path,
                                 expect=os.path.basename(self.binary))
        if reaped:
            self.error = None
            log.warning("cleared an orphaned InferScope llama-server (pid %s) on port %s",
                        reaped, self.port)
        busy = port_in_use(self.port)
        if busy:
            self.error = (f"port {self.port} is already in use by pid {busy}; "
                          f"stop it first — InferScope will not adopt a server "
                          f"it does not own (kernel tracing needs its own pid)")
            return
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.log_path = log_path
        self.error = None
        out = open(log_path, "ab")
        self.proc = subprocess.Popen(
            self.argv(), stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        self.started_at = time.time()

    def wait_ready(self, timeout: float = 300.0) -> bool:
        """Wait until *this* server is serving the *configured* model.

        Health alone is not enough: a foreign server on the same port would
        answer it, and then every turn would be attributed to a process InferScope
        cannot trace. /props reports the loaded model path, so that is checked.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is None:
                self.error = self.error or "server was not started"
                return False
            if self.proc.poll() is not None:
                self.error = (f"server exited with code {self.proc.returncode}"
                              + (f"; see {self.log_path}" if self.log_path else ""))
                return False
            props = _props(self.base_url)
            if props.get("model_path") and props["model_path"] == self.model_path:
                return True
            time.sleep(1.0)
        self.error = (f"timed out waiting for {self.model_path} to serve on "
                      f"port {self.port}")
        return False

    def stop(self, timeout: float = 20.0) -> dict[str, Any]:
        if not self.proc:
            return {"stopped": False, "detail": "not running"}
        rc = self.proc.poll()
        if rc is None:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=10)
        return {"stopped": True, "returncode": self.proc.returncode}

    def status(self) -> dict[str, Any]:
        attach_ok, attach_why = ptrace_capable()
        return {
            "managed": True,
            "running": self.running,
            "pid": self.proc.pid if self.proc else None,
            "base_url": self.base_url,
            "model_path": self.model_path,
            "model": os.path.basename(self.model_path),
            "ctx": self.ctx,
            "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else None,
            "log_path": self.log_path,
            "error": self.error,
            "profiling": self.profile,
            "trace_out_dir": self.out_dir if self.profile else None,
            "kernel_trace": ("session-level (server runs under rocprofv3; "
                             "the trace spans the whole server lifetime, not one turn)"
                             if self.profile else "not capturing"),
            "per_turn_trace": False,
            "per_turn_trace_reason": attach_why if not attach_ok else None,
        }


async def stop_managed(server: ManagedServer | None) -> None:
    if server and server.running:
        await asyncio.to_thread(server.stop)