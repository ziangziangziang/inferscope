"""Engine control: load and offload inference engines, one at a time.

This is the layer the UI drives. It is deliberately separate from
`engines.py`, which only *reads* Prometheus metrics from an engine that is
already running, and from `chat.py`, which owns the language-model server.

The model it encodes is the one the machine actually has: **one GPU, one heavy
engine resident at a time**. Loading an engine offloads whatever was there
first, and the UI is told that happened rather than being left to guess why VRAM
disappeared. Engines that are cheap enough to coexist (none today) could opt out
via ``exclusive``.

Every engine is a local HTTP service on loopback with the same contract:

    GET  {base}/health          -> 200 once it is serving
    POST {base}/<task-specific> -> the actual work

That uniformity is what lets one manager load a speech recogniser, a voice
synthesiser and a diffusion pipeline without knowing anything about any of them.
Adding an engine is one :class:`EngineSpec` and, when its request shape differs,
one branch in the API layer.

Ownership rules, unchanged from the language-model server and for the same
reason — a profiler has to attach to a pid we spawned:

* never adopt a listener we did not start;
* clear a listener only when it is *provably* our own leftover (see
  ``chat.reap_own_orphan``);
* otherwise refuse, and say which pid holds the port.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .chat import port_in_use, reap_own_orphan

log = logging.getLogger("inferscope.enginectl")

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_json(url: str, timeout: float = 4.0) -> tuple[int, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    ctx = None
    with contextlib.suppress(Exception):
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            body = r.read()
            try:
                return r.status, json.loads(body or b"null")
            except Exception:                              # noqa: BLE001
                return r.status, body.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:                                      # noqa: BLE001
        return 0, None


def _get_ok(url: str, timeout: float = 4.0) -> bool:
    st, _ = _get_json(url, timeout)
    return st == 200


@dataclass(frozen=True)
class EngineSpec:
    """One loadable inference engine."""
    key: str                      # "asr"
    label: str                    # "whisper.cpp"
    family: str                   # "speech" | "vision" | "language"
    purpose: str                  # one plain sentence, shown in the UI
    port: int
    binary: str                   # "" when not configured
    model: str                    # "" when not configured
    argv: Callable[["EngineSpec"], list[str]]
    #: path that answers 200 once the engine is serving
    health_path: str = "/health"
    #: call the engine's own readiness route; None means use health_path
    probe: Optional[Callable[["EngineSpec"], bool]] = None
    exclusive: bool = True        # offload others before loading this one
    #: extra config flags surfaced in the UI
    extra: dict[str, str] = field(default_factory=dict)
    hint: str = ""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def configured(self) -> tuple[bool, str]:
        """Whether this engine could actually be started, and why not."""
        if not self.binary:
            return False, "no binary configured"
        if not os.path.exists(self.binary):
            return False, f"binary not found: {self.binary}"
        if not os.access(self.binary, os.X_OK):
            return False, f"binary not executable: {self.binary}"
        if not self.model:
            return False, "no model configured"
        if not os.path.exists(self.model):
            return False, f"model not found: {self.model}"
        if self.key == "asr" and os.path.isdir(self.model):
            return False, f"model path is a directory: {self.model}"
        return True, ""

    def ready(self) -> bool:
        if self.probe is not None:
            return bool(self.probe(self))
        return _get_ok(self.base_url + self.health_path)

    def as_dict(self) -> dict[str, Any]:
        ok, why = self.configured()
        return {
            "key": self.key, "label": self.label, "family": self.family,
            "purpose": self.purpose, "port": self.port,
            "binary": self.binary, "model": self.model,
            "configured": ok, "problem": why,
            "ready": self.ready() if ok else False,
            "exclusive": self.exclusive,
            "extra": self.extra, "hint": self.hint,
        }


# --------------------------------------------------------------------------
# argv builders
# --------------------------------------------------------------------------

def _asr_argv(s: EngineSpec) -> list[str]:
    return [s.binary, "-m", s.model, "--host", "127.0.0.1",
            "--port", str(s.port), "-t", str(int(s.extra.get("threads", "8")))]


class EngineManager:
    """Owns at most one resident engine and the process behind it."""

    def __init__(self, specs: list[EngineSpec]):
        self._specs = {s.key: s for s in specs}
        self._proc: Optional[subprocess.Popen] = None
        self._loaded: Optional[str] = None
        self._started_at: Optional[float] = None
        self._log_path: str = ""
        self._last_error: str = ""
        self._evicted: Optional[str] = None

    # -- registry ---------------------------------------------------------
    def spec(self, key: str) -> Optional[EngineSpec]:
        return self._specs.get(key)

    def specs(self) -> list[EngineSpec]:
        return list(self._specs.values())

    # -- resident state ---------------------------------------------------
    @property
    def loaded(self) -> Optional[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._loaded
        return None

    def uptime(self) -> float:
        if self.loaded and self._started_at:
            return time.time() - self._started_at
        return 0.0

    def evicted_note(self) -> Optional[str]:
        return self._evicted

    def clear_note(self) -> None:
        self._evicted = None

    def _reap_if_dead(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            self._proc = None
            self._loaded = None
            self._started_at = None

    # -- lifecycle --------------------------------------------------------
    def offload(self, grace: float = 12.0) -> dict[str, Any]:
        """Stop the resident engine and wait for the port to actually free."""
        self._reap_if_dead()
        if self._proc is None:
            return {"stopped": False, "detail": "nothing was loaded"}
        key = self._loaded
        pid = self._proc.pid
        with contextlib.suppress(Exception):
            os.kill(pid, signal.SIGTERM)
        deadline = time.time() + grace
        while time.time() < deadline:
            if self._proc.poll() is not None:
                break
            time.sleep(0.2)
        if self._proc.poll() is None:
            with contextlib.suppress(Exception):
                os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=5)
        self._proc = None
        self._loaded = None
        self._started_at = None
        # the port can linger a moment behind the process
        for _ in range(20):
            if port_in_use(self._specs[key].port if key in self._specs else 0) is None:
                break
            time.sleep(0.25)
        return {"stopped": True, "engine": key, "pid": pid}

    def load(self, key: str, log_dir: str, wait: float = 180.0) -> dict[str, Any]:
        """Make *key* the resident engine, offloading whatever held the GPU."""
        self._reap_if_dead()
        spec = self._specs.get(key)
        if spec is None:
            return {"ok": False, "error": f"unknown engine: {key}"}
        ok, why = spec.configured()
        if not ok:
            return {"ok": False, "error": why}

        if self.loaded == key and spec.ready():
            return {"ok": True, "engine": key, "pid": self._proc.pid,
                    "already": True, "error": None}

        evicted = None
        if self.loaded and self.loaded != key and spec.exclusive:
            evicted = self.loaded
            self.offload()

        # clear an orphan of our own making (see chat.reap_own_orphan)
        reaped = reap_own_orphan(spec.port, app_root=APP_ROOT, model_path=spec.model,
                                 expect=os.path.basename(spec.binary))
        if reaped:
            log.warning("cleared orphaned %s (pid %s) on port %s",
                        spec.key, reaped, spec.port)

        busy = port_in_use(spec.port)
        if busy:
            return {"ok": False, "engine": key, "evicted": evicted,
                    "error": (f"port {spec.port} is held by pid {busy}, which "
                              f"InferScope does not own — stop it first")}

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, f"engine-{key}.log")
        out = open(self._log_path, "ab")
        try:
            self._proc = subprocess.Popen(
                spec.argv(spec), stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env={**os.environ, "LD_LIBRARY_PATH": _ld_path(spec)})
        except OSError as e:
            self._proc = None
            return {"ok": False, "engine": key, "evicted": evicted,
                    "error": f"could not start {spec.label}: {e}"}
        finally:
            with contextlib.suppress(Exception):
                out.close()

        self._loaded = key
        self._started_at = time.time()
        self._last_error = ""

        deadline = time.time() + wait
        while time.time() < deadline:
            if self._proc.poll() is not None:
                tail = _tail(self._log_path)
                self._proc = None
                self._loaded = None
                self._started_at = None
                return {"ok": False, "engine": key, "evicted": evicted,
                        "error": f"{spec.label} exited during startup"
                                  + (f": {tail}" if tail else "")}
            if spec.ready():
                return {"ok": True, "engine": key, "pid": self._proc.pid,
                        "evicted": evicted, "error": None}
            time.sleep(0.5)
        return {"ok": False, "engine": key, "evicted": evicted,
                "error": f"{spec.label} did not answer {spec.health_path} "
                         f"within {int(wait)}s"}

    # -- reporting --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        self._reap_if_dead()
        return {
            "loaded": self.loaded,
            "pid": self._proc.pid if self._proc is not None else None,
            "uptime_s": round(self.uptime(), 1),
            "log_path": self._log_path,
            "engines": [s.as_dict() for s in self._specs.values()],
        }


def _ld_path(spec: EngineSpec) -> str:
    """Shared libraries usually sit next to the binary."""
    d = os.path.dirname(spec.binary)
    return os.pathsep.join(filter(None, [d, os.environ.get("LD_LIBRARY_PATH", "")]))


def _tail(path: str, n: int = 400) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            return f.read().decode("utf-8", "replace").strip().splitlines()[-4:]
    except Exception:                                          # noqa: BLE001
        return []

# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

def build_engine_specs(cfg) -> list[EngineSpec]:
    """Every engine InferScope knows how to load.

    An engine with an empty binary/model path still appears, marked
    ``configured: false`` with the reason, so the UI can say *why* rather than
    silently omitting a capability the user was expecting.
    """
    specs: list[EngineSpec] = [
        EngineSpec(
            key="asr", label="whisper.cpp", family="speech",
            purpose="Transcribes speech audio to text, on the GPU.",
            port=cfg.asr_port, binary=cfg.asr_bin, model=cfg.asr_model,
            argv=_asr_argv, health_path="/health",
            extra={"threads": str(cfg.asr_threads)},
            hint="POST /inference with a multipart audio file.",
        ),
        EngineSpec(
            key="tts", label="piper", family="speech",
            purpose="Synthesises speech from text.",
            port=cfg.tts_port, binary=cfg.tts_bin, model=cfg.tts_model,
            argv=lambda s: [s.binary, "--model", s.model, "--port", str(s.port)],
            extra={"voice": cfg.tts_voice},
        ),
        EngineSpec(
            key="image", label="diffusers", family="vision",
            purpose="Generates still images from a text prompt.",
            port=cfg.image_port, binary=cfg.image_bin, model=cfg.image_model,
            argv=lambda s: [s.binary, "--model", s.model, "--port", str(s.port)],
        ),
        EngineSpec(
            key="video", label="h3-hip", family="vision",
            purpose="Generates short video clips from a text prompt.",
            port=cfg.video_port, binary=cfg.video_bin, model=cfg.video_model,
            argv=lambda s: [s.binary, "--model", s.model, "--port", str(s.port)],
        ),
    ]
    return specs
