"""Configuration from environment variables with safe, non-secret defaults.

Everything is loopback-bound by default and no endpoint requires a credential.
The application must start and be useful with none of these set (demo mode).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default data dir lives next to the package so the app is self-contained.
_DEFAULT_DATA = Path(os.environ.get("INFERSCOPE_DATA",
                                    str(Path(__file__).resolve().parent.parent / "data")))


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(_env("INFERSCOPE_DATA", str(_DEFAULT_DATA))))
    host: str = field(default_factory=lambda: _env("INFERSCOPE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("INFERSCOPE_PORT", 8090))

    # Telemetry cadence. ~1 s default per the brief; keep it configurable.
    poll_interval_s: float = field(default_factory=lambda: _env_float("INFERSCOPE_POLL_S", 1.0))

    # Engine metric endpoints. Empty == adapter reports "not configured",
    # which is a first-class, honest state — not an error.
    llamacpp_metrics_url: str = field(default_factory=lambda: _env("INFERSCOPE_LLAMACPP_METRICS", ""))
    vllm_metrics_url: str = field(default_factory=lambda: _env("INFERSCOPE_VLLM_METRICS", ""))

    # Model config dirs used to derive architecture, comma separated.
    model_dirs: str = field(default_factory=lambda: _env("INFERSCOPE_MODEL_DIRS", "/workspace/models"))

    # Profiling is opt-in and bounded. Never continuous, never on foreign procs.
    profiling_enabled: bool = field(default_factory=lambda: _env_bool("INFERSCOPE_PROFILING", False))
    capture_max_s: int = field(default_factory=lambda: _env_int("INFERSCOPE_CAPTURE_MAX_S", 20))

    # Retention / sampling for high-volume traces.
    retention_sessions: int = field(default_factory=lambda: _env_int("INFERSCOPE_RETENTION_SESSIONS", 24))
    sample_stride: int = field(default_factory=lambda: _env_int("INFERSCOPE_SAMPLE_STRIDE", 1))

    # Demo mode boots the UI with clearly-labelled synthetic data.
    demo: bool = field(default_factory=lambda: _env_bool("INFERSCOPE_DEMO", True))

    # ---- chat / per-turn profiling -------------------------------------
    # Point at an already-running OpenAI-compatible server, or let InferScope
    # start its own (managed mode). Managed mode is what enables per-turn
    # kernel tracing, because only a direct parent may ptrace the server.
    chat_base_url: str = field(default_factory=lambda: _env("INFERSCOPE_CHAT_BASE_URL", ""))
    chat_manage: bool = field(default_factory=lambda: _env_bool("INFERSCOPE_CHAT_MANAGE", True))
    chat_server_bin: str = field(default_factory=lambda: _env(
        "INFERSCOPE_CHAT_SERVER_BIN",
        "/workspace/build/llama.cpp/build-hip/bin/llama-server"))
    chat_model_path: str = field(default_factory=lambda: _env(
        "INFERSCOPE_CHAT_MODEL", ""))
    chat_port: int = field(default_factory=lambda: _env_int("INFERSCOPE_CHAT_PORT", 8081))
    chat_ctx: int = field(default_factory=lambda: _env_int("INFERSCOPE_CHAT_CTX", 8192))
    chat_parallel: int = field(default_factory=lambda: _env_int("INFERSCOPE_CHAT_PARALLEL", 1))
    chat_max_tokens: int = field(default_factory=lambda: _env_int("INFERSCOPE_CHAT_MAX_TOKENS", 512))
    # Per-turn kernel capture is opt-in and bounded by capture_max_s.
    chat_profile_default: bool = field(default_factory=lambda: _env_bool("INFERSCOPE_CHAT_PROFILE", False))

    # ---- other inference engines ----------------------------------------
    # One GPU, one heavy engine resident at a time; loading one offloads
    # whatever held it. Paths are configurable per engine so a model that does
    # not fit on /workspace can live on the larger volume.
    asr_bin: str = field(default_factory=lambda: _env(
        "INFERSCOPE_ASR_BIN",
        "/workspace/build/whisper.cpp/build-hip/bin/whisper-server"))
    asr_model: str = field(default_factory=lambda: _env(
        "INFERSCOPE_ASR_MODEL", "/workspace/models/whisper/ggml-base.en.bin"))
    asr_port: int = field(default_factory=lambda: _env_int("INFERSCOPE_ASR_PORT", 8082))
    asr_threads: int = field(default_factory=lambda: _env_int("INFERSCOPE_ASR_THREADS", 8))

    tts_bin: str = field(default_factory=lambda: _env("INFERSCOPE_TTS_BIN", ""))
    tts_model: str = field(default_factory=lambda: _env("INFERSCOPE_TTS_MODEL", ""))
    tts_port: int = field(default_factory=lambda: _env_int("INFERSCOPE_TTS_PORT", 8083))
    tts_voice: str = field(default_factory=lambda: _env("INFERSCOPE_TTS_VOICE", ""))

    image_bin: str = field(default_factory=lambda: _env("INFERSCOPE_IMAGE_BIN", ""))
    image_model: str = field(default_factory=lambda: _env("INFERSCOPE_IMAGE_MODEL", ""))
    image_port: int = field(default_factory=lambda: _env_int("INFERSCOPE_IMAGE_PORT", 8084))

    video_bin: str = field(default_factory=lambda: _env("INFERSCOPE_VIDEO_BIN", ""))
    video_model: str = field(default_factory=lambda: _env("INFERSCOPE_VIDEO_MODEL", ""))
    video_port: int = field(default_factory=lambda: _env_int("INFERSCOPE_VIDEO_PORT", 8085))

    @property
    def bind_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def ssh_tunnel_hint(self) -> str:
        # The remote is reached over SSH; the app binds loopback there.
        return (f"ssh -N -L {self.port}:localhost:{self.port} "
                f"-p 31699 root@36.150.116.206")

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        c.data_dir.mkdir(parents=True, exist_ok=True)
        return c
