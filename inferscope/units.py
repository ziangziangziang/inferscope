"""The catalogue of loadable units.

A *unit* is one engine paired with one specific model: `whisper.cpp` +
`ggml-base.en`, `llama.cpp` + `Qwen3.8-27B`, `diffusers` + `SDXL`. Everything
the UI loads, animates and reports is keyed on a unit, which is why swapping
engines and swapping models are the same operation in the scene — the block you
see is a unit, not a hardcoded llama.cpp slab.

Models are discovered on disk rather than configured one at a time, so adding a
checkpoint is a file drop, not a redeploy. An engine with nothing installed
still appears in the catalogue, marked unavailable with the reason, so the UI can
say *why* rather than silently omitting a capability.

The language-model server is included here as an ordinary unit even though
`chat.py` still owns its process lifecycle (it has the /props identity check and
the profiler wrapping). Unifying those two managers is a bigger change than this
catalogue needs, and pretending otherwise would be worse than saying so.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UnitPart:
    """One component of a model, sized by its real share of the weights."""
    key: str
    label: str
    weight: float = 1.0          # relative; normalised across parts


@dataclass
class Unit:
    key: str                     # "asr/ggml-base.en"
    engine: str                  # "asr"
    engine_label: str            # "whisper.cpp"
    model_label: str             # "ggml-base.en"
    model_path: str
    port: int
    binary: str
    size_bytes: int = 0
    params: int = 0
    parts: list[UnitPart] = field(default_factory=list)
    note: str = ""
    engine_role: str = ""
    problem: str = ""
    configured: bool = False
    managed: bool = True         # False for the language engine (chat.py owns it)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "engine": self.engine,
            "engine_label": self.engine_label, "model_label": self.model_label,
            "model_path": self.model_path, "port": self.port, "binary": self.binary,
            "size_bytes": self.size_bytes, "params": self.params,
            "parts": [{"key": p.key, "label": p.label, "weight": p.weight} for p in self.parts],
            "note": self.note, "engine_role": self.engine_role,
            "problem": self.problem,
            "configured": self.configured, "managed": self.managed,
        }


#: filename patterns per engine, so a directory of unrelated checkpoints does not
#: turn into a catalogue full of nonsense.
PATTERNS: dict[str, tuple[str, ...]] = {
    "asr": ("*.bin", "*.gguf"),
    "tts": ("*.onnx", "*.pth", "*.pt", "*.ckpt"),
    "image": (),                 # directories
    "video": (),
    "llm": ("*.gguf",),
}


def _dir_size(path: str, cap: int = 1 << 30) -> int:
    """Bytes under a file or directory, giving up after *cap*."""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    continue
                if total > cap:
                    return total
        return total
    except OSError:
        return 0


def _match(path: str, engine: str) -> bool:
    import fnmatch
    pats = PATTERNS.get(engine, ())
    if not pats:
        return True
    base = os.path.basename(path)
    return any(fnmatch.fnmatch(base, p) for p in pats)


def discover(cfg, specs) -> list[Unit]:
    """Every (engine, model) pair this machine can actually load."""
    from .enginectl import APP_ROOT            # noqa: F401  (documents the link)

    units: list[Unit] = []
    seen: set[str] = set()

    for spec in specs:
        model_path = spec.model
        models_dir = getattr(cfg, f"{spec.key}_models_dir", "") or ""
        candidates: list[str] = []
        if model_path and os.path.exists(model_path):
            candidates.append(model_path)
        if models_dir and os.path.isdir(models_dir):
            for name in sorted(os.listdir(models_dir)):
                full = os.path.join(models_dir, name)
                if _match(full, spec.key) and os.path.exists(full):
                    candidates.append(full)
        # de-duplicate while preserving order
        uniq: list[str] = []
        for c in candidates:
            rp = os.path.realpath(c)
            if rp not in seen:
                seen.add(rp)
                uniq.append(c)

        if not uniq:
            units.append(Unit(
                key=f"{spec.key}/-", engine=spec.key, engine_label=spec.label,
                model_label="no model installed", model_path="", port=spec.port,
                binary=spec.binary, engine_role=spec.purpose,
                problem=("no binary configured" if not spec.binary
                         else (spec.configured()[1] or "no model installed")),
                configured=False))
            continue

        for path in uniq:
            label = os.path.basename(path.rstrip("/"))
            if os.path.isdir(path):
                label = os.path.basename(path.rstrip("/"))
            units.append(Unit(
                key=f"{spec.key}/{label}", engine=spec.key, engine_label=spec.label,
                model_label=label, model_path=path, port=spec.port,
                binary=spec.binary, size_bytes=_dir_size(path),
                engine_role=spec.purpose,
                configured=bool(spec.binary) and os.access(spec.binary, os.X_OK),
                problem="" if spec.binary and os.access(spec.binary, os.X_OK)
                        else (spec.configured()[1] or "no binary configured"),
            ))

    units.extend(_llm_units(cfg))
    return units


def _llm_units(cfg) -> list[Unit]:
    """The language-model server, expressed as units too.

    Not managed by `enginectl` — `chat.py` owns that process — but it still has
    to appear in the catalogue or "choose a model" would be blind to the biggest
    model on the box.
    """
    if not cfg.chat_model_path or not os.path.exists(cfg.chat_model_path):
        return []
    label = os.path.basename(cfg.chat_model_path)
    return [Unit(
        key=f"llm/{label}", engine="llm", engine_label="llama.cpp",
        model_label=label, model_path=cfg.chat_model_path,
        port=cfg.chat_port, binary=cfg.chat_server_bin,
        engine_role="reads words, predicts the next one",
        size_bytes=_dir_size(cfg.chat_model_path),
        configured=os.access(cfg.chat_server_bin, os.X_OK),
        managed=False,
        note="loads through Run & capture",
    )]