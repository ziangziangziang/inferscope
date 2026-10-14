"""Read a GGUF checkpoint's own metadata.

llama.cpp serves quantised GGUF files, and those files carry their architecture:
the model name, the family it belongs to, and the dimensions that define the
stack. Reading them is the difference between describing the checkpoint on the
GPU and describing a *different* checkpoint that happens to share a name.

That distinction matters here, because this catalogue also sees full-precision
checkpoints in a model registry, and picking one of those when the user chose a
quantised GGUF of a different model is simply the wrong answer presented
confidently.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

# GGUF metadata value types. Types 8 and 9 are strings and arrays; everything
# else is a fixed-width scalar.
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    10: ("h", 2), 11: ("i", 4), 12: ("?", 8),
}
_STRING = 8
_ARRAY = 9


def read_kv(path: str | Path) -> dict[str, Any]:
    """The metadata block of a GGUF file, or {} if it is not one."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            _version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
            out: dict[str, Any] = {}
            for _ in range(min(n_kv, 4096)):
                key = _read_str(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                try:
                    out[key] = _read_value(f, vtype)
                except (struct.error, EOFError, MemoryError, ValueError):
                    break
            return out
    except (OSError, struct.error, ValueError, IndexError):
        return {}


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 20:                      # nonsense length: stop rather than hang
        raise ValueError("implausible string length")
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, vtype: int) -> Any:
    if vtype == _STRING:
        return _read_str(f)
    if vtype == _ARRAY:
        (elem,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if count > 1 << 22:
            raise ValueError("implausible array length")
        if elem == _STRING:
            return [_read_str(f) for _ in range(count)]
        for _ in range(count):
            _read_value(f, elem)
        return None
    fmt, size = _SCALARS[vtype]
    return struct.unpack("<" + fmt, f.read(size))[0]


def arch_name(kv: dict[str, Any]) -> str:
    """``qwen3vl`` for ``qwen3vl.block_count``, and so on."""
    return str(kv.get("general.architecture", "") or "")


def g(kv: dict[str, Any], name: str, default: int = 0) -> int:
    """One integer from the metadata, with a default for absent keys."""
    try:
        return int(kv.get(name, default) or default)
    except (TypeError, ValueError):
        return default


def family_of(kv: dict[str, Any]) -> str:
    """``image-text-to-text`` beats guessing from the architecture name."""
    tags = kv.get("general.tags") or []
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, str) and "-" in t and "text" in t:
                return t
    return str(kv.get("general.type", "") or "llm")


def is_vision(kv: dict[str, Any]) -> bool:
    return "image" in family_of(kv) or "vision" in family_of(kv)