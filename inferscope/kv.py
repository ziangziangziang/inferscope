"""KV-cache memory model.

Baseline for conventional full-attention caching:

    KV bytes = 2 * layers * batch * context * kv_heads * head_dim * bytes_per_element

The 2 is K and V. Everything that bends this — grouped-query attention,
sliding-window attention, cache quantisation, shared prefixes — is applied as an
explicit, documented adjustment so the estimate stays honest and the reasoning is
visible, never folded silently into the number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2,
    "float16": 2, "fp16": 2,
    "float8_e4m3fn": 1, "fp8": 1,
    "int8": 1, "uint8": 1,
    "q8_0": 1, "q4_0": 0.5 + 0.125,   # 4-bit + scale/zero overhead, ~0.5625 B/elem
    "q4_0": 0.5,
    "q8": 1, "q4": 0.5,
}


def dtype_bytes(dtype: str) -> float:
    return DTYPE_BYTES.get(dtype.lower(), 2.0)


@dataclass
class KVInputs:
    n_layers: int
    context: int
    batch: int = 1
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"
    # Adjustments (all default to "plain full attention").
    sliding_window: Optional[int] = None      # attend to last W tokens only
    swa_layer_fraction: float = 1.0           # fraction of layers that are SWA
    shared_prefix: Optional[int] = None       # tokens cached once across batch
    quant_scale_overhead: float = 0.0         # extra bytes/elem for scale/zero

    def validate(self) -> list[str]:
        errs = []
        if self.n_layers <= 0:
            errs.append("n_layers must be positive")
        if self.context <= 0:
            errs.append("context must be positive")
        if self.batch <= 0:
            errs.append("batch must be positive")
        if self.kv_heads <= 0:
            errs.append("kv_heads must be positive")
        if self.head_dim <= 0:
            errs.append("head_dim must be positive")
        return errs


@dataclass
class KVResult:
    bytes: int
    gib: float
    baseline_bytes: int
    breakdown: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    formula: str = ""


def kv_cache_bytes(inp: KVInputs) -> KVResult:
    """Return measured-style KV memory with the derivation exposed."""
    bpe = dtype_bytes(inp.dtype) + inp.quant_scale_overhead
    # Plain baseline, before any adjustment.
    baseline = int(round(2 * inp.n_layers * inp.batch * inp.context
                         * inp.kv_heads * inp.head_dim * bpe))

    notes = [
        f"baseline = 2·L·B·C·Hkv·D·bytes = 2·{inp.n_layers}·{inp.batch}·"
        f"{inp.context}·{inp.kv_heads}·{inp.head_dim}·{bpe:g}",
        f"dtype {inp.dtype} = {dtype_bytes(inp.dtype):g} B/elem"
        + (f" (+{inp.quant_scale_overhead:g} scale overhead)" if inp.quant_scale_overhead else ""),
    ]
    breakdown = {"baseline": baseline}
    total = float(baseline)

    if inp.sliding_window and inp.sliding_window < inp.context and inp.swa_layer_fraction < 1.0:
        # SWA layers only keep a window of tokens; the rest keep full context.
        eff_context = (inp.swa_layer_fraction * inp.sliding_window
                       + (1 - inp.swa_layer_fraction) * inp.context)
        adjusted = int(round(2 * inp.n_layers * inp.batch * eff_context
                             * inp.kv_heads * inp.head_dim * bpe))
        breakdown["sliding_window"] = adjusted - baseline
        total = float(adjusted)
        notes.append(
            f"sliding-window attention: {inp.swa_layer_fraction:.0%} of layers keep a "
            f"{inp.sliding_window}-token window → effective context {eff_context:.0f}")
    elif inp.sliding_window and inp.sliding_window < inp.context:
        adjusted = int(round(2 * inp.n_layers * inp.batch * inp.sliding_window
                             * inp.kv_heads * inp.head_dim * bpe))
        breakdown["sliding_window"] = adjusted - baseline
        total = float(adjusted)
        notes.append(f"all layers sliding-window {inp.sliding_window} tokens")

    if inp.shared_prefix and inp.shared_prefix > 0 and inp.batch > 1:
        # A shared prefix is stored once, not once per sequence.
        saved = int(round(2 * inp.n_layers * (inp.batch - 1) * inp.shared_prefix
                          * inp.kv_heads * inp.head_dim * bpe))
        breakdown["shared_prefix_saving"] = -saved
        total -= saved
        notes.append(
            f"shared prefix of {inp.shared_prefix} tokens cached once across "
            f"batch {inp.batch} (saves {saved/2**30:.2f} GiB)")

    total = max(0, int(round(total)))
    return KVResult(
        bytes=total,
        gib=total / 2**30,
        baseline_bytes=baseline,
        breakdown=breakdown,
        notes=notes,
        formula="2 * layers * batch * context * kv_heads * head_dim * bytes_per_element",
    )


def gqa_note(n_heads: int, kv_heads: int) -> str:
    """Explain grouped-query attention's effect on the KV footprint."""
    if kv_heads >= n_heads:
        return "multi-head attention: every query head has its own KV head (no GQA saving)."
    ratio = n_heads / max(1, kv_heads)
    return (f"grouped-query attention: {n_heads} query heads share {kv_heads} KV heads "
            f"({ratio:.1f}:1), so KV is {kv_heads}/{n_heads} of the MHA footprint.")
