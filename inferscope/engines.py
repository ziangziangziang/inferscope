"""llama.cpp and vLLM Prometheus adapters.

Both engines expose a /metrics endpoint in the standard exposition format, so one
parser serves both; each adapter maps the family names it knows into a common
normalized dict. Rates (tokens/s) are derived as deltas between polls — that is
`derived`, not `measured`, and is labelled as such. A metric an engine version
renamed or dropped simply comes back None; the adapter never substitutes a value.

Nothing here starts or restarts a server. It only GETs a metrics URL.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import Provenance, Sample, Source, now_mono_ns, now_wall_ns


@dataclass
class AdapterResult:
    available: bool
    detail: str = ""
    state: str = "ok"          # ok | not_configured | unreachable | malformed
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"available": self.available, "detail": self.detail,
                "state": self.state, "metrics": self.metrics}


def _parse_prom(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse exposition text into {family: [{labels, value}], ...}.

    Uses prometheus_client's parser; falls back to a tiny manual line parser so a
    missing dependency or a slightly-off response never crashes collection.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        from prometheus_client.parser import text_string_to_metric_families
        for fam in text_string_to_metric_families(text):
            samples = out.setdefault(fam.name, [])
            for s in fam.samples:
                samples.append({"labels": dict(s.labels), "value": s.value,
                                "suffix": s.name[len(fam.name):]})
        return out
    except Exception:  # noqa: BLE001
        pass
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, val = line.rpartition(" ")
        try:
            v = float(val)
        except ValueError:
            continue
        name = head.split("{", 1)[0].strip()
        if not name:
            continue
        out.setdefault(name, []).append({"labels": {}, "value": v, "suffix": ""})
    return out


def _sum_family(fams: dict, name: str, suffix: str = "_sum") -> Optional[float]:
    for s in fams.get(name, []):
        if s["suffix"] == suffix:
            return s["value"]
    # fallback: a flat `name_sum` family (no TYPE line / non-grouping parser)
    for s in fams.get(name + suffix, []):
        if s["suffix"] == "":
            return s["value"]
    return None


def _count_family(fams: dict, name: str) -> Optional[float]:
    for s in fams.get(name, []):
        if s["suffix"] == "_count":
            return s["value"]
    for s in fams.get(name + "_count", []):
        if s["suffix"] == "":
            return s["value"]
    return None


def _gauge(fams: dict, name: str) -> Optional[float]:
    for s in fams.get(name, []):
        if s["suffix"] == "":
            return s["value"]
    return None


def _counter_total(fams: dict, name: str) -> Optional[float]:
    # A counter declared as `name` in the TYPE line often appears as a family
    # `name_total` (prometheus_client splits these), so check both spellings.
    vals: list[float] = []
    for fam in (name, name + "_total"):
        for s in fams.get(fam, []):
            if s["suffix"] in ("", "_total"):
                vals.append(s["value"])
    return sum(vals) if vals else None


class _BaseAdapter:
    source: Source
    name: str

    def __init__(self, url: str):
        self.url = url
        self._last: dict[str, tuple[float, float]] = {}  # key -> (value, mono_ns)

    def _rate(self, key: str, value: Optional[float]) -> Optional[float]:
        """Per-second rate from a monotonically increasing counter."""
        if value is None:
            return None
        now = now_mono_ns()
        prev = self._last.get(key)
        self._last[key] = (value, now)
        if not prev:
            return None
        pv, pt = prev
        dt = (now - pt) / 1e9
        if dt <= 0:
            return None
        return max(0.0, (value - pv) / dt)

    def scrape(self) -> AdapterResult:
        if not self.url:
            return AdapterResult(False, "no metrics URL configured", "not_configured")
        try:
            import httpx
            r = httpx.get(self.url, timeout=5.0)
            if r.status_code != 200:
                return AdapterResult(False, f"HTTP {r.status_code}", "unreachable")
            fams = _parse_prom(r.text)
        except Exception as e:  # noqa: BLE001
            return AdapterResult(False, f"{type(e).__name__}: {e}", "unreachable")
        if not fams:
            return AdapterResult(False, "no metrics parsed", "malformed")
        return AdapterResult(True, f"{len(fams)} families", "ok", self.map_metrics(fams))

    def map_metrics(self, fams: dict) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_sample(self, res: AdapterResult, session_id: str) -> Optional[Sample]:
        if not res.available:
            return None
        m = res.metrics
        return Sample(
            session_id=session_id, ts_wall_ns=now_wall_ns(), ts_mono_ns=now_mono_ns(),
            source=self.source, provenance=Provenance.MEASURED,
            active_requests=m.get("active_requests"),
            queue_depth=m.get("queue_depth"),
            tokens_per_s=m.get("gen_tokens_per_s"),
            extra={"engine_metrics": {k: v for k, v in m.items()
                                      if k != "active_requests"}},
        )


class LlamaCppAdapter(_BaseAdapter):
    name = "llamacpp"
    source = Source.LLAMA_CPP

    # Metric names have moved across llama.cpp versions; accept the current
    # (measured against a live 2026 build) and the older spellings.
    _GEN = ("llamacpp:tokens_predicted", "llamacpp:generated_tokens")
    _PROMPT = ("llamacpp:prompt_tokens",)
    _TTFT = ("llamacpp:prompt_tokens_seconds", "llamacpp:time_to_first_token_seconds")
    _TPOT = ("llamacpp:predicted_tokens_seconds", "llamacpp:time_per_output_token_seconds")

    def _first(self, fams, names, fn):
        for n in names:
            v = fn(fams, n)
            if v is not None:
                return v
        return None

    def map_metrics(self, fams: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["active_requests"] = _gauge(fams, "llamacpp:requests_processing")
        out["queue_depth"] = _gauge(fams, "llamacpp:requests_deferred")
        out["busy_slots_per_decode"] = _gauge(fams, "llamacpp:n_busy_slots_per_decode")
        p_tok = self._first(fams, self._PROMPT, _counter_total)
        g_tok = self._first(fams, self._GEN, _counter_total)
        out["prompt_tokens_total"] = p_tok
        out["gen_tokens_total"] = g_tok
        out["prompt_tokens_per_s"] = self._rate("llama_prompt", p_tok)
        out["gen_tokens_per_s"] = self._rate("llama_gen", g_tok)
        for nm in ("llamacpp:kvcache_usage_pct", "llamacpp:kv_cache_usage"):
            v = _gauge(fams, nm)
            if v is not None:
                out["kv_cache_usage"] = v
                break
        # Histogram means (client-observed timing, not kernel timing).
        for key, names in (("ttft_mean_s", self._TTFT), ("tpot_mean_s", self._TPOT)):
            c = self._first(fams, names, _count_family)
            s = self._first(fams, names, _sum_family)
            if c and s and c > 0:
                out[key] = s / c
        out["_provenance"] = {"active_requests": "measured",
                              "gen_tokens_per_s": "derived",
                              "prompt_tokens_per_s": "derived",
                              "ttft_mean_s": "derived", "tpot_mean_s": "derived"}
        return {k: v for k, v in out.items() if v is not None}


class VLLMAdapter(_BaseAdapter):
    name = "vllm"
    source = Source.VLLM

    def map_metrics(self, fams: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["active_requests"] = _gauge(fams, "vllm:num_requests_running")
        out["queue_depth"] = _gauge(fams, "vllm:num_requests_waiting")
        out["preemptions_total"] = _counter_total(fams, "vllm:num_preemptions")
        p_tok = _counter_total(fams, "vllm:prompt_tokens")
        g_tok = _counter_total(fams, "vllm:generation_tokens")
        out["prompt_tokens_total"] = p_tok
        out["gen_tokens_total"] = g_tok
        out["prompt_tokens_per_s"] = self._rate("vllm_prompt", p_tok)
        out["gen_tokens_per_s"] = self._rate("vllm_gen", g_tok)
        kv = _gauge(fams, "vllm:gpu_cache_usage")
        if kv is None:
            kv = _gauge(fams, "vllm:gpu_cache_usage_perc")
        out["kv_cache_usage"] = kv
        ttft_c = _count_family(fams, "vllm:time_to_first_token_seconds")
        ttft_s = _sum_family(fams, "vllm:time_to_first_token_seconds")
        if ttft_c and ttft_s and ttft_c > 0:
            out["ttft_mean_s"] = ttft_s / ttft_c
        itp_c = _count_family(fams, "vllm:inter_token_latency_seconds")
        itp_s = _sum_family(fams, "vllm:inter_token_latency_seconds")
        if itp_c and itp_s and itp_c > 0:
            out["itl_mean_s"] = itp_s / itp_c
        e2e_c = _count_family(fams, "vllm:e2e_request_latency_seconds")
        e2e_s = _sum_family(fams, "vllm:e2e_request_latency_seconds")
        if e2e_c and e2e_s and e2e_c > 0:
            out["e2e_mean_s"] = e2e_s / e2e_c
        out["_provenance"] = {"active_requests": "measured",
                              "gen_tokens_per_s": "derived",
                              "ttft_mean_s": "derived",
                              "itl_mean_s": "derived"}
        return {k: v for k, v in out.items() if v is not None}
