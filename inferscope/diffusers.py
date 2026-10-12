"""Diffusers instrumentation — an explicit wrapper, never a global monkey-patch.

Diffusers pipelines accept ``callback_on_step_end`` and expose components, so we
capture a generation run by *wrapping* the call and registering a step callback,
not by patching the library. The recorder core is plain Python and testable with
a stub pipeline; torch/diffusers are imported lazily so this module loads (and
the app boots) on a machine without them.

Emitted events (all provenance=measured, source=diffusers):
  pipeline_stage  — one per wrapped component call, with duration + memory delta
  denoise_step    — one per scheduler step via the step callback
  alloc           — peak device memory at the end of the run
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Optional

from .schema import Event, EventKind, Provenance, Source, now_mono_ns, now_wall_ns

Emit = Callable[[Event], None]


def _eid(*parts) -> str:
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return "is-" + h[:16]


class PipelineRecorder:
    """Records stages/steps/memory into Events. Usable standalone (tests) or via
    :func:`instrument` around a real pipeline."""

    def __init__(self, session_id: str, model_id: str, emit: Emit,
                 mem_fn: Optional[Callable[[], int]] = None):
        self.session_id = session_id
        self.model_id = model_id
        self.emit = emit
        self.mem_fn = mem_fn          # returns device bytes allocated, or None
        self._t0 = now_mono_ns()
        self._peak_mem = 0

    def _mem(self) -> Optional[int]:
        if not self.mem_fn:
            return None
        try:
            m = int(self.mem_fn())
            self._peak_mem = max(self._peak_mem, m)
            return m
        except Exception:  # noqa: BLE001
            return None

    def stage(self, name: str, fn: Callable[..., Any], *a, **kw) -> Any:
        """Time a component/stage call and emit a pipeline_stage event."""
        start = now_mono_ns()
        mem_before = self._mem()
        try:
            out = fn(*a, **kw)
            ok = True
        except Exception:
            ok = False
            raise
        finally:
            end = now_mono_ns()
            self.emit(Event(
                event_id=_eid(self.session_id, name, start), session_id=self.session_id,
                kind=EventKind.PIPELINE_STAGE, source=Source.DIFFUSERS,
                provenance=Provenance.MEASURED,
                ts_wall_ns=now_wall_ns(), ts_mono_ns=start,
                engine="diffusers", model_id=self.model_id, component=name,
                start_ns=start, end_ns=end, duration_us=(end - start) / 1000, unit="us",
                extra={"ok": ok, "mem_before": mem_before, "mem_after": self._mem()},
            ))
        return out

    def step(self, step_idx: int, duration_us: float, extra: dict | None = None) -> None:
        self.emit(Event(
            event_id=_eid(self.session_id, "step", step_idx), session_id=self.session_id,
            kind=EventKind.DENOISE_STEP, source=Source.DIFFUSERS,
            provenance=Provenance.MEASURED,
            ts_wall_ns=now_wall_ns(), ts_mono_ns=now_mono_ns(),
            engine="diffusers", model_id=self.model_id, step_idx=step_idx,
            duration_us=duration_us, unit="us", extra=extra or {},
        ))

    def finish(self) -> None:
        self._mem()  # sample once more so peak reflects the end of the run
        if self._peak_mem:
            self.emit(Event(
                event_id=_eid(self.session_id, "peak"), session_id=self.session_id,
                kind=EventKind.ALLOC, source=Source.DIFFUSERS, provenance=Provenance.MEASURED,
                ts_wall_ns=now_wall_ns(), ts_mono_ns=now_mono_ns(),
                engine="diffusers", model_id=self.model_id,
                bytes_alloc=self._peak_mem, extra={"what": "peak device memory"},
            ))


def _torch_mem_fn():
    try:
        import torch
        if torch.cuda.is_available():
            return lambda: torch.cuda.memory_allocated()
    except Exception:  # noqa: BLE001
        pass
    return None


class InstrumentedPipeline:
    """Thin wrapper: ``out = wrapped(prompt=..., num_inference_steps=20)``.

    Times the whole call, registers a denoise-step callback, and records peak
    device memory. Works with any diffusers pipeline that supports
    ``callback_on_step_end``; degrades to whole-call timing if it does not.
    """

    def __init__(self, pipeline, session_id: str, model_id: str, emit: Emit):
        self._p = pipeline
        self.rec = PipelineRecorder(session_id, model_id, emit, mem_fn=_torch_mem_fn())
        self._last_step_ns: Optional[int] = None

    def _on_step_end(self, pipeline, step: int, timestep, callback_data):
        now = now_mono_ns()
        if self._last_step_ns is not None:
            self.rec.step(step, (now - self._last_step_ns) / 1000)
        self._last_step_ns = now
        return callback_data

    def __call__(self, *a, **kw):
        self._last_step_ns = now_mono_ns()
        # Try to register the step callback; if the pipeline does not accept it,
        # fall back to whole-call timing rather than failing the run.
        try:
            out = self.rec.stage("pipeline", lambda: self._p(
                *a, callback_on_step_end=self._on_step_end,
                callback_on_step_end_tensor_inputs=["latents"], **kw))
        except TypeError:
            out = self.rec.stage("pipeline", lambda: self._p(*a, **kw))
        self.rec.finish()
        return out

    def __getattr__(self, name):
        return getattr(self._p, name)


def instrument(pipeline, session_id: str, model_id: str, emit: Emit) -> InstrumentedPipeline:
    """Wrap a diffusers pipeline for measurement. The only supported entry point."""
    return InstrumentedPipeline(pipeline, session_id, model_id, emit)
