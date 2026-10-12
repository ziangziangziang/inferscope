from inferscope.diffusers import PipelineRecorder, instrument
from inferscope.schema import Event, EventKind


def test_recorder_emits_stage_event():
    out: list[Event] = []
    rec = PipelineRecorder("s", "m", out.append, mem_fn=lambda: 12345)
    assert rec.stage("text_encoder", lambda x: x * 2, 21) == 42
    assert len(out) == 1
    e = out[0]
    assert e.kind == EventKind.PIPELINE_STAGE and e.component == "text_encoder"
    assert e.provenance.value == "measured"
    assert e.duration_us is not None and e.duration_us >= 0
    assert e.extra["ok"] is True


def test_recorder_records_failure_but_still_emits():
    out: list[Event] = []
    rec = PipelineRecorder("s", "m", out.append)
    def boom():
        raise ValueError("x")
    try:
        rec.stage("unet", boom)
    except ValueError:
        pass
    assert len(out) == 1 and out[0].extra["ok"] is False


def test_step_and_peak_events():
    out: list[Event] = []
    rec = PipelineRecorder("s", "m", out.append, mem_fn=lambda: 999)
    rec.step(0, 120.0); rec.step(1, 130.0)
    rec.finish()
    kinds = [e.kind for e in out]
    assert kinds.count(EventKind.DENOISE_STEP) == 2
    assert EventKind.ALLOC in kinds
    assert out[-1].bytes_alloc == 999


def test_instrument_wrapper_times_call_and_steps():
    class FakePipeline:
        def __init__(self): self.calls = 0
        def __call__(self, callback_on_step_end=None, **kw):
            # simulate 3 denoise steps invoking the callback
            for step in range(3):
                if callback_on_step_end:
                    callback_on_step_end(self, step, step, {})
            self.calls += 1
            return "image"
    out: list[Event] = []
    wrapped = instrument(FakePipeline(), "s", "m", out.append)
    assert wrapped(prompt="a cat") == "image"
    kinds = [e.kind for e in out]
    assert EventKind.PIPELINE_STAGE in kinds          # whole-call timing
    assert kinds.count(EventKind.DENOISE_STEP) == 3    # one measured duration per step


def test_wrapper_degrades_without_callback_support():
    class Plain:
        def __call__(self, **kw):
            return "x"
    out: list[Event] = []
    wrapped = instrument(Plain(), "s", "m", out.append)
    assert wrapped() == "x"
    assert any(e.kind == EventKind.PIPELINE_STAGE for e in out)
