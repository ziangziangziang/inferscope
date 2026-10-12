# InferScope

A local-first, interactive workbench for understanding how LLMs, vision-language
models, and image/video generators actually execute on real hardware. It links
three perspectives — **structure** (an interactive computational graph),
**time** (an execution timeline), and **resources** (CPU/RAM/GPU/VRAM) — so that
selecting a block, a decode step, or a moment updates every view together.

Its organising rule: **a value's origin is part of the value.** Every number is
tagged `measured`, `derived`, `estimated`, `simulated`, or `unavailable`, and the
UI never lets an architecture-only estimate or a missing reading masquerade as a
hardware measurement.

---

## Architecture

```
 collectors ┐                                   ┌─ SQLite  (sessions, experiments, health)
 engines    ├─► common Event/Sample schema ─────┤
 profiler   ┘   (provenance on every field)      └─ Parquet + DuckDB  (events, samples, arch)
                                                        │
                                              FastAPI  ─┤  REST + SSE  ─►  React/TS UI (SVG)
```

- **Backend** — Python 3.12, FastAPI, uvicorn. No new packages installed; uses the
  host's `/opt/venv` (fastapi, uvicorn, duckdb, pyarrow, psutil, prometheus-client,
  httpx, pydantic). `amdsmi` is used when importable, with sysfs/rocm-smi fallbacks.
- **Storage** — SQLite (WAL) for the low-volume registry; Parquet (source of truth)
  queried through DuckDB views for events/samples/architecture. The API reads
  DuckDB in-memory over Parquet globs so a live collector and the API never fight
  over a single-writer lock.
- **Frontend** — React 18 + TypeScript + Vite, hand-rolled SVG (no chart library).
  Built on the workstation and shipped to the backend as static `dist/`.
- **Adapters** — llama.cpp and vLLM (Prometheus `/metrics`), Diffusers (explicit
  wrapper), AMD/Linux telemetry (amdsmi/rocm-smi/sysfs/psutil), rocprofv3 (bounded).

Layout:
```
inferscope/
  inferscope/        backend package
    schema.py        Event/Sample + Provenance/Source/EventKind  (the contract)
    config.py        env-driven config, loopback defaults
    capability.py    runtime probes; drives unavailable/unsupported states
    store.py         SQLite registry + Parquet/DuckDB trace store
    collectors.py    SystemCollector, GpuCollector (amdsmi→sysfs→rocm-smi)
    engines.py       LlamaCppAdapter, VLLMAdapter (Prometheus)
    arch.py          config → computational graph (text/vlm/image/video, ×N collapse)
    kv.py            KV-cache memory model
    diffusers.py     PipelineRecorder / instrument() wrapper
    profiler.py      bounded rocprofv3 capture → events
    demo.py          synthetic session
    api.py           FastAPI app (REST + SSE + static)
    cli.py           check / serve / demo / models / capture
  ui/                React app (src/views, src/components, src/lib)
  tests/             pytest suite + fixtures
  docs/              ENVIRONMENT.md, STATUS.md
```

## Install

No global installs. On the GPU host the deps already exist in `/opt/venv`. On a
fresh machine:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # backend
cd ui && npm install                      # frontend (build on workstation)
```

## Run

**Backend + UI (production — backend serves the built frontend):**
```bash
/opt/venv/bin/python -m inferscope serve          # http://127.0.0.1:8090
```

**Self-check first (recommended):**
```bash
/opt/venv/bin/python -m inferscope check
```
Prints which collectors, engines, and profilers are available, and the platform
caveats (e.g. gfx1100 util/bandwidth unavailability).

**Frontend dev (hot reload, proxies /api to :8090):**
```bash
cd ui && npm run dev                       # http://127.0.0.1:5180
```

**Build the frontend and deploy to the host:**
```bash
cd ui && npm run build                     # -> ui/dist
scp -r ui/dist  host:/workspace/inferscope/ui/
```

## Access — through the tunnel (not SSH port-forward)

The app binds loopback on the host and is reached **through the rc-tunnel**, which
hands out an ephemeral public URL. That URL is the canonical host — it is **not**
hard-coded; InferScope reads it live from `rc-tunnel status` and shows it in the
top bar and at `/api/health`.

```bash
# start/refresh the tunnel for the app port (on the host)
bash /workspace/fathom/tunnel.sh expose --port 8090
# read the current URL
bash /workspace/fathom/tunnel.sh status      # -> URL: https://rc-<id>.radeon.firstdg.ai
```
Open that URL in your browser. `inferscope check` prints it too.

Do **not** bind a public interface — the tunnel is the only intended entry point
and there is no auth on the API. Access is through the tunnel URL only; the app
always binds loopback (`127.0.0.1:8090`) on the remote host.

## The four sections

The UI is organised as the workflow actually runs, not as a flat list of
topics:

| section | what it is for |
|---|---|
| **Run & capture** | the only thing that *produces* data: chat a local model, start/stop the engine, capture kernels |
| **Analyze** | where the time went — op-class summary above the execution timeline (one dataset, one view) |
| **Model** | reference facts: Architecture / Memory & KV / Multimodal |
| **Status** | the landing page: is anything running, and what did the last capture find |

The session picker shows what each session *contains* (`164,320 kernels · gemm`)
rather than a bare id, and the app opens on the session holding the most measured
kernel evidence.

## Chat & per-turn profiling

## Connecting llama.cpp / vLLM

Start the engine with metrics, then point InferScope at its `/metrics` (no restart
of InferScope needed — it scrapes live):
```bash
export INFERSCOPE_LLAMACPP_METRICS=http://127.0.0.1:8080/metrics   # llama-server --metrics
export INFERSCOPE_VLLM_METRICS=http://127.0.0.1:8000/metrics        # vllm serve
```
Adapters read request counts, queue depth, prompt/generation token totals,
TTFT/ITL histograms and KV-cache usage where the version exposes them. Missing or
renamed metrics come back absent, never invented. Rates (tokens/s) are
**derived** from counter deltas.

## Instrumenting Diffusers

Wrap a pipeline explicitly — InferScope does not monkey-patch the library:
```python
from inferscope.diffusers import instrument
from inferscope.store import TraceStore
trace = TraceStore("data")
emit = lambda e: trace.append_events([e])       # append_events takes an iterable
pipe = instrument(sd_pipeline, session_id, model_id, emit)
image = pipe(prompt="...", num_inference_steps=20)   # emits pipeline_stage,
                                                      # denoise_step, alloc events
```
It times the whole call, records per-step latency via `callback_on_step_end`, and
samples peak device memory. Degrades to whole-call timing if the pipeline has no
step callback.

## Profiling sessions (bounded, opt-in)

```bash
inferscope capture --yes --seconds 10 -- python your_infer_script.py
```
Validates rocprofv3 arguments against the installed CLI, runs the target under a
hard `timeout`, parses the kernel trace into measured events, and writes CSV (or
Perfetto `.pftrace` where supported). The API exposes `/api/capture/status`
(capability only) — it never executes browser-supplied commands.

## Telemetry, provenance, and limitations

See `docs/STATUS.md` for the full list. The headline ones:

- **GPU utilisation is unavailable on gfx1100** — both rocm-smi and amdsmi report
  0 under load. InferScope shows `unavailable`, not `0`, and uses VRAM occupancy.
- **Memory-bandwidth pressure is not measurable** here (rocprofv3 DRAM counters
  read 0) — only modelled, and labelled `estimated`.
- **Per-layer kernel attribution** requires a trace whose time buckets contain a
  whole forward pass; when they don't, layer attribution is refused rather than
  faked.
- **Kernel events are never attributed to a request/token** without explicit
  correlation from the instrumentation.

## Tests

```bash
/opt/venv/bin/python -m pytest -q          # 50 backend tests
python inferscope_smoke.py                 # Playwright UI smoke (needs the server)
```

## Security

Loopback-only by default; read-only APIs; no shell execution from HTTP; no engine
control; no secrets required or stored. Profiling is opt-in and bounded. See
`docs/STATUS.md`.
