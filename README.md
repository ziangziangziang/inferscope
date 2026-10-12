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

### Tunnel watchdog

The tunnel's `frpc` client repeatedly loses its broker credentials
(`oauth2: "invalid_client"`), which strands the hostname on frp's 404 page even
though InferScope is still healthy on loopback. This happens roughly every
30–60 minutes.

`tunnel-watchdog.sh` re-issues the tunnel automatically when the broker reports
no active tunnel for this pod:

```bash
# started automatically by run.sh; to run or inspect it by hand
setsid nohup /workspace/inferscope/tunnel-watchdog.sh > /dev/null 2>&1 &
tail -f /workspace/inferscope/tunnel-watchdog.log
```

Design notes:

- **It only touches the tunnel**, never the app, models, or profiling config.
- **It asks the broker** (`rc-tunnel status`), which is the authority. It does
  not HTTP-probe the public URL: the pod has no hairpin route to its own public
  host, so such a probe always fails.
- **A new URL means the hostname changes.** Read it back with
  `curl -s localhost:8090/api/access` — the app re-detects it automatically.
- **Broker env** comes from `/proc/1/environ`; a non-interactive shell does not
  inherit it, which is why a bare `rc-tunnel` invocation fails.
- **Single instance** via a pidfile checked against `/proc/<pid>/cmdline`. This
  is deliberately not `flock`: a SIGKILLed watchdog leaves a zombie whose open
  file descriptor keeps the advisory lock held, so `flock` deadlocks and refuses
  to start a new watchdog while none is running.
- **Backs off** 30s → 60s → … capped at 10 min, so a broker outage cannot
  become a re-issue loop.
- **Refuses to stop a tunnel it cannot rebuild** — if the broker env is missing
  it aborts before `stop`, so a failed re-issue can never leave you with no
  tunnel at all.

## The scene, and opening the model

The front page carries one isometric diagram, all of it through a single
projection (`ui/src/components/iso.tsx`), so nothing can drift out of
perspective:

```
HOST plate ── CPU · GPU · RAM · VRAM      fill = how busy
     │
     ├─ engine slab  (llama.cpp)  ─┐     same footprint, same height
     └─ model slab   (weights)    ─┘     dashed = software, not hardware
```

The engine and the model are **separate on purpose**. They are different things
that live in the same place — swap the engine and the weights still describe the
same model — so the diagram does not merge them.

The tower hangs close enough to overlap the hardware, which is the point: it is
running *on* this machine. Hovering it lifts it clear (dock-style), and the
slabs are glass so the parts underneath stay readable either way.

Click the model and a panel takes half the screen — **the architecture is
drawn there, not on top of the scene**. The scene stays the overview; the
detail belongs beside it.

- every box is drawn **at the size of the parameters it holds**
  (edge ∝ `params ** 0.36`, chosen so a 5,120-parameter norm and a 1.27B
  embedding can share one diagram);
- repetitive units **wrap into grids** — 64 transformer blocks become an 8×8
  grid of individually-sized boxes, 27 vision blocks a 6×5 one;
- clicking any box opens a panel on the right with its detail, and for a wrapped
  block, **its sub-layers**.

For `Qwen3.8-27B-INT4` that is 101 boxes over two branches — vision behind, text
in front, joined by a dashed feed.

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
