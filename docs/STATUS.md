# InferScope — implementation status

## What was built (and is running)

A working local-first inference observability workbench: FastAPI backend +
React/TS SVG frontend, six linked views, real telemetry adapters, a
config-derived architecture engine, and a bounded profiling path. It boots and is
fully navigable with no GPU and no engine (demo mode), and collects real AMD/CPU
telemetry when run on the host.

### Backend
- **Versioned event schema** (`schema.py`) — `Event` + `Sample` with dual
  wall/monotonic timestamps and a `Provenance` enum on every value. Unknown is
  `None`/`unavailable`, never `0`.
- **Capability probe** (`capability.py`) — detects amdsmi/rocm-smi/rocprofv3/
  sysfs/psutil/engine endpoints at runtime and records per-field support; drives
  the "unavailable vs unsupported vs not configured" states everywhere.
- **Storage** (`store.py`) — SQLite (WAL) registry + Parquet/DuckDB facts. The API
  reads DuckDB in-memory over Parquet globs so the live collector and API don't
  contend on a single writer.
- **Collectors** (`collectors.py`) — `SystemCollector` (psutil: cpu, per-core,
  load, RAM, RSS) and `GpuCollector` (amdsmi exact VRAM/clocks/power, sysfs temp,
  rocm-smi fallback).
- **Engine adapters** (`engines.py`) — llama.cpp + vLLM Prometheus scrapers,
  header/`# TYPE`-driven parsing, delta-derived rates, absent metrics stay absent.
- **Architecture** (`arch.py`) — config → graph for text LLM / VLM / image /
  video, with `× N` collapse + expandable single-instance templates and
  per-node/edge provenance.
- **KV model** (`kv.py`) — the documented baseline plus GQA / sliding-window /
  shared-prefix / cache-quantisation adjustments, each explained in `notes`.
- **Diffusers** (`diffusers.py`) — `instrument()` wrapper + `PipelineRecorder`;
  explicit, no global monkey-patching; testable with a stub pipeline.
- **Profiler** (`profiler.py`) — bounded rocprofv3 capture → measured events;
  validates args against the installed CLI; Perfetto output when supported.
- **API** (`api.py`) — REST + SSE live stream + static hosting; per-adapter
  health; loopback by default; no shell/paths from HTTP.
- **CLI** (`cli.py`) — `check`, `serve`, `demo`, `models`, `capture`.

### Frontend (six views, one design system)
Live overview · Architecture explorer · Execution timeline · Memory & KV ·
Multimodal lab · Bottleneck/optimisation lab. Persistent nav, a provenance legend
always visible, linked selection (graph node ↔ inspector ↔ timeline), zoom/pan
SVG graphs, and explicit empty/unsupported/not-configured/simulated states.

## Design decisions
- **Reuse Fathom's proven patterns** (DuckDB+Parquet store, rocprofv3 parsing,
  Playwright smoke harness) rather than greenfield, but as a new, independent
  project; Fathom is untouched.
- **Provenance is a first-class column**, not a UI afterthought — the product's
  correctness guarantee.
- **Hand-rolled SVG** for the graph/timeline (a chart library would fight the
  TikZ-style, collapsible-block requirement).
- **Architecture is a property of the model**, so a session without a running
  engine still shows the host's discovered checkpoint graph, labelled
  config-derived.
- **No new installs** — the host has no reliable egress; everything runs on the
  existing `/opt/venv` + a workstation-built `dist/`.

## Verification performed (on the real host unless noted)
- `inferscope check` → amdsmi 7.0.2 (vram/power/clock/activity probed),
  rocprofv3 1.1.0 (kernel-trace + Perfetto), gfx1100, engines "not configured",
  gfx1100 util/bandwidth caveats surfaced.
- **Live GPU telemetry**: real VRAM occupancy, temp, SOC/MEM clocks, RAM/RSS/CPU —
  provenance `measured`. **GPU utilisation now reads correctly via amdsmi
  `gpu_activity`** (verified tracking 0→100→98→0 across a generation); the earlier
  "unavailable" claim was a mistake (idle sample + rocm-smi's broken `--showuse`).
- **Live llama.cpp adapter**: started `llama-server` (Qwen3-VL-8B IQ4_XS GGUF,
  HIP, `--metrics`) on loopback:8081, generated ~600-token completions (~87 tok/s,
  coherent output), and confirmed InferScope's adapter reads the **real** metric
  names (`tokens_predicted_total`, `prompt_tokens_total`, `requests_processing`,
  `n_busy_slots_per_decode`) — `gen_tokens_total` climbed 600→1200 and
  `active_requests` tracked 0→1→0 live. The fixture was corrected to the real
  names and a legacy-name fallback test added. Server stopped afterwards; GPU
  restored to idle.
- **Real VLM architecture**: Qwen3.8-27B-INT4 → family `vlm`, `Vision block × 27`
  and `Transformer block × 64`, `image_token_id` splice edge, provenance `derived`.
- **KV calc**: 64 L × 262144 ctx × 4 kv × 256 dim bf16 = 64 GiB (exact).
- **Backend tests**: `pytest -q` → **50 passed** (schema, kv, arch, engines
  parsing incl. missing/malformed, store round-trip + reader-refresh, profiler CSV
  parse, diffusers recorder, API endpoints).
- **UI smoke** (Playwright against the tunnel): all six views render, **0 console
  / 0 page errors**; architecture shows 12 nodes, drill-in yields 10 sub-nodes;
  node click populates the inspector; demo timeline renders 5 lanes / 252 events;
  memory shows a GiB value.

## Not verified live (and why)
- **vLLM adapter** — no vLLM server was running; verified against realistic
  Prometheus fixtures + config wiring. (llama.cpp *is* live-verified — see above.)
- **Diffusers end-to-end** — `diffusers` is not installed on the host and there is
  no image/video model on disk; a real generation run would need a large download
  over unreliable egress. The wrapper is unit-tested with a stub pipeline.
- **rocprofv3 capture against a live GPU kernel** — plumbing unit-tested on a
  trace fixture and args validated against the installed CLI, but no capture was
  executed (would require launching GPU work; not done to avoid disturbing the host).

## Known limitations
- `rocm-smi --showuse` GPU-use% is broken on gfx1100 (always 0). InferScope uses
  amdsmi `gpu_activity` (works) as its primary path; util is only `unavailable`
  when falling back to rocm-smi.
- rocprofv3 DRAM/FETCH_SIZE **byte** counters read 0 on gfx1100 → measured
  memory-bandwidth bytes are unavailable; amdsmi `umc_activity`% is a proxy.
- llama.cpp `tokens_predicted_total` updates at request-completion granularity on
  this build, so the **derived** tokens/s reads 0 mid-request and steps up at the
  end; instantaneous throughput needs the decode-timing histogram deltas (labelled
  `derived` either way).
- Per-layer kernel attribution is refused when a time bucket doesn't span a full
  forward pass (the honest behaviour; recovering it needs finer buckets or real
  pass boundaries).
- Kernel events are never attributed to a request/token without explicit
  correlation from the instrumentation.

## Chat & per-turn profiling (added)
- **Chat view** talks to a local llama-server and records every turn as a
  `request` event (TTFT, wall time, prompt/completion tokens, tok/s) in the
  session selected in the top bar, so turns appear in Timeline and Bottlenecks.
  TTFT and wall time are **measured** client-side; token counts are **measured**
  deltas of the engine's own `/metrics` counters, bracketing the turn.
- **Managed mode** starts `llama-server` as a child of InferScope. Two guards:
  it refuses to start if the port is already taken (it will not adopt a server it
  does not own), and readiness is confirmed via the `/props` `model_path`, not
  just `/health`, so a foreign server on the same port can never be mistaken for
  ours.
- **Kernel trace for a chat session**: "start + kernel trace" runs the server
  *under* rocprofv3 (wrap mode). On stop the trace is parsed and ingested.
  Verified end to end: a real 27B chat turn ingested **466,494 kernel events**
  (gemm 84.6% of 4.9 s).
- **Per-turn kernel traces are unavailable here, and the UI says so.** They need
  `rocprofv3 --attach`, which runs from a *separate* helper process; with
  `ptrace_scope=1` only a direct parent may ptrace, and this container's `CapEff`
  lacks `CAP_SYS_PTRACE`. The session-level trace is the honest substitute; the
  per-turn checkbox was removed rather than left failing opaquely.

## Layer size and compute (added)
- `cost.py` computes **exact analytic parameter counts** from the checkpoint
  config, plus per-token FLOPs at a stated context length. Verified against real
  configs: Qwen3.8-27B-INT4 = 24.76 B params, 340.8 M/block; Qwen3.5-4B =
  4.03 B params, 97.0 M/block; the SwiGLU MLP is 78% of a block.
- Architecture nodes show the count, a bar encoding relative parameter share, and
  per-token compute; the header shows model totals. These are **derived**
  (analytic from config), never measured, and labelled as such.
- Parameters are exact. FLOPs are static analytic: they ignore batching,
  sparsity and quantization, and during decode the real limiter is weight
  bandwidth, not FLOPs. FLOPs for **linear/recurrent attention are reported
  unavailable**, because a recurrent state's per-token cost is not implied by the
  config; an earlier placeholder figure was removed rather than presented.

## Robustness fixes (added)
- A truncated Parquet part file (from a process killed mid-flush) used to poison
  the whole DuckDB catalog, so the collector reported `state: error`. The catalog
  now validates each file's magic bytes and skips only the corrupt one.
- Startup pruning is fault-tolerant per session, and superseded live sessions are
  **reclassified to `profile`, never deleted** — sessions that carry captured
  events are evidence, so the picker shows the sessions that matter instead of a
  dozen identical `live` rows.
- Tunnel discovery reads rc-tunnel's own `state.json`, inherits the broker
  environment from PID 1 (a non-interactive shell has no `FRP_BROKER_*`), and
  reports liveness as a tri-state from the tunnel client's own state. It
  deliberately does **not** HTTP-probe its own public URL: the pod has no
  hairpin route to it, so such a probe always fails. Log lines older than the
  tunnel's start are ignored, and the last decisive line wins.
- `index.html` is served `no-cache` while hashed bundles are immutable, so UI
  fixes actually reach a browser that has cached the shell.

## Tunnel watchdog (`tunnel-watchdog.sh`)

The public URL kept dying while the app stayed perfectly healthy. Cause is
upstream: the tunnel's `frpc` client loses its broker credentials every ~30–60
min (`oauth2: "invalid_client"`), so the hostname starts serving frp's 404 page
while `127.0.0.1:8090` still answers normally. `/api/access` correctly reported
`reachable: false` with *"broker credentials rejected; re-issue with
rc-tunnel expose"* — the failure was always visible, it just was not acted on.

The watchdog closes that loop. Every 60s it asks the broker
(`rc-tunnel status`) whether this pod still has an active tunnel, and re-issues
when it does not. Verified end-to-end: `rc-tunnel stop` at 14:55:01 →
detected at 14:55:22 → re-issued as a new URL by 14:55:26 → broker confirmed
active at 14:55:34 → external `GET /api/health` returned **200** on the new
URL, and the app auto-detected it via `/api/access`.

Constraints that shaped it:

- **Only the tunnel is touched** — never the app, the models, or ROCm/driver
  config.
- **The broker is the authority, not an HTTP probe.** The pod has no hairpin
  route to its own public host, so probing the public URL from inside always
  fails and would have caused a permanent re-issue loop.
- **Never tears down a tunnel it cannot rebuild.** The broker env
  (`FRP_BROKER_*`, from `/proc/1/environ`) is loaded *before* any `stop`, and the
  re-issue aborts if it is missing. This guard was added after the first version
  did the opposite: `export "$kv"` on a multi-line string exports one invalid
  name, so `expose` failed *after* `stop` had already killed a working tunnel.
- **Backs off** 30s → 60s → … capped at 10 min, so a broker outage cannot become
  a re-issue loop.
- **Idempotent single instance**, via a pidfile validated against
  `/proc/<pid>/cmdline`. `flock` was tried first and **deadlocked**: a SIGKILLed
  watchdog leaves a zombie (this container accumulates ~170 unreaped zombies)
  whose open descriptor keeps the advisory lock held, so a new watchdog refused
  to start with *"another watchdog already running"* while nothing was running.

`run.sh` starts it on boot behind an idempotent `pgrep` guard, so it survives app
and pod restarts. Log: `/workspace/inferscope/tunnel-watchdog.log`.

**Operational consequence:** a re-issue yields a **new URL**. It is never
hard-coded anywhere; `/api/access` and the UI top bar pick it up automatically.

## The scene: engine, model, and the model opened up

The front-door diagram was rebuilt around one request: **decouple the model from
the inference engine**, and let the model's own structure be inspectable.

### One camera, one set of primitives

`ui/src/components/iso.tsx` holds the projection and a single generic `Box`
(width/depth/height in world units). The earlier scene forced every box to be a
cube, which was fine while every part happened to be the same size and stopped
being fine the moment a component had to be drawn at its *own* size. Hardware
cubes, the two tower slabs and every model component now go through the same
primitive, so the perspective cannot drift between them.

### The tower

The engine and the model are deliberately **not merged**. A plate-sized engine
slab (dashed — it is software, with no size of its own) carries a model slab of
identical footprint and identical height. They are drawn as separate objects
because they are different things that happen to live in the same place: swap
the engine and the weights still describe the same model.

The tower hangs close enough to overlap the hardware, because that overlap is
the claim — *this is running on this machine*. Two things keep it honest:

- the tower is drawn as **glass**, not solid, so the parts underneath stay
  readable through it;
- hovering it **lifts it clear** with macOS-dock easing.

The lift has hysteresis. Without it the tower moves *away* from the cursor as it
rises, so a pointer near the lower edge knocks it out of its own hover target,
which drops it, which picks it up again — a visible flicker that also made the
model box unclickable. Playwright surfaced this as
`element is not stable`; it was a real interaction bug, not a test artefact.

Hardware captions are drawn in a **separate layer after the tower**, with a
paper-coloured halo. The tower deliberately covers the back of the host, so a
caption drawn with its cube would simply vanish — CPU's label disappeared
entirely until the layer was split out.

### The model, opened — in the panel, not on the stack

Clicking the model opens the detail panel, and **the architecture is drawn
there**, not unfolded above the scene. Unfolding it in place was tried first
and was wrong: a 101-box tier on top of the host buried the hardware it was
supposed to sit above, and still had to be scaled to ~40% to fit beside the
panel. The scene is the *overview* — what machine, what engine, what model — and
the panel is where detail belongs. In the panel the same diagram gets half the
screen to itself and the host stays legible.

The scene therefore keeps only the plate, the four hardware cubes and the two
tower slabs; the model slab stays lit while the selection is anywhere inside
the model, so the diagram still answers "where am I".

Two rules drive the layout, both in `sceneModel.ts`:

**A block is drawn at the size of the thing it holds.** Edge length scales as
`params ** 0.36`. Raw analytic counts span six orders of magnitude here — a
1.27B token embedding against a 5,120-parameter RMSNorm — so a linear or
square-root scale buries everything small. The exponent keeps ordering and
ratio meaningful while the small parts stay legible, and every block is labelled
with its true count so nothing is lost to the scaling.

**Repetitive units wrap.** 64 transformer blocks drawn as 64 towers is an
unreadable smear, so a repeat group becomes a near-square grid (64 → 8×8, 27
vision blocks → 6×5) of unit boxes, each still sized by its own per-unit
parameter count. The grid's footprint then encodes the count, which is the
honest thing for a grid to encode. For `Qwen3.8-27B-INT4` that is 101 boxes:
64 transformer + 27 vision + embedding + LM head + merger + patch embed + norms
and markers.

The two branches are stacked back to front — the vision branch behind, because
the backend prepends it and it *feeds* the text stream. A dashed connector
joins them, so the parallel rows do not read as unrelated.

### Selection and the detail panel

Clicking anything in the scene opens a panel that takes half the screen. The
selected part stays highlighted and everything else dims. One `Sel` value drives
both, so the panel and the diagram cannot disagree about what is selected.

For a wrapped unit the panel opens the block's sub-layers (11 for a transformer
block) with a proportional bar each, and — because this model interleaves linear
and full attention — it reports *which kind of attention this particular unit
has*, read from `meta.layer_types[unitIndex]`. Clicking unit 8 says "full
attention"; unit 38 says "linear attention". Same weights, different layer.

Opening the panel widens the page column from 760px to 1180px: split at the
narrow width, the diagram scaled down to about 40% and became unreadable.

### Notes from the build

A few things were wrong in ways that only showed up on screen:

- Caption positions were computed **twice** — once to render, once to size the
  viewBox — and the two drifted. Captions were drawn at `z=0` while their boxes
  sat at `z=archZ`, putting labels a long way below what they belonged to. There
  is now one function, `layoutCaptions`, used by both.
- Caption bounds were reserved with a guessed constant. An `end`-anchored
  caption extends its **full** width to the left of its anchor, not half, which
  clipped "reads the image · 27 blocks" off the edge. Widths are now computed
  from the text.
- Captions were sorted "longest last so they win" — which silently painted one
  caption over another and hid the collision instead of revealing it.
- The back row's caption, anchored at its left end, landed in the same screen
  patch as the front row's first component. Both captions were drawn there and
  overprinted. Back-row captions are centred above their own row now.

Two of those were only ever a problem *because* the tier was stacked on the
scene; moving it into the panel deleted them outright rather than fixing them.

## Engines: one GPU, one resident engine

The machine has one GPU, so only one heavy engine can be resident at a time.
`inferscope/enginectl.py` owns that: a registry of `EngineSpec`s (binary,
model, port, how to prove it is serving) and an `EngineManager` that keeps at
most one process alive. Loading an engine **offloads whatever held the card
first and says so in the response** — that eviction is the thing a user cannot
deduce from a VRAM number, so it is stated rather than implied.

Every engine is a local HTTP service on loopback with one contract
(`GET /health`, then a task-specific POST). That uniformity is what lets one
manager load a speech recogniser and, later, a diffusion pipeline, without
knowing anything about either. Adding an engine is one `EngineSpec` plus one
branch in the API.

It is deliberately separate from `engines.py`, which only *reads* Prometheus
metrics from an engine that is already running, and from `chat.py`, which owns
the language-model server.

Ownership rules carried over from the language-model server, for the same
reason — a profiler must attach to a pid we spawned:

* never adopt a listener we did not start;
* clear a listener only when it is provably our own leftover;
* otherwise refuse, naming the pid.

### The orphan reaper had to become generic

`reap_own_orphan` cleared a leftover server on the chat port after a hard
`pkill -9` of InferScope orphaned it. Its ownership test hardcoded
`"llama-server"`, which was fine when llama.cpp was the only managed process and
wrong the moment a second engine existed: a `whisper-server` orphan sitting on
the ASR port matched our model path exactly and was still dismissed as "not an
inference server at all".

The binary name now comes from the caller (`expect=`), so the rule is "the
process holds our model **and** is the binary we would have started". Two tests
pin both directions: ours gets reaped, a same-model different-binary does not.

### ASR (speech to text)

whisper.cpp built from source against ROCm 7.2 (`-DGGML_HIP=ON`,
`gfx1100`), model `ggml-base.en`. `/api/asr/transcribe` posts multipart audio to
the engine's `/inference`, and returns the text plus **measured** timings: audio
duration and engine wall time, so the view can show a real factor rather than a
round trip.

Measured on the 11s JFK sample: **0.23s, ~48× realtime**, on the GPU.

Two things worth recording:

- `huggingface_hub` could not fetch the model (its client does not survive this
  network), but `curl -L` through the same proxy downloads it fine. Large model
  pulls here go through curl.
- The WAV duration parser read `0.0s` for every clip until it walked the chunk
  list instead of assuming a 44-byte header. Fixing the offset then needed two
  more passes — the byte rate is at `body+8` in a WAVEFORMATEX, not `+10` or
  `+28` — and the test helper itself was emitting a malformed `fmt` chunk with
  an extra WORD and no word-alignment padding. Three bugs in a row, all of them
  invisible until a real file went through.

### Everything on the first page

The engine console and the speech panel were separate views. They are now part
of the front door: the rail sits under the scene, and the task panel below it
shows whatever engine is resident. One page drives the machine — scene, load and
offload, run the task, read the result — instead of hunting across tabs for a
button whose effect is global.

The rail is a rail rather than a list because the resident engine is the thing
that matters; the rest are one click away, and the eviction notice appears
exactly where the load happened.

### Voice input

`useRecorder` captures raw PCM with a `ScriptProcessor` (deprecated, but the
only node that returns samples synchronously everywhere we need, and it keeps a
codec out of the path), resamples to 16 kHz mono through an
`OfflineAudioContext`, and writes the WAV header by hand. The browser already
knows how to do the decode-and-resample, so the host gains no ffmpeg dependency
for a container format the GPU path does not want.

Each way a microphone can fail gets its own sentence. The generic branch used to
surface the browser's raw message, which on this box was literally
`Not supported` — technically true and useless. It now distinguishes refused
permission, no device, device busy, and unsupported browser, and names the error
code only when it is genuinely unrecognised.

## Units: the blocks are data, not hardcoded

The scene drew a llama.cpp slab with a Qwen model on top. That worked until
there was a second engine: nothing could be swapped, nothing was loaded by
default, and a "load" had no starting state to animate from.

### The unit

A **unit** is one engine paired with one checkpoint — `whisper.cpp` +
`ggml-base.en`, `llama.cpp` + `Qwen3.8-27B`. `inferscope/units.py` discovers
them from disk rather than configuring one model per engine, so adding a
checkpoint is a file drop. An engine with nothing installed still appears,
marked unavailable with the reason.

The scene now takes a unit and draws from it. Geometry is unchanged from the two
fixed slabs: the engine is a plate-sized outline (software has no size of its
own), the model sits on it with the same footprint, subdivided into one block
per component when the model has parts, each block's width being its real share
of the weights. So a single-file model is one block, and a diffusion model will
be several side by side whose relative widths are the honest ones.

### The choreography

Blocks rise out of the plate to load and sink back into it to evict, staggered
by tier. Captions settle last, so the stack reads before the text does.

A swap does **not** cross-fade two towers. It sinks the old one into the plate
and only then raises the new one, because two towers briefly sharing the GPU is
a state this machine can never actually be in. The scene follows the user's
*choice*, not the boot completing — tying it to `resident` left the plate bare
for the full ~20s a whisper.cpp load takes, with nothing to animate.

Nothing loaded is drawn as an explicit empty state with a sentence, since the
space above the plate is reserved either way and a bare plate otherwise reads
as broken.

### Two owners of residency

The one-engine rule is enforced by `enginectl`, but the language-model server
is owned by `chat.py`, which also owns the profiler's identity check and kernel
wrapping. Unifying those managers is a bigger change than the catalogue needs,
so the scene reads **both** and, when both are resident, says so rather than
quietly claiming a single occupant. That bug was real: before it, the plate
showed nothing at all while llama.cpp was running.

## Information architecture (restructured)

The UI had **seven equal tabs** that hid the fact that only one of them
produced anything and that two of them were the same data twice. Mapped by what
each view actually called, there were only three jobs:

| job | views |
|---|---|
| produce data | Chat |
| read measured data | Overview, Timeline, Bottlenecks |
| refer to model facts | Architecture, Memory & KV, Multimodal |

They are now **four workflow-ordered sections**:

- **Run & capture** — the only producer (chat + engine + capture).
- **Analyze** — op-class summary *above* the execution timeline. These were two
  tabs over one dataset (`opclasses` is an aggregate of the same `events`), so
  they were split arbitrarily and made a user hunt for the other half.
- **Model** — Architecture / Memory & KV / Multimodal as sub-tabs. Reference
  material, one click away, without implying you must visit all three.
- **Status** — the landing page.

Merging Timeline+Bottlenecks would have silently dropped the experiment
recorder, so it was extracted to `ExperimentRecorder.tsx` and kept in Analyze,
where it belongs (an experiment is recorded *after* reading the analysis).

### Landing page

Opening the app used to show GPU 0% / 27 MiB / 64 idle cores, which reads as
"nothing happened" when it actually means "no engine is attached". **Status**
now answers the two questions on arrival — *is anything running* and *what did the
last capture find* — before the telemetry detail. Telemetry is read from the
**live** session, not the selected one: a profiled session holds kernels but no
samples, so reading it made a healthy idle GPU look broken.

### Session spine

Captured data was siloed — you captured, then had to know to switch tabs and pick
the right session:

- `/api/sessions` now returns per-session `event_count`, `kernel_events`,
  `gpu_ms`, `span_s`, `top_op`, `has_events`, plus `best_session`.
- The picker shows **what is in each session** (`prof-… — 164,320 kernels ·
  gemm`) instead of a column of identical ids.
- The app lands on `best_session`: the one with the most **measured kernel**
  evidence. Deliberately not "most recent with data" — demo is re-seeded every
  boot, so that always selected the synthetic session.
- Empty sessions get an empty state that names the action ("send a prompt under
  Run") instead of an empty panel.

### Bug found while doing this

Pruning superseded `live` sessions (added earlier to stop the picker filling up)
**deleted real chat captures**, because a capture is written into the live
session. A session that holds events is now *reclassified* to `profile` instead
of deleted; only empty ones are removed. Verified by capturing 47,366 kernel
events, restarting, and finding them intact.

Also: the persisted architecture graph is a snapshot that went stale (it
predated the cost/params work), so the endpoint now rebuilds from the checkpoint
config rather than serving the stored JSON.

## Density / progressive disclosure

The pages carried their own documentation inline, which is what made them feel
heavy. Explanations moved behind affordances:

- `Info` — a small `i` glyph. Hover, tap or focus opens a panel. Rendered in a
  **portal** and positioned against the trigger's viewport rect: the sidebar is
  `overflow: auto`, so an absolutely-positioned tooltip was clipped mid-sentence.
- `Reveal` — a `› label` row for secondary detail that is referenceable rather
  than explanatory (timeline colours, launch counts, the experiment recorder,
  the Diffusers snippet, the collector health table).
- Jargon carries a plain-language `title`: `gemm` = "the big matrix maths a
  transformer spends most of its time on", `ttft` = "how long before the first
  word appeared", `tok/s` relabelled "words/s".
- Labels moved toward plain speech: "GPU utilisation" → "GPU busy",
  "time by op class" → "time by kind of work", "prompt/gen tok" → "in/out".
- The five always-visible provenance badges in the sidebar collapsed to one
  glyph carrying the glossary.

Measured text volume per view (Playwright `inner_text` length):

| view | before | after |
|---|---|---|
| Status | 913 | 234 |
| Run & capture | 1406 | 635 |
| Analyze | 835 | 350 |
| Multimodal | 1292 | 921 |

### Bug found while decluttering

The session summary reported a **span of 1,782,734,402 s**. Kernel rows carry the
raw GPU clock while request rows carry wall-clock time, so `min`/`max` across a
mixed session is meaningless. Span is now computed only from wall-clock events
and is `None` for a kernel-only capture — the pill is hidden rather than showing
a wrong number. Covered by a test that rejects any span ≥ 1 day.

Also: VRAM showed "0.0 GiB" for a 27 MiB reading; units are now chosen by
magnitude.

## Landing page + light-green theme

The app opened straight into a perspective with no explanation of what
inference is, in a near-black, rounded, monospace-heavy skin that read as a
serious instrument panel. Both were changed.

### Start-here page (`views/Index.tsx`)

- Opens with one paragraph defining inference in plain language.
- **CPU / MEM / GPU as isometric 3D boxes on a ground slab.** Fill level =
  measured utilisation; a hatched inner block = memory already used (VRAM inside
  the GPU box). Busy parts animate.
- **Engine bar on top of the ground**: running model, a model chooser, and a
  start/stop affordance.
- **Tokens grow leaves.** Every streamed token from a real turn becomes a leaf;
  the trunk grows with the token count. Verified: 42 leaves from a real
  Qwen3.8-27B turn.
- Three plain cards lead onward.

### Visual re-skin

- Pale green palette, square corners everywhere, sans-serif prose (monospace
  retained for numbers/ids only).
- Re-skinned by overriding the existing CSS variables, so every component
  followed without being rewritten.

### Two bugs found while building the scene

- **Only 2 of 3 boxes were visible.** Each part drew its own copy of the full
  ground plate, so every later plate painted over the previous part's cube. The
  plate is now drawn once, behind everything. This one is only obvious in a
  screenshot — the DOM was correct.
- **The first attempt used CSS 3D** (`rotateX/rotateZ` + `preserve-3d`). The
  ground rendered as diagonal wedges over the hero text and could not be
  reasoned about. Replaced with an SVG isometric projection: `iso(x,y,z)` is a
  pure function, and the viewBox is computed from the real projected bounds
  rather than guessed (guessing clipped the right-hand cube and its labels).

### "span" removed

The session summary showed `span 1327.56 s`. Not a bug — the session had 3
request events 22 minutes apart while its kernels spanned 3.3 s — but the
number invites the conclusion "this capture took 22 minutes". A figure that
encourages a wrong reading is worse than no figure, so the span pill was dropped
from Analyze and Status. The backend still returns `span_s` (wall-clock events
only, `None` for a kernel-only capture) and a test rejects spans >= 1 day.

### Start-here refinements

- **Four equal hardware blocks.** VRAM was previously a hatched texture *inside*
  the GPU box, which under-weighted it. CPU / RAM / GPU / VRAM are now peers of
  the same size, each filled from its own measurement, and blocks are drawn with
  solid faces (opacity raised from 0.1/0.2 to 0.95) so an idle part still reads
  as a solid block rather than a ghost.
- **One page does everything.** `start engine` / `stop engine` and the chat box
  both live on the start-here page, so the diagram fills in without navigating
  away. The engine is started through the same managed path the Run page uses.
- **The stack reads top to bottom**: HARDWARE -> ENGINE (llama.cpp) -> MODEL,
  with the conversation layer growing beneath it. `z` is "up", so the layer
  heights are `zHw = 3LH, zEngine = 2LH, zModel = LH` — the first version had
  engine and model swapped.
- Rows of blocks step `+x` and `-y` together (`STEP = (W+D)/2 + 10`) so a row
  marches horizontally; placing them on a single axis made the row slide down the
  page.
- Layer captions are end-anchored and extend left of the slab corner, so the
  bounds calculation reserves room for their width — without that they were
  clipped to "RE" and "wen3.8-27B...".

Verified: a real Qwen3.8-27B turn produced **105 leaves** and a 540-character
reply from the start-here page alone.

### The tree is a CS tree, not a botanical one

An earlier pass drew a trunk-and-leaves tree, which was the wrong idea: the
diagram wanted a **hierarchical** tree — root at the top, fanning out to
children. Rebuilt as `CpuTree`:

- **Root** = the prompt being answered, labelled with the model, engine and
  tokens so far.
- **Four children in a 田 (2×2) grid**: CPU / RAM on the top row, GPU / VRAM
  below, each joined to the root by an arrowed edge.
- **Each node is drawn as the thing it actually is**, via artwork on its top
  face (`ART` in `Scene.tsx`):
  - CPU — a die with a few large cores and a cache strip
  - GPU — a dense array of many small compute units
  - RAM — a DIMM: chips in one row along a board
  - VRAM — a card: fewer, larger stacks on a substrate
- Utilisation is the fill, drawn **under** the artwork so a busy part still
  shows its structure.

### Bug: reasoning models looked silent

A turn returned `"text": ""` with `completion_tokens: 16`. The model is a
reasoning model: it streams `reasoning_content` before any `content`, and the
parser only read `content`. So a short turn showed an empty reply and an empty
tree even though real tokens had been generated.

- `chat.py` now forwards `reasoning_content` as its own stream and reports
  `reasoning` / `reasoning_only` on the turn.
- Both chat surfaces render thinking separately and count reasoning tokens as
  work done.
- `INFERSCOPE_CHAT_MAX_TOKENS` raised 256 → 512: at 256 this model spent the
  whole budget thinking and never produced an answer.

### Two smaller fixes

- A swallowed `catch` reported "network error" for every dropped stream. The
  count now comes from a ref (reading React state in the catch would always see
  0), and the message says what actually happened: *"connection dropped after
  N tokens — showing what arrived"*. Long SSE streams are still truncated by the
  tunnel's HTTP/2 layer; the UI degrades honestly rather than pretending.
- Graph node fills were authored for the dark theme, so matmul/projector nodes
  rendered as dark slabs with light text on the new light page. Re-authored per
  node kind.

**92 backend tests, 7/7 views, 0 console errors** on the smoke run.

### 田 grid layout

Confirmed by ASCII mock-up before implementing, then built as `HostTree`:

- **Root = a plate labelled HOST**, carrying the model, the engine and the token
  count. The four components sit beneath it. No arrows, per the design.
- **Four square, equal cells** in a 田 (2×2) grid — `CPU | GPU` on the top row,
  `RAM | VRAM` below — drawn as an outer frame plus the two rules that form the
  cross. `CELL = 152` for both axes, so the cells are genuinely square.
- The grid is laid out in **screen space, not on an isometric ground plane**. A
  square grid in isometric projection renders as a tilted diamond and would not
  read as a 田; the 3D is kept *inside* each cell instead.

Two layout bugs found by looking:

- The plate's top vertex sits `(w+depth)/2` **above** its centre. At `cy=64`
  that was `y = -46`, outside the viewBox, so the plate was clipped at the top.
- Captions for the bottom row sat within a few pixels of the bottom rule.

### One scene, one perspective

The previous version shared the viewing *angle* but not the box *convention* —
a wide slab for the host, squat boxes for the parts — which is what made the
perspective read wrong. Rebuilt as `HostScene`, following the `BaseBox`
convention from the reference schematic figures (`width = height = depth`,
semi-transparent faces, hairline border):

- **One ground plane.** Every cube stands on the same plate, drawn once.
- **Identical cubes**: `C = 56` for width, height and depth. The host is a
  cube too, not a slab.
- **The host sits behind** the group, so it reads higher on screen.
- The four parts occupy a **real 2x2 in world space**, which projects as a
  diamond of four cubes. That is the honest consequence of a single camera —
  it is what the reference figures do too, and it is why the earlier "square
  田 in screen space" version had to mix a flat 2D frame with 3D cubes.
- Painter's order sorts cubes by `x + y`, so nearer cubes overdraw further ones.
- Per-component artwork stays on each top face; utilisation is the fill on the
  two visible side faces.

Captions are anchored to each cube's **outer** side (CPU/RAM left, GPU right,
VRAM below). A single shared anchor put the back cube's label on the front
cube's body.

The host's model/engine/token detail moved to the bar beneath the scene: a cube
is only ~97 px wide across, so a model filename cannot fit on its top face, and
the bar already showed it.

### Smoke test made deterministic

A run failed with "architecture: no graph nodes rendered" while the view rendered
12 nodes on inspection — the smoke used fixed `wait_for_timeout` sleeps, and a
cold load through the bandwidth-limited tunnel can exceed 1.8 s. It now waits
for the selector each view actually produces.

### HOST is the plate, not a fifth cube

A host drawn as an identical cube read as a fifth component rather than the
thing that contains the other four. It is now the **plate itself**: one slab
with visible thickness, labelled `HOST` on its front edge, with the four part
cubes standing on it.

```
                    ╱▔▔▔▔▔╲
                   │  CPU  │              CPU above (behind)
      ╱▔▔▔▔▔╲      ╱▔▔▔▔▔▔▔▔╲                ╱▔▔▔▔▔╲
     │  RAM  │    ╱           ╲              │  GPU  │   GPU right
      ╲_____╱    ╲   ╱▔▔▔▔╲   ╱                ╲_____╱
╱▔▔▔▔▔▔╔▔▔▔▔▔▔╔▔▔╔▔▔▔▔▔▔▔╔▔▔▔▔╔▔▔▔▔▔▔▔╲   VRAM below
    ╲  VRAM  ╱  ╚══════════╝   ╚══╝   ╚══╝   ╱ ▔▔▔▔▔▔▔
      ╲______╱ ╲_________________________╱
                HOST
```

Gaps widened (`G` 18 → 36, plate margin 26 → 34) so each caption sits on clear
plate instead of on a neighbouring cube. Anchors: CPU above, RAM left, GPU
right, VRAM below.

### The stack grows when the engine starts

Pressing **start engine** appends an engine tier above the hardware. The scene
reserves that slot at all times, so starting the engine never resizes the
diagram under the user — the layer simply appears in place.

```
                    ╱╌╌╌╌╌╌╌╲      engine tier — dashed, translucent:
                  ╱  llama.cpp ╲    software sitting on hardware, not a part
                 │ Qwen3.8-27B  │  → 128 tokens
                 ╲     0 tokens ╱
                  ╰╌╌╌╌╌╌╌╱

                    ╱▔▔▔▔▔╲       CPU / GPU / RAM / VRAM on the HOST plate
      ╱▔▔▔▔▔╲     ╱▔▔▔▔▔▔▔▔╲     ╱▔▔▔▔▔╲
     │  RAM  │   ╱           ╲   │  GPU  │
       ╲_____╱   ╲  ╱▔▔▔╲   ╱     ╲_____╱
╱▔▔▔▔▔╔╔▔▔▔▔╔╔▔╔▔▔▔▔╔╔▔╔▔▔╔▔▔▔▔▔▔╔▔▔╲   VRAM
    ╲  VRAM  ╱  ╚═════════╝  ╚══╝  ╲  HOST   ╱
      ╲______╱ ╲____________________╱
```

The tier is drawn as a dashed, translucent outline rather than a solid cube: it
is the inference engine running *on* the hardware, not a fifth component. It
carries the engine name, the model it loaded, and the running token count.

Geometry note: `iso(C, C, z)` projects onto the same screen point as
`iso(0, 0, C)` when `z == 2C`, so the first attempt at a 34-unit clearance left
the tier exactly touching the CPU cube. It sits at `z0 = 2C + 46` instead.

## How to extend
- **New engine adapter**: subclass `_BaseAdapter` in `engines.py`, map its metric
  families into the normalized dict, register it in `AppState.engines`.
- **New architecture**: add a template + a `detect_family` branch in `arch.py`;
  the graph format is shared by the frontend.
- **New metric source**: add a probe in `capability.py` and a field in
  `Sample`/`schema.py` with an explicit provenance.
