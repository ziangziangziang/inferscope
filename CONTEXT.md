# Context for the next agent

Read this before changing anything. It records what the two projects are for,
how they connect, and — more usefully — which parts of this setup will waste
your time if you do not know about them.

---

## 1. What this is

One repository, two related projects.

### InferScope — the workbench (`https://inferscope.radeon.firstdg.ai`)

An inference-observability workbench for the GPU box. It runs models, watches
them, and draws what it finds: a live GPU rail, a bounded collector writing
Parquet, per-engine views for chat / ASR / TTS / image / video, and an
architecture view of whichever model is loaded.

Its reason for existing: **inference is opaque**. You can see a GPU is busy
and a token rate is coming out, but not what the machine is actually doing, how
the model is shaped, or whether a number in a config is real. InferScope tries
to make the path from a prompt to a kernel trace visible and honest about its
provenance.

### Model Atlas / `hfarch` — the architecture viewer (`…/hfarch`)

The first step of the current work. Type a Hugging Face model name, get a
picture of what is inside it. Runs entirely in the browser: a model's shape is
declared in its `config.json`, the Hub serves that with permissive CORS, so no
weights, no backend, no build step.

It is deliberately a *separate* app rather than another tab in the workbench,
because it answers a different question. InferScope describes **the model you
are running**. hfarch describes **any model you can name**, including ones that
were never downloaded and will never run on this box. The shared part is the
idea: every number shown is derived from a config or measured at runtime, and
anything unknown is reported as unknown.

**Current assignment: improve hfarch.** Treat everything below as the
starting point.

---

## 2. Connection strings

| what | value |
|---|---|
| git remote | `git@github.com:ziangziangziang/inferscope.git` (branch `main`) |
| ssh to the GPU box | `root@36.150.116.206`, port `31699` |
| deployed source | `/workspace/inferscope/` on that box |
| public URL | `https://inferscope.radeon.firstdg.ai` |
| hfarch URL | `https://inferscope.radeon.firstdg.ai/hfarch/` |
| local checkout | `/Users/zziang/Documents/projects/test/test/inferscope` |

Egress from the box is **only** through a cluster proxy. Neither the server nor
a browser on this network can reach `huggingface.co` directly:

```
http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
```

Note `lablab`, not `lab`. This is set in `run.sh`, which is **not in the repo**.

GPU box, for context: ~48 GiB VRAMD, ~500 GiB RAM. One heavy engine at a time —
chat `8081`, ASR `8082`, TTS `8083`, image `8084`, video `8085`; llama.cpp is
owned by `chat.py`. `/workspace` is small (~24 GiB free), so large models live
on the root overlay under `/models` and are **not** persistent.

---

## 3. Repository layout

```
inferscope/            backend package (FastAPI, collectors, engines)
  api.py               REST + SSE, static frontend mount, /api/hf/config
  chat.py asr_engine.py tts_server.py image_server.py video_server.py
  enginectl.py         load/offload/ownership for the non-chat engines
  gguf.py gguf_arch.py diffusers_arch.py asr_arch.py   architecture builders
ui/                    React + Vite workbench
  src/components/iso.tsx       the isometric camera — read this first
  src/components/sceneModel.ts caption layout rules
hfarch/                the browser viewer (no build step, plain ES modules)
archviz/               standalone architecture CLI (earlier, partly superseded)
tests/                 14 pytest files
```

**`ui/src/components/iso.tsx` is the reference for anything visual.** hfarch
deliberately uses the same camera (`iso(x,y,z)`), the same three-face boxes, the
same painter's ordering and the same ground, so the two views of the same model
look like one product.

---

## 4. hfarch in detail

Nine source files, ~1,700 lines total. **Read `hfarch/CONTEXT.md` first** —
it is the current, detailed record of the app's state, decisions and traps;
the notes below are the short form.

| file | role |
|---|---|
| `src/blocks.js` | the component library and every parameter formula, `family()` |
| `src/families.js` | speech, diffusion, VAE, DiT, encoder and contrastive builders |
| `src/graph.js` | config → flow graph; dispatches to the family builders |
| `src/hub.js` | fetching `config.json`, classified failures, mirror failover, cache |
| `src/prim.js` | drawing primitives: cards, wires, wrap/skip/drop routes |
| `src/modules.js` | the drawing modules, shared with the gallery |
| `src/render.js` | grid layout with row wrapping, 2D and 3D renderers |
| `src/main.js` | status line, error panel, drop-a-config, mode toggle |
| `serve.py` | static server + same-origin `/api/hf/config`, 127.0.0.1 only |

### The component library is the substance of the project

Transformers are assembled from the same handful of parts, so the drawing is
assembled from the same vocabulary. `blocks.js` has one entry per part —
attention, grouped-query attention, gated feed-forward, RMSNorm, rotary
positions, mixture of experts, patch embedding, merger, tied output head — each
knowing which config keys describe it and how many parameters it holds. A model
nobody has seen still draws from its config alone.

### The parameter formulas are where a diagram goes quietly wrong

They are written out deliberately:

- attention is sized by **key/value** heads, not query heads;
- SwiGLU-family feed-forward has **three** matrices where plain GELU has two;
- biases counted only when the config declares them;
- a tied embedding and output head are **one** matrix.

Two Mixture-of-Experts traps, both found by the test suite, both large:

1. **Experts replace the dense feed-forward block; they do not sit beside it.**
   Counting both turned Qwen3-30B-A3B into a 235B model.
2. **Experts are sized by `moe_intermediate_size` (768), not
   `intermediate_size` (6144).** Reading the wrong one is an 8× error.

Counts currently land on the published sizes: 4.02B / 4.0B, 30.53B / 30.5B,
8.46B / ~8.2B.

### Running it

```
cd hfarch && python3 -m http.server 8811     # any static server
node test/graph.test.mjs                     # parameter counts vs published sizes
```

There is no build step and no dependency install. ES modules need `http://`,
not `file://`.

### Deployment

`hfarch/` is copied into `ui/dist/hfarch/` **after every build**, by
`ui/scripts/copy-hfarch.mjs` (wired into `npm run build`). Vite empties
`dist/`, so without that step a rebuild silently deletes `/hfarch`.

```
npm --prefix ui run build          # builds UI + copies hfarch
rsync -az -e "ssh -p 31699" hfarch/ root@36.150.116.206:/workspace/inferscope/hfarch/
rsync -az --delete -e "ssh -p 31699" hfarch/ \
  root@36.150.116.206:/workspace/inferscope/ui/dist/hfarch/
```

Restart the app (see the trap in §7).

---

## 5. How to verify a change

Use a browser. `node test/graph.test.mjs` covers the arithmetic; nothing covers
how it looks.

**The remote browser is broken.** Chromium on the GPU box crashes on *any*
page, including a static one — not an app bug, the host. Install Playwright's
Chromium locally instead:

```
python3 -m playwright install chromium
```

**Testing against a stub.** From this machine `huggingface.co` is unreachable
and the mirror's CDN redirect drops CORS, so a live fetch never completes in the
browser. Inject a stub before the module loads:

```js
// init script: return a local config for any config.json request
(function(){var CFG=<contents of test/cfg/….json>;var real=window.fetch;
window.fetch=async function(u){var s=String(u);
if(s.indexOf('config.json')>=0)return new Response(JSON.stringify(CFG),
  {headers:{'Content-Type':'application/json'}});
return real(u);};})();
```

Two traps here, both hit already: an init script cannot use top-level `await`,
and `new Response(CFG)` where `CFG` is a JS **object** stringifies to
`[object Object]` — you need `JSON.stringify(CFG)`. Either one fails as
`config.json not readable`, which looks like a config problem.

**Screenshot, then look.** Several bugs in this project were invisible in code
and obvious in a picture: cells collapsed into a dot because the patch helper
used world-unit edges; a caption rendering as `GUAGE STACK` because the viewBox
was computed before the captions reserved their space.

Prefer assertions over eyeballing where possible — the connection check walks
every path and asserts each segment moves on one axis only:

```js
const pts = [...document.querySelectorAll('#svg .edge')]
  .flatMap(p => p.getAttribute('d').match(/-?[\d.]+/g).map(Number));
```

---

## 6. Open issues in hfarch

Ordered by how much they hurt. **Read `hfarch/CONTEXT.md` for the full
state** — several items that used to live here (families that were detected
but not drawn, the silent network failure, letterboxed single-lane diagrams,
text encoders and DiT) are done as of October 2026.

1. **Very deep graphs may want more columns than fit at reading zoom.** They
   wrap now, but a 48-layer MoE plus a vision tower still ends up wide; pan
   and zoom cover it.
2. **A gated repo is only recognisable through the local route.** The mirror
   answers 403 without CORS headers, so the browser reports a CORS failure
   and cannot see the reason; `serve.py` can see it and says so.
3. **`/api/hf/config` is unauthenticated.** `serve.py` binds 127.0.0.1, three
   filenames, two hosts, validated segments — but if this app is ever
   deployed publicly again, that route needs a token or a real allowlist.
4. **Arrow heads can clip behind blocks** when a wire passes over one; a
   trade that has not been worth un-trading yet.
5. **A parallel `archviz/` CLI exists** and is superseded, but kept: it emits
   JSON/markdown tables, which is a different output. Delete it before
   extending it.

---

## 7. Traps

**`run.sh` is not in the repo.** It lives only at
`/workspace/inferscope/run.sh` and holds the environment: proxy, model paths,
ports. Changes to it cannot be committed and will be lost if the box is
rebuilt.

**Never restart with `pgrep -f "inferscope serve"`.** That pattern matches the
shell you are running it in, so you kill your own command and conclude the
service is down. Use:

```sh
kill -9 $(ps -eo pid,cmd | grep "[i]nferscope serve" | awk '{print $1}')
cd /workspace/inferscope && setsid ./run.sh > /tmp/inf.log 2>&1 < /dev/null & disown
```

**The static mount is last.** `app.mount("/", StaticFiles(ui/dist))` catches
everything, so a mistyped API path returns the UI's `index.html` with a 200 —
not a 404. If a new endpoint 404s, check you registered it *above* the mount.

**Do not trust a bare `except`.** A `{r.json()}` bug (building a *set*
containing a dict) surfaced as `TypeError: unhashable type: 'dict'`, but a
catch-all reported it as "config.json is not JSON" and sent me hunting the
wrong host for several rounds. Name the exception.

**Residual risk in the parameters, not the picture:** anything a config does
not declare is reported as unknown rather than guessed. Keep it that way.

---

## 8. Where to start

Improve hfarch. In rough order of value:

1. Fit-to-window / column-count choice for very deep graphs.
2. Wire `family()`'s `omni` case (audio_config) — still detected, still not drawn.
3. More families: a T5-style encoder-decoder would slot into the speech
   builder's shape with a different config surface.
4. Deployment does not exist right now (the GPU box is gone). If the app ever
   goes public again, `/api/hf/config` needs a token first.

Whatever you change, keep the two invariants that make this project worth
trusting: every number is **derived from a config or measured at runtime**, and
whatever is not known is **shown as not known**.