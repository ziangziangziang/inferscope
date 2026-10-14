# Context for the next agent

Read this before changing anything. It records where hfarch stands now, what
was just done and why, and which parts of this setup will waste your time if
you do not know about them. The repository-wide context (both projects, the
GPU box, deployment) is in `../CONTEXT.md`.

---

## 1. What this is

A browser-only viewer for Hugging Face model architectures. Type a model name,
get a picture of what is inside it. No backend, no build step, no weights: a
model's shape is declared in `config.json`, and the Hub serves it with
permissive CORS, so it runs client-side — on networks where the browser can
reach the Hub. On networks where it cannot, `serve.py` adds one narrow
same-origin route that reads the same three metadata files server-side.

Two invariants, and they are the whole point of the project:

- Every number is **derived from a config, or measured at runtime**. Never
  estimated to match a published figure.
- Whatever is not known is **shown as not known**. Where a config is silent
  (a diffusers feed-forward, CLIP's colour channels), the card says so. The
  U-Net total reads 382M against a published ~860M on purpose. Do not "fix"
  it by guessing.

## 2. Current state — what changed last session

The GPU box is gone; this now runs locally only.

**The Hub failure is legible and the app works on this network.**
`huggingface.co` is unreachable here and `hf-mirror.com` answers its CDN
redirect with `Access-Control-Allow-Origin: https://hf-mirror.com`, so a
browser on localhost gets nothing and used to report it as a silent
`TypeError`. Now:

- `src/hub.js` classifies every failure (unreachable / cors / timeout /
  gated / missing / no route on this server) — a no-cors probe tells "the
  network is fine and the refusal is CORS" apart from "unreachable".
- A failed load shows a panel: each host tried, what it answered, how long it
  took, and the remedies. It stays up until the next load.
- `serve.py` (stdlib only, binds 127.0.0.1) serves the static files plus
  `/api/hf/config?repo=owner/name[/subfolder]`, reading config.json with
  urllib, where the network is fine. Deliberately narrow: three filenames,
  two allowlisted hosts, segments validated so no `..` can get through.
- The per-host outcome is remembered in localStorage; once the Hub is known
  blocked, the origin route is tried first (seconds instead of 25+).
- A `config.json` from this machine works: drop it anywhere on the page, or
  use the *open a config.json* button. The source line says where the config
  came from — cached, hub host, this server's route, or a dropped file.
- Repo names may carry one subfolder (`facebook/DiT-XL-2-256/transformer`),
  which is how diffusers pipelines expose their parts.

**Row wrapping.** Single-lane diagrams used to letterbox (a language model
read 8.6:1). `layout()` now flows each lane's columns into rows of a width
chosen to land the whole diagram near the window's aspect (target 1.62, never
wrapping if it fits 3.1 unwrapped). Column pitch is untouched, so columns
align across rows; reading order is untouched. The connector out of a row's
end goes round the right margin and back along the gutter — `wrapWire`.
Lane-to-lane joins drop into the target's top edge (`dropWire`), or round the
margin when lanes intervene (`marginDrop`). U-Net skips route through the row
gutters. Both 2D and 3D share this layout; in 3D a wrapped row is a step back
in depth.

**New families.** Text encoders, CLIP, and DiT are built now:

- `encoder` — CLIPTextModel, BERT-style base models. No output head, no
  logits, no sampling; the drawing ends at the vector it emits.
- `contrastive` — CLIPModel. Two towers, each ending in a projection, meeting
  in a shared space with the one learned logit-scale parameter. SD's
  text encoder config (`sd-clip-text.json`) is the `encoder` case; the full
  CLIP configs are the `contrastive` case.
- `dit` — diffusers `Transformer2DModel` over patches: DiT, PixArt. Width is
  derived as `num_attention_heads × attention_head_dim`. The feed-forward is
  undeclared in these configs and is not counted (DiT-XL/2 reads 375.8M
  against a published 675M; the card says why). PixArt's `ada_norm_single`
  carries a learned position table and a caption projection; DiT's
  `ada_norm_zero` carries a fixed sin-cos grid.

**Corrections to shared formulas** (all landed by the tests, none by eye):

- `norm()` now counts a LayerNorm as 2×width (scale *and* shift) and RMSNorm
  as 1×width. Whisper's builder had been declaring `rms_norm_eps` to reuse
  the formula, which silently halved every norm the model has.
- `vision()`'s per-block norm width follows the same rule, and the VLM vision
  template in `graph.js` uses `vis.nnW` so the template × count check holds.

Counts now land: CLIP ViT-L/14 at 427,310,849 against a published
427,616,846 (−0.07%, the undeclared biases); SD's text encoder at 123.0M
against ~123M; PixArt at 308.4M + its undeclared feed-forward ≈ 611.8M
published.

## 3. Layout of the source

| file | role |
|---|---|
| `src/blocks.js` | component library, parameter formulas, `family()` |
| `src/families.js` | speech, diffusion, VAE, DiT, encoder, contrastive builders |
| `src/graph.js` | config → flow graph; dispatches to the family builders |
| `src/hub.js` | classified Hub failures, origin route, localStorage cache |
| `src/prim.js` | primitives: cards, wires, wrap/skip/drop routes, bounds |
| `src/modules.js` | the drawing modules (flat + 3D), shared with the gallery |
| `src/render.js` | layout (stages, lanes, wrapping), 2D and 3D renderers |
| `src/main.js` | status line, error panel, drop-a-file, mode toggle |
| `serve.py` | static server + `/api/hf/config`, 127.0.0.1 only |
| `test/graph.test.mjs` | counts against published sizes; template × count |
| `test/shoot.py` | browser screenshots + in-DOM assertions (playwright) |
| `test/cfg/` | real configs fetched from the Hub by curl, incl. a gated 403 |
| `gallery.html` | each drawing module alone; compose only once parts look right |

## 4. How to run and verify

```
python3 serve.py 8811                 # then open http://localhost:8811
node test/graph.test.mjs              # the arithmetic, no dependencies
python3 test/shoot.py                 # draw every family, 2D
python3 test/shoot.py --mode 3d       # and 3D
```

`shoot.py` uses playwright with `--no-proxy-server` — this sandbox exports
`HTTP_PROXY`, which puts `127.0.0.1` through a proxy that answers for it, and
that proxy has cost real debugging time already. Chromium headless shell is
installed under `~/Library/Caches/ms-playwright/`.

`shoot.py` asserts, in the live DOM: every card inside the viewBox (all four
corners), no label truncated with an ellipsis, no text wider than its card,
and aspect inside a usable range. It also screenshots into `test/shots/` —
look at them; several bugs here were invisible in code and obvious in a
picture.

The rest of the old verification advice still applies: a test stub injected
via `addInitScript` cannot use top-level await, and `new Response(obj)` needs
`JSON.stringify`. Either mistake surfaces as "config.json not readable".

## 5. Decisions made, and the reasoning

- **`serve.py` instead of a proxy in the page.** A page-side proxy cannot
  exist (CORS is enforced by the browser); a server-side read can. The route
  is narrower than the one the old GPU box ran: same three filenames, but
  subfolders allowed, segments validated against `[A-Za-z0-9][\w.-]*`, bound
  to 127.0.0.1, answers cached per repo in memory.
- **Origin-first once the Hub is known blocked.** Direct attempts still
  happen first on a fresh browser (the Hub is the honest source), but each
  host's outcome is remembered, and a known-blocked Hub no longer costs two
  timeouts per load.
- **Biases stay uncounted unless declared**, even though CLIP's biases are
  certain. The rule is what keeps the count trustworthy on models nobody has
  checked; 0.07% on CLIP is the price, and it is stated on the record here.
- **`archviz/` stays, explicitly superseded.** It is a CLI that emits JSON
  and markdown tables — a different output from a different era, but small,
  working, and sometimes the right shape for the job. Delete it if you find
  yourself fixing a bug in it rather than in hfarch.
- **The joint "Shared space" node is its own lane**, so both towers drop into
  it without a lane's cards sitting between a tower and its target.

## 6. Traps — each one cost real time and none is visible in the source

- `txt()` is a CONSTRUCTOR, not an appender. Every call must be
  `g.append(txt(...))`. Discarding the return value builds an element and
  silently drops it; the drawing looks fine and all the text is missing.
- Never name a helper `chrome`. `window.chrome` is a real object in every
  Chromium browser: no error, no text.
- `getBBox()` on a detached SVG throws or returns nothing, silently — bounds
  stay Infinity and the cell renders as a crop of nothing. Measure only after
  the node is in the document.
- CSS `[hidden]` loses to a later `display:` rule. The `.why` panel has an
  explicit `.why[hidden] { display: none }` for exactly this reason; keep it
  when adding overlay panels.
- `getBBox()` returns a rect whose `.x`/`.y` are already the min corner; do
  not add `w`/`h` twice.
- The Hub answers a gated repo with 403 and no CORS headers, so the browser
  reports a *CORS* failure, not a 403 — the gate is only recognisable through
  the server-side route. That is why `viaOrigin` matches `gated` against the
  body's `detail` *and* its per-host detail.
- `python3 -m http.server` answers `/api/hf/config` with a 404, which used to
  surface as a generic failure; it is now its own diagnostic ("served by a
  plain static server — restart as serve.py").
- Playwright in this sandbox: launch with
  `--no-proxy-server --proxy-server=direct:// --proxy-bypass-list=*`, or
  localhost requests intermittently die with ERR_CONNECTION_RESET.
- hf-mirror rate-limits bursts of config reads (HTTP 429 wrapped in an HTML
  page). `serve.py` skips the two optional files when config.json fails, so a
  dead repo costs one request per host rather than six.
