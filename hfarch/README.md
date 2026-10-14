# Model Atlas

Type a Hugging Face model name, get a picture of what is inside it.

Everything runs in the browser. No build step, no weights — a model's shape is
declared in its `config.json`. The Hub serves that file with permissive CORS,
so on a normal network the whole thing is client-side. On a network like this
one, where the browser cannot read the Hub at all, `serve.py` adds a narrow
same-origin route that reads the same file server-side.

```
python3 serve.py 8811           # static files + /api/hf/config (127.0.0.1 only)
open http://localhost:8811      # plain `python3 -m http.server` works too,
                                # where the browser can reach the Hub
```

When the Hub cannot be read, the page says exactly that: what was tried, in
order, what each host answered, and what would fix it — rather than a bare
`TypeError` or nothing. A `config.json` from this machine works too: drop one
anywhere on the page, or use *open a config.json*. The drawing is derived from
exactly the file you give it, and the source line says which.

Names follow the Hub: `Qwen/Qwen3-4B`, or `owner/name/unet` for one component
of a diffusers pipeline — `transformer`, `vae`, `text_encoder`.

## What it draws

The diagram is arranged in the order data moves. Components are boxes carrying
their parameter counts, connected by arrows, and a repeated component is drawn
as its first instance, a marker for the ones in between, and its last:

```
Decoder block ×36  →  [first]  ··· ×34 ···  [last]
```

Sixty-five identical boxes say "there are sixty-five" and take more room than
everything else on the page. The marker says the same in a quarter of the
space.

Long single-lane chains — a language model, a U-Net — wrap into rows of a
fixed width, the way a line of text wraps at the page edge: each row reads
left to right, the wrap connector leaves by the right margin and comes back
along the gutter. The column pitch never changes, so column *n* of every row
still lines up.

Families drawn, from the config alone:

| family | what it is | e.g. |
|---|---|---|
| language | a decoder stack | Qwen3, Llama |
| vision-language | a decoder stack with a vision tower joining it | Qwen3-VL |
| contrastive | two encoders compared in one shared space | CLIP |
| encoder | an encoder with no output head and nothing to sample | SD's text encoder |
| speech | an encoder over audio, a decoder over text | Whisper |
| diffusion | a U-Net, with real skip connections | SD's unet |
| dit | a transformer over patches | DiT-XL/2, PixArt |
| vae | the decoder half of a latent pipeline | SD's vae |

Click any box for its dimensions, its parameter share, and — for a repeated
block — what is inside one instance. Scroll to zoom, drag to pan. There is a
3D mode sharing the same layout; the toggle is in the header.

## The component library

`src/blocks.js` is where the drawing scheme lives: one entry per component
transformers are actually built from — attention, grouped-query attention,
gated feed-forward, RMSNorm, rotary positions, mixture of experts, patch
embedding, merger, tied output head. Each knows which config keys describe it
and how many parameters it holds, so a model nobody has seen before still
draws from its config alone.

The parameter formulas are the part that goes quietly wrong, so they are the
part that is written out:

- Attention is sized by **key/value** heads, not query heads. Grouping is the
  norm now, and sizing by the query count overstates most current models.
- SwiGLU-family feed-forward blocks have **three** matrices where plain GELU
  has two.
- Biases are counted only when the config declares them. Assuming them
  inflates every block of most current models — and makes the total read
  slightly low on models that have them but do not say so (CLIP, by 0.07%).
- A LayerNorm holds a scale **and** a shift; an RMSNorm holds a scale only.
- A tied embedding and output head are **one** matrix.

## What the numbers are allowed to be

Every number is derived from `config.json` or measured at runtime. Nothing is
estimated to match a published figure, and where a config is silent the card
says so instead of guessing:

- A diffusers U-Net's feed-forward is not in the config, so the total reads
  382M against a published ~860M, and the card says why.
- A DiT block's feed-forward width is likewise undeclared; DiT-XL/2 reads
  375.8M against a published 675M, for the same stated reason.
- CLIP's patch embedding assumes three colour channels; the config does not
  declare them, and the card says so.

This is deliberate. A viewer that quietly rounds up to match model cards is a
viewer you cannot trust on a model you do not know.

## Running the checks

```
node test/graph.test.mjs        # parameter counts against published sizes,
                                # and template × count == block for everything
python3 test/shoot.py           # draw every family in a real browser and
                                # assert no card is clipped or truncated
python3 test/shoot.py --mode 3d # same, in the 3D mode
```

`shoot.py` needs playwright (`pip install playwright && playwright install
chromium`). Both checks run against real configs fetched from the Hub with
curl into `test/cfg/` — including a gated-repo 403 body, which is worth having.

## Layout

| file | |
|---|---|
| `src/blocks.js` | the component library and the parameter formulas |
| `src/graph.js` | config → flow graph, dispatching to the family builders |
| `src/families.js` | speech, diffusion, VAE, DiT, encoder and contrastive builders |
| `src/hub.js` | fetching `config.json`, classified failures, the local cache |
| `src/render.js` | stage/lanes, row wrapping, arrow routing, pan and zoom |
| `src/prim.js` | drawing primitives: cards, wires, wraps, skips, bounds |
| `serve.py` | static server + the same-origin `/api/hf/config` route |
| `test/graph.test.mjs` | parameter counts against published sizes |
| `test/shoot.py` | browser screenshots plus in-DOM visual assertions |
| `gallery.html` | every drawing module alone, for looking before composing |

## Known gaps

- Gated repos: the mirror answers 403 with plain text and the browser refuses
  to show it cross-origin, so the gate is recognised via the local route; a
  direct-Hub read cannot see the reason at all.
- The 3D mode draws wrapped rows as depth steps — consistent with the flat
  mode's grid, but a staircase rather than a page.
- Very deep graphs (48-layer MoE plus a vision tower) wrap but may want more
  columns than fit at reading zoom; pan and zoom cover it.
