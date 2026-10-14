# Model Atlas

Type a Hugging Face model name, get a picture of what is inside it.

Everything runs in the browser. No backend, no build step, no weights — a
model's shape is declared in its `config.json`, and the Hub serves that file
with permissive CORS, so the whole thing is client-side.

```
python3 -m http.server 8811     # any static server; ES modules need http://
open http://localhost:8811
```

## What it draws

The diagram is arranged in the order data moves. Components are boxes sized by
the parameters they hold, connected by arrows, and a repeated component is drawn
as its first instance, a marker for the ones in between, and its last:

```
Decoder block ×36  →  [first]  ··· ×34 ···  [last]
```

Sixty-five identical boxes say "there are sixty-five" and take more room than
everything else on the page. The marker says the same in a quarter of the space.

For a vision-language model the picture and the text run in two column-aligned
lanes, so the point where visual tokens join the text stream is visible rather
than described.

Click any box for its dimensions, its parameter share, and — for a repeated
block — what is inside one instance. Scroll to zoom, drag to pan.

## The component library

`src/blocks.js` is where the drawing scheme lives: one entry per component that
transformers are actually built from — attention, grouped-query attention,
gated feed-forward, RMSNorm, rotary positions, mixture of experts, patch
embedding, merger, tied output head. Each knows which config keys describe it
and how many parameters it holds, so a model nobody has seen before still draws
from its config alone.

The parameter formulas are the part that goes quietly wrong, so they are the
part that is written out:

- Attention is sized by **key/value** heads, not query heads. Grouping is the
  norm now, and sizing by the query count overstates most current models.
- SwiGLU-family feed-forward blocks have **three** matrices where plain GELU
  has two.
- Biases are counted only when the config declares them. Assuming them inflates
  every block of most current models.
- A tied embedding and output head are **one** matrix.

## Two bugs worth recording

Both were caught by `test/graph.test.mjs` against published model sizes, and
both are the kind that look fine on screen:

1. **MoE double-counting.** Experts *replace* the dense feed-forward block;
   they do not sit beside it. Counting both turns Qwen3-30B-A3B into a 235B
   model.
2. **Wrong expert width.** Experts are sized by `moe_intermediate_size` (768),
   not `intermediate_size` (6144). Reading the wrong one is an 8× error.

```
$ node test/graph.test.mjs

Qwen_Qwen3-30B-A3B.json  [language]
  48 layers x 2048 wide, 32q/4kv heads, 128 experts top-8
  computed 30.53B                      (published: 30.5B)
Qwen_Qwen3-4B.json  [language]
  36 layers x 2560 wide, 32q/8kv heads, vocab 151936, tied
  computed 4.02B                       (published: 4.0B)
Qwen_Qwen3-VL-8B-Instruct.json  [vision-language]
  computed 8.46B                       (published: ~8.2B)

all checks passed
```

Every number above is derived from `config.json`. Nothing is measured, and
nothing is guessed: where a config is silent the drawing says so.

## Layout

| file | |
|---|---|
| `src/blocks.js` | the component library and the parameter formulas |
| `src/graph.js` | config → flow graph, including the vision branch and MoE |
| `src/hub.js` | fetching `config.json`, with mirror failover and a deadline |
| `src/render.js` | stage/lane layout, elision, arrows, pan and zoom |
| `test/graph.test.mjs` | parameter counts against published sizes |

## Known gaps

- Only `huggingface.co` sends CORS headers for the mirror's CDN redirect, so
  the fallback works from some networks and not others.
- Gated repos cannot be read without a token; the app says so rather than
  drawing an empty page.
- Only the transformer families are handled. Diffusion UNets, VAEs and the
  speech encoders are recognised but not drawn yet.
- The drawing does not fit itself to the window; pan and zoom cover it.