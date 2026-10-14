"""Render an archviz graph as a standalone page.

The point of the diagram is the *flow*: what goes in on the left, what comes out
on the right, and which stage does what in between. Repetition is drawn once and
labelled, because "Transformer block × 36" is a fact about the model and
thirty-six identical rectangles is a fact about the drawing.

Colours are the workbench's: a pale ground, green for the language stack, blue
for vision, grey for tensors. Nothing here is decorative — a colour is a claim
about what a node is.
"""
from __future__ import annotations

import html
import json
from typing import Any

PALETTE = {
    "input": ("#e8efe6", "#3d4a3e"),
    "embed": ("#cfe0d2", "#2e5d43"),
    "repeat_group": ("#bcd6c6", "#24503a"),
    "attention": ("#c6d9e8", "#26506a"),
    "norm": ("#e2e6dc", "#4a5445"),
    "matmul": ("#d8dcc9", "#4f5a3c"),
    "projector": ("#dcd0e4", "#5a3f6b"),
    "patchify": ("#c9d6e4", "#2b4c66"),
    "tensor": ("#f0f1e9", "#5c6255"),
    "sample": ("#e6e2d2", "#5c5738"),
}

W, H = 1180, 720
PAD = 28
BOX_W, BOX_H = 186, 54
GAP_X, GAP_Y = 92, 26


def _fmt(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}".rstrip("0").rstrip(".")
    return str(int(n))


def render(graph: dict[str, Any]) -> str:
    nodes = graph["nodes"]
    cols: dict[int, list[dict[str, Any]]] = {}
    for n in nodes:
        cols.setdefault(n.get("column", 0), []).append(n)
    order = sorted(cols)

    # lay each column out top to bottom, then place columns left to right
    laid: dict[str, tuple[float, float]] = {}
    widths = {c: max((len(str(n.get("label", ""))) for n in cols[c]), default=4)
              for c in order}
    x = PAD
    col_x = {}
    for c in order:
        col_x[c] = x
        x += BOX_W + GAP_X

    y = PAD + 40
    for c in order:
        col = sorted(cols[c], key=lambda n: n.get("order", 0))
        col_h = len(col) * BOX_H + (len(col) - 1) * GAP_Y
        y = PAD + 40 + max(0, (H - 120 - col_h) / 2)
        for n in col:
            laid[n["id"]] = (col_x[c], y)
            y += BOX_H + GAP_Y

    parts: list[str] = []
    parts.append(f'<svg viewBox="0 0 {W} {H}" width="100%" '
                 f'xmlns="http://www.w3.org/2000/svg" font-family="Inter, system-ui, sans-serif">')
    parts.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
                 'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                 '<path d="M0,0 L10,5 L0,10 z" fill="#7d8a79"/></marker></defs>')
    parts.append(f'<rect width="{W}" height="{H}" fill="#eef4ea" rx="10"/>')

    # column captions
    branch_of: dict[int, str] = {}
    for c in order:
        b = next((n.get("branch") for n in cols[c] if n.get("branch")), "")
        branch_of[c] = b
        parts.append(f'<text x="{col_x[c]}" y="{PAD + 20}" font-size="13" '
                     f'fill="#6b7a66" letter-spacing=".08em">'
                     f'{html.escape((b or "stage").upper())}</text>')

    # edges first, so boxes sit on top
    for e in graph["edges"]:
        a, b = e["from"], e["to"]
        if a not in laid or b not in laid:
            continue
        x1, y1 = laid[a]
        x2, y2 = laid[b]
        x1 += BOX_W
        y1 += BOX_H / 2
        x2 += 0
        y2 += BOX_H / 2
        mid = (x1 + x2) / 2
        d = f"M {x1} {y1} C {mid} {y1}, {mid} {y2}, {x2} {y2}"
        parts.append(f'<path d="{d}" fill="none" stroke="#9aa896" stroke-width="1.6" '
                     f'marker-end="url(#a)"/>')
        shape = (e.get("tensor") or {}).get("shape")
        # only label an edge with room for the label; a cramped one just adds
        # noise over the boxes it is describing
        if shape and abs(x2 - x1) > 64 and abs(y2 - y1) > 22:
            label = "×".join(str(s) for s in shape)
            parts.append(f'<text x="{mid}" y="{(y1 + y2) / 2 - 5}" font-size="10" '
                         f'fill="#8b9786" text-anchor="middle" '
                         f'font-family="ui-monospace, monospace">{html.escape(label)}</text>')

    for n in nodes:
        x, y = laid[n["id"]]
        fill, ink = PALETTE.get(n.get("kind"), ("#e6e9e0", "#3d4a3e"))
        parts.append(f'<g><rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" '
                     f'rx="7" fill="{fill}" stroke="{ink}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x + 12}" y="{y + 23}" font-size="13.5" fill="{ink}" '
                     f'font-weight="600">{html.escape(str(n["label"])[:22])}</text>')
        if n.get("params"):
            parts.append(f'<text x="{x + 12}" y="{y + 41}" font-size="11" fill="{ink}" '
                         f'opacity=".78" font-family="ui-monospace, monospace">'
                         f'{_fmt(n["params"])} params</text>')
        # the label already reads "block × 36" for most builders; only add the
        # badge when it would not be a repeat of what is already there
        # "× 36" and "×36" are the same fact; match either spacing
        lbl = str(n.get("label", "")).replace(" ", "")
        if n.get("repeat") and f"×{n['repeat']['count']}" not in lbl:
            parts.append(f'<text x="{x + BOX_W - 12}" y="{y + 23}" font-size="11" '
                         f'fill="{ink}" text-anchor="end" opacity=".8">'
                         f'×{n["repeat"]["count"]}</text>')
        parts.append("</g>")

    m = graph.get("meta", {})
    head = (f'{html.escape(graph["model_id"])} — {graph["family"]} — '
            f'{m.get("num_hidden_layers", "?")} layers × {m.get("hidden_size", "?")} wide, '
            f'{m.get("num_attention_heads", "?")} heads ({m.get("num_key_value_heads", "?")} kv)'
            f' — {_fmt(graph.get("params_total", 0))} parameters, derived from config')
    parts.append(f'<text x="{PAD}" y="{H - 16}" font-size="11.5" fill="#6b7a66">'
                 f'{head}</text>')
    parts.append("</svg>")
    return "".join(parts)


def page(graph: dict[str, Any]) -> str:
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in graph.get("warnings", []))
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(graph['model_id'])} architecture</title>
<style>
  body {{ margin:0; background:#e7eee2; font-family:Inter,system-ui,sans-serif;
         color:#2f3a2c; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:24px; }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  p.sub {{ margin:0 0 18px; color:#66735f; font-size:13px; }}
  .card {{ background:#f6f8f3; border:1px solid #d5ddcd; border-radius:12px; padding:8px; }}
  ul.warn {{ color:#7a6a3a; font-size:12.5px; margin:14px 2px 0; }}
</style>
<div class="wrap">
  <h1>{html.escape(graph['model_id'])}</h1>
  <p class="sub">computed from published metadata only — no weights downloaded</p>
  <div class="card">{render(graph)}</div>
  {f'<ul class="warn">{warn}</ul>' if warn else ''}
</div>"""


if __name__ == "__main__":
    import sys
    from graph import build_graph, fetch_configs

    repo = sys.argv[1]
    g = build_graph(repo, fetch_configs(repo)["config.json"])
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/{repo.split('/')[-1]}.html"
    with open(out, "w") as f:
        f.write(page(g.as_dict()))
    print(out)