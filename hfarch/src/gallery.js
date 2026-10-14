/**
 * The module gallery.
 *
 * Each element drawn alone, at a size where its problems are obvious, before
 * anything is composed. Imports the same modules the renderer composes with,
 * so a cell cannot drift from what ships.
 */
import { el, txt } from "./prim.js";
import { block, defs, plate, residual, tower, wire } from "./modules.js";

const NS = "http://www.w3.org/2000/svg";

const CELLS = [
  ["block", "a plain component — name, count, parameter share",
    (g) => block(g, { x: 30, y: 30, w: 180, h: 104, label: "Token embedding",
      sub: "622.3M parameters", kind: "embed", dims: { width: 4096 } })],
  ["tower", "a repeated block — count on the card, sub-layers inside",
    (g) => tower(g, { x: 30, y: 20, w: 180, h: 104, label: "Decoder block",
      sub: "192.9M each · 6.95B total", count: 36,
      parts: [{ label: "norm", kind: "norm" }, { label: "attn", kind: "attention" },
        { label: "mlp", kind: "mlp" }],
      weights: [1, 34, 49] })],
  ["plate", "an input plate — one bar per token, one pulled out",
    (g) => plate(g, { x: 30, y: 24, w: 180, h: 104, label: "Tokens", shape: "B×T" })],
  ["residual", "the bypass and the add where it rejoins",
    (g) => residual(g, { x: 40, y: 60, w: 180, h: 104 })],
  ["wire", "a connection — straight when level, one bend when not",
    (g) => wire(g, { p0: [200, 40], p1: [340, 40], shape: "B×T×4096" })
      || wire(g, { p0: [200, 130], p1: [340, 130], shape: "S×1152" })],
];

const host = document.getElementById("cells");
for (const [name, note, draw] of CELLS) {
  const wrap = document.createElement("div");
  wrap.className = "cell";
  wrap.id = `cell-${name}`;
  const svg = document.createElementNS(NS, "svg");
  svg.append(defs());
  const root = el("g", {});
  svg.append(root);
  draw(root);
  wrap.append(svg);
  const h = document.createElement("h2");
  h.textContent = name;
  wrap.append(h);
  const p = document.createElement("div");
  p.className = "note";
  p.textContent = note;
  wrap.append(p);
  host.append(wrap);
  // measured only after the cell is in the document: getBBox on a detached
  // SVG throws, and the failure is silent -- the cell renders as a crop of
  // nothing.
  const bb = root.getBBox();
  svg.setAttribute("viewBox",
    `${(bb.x - 24).toFixed(0)} ${(bb.y - 20).toFixed(0)} `
    + `${(bb.width + 48).toFixed(0)} ${(bb.height + 40).toFixed(0)}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
}

window.__galleryReady = document.querySelectorAll(".cell").length;
