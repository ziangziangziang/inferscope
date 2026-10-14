/**
 * The module gallery.
 *
 * Composing first and judging afterwards does not work: a module that looks
 * fine alone can be unreadable in a row of twelve, and a layout mistake gets
 * mistaken for a bad module. So every piece is built and looked at here, on its
 * own, at a size where its problems are obvious, before anything is assembled.
 *
 * It imports the same modules the renderer composes with. Nothing is
 * re-implemented here, so a cell cannot drift from what actually ships.
 */
import { Bounds, KINDS, box, el } from "./iso.js";
import { edges, elision, plate, residual, tower, wire } from "./modules.js";

const NS = "http://www.w3.org/2000/svg";

const CELLS = [
  ["block", "a plain component — three faces, solid enough to read as a volume",
    (r) => box(r, { x: 60, y: 20, z: 0, w: 74, d: 46, h: 60, color: KINDS.embed })],
  ["tower", "a repeated block, sectioned — one block, faces divided by sub-layer",
    (r) => tower(r, {})],
  ["plate", "an input plate — one sheet per token, the current one pulled out",
    (r) => plate(r, {})],
  ["elision", "first … ×N … last", (r) => elision(r, {})],
  ["residual", "the bypass and the add where it rejoins", (r) => residual(r, {})],
  ["wire", "a connection, orthogonal, tensor written underneath", (r) => wire(r, {})],
  ["edges", "dimensions on the three edges they measure", (r) => edges(r, {})],
];

const host = document.getElementById("cells");
for (const [name, note, draw] of CELLS) {
  const wrap = document.createElement("div");
  wrap.className = "cell";
  wrap.id = `cell-${name}`;
  const svg = document.createElementNS(NS, "svg");
  svg.append(el("defs", {}, [
    el("marker", { id: "gh", viewBox: "0 0 10 10", refX: "8.6", refY: "5",
      markerWidth: "4.2", markerHeight: "4.2", orient: "auto-start-reverse" },
      [el("path", { d: "M 0 2 L 9 5 L 0 8 z", fill: "#8fa787" })]),
  ]));
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

  // The viewBox has to be measured *after* the cell is in the document.
  // getBBox on a detached SVG throws or returns zeroes, and the failure is
  // silent: the bounds stay Infinity and the cell renders as a crop of
  // nothing, which is exactly what happened the first time.
  const bb = root.getBBox();
  svg.setAttribute("viewBox",
    `${(bb.x - 30).toFixed(0)} ${(bb.y - 24).toFixed(0)} `
    + `${(bb.width + 60).toFixed(0)} ${(bb.height + 48).toFixed(0)}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
}

window.__galleryReady = document.querySelectorAll(".cell").length;