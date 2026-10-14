/**
 * The drawing modules.
 *
 * Every element of the diagram, drawn once, here. The gallery imports them to
 * test each alone; the renderer imports them to compose. An earlier version
 * kept a copy in each, which guarantees they drift and that what you looked at
 * is not what ships.
 *
 * All flat. The isometric camera is gone; see prim.js for why.
 */
import { INK, KINDS, badge, band, card, el, soft, txt, wire } from "./prim.js";

const TITLE = 15;
const SUB = 12.5;

/** Text laid out inside a card: title, count badge, optional sub.
 *
 *  Named cardText, not chrome. `window.chrome` is a real object in every
 *  Chromium browser, so calling chrome() here silently invoked the
 *  browser's own API with our arguments and returned undefined: no error,
 *  no text, and cards that looked fine because card() had already run. */
function cardText(g, { x, y, w, h, label, sub, count }) {
  const pad = 12;
  const bw = count ? 10 + String(count).length * 9 + 8 : 0;
  const fit = (s, px, size) => {
    const room = Math.max(4, Math.floor(px / (size * 0.56)));
    return s.length > room ? s.slice(0, room - 1) + "…" : s;
  };
  g.append(txt(x + pad, y + 24, fit(label, w - pad * 2 - bw, TITLE), { class: "card-t" }));
  if (count) badge(g, { x: x + w - pad - (10 + String(count).length * 9), y: y + 10,
    label: `×${count}` });
  if (sub) g.append(txt(x + pad, y + 42, fit(sub, w - pad * 2, SUB), { class: "card-s" }));
  return y + (sub ? 54 : 40);
}

/** A plain component. */
export function block(g, { x, y, w, h, label, sub, count = 0, kind = "tensor",
  dims = null }) {
  const base = KINDS[kind] || KINDS.tensor;
  card(g, { x, y, w, h, fill: soft(base, 0.74), stroke: base });
  const bodyY = cardText(g, { x, y, w, h, label, sub, count });
  // the declared width goes inside its own card, not floating beside it
  if (dims) {
    const d = [dims.width && `width ${dims.width}`, dims.vocab && `vocab ${dims.vocab}`,
      dims.patch && `patch ${dims.patch}`].filter(Boolean).join("  ·  ");
    if (d) g.append(txt(x + 12, y + h - 11, d, { class: "card-d" }));
  }
  void bodyY;
}

/**
 * A repeated block.
 *
 * Once it was two blocks and a floating "×34" between them. On a grid that is
 * two boxes that have to be kept in step, and a number that belongs to neither.
 * It is now one card with the count on it and its sub-layer composition drawn
 * as a strip inside — which is what the reader actually wants from a repeated
 * block: what it is, how many, and what is in one.
 */
export function tower(g, { x, y, w, h, label, sub, count = 0, kind = "stack",
  parts = [], weights = [] }) {
  const base = KINDS[kind] || KINDS.stack;
  card(g, { x, y, w, h, fill: soft(base, 0.76), stroke: base });
  cardText(g, { x, y, w, h, label, sub, count });

  if (parts.length) {
    const tot = weights.reduce((a, b) => a + b, 0) || 1;
    const stripY = y + h - 28;
    const stripH = 16;
    const gap = 2;
    const inner = w - 24 - gap * (parts.length - 1);
    let cx = x + 12;
    parts.forEach((p, i) => {
      const pw = Math.max(3, (weights[i] / tot) * inner);
      band(g, { x: cx, y: stripY, w: pw, h: stripH, fill: KINDS[p.kind] || base,
        opacity: 0.85 });
      if (pw > 26) {
        g.append(txt(cx + pw / 2, stripY + 12, p.label,
          { class: "band-t", "text-anchor": "middle" }));
      }
      cx += pw + gap;
    });
  }
}

/**
 * An input plate: tokenised text as a deck of vectors, one pulled out.
 *
 * Kept in flat form as a stack of bars because the idea survives the projection
 * change — it is a stack, and one of them is the one being read.
 */
export function plate(g, { x, y, w, h, label, shape, count = 7, focus = 3 }) {
  const base = KINDS.tensor;
  card(g, { x, y, w, h, fill: soft(base, 0.84), stroke: base });
  g.append(txt(x + 12, y + 24, label, { class: "card-t" }));
  if (shape) g.append(txt(x + 12, y + 41, shape, { class: "card-s" }));

  const bx = x + 14, bw = w - 28;
  const top = y + 52;
  const rows = Math.max(4, Math.min(count, h - 70));
  const gap = 3;
  const bh = Math.max(5, (h - 60 - gap * (rows - 1)) / rows);
  for (let i = 0; i < rows; i++) {
    const on = i === focus;
    const bwRow = on ? bw : bw - (i > focus ? 0 : 10);
    band(g, {
      x: on ? bx - 6 : bx, y: top + i * (bh + gap), w: bwRow, h: bh,
      fill: on ? KINDS.embed : "#9fb79a", rx: 2,
    });
    if (on) {
      g.append(txt(bx + 6, top + i * (bh + gap) + bh - 1, "t₀",
        { class: "plate-t" }));
    }
  }
}

/**
 * The residual and the add where it rejoins.
 *
 * The add is the point, so it is drawn as an operator on the connection the way
 * `\elementsum` does — a disc with a + in it — and the bypass wraps over the
 * block it belongs to rather than crossing its face.
 */
export function residual(g, { x, y, w, h }) {
  const over = y - 26;
  const ax = x + 14, bx = x + w - 14;
  g.append(el("path", {
    class: "resid",
    d: `M ${ax} ${y - 4} L ${ax} ${over} L ${bx} ${over} L ${bx} ${y - 4}`,
  }));
  const r = 11;
  g.append(el("circle", { class: "sum", cx: bx, cy: over, r }));
  g.append(txt(bx, over + 5, "+", { class: "sum-t", "text-anchor": "middle" }));
}

export { wire };

/** The arrowhead, defined once so every connection has the same one. */
export function defs() {
  return el("defs", {}, [
    el("marker", {
      id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5",
      markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse",
    }, [el("path", { d: "M 0 1.5 L 9.5 5 L 0 8.5 z", fill: "#7f9478" })]),
  ]);
}

void INK;
/* ------------------------------------------------------------------ *
 * The 3D card. Same grid, same ordering, seen through a camera.
 * ------------------------------------------------------------------ */
import { iso, isoBox, isoRule } from "./prim.js";

/**
 * A component as a solid.
 *
 * The label goes on the *front face*, upright and unrotated, which is where
 * Box.sty puts its caption. Rotating text into the face's perspective is what
 * made the isometric version unreadable; putting it on the face and leaving it
 * horizontal keeps the solid and recovers the text.
 */
export function isoCard(g, { x, y, z = 0, w, d, h, label, sub, count = 0,
  kind = "tensor", parts = [], weights = [] }) {
  // start from a light tint of the hue: the faces are then all light and
  // the dark label on the front face reads on every one of them
  const base = soft(KINDS[kind] || KINDS.tensor, 0.66);
  isoBox(g, { x, y, z, w, d, h, color: base });

  // front face: the plane at y+d, spanning x..x+w and z..z+h
  const face = iso(x + w / 2, y + d, z + h * 0.42);
  const room = Math.max(4, Math.floor((w * 0.86) / (12.5 * 0.56)));
  const title = label.length > room ? label.slice(0, room - 1) + "…" : label;
  g.append(txt(face[0], face[1], title,
    { class: "iso-t", "text-anchor": "middle" }));
  if (sub) {
    g.append(txt(face[0], face[1] + 15, sub, { class: "iso-s", "text-anchor": "middle" }));
  }

  // the count, on a badge above the block so it never fights the label
  if (count) {
    const p = iso(x + w - 16, y + d - 10, z + h + 10);
    g.append(el("circle", { class: "iso-badge", cx: p[0], cy: p[1], r: 15 }));
    g.append(txt(p[0], p[1] + 5, `×${count}`, { class: "iso-badge-t", "text-anchor": "middle" }));
  }

  // a repeated block's composition, as divisions across its two visible faces
  if (parts.length && h >= 60) {
    const tot = weights.reduce((a, b) => a + b, 0) || 1;
    const FLOOR = 0.08;
    const free = Math.max(0, 1 - FLOOR * parts.length);
    let zz = 0;
    parts.forEach((p, i) => {
      zz += (FLOOR + ((weights[i] || 1) / tot) * free) * h;
      if (i < parts.length - 1) isoRule(g, { x, y, z: z + zz, w, d });
    });
  }
}

/** A plate as a standing deck of sheets, the current one pulled forward. */
export function isoPlate(g, { x, y, z = 0, w, d, h, label, shape, count = 7, focus = 3 }) {
  const base = soft(KINDS.tensor, 0.6);
  const rows = Math.max(4, Math.min(count, 7));
  const gap = 1.5;
  const bh = Math.max(4, (h - 16 - gap * (rows - 1)) / rows);
  for (let i = 0; i < rows; i++) {
    const on = i === focus;
    isoBox(g, {
      x: x + (on ? -10 : 0), y: y + (on ? d + 7 : i * ((d * 0.7) / rows)),
      z, w: on ? w * 0.78 : w * 0.7, d: (d * 0.7) / rows - 1,
      h: bh, color: on ? soft(KINDS.embed, 0.5) : base, top: 0.4, sw: 0.6,
    });
  }
  const p = iso(x + w * 0.4, y + d + 12, z + h * 0.5);
  g.append(txt(p[0], p[1], label, { class: "iso-t", "text-anchor": "middle" }));
  if (shape) {
    g.append(txt(p[0], p[1] + 14, shape, { class: "iso-s", "text-anchor": "middle" }));
  }
}

/** The residual, and the add where it rejoins — the same operator, in 3D. */
export function isoResidual(g, { x, y, z = 0, w, d, h }) {
  const a = iso(x - 10, y + d / 2, z);
  const b = iso(x + w + 10, y + d / 2, z);
  const top = a[1] - 18;
  g.append(el("path", { class: "resid",
    d: `M ${a[0]} ${a[1]} L ${a[0]} ${top} L ${b[0]} ${top} L ${b[0]} ${b[1]}` }));
  g.append(el("circle", { class: "sum", cx: b[0], cy: top, r: 11 }));
  g.append(txt(b[0], top + 5, "+", { class: "sum-t", "text-anchor": "middle" }));
}
