/**
 * The shared isometric primitives.
 *
 * One camera for the whole viewer, lifted out of the renderer so that every
 * piece — and the gallery that tests each piece alone — goes through it. If the
 * camera lives in the layout code, every module quietly acquires its own
 * perspective and the drawing stops looking like one object.
 */

const COS30 = Math.cos(Math.PI / 6);
const SIN30 = 0.5;

/** World point to screen point. Larger z is higher on screen. */
export const iso = (x, y, z) => [(x - y) * COS30, (x + y) * SIN30 - z];

export const poly = (pts) =>
  pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");

export const INK = "#2f3a2c";

/** Painter's-algorithm depth: larger x+y is nearer the viewer. */
export const depth = (x, y) => x + y;

/** Face colours, one per kind of component. */
export const KINDS = {
  tensor: "#a9b4a0",
  embed: "#7cb98f",
  stack: "#4c9e69",
  attention: "#6d9fd2",
  norm: "#aab59f",
  mlp: "#b2bd7d",
  projector: "#9a7ec1",
  moe: "#c99962",
};

/** Lighten (amt > 0) or darken (amt < 0) a hex colour, staying in gamut. */
export function tint(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const t = amt >= 0 ? v + (255 - v) * amt : v * (1 + amt);
    return Math.max(0, Math.min(255, Math.round(t)));
  });
  return `#${ch.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

export function el(tag, attrs = {}, kids) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v);
  for (const k of [].concat(kids || [])) if (k) e.append(k);
  return e;
}

export function txt(x, y, s, attrs = {}) {
  return el("text", { x, y, ...attrs }, document.createTextNode(s));
}

/**
 * One box: the three faces this camera can see.
 *
 * The faces are **opaque and tinted**, not the same colour at three alphas.
 * The workbench gets away with alpha because its parts sit on a solid ground
 * plate with nothing behind them; here a connector runs behind every block, so
 * a translucent face shows the wire through the block and the drawing reads as
 * a mistake. Tinting gives the same three-tone volume with nothing behind it
 * showing.
 *
 * Light comes from above: the lid is the lightest, the left face mid, the right
 * face darkest.
 */
export function box(g, { x, y, z, w, d, h, color, solid = true,
  top = 0.52, left = -0.04, right = -0.26, stroke = INK, sw = 0.9 }) {
  const x1 = x + w, y1 = y + d, z1 = z + h;
  const face = solid
    ? [tint(color, left), tint(color, right), tint(color, top)]
    : [color, color, "#ffffff"];
  const op = solid ? 1 : null;
  g.append(el("polygon", {
    points: poly([iso(x, y1, z), iso(x, y1, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
    fill: face[0], "fill-opacity": solid ? op : left === -0.04 ? 0.22 : 0.22,
    stroke, "stroke-width": sw,
  }));
  g.append(el("polygon", {
    points: poly([iso(x1, y, z), iso(x1, y, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
    fill: face[1], "fill-opacity": solid ? op : 0.14,
    stroke, "stroke-width": sw,
  }));
  g.append(el("polygon", {
    points: poly([iso(x, y, z1), iso(x1, y, z1), iso(x1, y1, z1), iso(x, y1, z1)]),
    fill: face[2], "fill-opacity": solid ? op : 0.62,
    stroke, "stroke-width": sw,
  }));
}

/** Accumulator for a viewBox that actually fits what was drawn. */
export class Bounds {
  x0 = Infinity; y0 = Infinity; x1 = -Infinity; y1 = -Infinity;
  add(x, y) {
    this.x0 = Math.min(this.x0, x); this.x1 = Math.max(this.x1, x);
    this.y0 = Math.min(this.y0, y); this.y1 = Math.max(this.y1, y);
    return this;
  }
  addBox(x, y, z, w, d, h) {
    for (const p of [iso(x, y, z), iso(x + w, y + d, z + h),
      iso(x, y + d, z), iso(x + w, y, z)]) this.add(p[0], p[1]);
    return this;
  }
  addP(p, r = 0) { return this.add(p[0] - r, p[1] - r).add(p[0] + r, p[1] + r); }
  viewBox(pad = 40) {
    return `${(this.x0 - pad).toFixed(0)} ${(this.y0 - pad).toFixed(0)} `
      + `${(this.x1 - this.x0 + pad * 2).toFixed(0)} ${(this.y1 - this.y0 + pad * 2).toFixed(0)}`;
  }
}