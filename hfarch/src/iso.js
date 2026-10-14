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
 * Face opacity is the whole trick. A solid box drawn in isometric reads as a
 * flat hexagon with three lines in it; the same box reads as a solid only when
 * the top is nearly opaque and the two sides fall away from it. These numbers
 * come from the workbench's `iso.tsx`, which is what the two views share.
 */
export function box(g, { x, y, z, w, d, h, color, top = 0.62, left = 0.22,
  right = 0.14, stroke = INK, sw = 0.9 }) {
  const x1 = x + w, y1 = y + d, z1 = z + h;
  g.append(el("polygon", {
    points: poly([iso(x, y1, z), iso(x, y1, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
    fill: color, "fill-opacity": left, stroke, "stroke-width": sw,
  }));
  g.append(el("polygon", {
    points: poly([iso(x1, y, z), iso(x1, y, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
    fill: color, "fill-opacity": right, stroke, "stroke-width": sw,
  }));
  g.append(el("polygon", {
    points: poly([iso(x, y, z1), iso(x1, y, z1), iso(x1, y1, z1), iso(x, y1, z1)]),
    fill: "#ffffff", "fill-opacity": top, stroke, "stroke-width": sw,
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