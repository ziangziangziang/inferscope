/**
 * Two-dimensional primitives.
 *
 * This started as an isometric camera and became a flat grid, deliberately.
 * In isometric, no world axis projects to screen-horizontal -- that is why the
 * residual had to be drawn in screen space, why captions had nowhere to sit
 * that was not another block's face, and why text on a tilted face is read at
 * an angle by the eye and by nobody else. Every one of those problems was
 * worked around individually; removing the projection removes the class.
 *
 * The colour coding, the orthogonal routing and the elision carry over
 * unchanged. They were right.
 */

export const INK = "#22301f";
export const PAPER = "#f7faf4";
export const HAIR = "#c3d2bd";

/** Face colours, one per kind of component. */
export const KINDS = {
  tensor: "#9fb39a",
  embed: "#5f9c73",
  stack: "#2f7d4f",
  attention: "#4d86c4",
  norm: "#93a189",
  mlp: "#96a44f",
  projector: "#7c5aa8",
  moe: "#b8792f",
};

/** A colour light enough to carry black text on. */
export const soft = (hex, amt) => {
  const n = parseInt(hex.slice(1), 16);
  const ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) =>
    Math.round(v + (255 - v) * amt));
  return `#${ch.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
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
 * A card: the one solid in the diagram.
 *
 * The shadow is the only thing doing three-dimensional work now. A flat
 * rectangle with a soft drop shadow reads as a component sitting on the page,
 * and text on it is horizontal, which is the whole point.
 */
export function card(g, { x, y, w, h, fill = PAPER, stroke = HAIR, rx = 9, shadow = true }) {
  if (shadow) {
    g.append(el("rect", {
      x: x + 2, y: y + 3, width: w, height: h, rx,
      fill: "#2f3a2c", opacity: 0.13,
    }));
  }
  g.append(el("rect", { x, y, width: w, height: h, rx, fill, stroke, "stroke-width": 1.4 }));
}

/** A tinted strip inside a card: one sub-layer, or one axis of information. */
export function band(g, { x, y, w, h, fill, rx = 4, opacity = 1 }) {
  g.append(el("rect", { x, y, width: w, height: h, rx, fill, opacity }));
}

/** A count badge — how many copies of this thing there are. */
export function badge(g, { x, y, label, fill = "#2f7d4f" }) {
  const w = 10 + label.length * 9;
  g.append(el("rect", { x, y, width: w, height: 22, rx: 11, fill }));
  g.append(txt(x + w / 2, y + 16, label, {
    class: "badge-t", "text-anchor": "middle",
  }));
  return w;
}

/**
 * A connection. Orthogonal, one bend, never more.
 *
 * Two things it will not do. It will not zig-zag: when the ends are level it
 * draws a straight line, and when they are not it turns exactly once, at the
 * midpoint of the horizontal run. And it will not run its label over the
 * line -- the tensor it carries sits above the run, clear of the stroke.
 */
export function wire(g, { p0, p1, shape = null, label = null, inset = 10 }) {
  const a = [p0[0] + inset, p0[1]];
  const b = [p1[0] - inset, p1[1]];
  let d;
  let mid;
  if (Math.abs(a[1] - b[1]) < 1) {
    d = `M ${a[0]} ${a[1]} L ${b[0]} ${b[1]}`;
    mid = [(a[0] + b[0]) / 2, a[1]];
  } else {
    const mx = (a[0] + b[0]) / 2;
    d = `M ${a[0]} ${a[1]} L ${mx} ${a[1]} L ${mx} ${b[1]} L ${b[0]} ${b[1]}`;
    mid = [mx, b[1] - 10];
  }
  g.append(el("path", { class: "wire", d, "marker-end": "url(#arrow)" }));
  if (shape) {
    g.append(txt(mid[0], mid[1], shape, { class: "wire-t", "text-anchor": "middle" }));
  } else if (label) {
    g.append(txt(mid[0], mid[1], label, { class: "wire-t", "text-anchor": "middle" }));
  }
  return { mid };
}

/** Bounds that actually fit what was drawn. */
export class Bounds {
  x0 = Infinity; y0 = Infinity; x1 = -Infinity; y1 = -Infinity;
  add(x, y) {
    this.x0 = Math.min(this.x0, x); this.x1 = Math.max(this.x1, x);
    this.y0 = Math.min(this.y0, y); this.y1 = Math.max(this.y1, y);
    return this;
  }
  addBox(x, y, w, h, pad = 0) {
    return this.add(x - pad, y - pad).add(x + w + pad, y + h + pad);
  }
  viewBox(pad = 34) {
    return `${(this.x0 - pad).toFixed(0)} ${(this.y0 - pad).toFixed(0)} `
      + `${(this.x1 - this.x0 + pad * 2).toFixed(0)} ${(this.y1 - this.y0 + pad * 2).toFixed(0)}`;
  }
}