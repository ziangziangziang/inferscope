/**
 * The drawing modules.
 *
 * Every visual element in the viewer lives here, once. The gallery imports them
 * to test each one alone; the renderer imports them to compose. An earlier
 * version had the gallery and the renderer each holding their own copy, which
 * guarantees they drift apart, and that what you looked at is not what ships.
 *
 * Each module is plain, takes world coordinates, and appends to a root group.
 * None of them knows about the model, the layout, or each other.
 */
import { INK, KINDS, box, el, iso, poly, txt } from "./iso.js";

/** A plate of thin sheets, one pulled forward as the current vector. */
export function plate(root, { x = 0, y = 0, z = 0, count = 8, w = 68, d = 48, t = 6,
  focus = 3, dim = "#b9c6b2", hot = "#7cb98f" } = {}) {
  const step = d / count;
  for (let i = 0; i < count; i++) {
    const on = i === focus;
    const sy = y + (on ? d + step * 0.8 : i * step);
    const sx = x + (on ? -12 : 0);
    const g = el("g");
    box(g, { x: sx, y: sy, z, w, d: step * 0.82, h: t,
      color: on ? hot : dim, top: on ? 0.95 : 0.7, sw: on ? 1.5 : 0.7 });
    if (on) g.classList.add("sheet-on");
    root.append(g);
  }
  const [lx, ly] = iso(x - 12, y + d + step * 0.8, z + t);
  root.append(txt(lx, ly - 8, "t₀", { class: "sheet-t", "text-anchor": "middle" }));
  return { anchor: [x + w, y - step * 0.55, z + t + 5] };
}

/**
 * A repeated block, sectioned.
 *
 * The obvious version -- one box per sub-layer, stacked -- does not work. Drawn
 * as separate boxes you see between them, so the stack reads as several blocks
 * hovering over each other rather than one block, and a sub-layer holding 1% of
 * the parameters collapses to an invisible line. Both faults showed up the
 * first time this was rendered.
 *
 * What works is how an architectural section is drawn: one solid box, with the
 * *faces* divided. The silhouette stays a block, and the divisions carry the
 * composition.
 */
export function tower(root, { x = 0, y = 0, z = 0, w = 74, d = 46, h = 84,
  parts = ["RMSNorm", "Attention", "RMSNorm", "MLP"],
  weights = [1, 34, 1, 49], kinds = ["norm", "attention", "norm", "mlp"] } = {}) {
  // Reserve a visible slice for every sub-layer before distributing what is
  // left. Strictly proportional heights are honest and useless: an RMSNorm is
  // 1% of a decoder block's parameters and rendered as an invisible sliver, so
  // the two norms simply disappeared from the block.
  const FLOOR = 0.07;
  const tot = weights.reduce((a, b) => a + b, 0);
  const free = Math.max(0, 1 - FLOOR * weights.length);
  const edges = [];
  let dz = 0;
  weights.forEach((wgt, i) => {
    const sh = (FLOOR + (wgt / tot) * free) * h;
    edges.push([dz, dz + sh, KINDS[kinds[i]]]);
    dz += sh;
  });

  // the silhouette first: this is what makes it a block and not a pile
  box(root, { x, y, z, w, d, h, color: KINDS.stack, top: 0.25, left: 0.1, right: 0.06 });
  // the lid belongs to the sub-layer sitting on it
  root.append(el("polygon", {
    points: poly([iso(x, y, z + h), iso(x + w, y, z + h),
      iso(x + w, y + d, z + h), iso(x, y + d, z + h)]),
    fill: edges[edges.length - 1][2], "fill-opacity": 0.34,
    stroke: INK, "stroke-width": 0.9 }));

  const x1 = x + w, y1 = y + d;
  for (const [a, b2, c] of edges) {
    // a tinted band on each of the two faces this camera can see
    root.append(el("polygon", {
      points: poly([iso(x, y1, z + a), iso(x, y1, z + b2),
        iso(x1, y1, z + b2), iso(x1, y1, z + a)]),
      fill: c, "fill-opacity": 0.3,
    }));
    root.append(el("polygon", {
      points: poly([iso(x1, y, z + a), iso(x1, y, z + b2),
        iso(x1, y1, z + b2), iso(x1, y1, z + a)]),
      fill: c, "fill-opacity": 0.2,
    }));
  }
  // and a rule at every division, so the boundaries read as drawn, not tinted
  for (const [a] of edges.slice(1)) {
    const p = iso(x, y1, z + a), q = iso(x1, y1, z + a);
    root.append(el("line", { class: "rule", x1: p[0], y1: p[1], x2: q[0], y2: q[1] }));
    const r = iso(x1, y, z + a), s = iso(x1, y1, z + a);
    root.append(el("line", { class: "rule", x1: r[0], y1: r[1], x2: s[0], y2: s[1] }));
  }
  return { top: z + h };
}

/** First, marker, last: the elision. */
export function elision(root, { x = 0, y = 0, z = 0, hidden = 34, w = 74, d = 46, h = 60 } = {}) {
  const g = el("g");
  box(g, { x: x - 86, y, z, w, d, h, color: KINDS.stack });
  root.append(g);
  const g2 = el("g");
  box(g2, { x: x + 34, y, z, w, d, h, color: KINDS.stack });
  root.append(g2);
  // the marker sits on the base line between the two, not across their faces
  const gl = iso(x - 86 + w, y + d, z), gr = iso(x + 34, y + d, z);
  const mx = (gl[0] + gr[0]) / 2;
  const my = gl[1] + 10;
  root.append(el("path", { class: "gap", d: `M ${gl[0] + 4} ${gl[1] + 10} L ${gr[0] - 4} ${gr[1] + 10}` }));
  root.append(txt(mx, my - 4, `×${hidden}`, { class: "gap-n", "text-anchor": "middle" }));
}

/**
 * The residual, and the add where it rejoins.
 *
 * First attempt ran it along world x, which produced a long diagonal. That is
 * not a styling mistake, it is the camera: under iso() neither world axis
 * projects to screen-horizontal, so *any* world-space run between a block's
 * input and output is necessarily slanted. Asking for a straight line and
 * asking for a world-accurate one are different requests, and the user asked
 * for the first.
 *
 * So the residual is drawn in screen space, along the block's own base and
 * level with it. It is a connector, not a solid, and connectors read best in
 * the plane they connect things in.
 */
export function residual(root, { x = 0, y = 0, z = 0, w = 74, d = 46 } = {}) {
  const near = iso(x + w, y + d, z);              // nearest bottom corner
  const l = iso(x, y + d, z), r = iso(x + w, y, z);
  const y0 = near[1] + 16;                         // just under the block
  const x0 = Math.min(l[0], r[0]), x1 = Math.max(l[0], r[0]) + 26;
  root.append(el("path", { class: "residual",
    d: `M ${x0} ${y0} L ${x1} ${y0}` }));
  root.append(el("circle", { class: "sum", cx: x1, cy: y0, r: 7 }));
  root.append(txt(x1, y0 + 4, "+", { class: "sum-t", "text-anchor": "middle" }));
}

/** A connection: orthogonal, with the tensor it carries written underneath. */
export function wire(root, { from = [0, 0, 0], to = [200, 40, 0], shape = ["B", "T", 4096] } = {}) {
  const p0 = iso(from[0], from[1], from[2]);
  const p1 = iso(to[0], to[1], to[2]);
  const tip = [p1[0] - 9, p1[1]];
  let d;
  if (Math.abs(p0[1] - p1[1]) < 1.5) {
    d = `M ${p0[0]} ${p0[1]} L ${tip[0]} ${tip[1]}`;
  } else {
    const mx = (p0[0] + tip[0]) / 2;
    d = `M ${p0[0]} ${p0[1]} L ${mx} ${p0[1]} L ${mx} ${tip[1]} L ${tip[0]} ${tip[1]}`;
  }
  root.append(el("path", { class: "edge", d, "marker-end": "url(#gh)" }));
  const mid = Math.abs(p0[1] - p1[1]) < 1.5
    ? [(p0[0] + tip[0]) / 2, p0[1] + 13]
    : [(p0[0] + tip[0]) / 2, tip[1] + 13];
  root.append(txt(mid[0], mid[1], shape.join("×"), { class: "shape", "text-anchor": "middle" }));
}

/** Dimensions on the three edges they measure, per Box.sty. */
export function edges(root, { x = 0, y = 0, z = 0, w = 74, d = 46, h = 60,
  width = 4096, params = "622.3M", depth = null } = {}) {
  const g = el("g");
  box(g, { x, y, z, w, d, h, color: KINDS.embed });
  root.append(g);
  const a = iso(x, y + d, z), b = iso(x + w, y + d, z);
  root.append(txt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 12, String(width),
    { class: "dim", "text-anchor": "middle" }));
  // the leftmost corner of the footprint -- iso(x, y+d/2, z) sits *inside* the
  // block's silhouette, so the label was cut in half by the face
  const c = iso(x, y + d, z), e = iso(x, y + d, z + h);
  root.append(txt(c[0] - 8, (c[1] + e[1]) / 2 + 3, params,
    { class: "dim", "text-anchor": "end" }));
  if (depth != null) {
    const f = iso(x + w, y + d, z), g2 = iso(x + w, y, z);
    const mx = (f[0] + g2[0]) / 2 + 12, my = (f[1] + g2[1]) / 2 + 4;
    root.append(txt(mx, my, String(depth), {
      class: "dim", "text-anchor": "middle", transform: `rotate(-30 ${mx} ${my})`,
    }));
  }
}
