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

/**
 * A plate: a run of thin sheets, one per element of the leading axis, with one
 * sheet pulled out as the vector being read right now.
 *
 * This is the difference between "Token ids" as a box and tokenised text as
 * what it is. A box says a tensor is here; a plate says there are B of them,
 * they are stacked, and this one is the current step -- which is the only
 * reason the rest of the diagram has a single highlighted path through it.
 *
 * The sheet count is representative, not the config's own batch size: "B" is
 * symbolic, and drawing 32 sheets to stand for it would be a lie about scale.
 * The axis label carries the real shape instead.
 */
export function plate(g, { x = 0, y = 0, z = 0, count = 8, w = 68, d = 48, t = 6,
  shape = null, width = null, label = "", dim = "#b9c6b2", hot = "#7cb98f" } = {}) {
  const step = d / count;
  const focus = count - 1;          // the front sheet is the near one
  for (let i = 0; i < count; i++) {
    const on = i === focus;
    // +y is *towards* the viewer under iso(); -y is away. Pulling the sheet the
    // other way made it float up behind the deck instead of out of it.
    const sy = y + (on ? d + step * 0.8 : i * step);
    const sx = x + (on ? -12 : 0);
    const sheet = el("g", { class: on ? "node sheet-on" : "node" });
    box(sheet, { x: sx, y: sy, z, w, d: step * 0.82, h: t,
      color: on ? hot : dim, top: on ? 0.95 : 0.7, sw: on ? 1.5 : 0.7 });
    if (label) {
      sheet.append(el("title", {}, document.createTextNode(label)));
    }
    g.append(sheet);
  }
  const [lx, ly] = iso(x - 12, y + d + step * 0.8, z + t);
  g.append(txt(lx, ly - 8, "t₀", { class: "sheet-t", "text-anchor": "middle" }));
  if (shape) {
    const [ax, ay] = iso(x + w / 2, y + d, z);
    g.append(txt(ax, ay + 14, shape.join("×"), { class: "dim", "text-anchor": "middle" }));
  }
  if (width) {
    const [vx, vy] = iso(x - 12, y + d + step * 0.8, z + t);
    g.append(txt(vx - 8, vy + 3, String(width), { class: "dim", "text-anchor": "end" }));
  }
  // where a connection should leave from: the highlighted sheet, not the middle
  return iso(x + w - 12, y + d + step * 0.8, z + t / 2);
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
export function tower(g, { x = 0, y = 0, z = 0, w = 74, d = 46, h = 84,
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
  box(g, { x, y, z, w, d, h, color: KINDS.stack });
  // the lid belongs to the sub-layer sitting on it
  g.append(el("polygon", {
    points: poly([iso(x, y, z + h), iso(x + w, y, z + h),
      iso(x + w, y + d, z + h), iso(x, y + d, z + h)]),
    fill: edges[edges.length - 1][2], "fill-opacity": 0.34,
    stroke: INK, "stroke-width": 0.9 }));

  const x1 = x + w, y1 = y + d;
  for (const [a, b2, c] of edges) {
    // a tinted band on each of the two faces this camera can see
    g.append(el("polygon", {
      points: poly([iso(x, y1, z + a), iso(x, y1, z + b2),
        iso(x1, y1, z + b2), iso(x1, y1, z + a)]),
      fill: c, "fill-opacity": 0.42,
    }));
    g.append(el("polygon", {
      points: poly([iso(x1, y, z + a), iso(x1, y, z + b2),
        iso(x1, y1, z + b2), iso(x1, y1, z + a)]),
      fill: c, "fill-opacity": 0.3,
    }));
  }
  // and a rule at every division, so the boundaries read as drawn, not tinted
  for (const [a] of edges.slice(1)) {
    const p = iso(x, y1, z + a), q = iso(x1, y1, z + a);
    g.append(el("line", { class: "rule", x1: p[0], y1: p[1], x2: q[0], y2: q[1] }));
    const r = iso(x1, y, z + a), s = iso(x1, y1, z + a);
    g.append(el("line", { class: "rule", x1: r[0], y1: r[1], x2: s[0], y2: s[1] }));
  }
  return { top: z + h };
}

/**
 * The elision mark: the "and N-2 more" between the first and last of a run.
 *
 * Drawn on the base line between the two blocks rather than across their faces,
 * where it read as a scratch across the artwork.
 */
export function gapMark(g, { x = 0, y = 0, z = 0, w = 40, d = 46, h = 60, hidden = 34 } = {}) {
  const l = iso(x - w / 2, y + d, z), r = iso(x + w / 2, y + d, z);
  // drawn *above* the blocks rather than on the base line between them, where
  // it sat behind the two it was counting
  const yTop = Math.min(l[1], r[1]) - 26;
  g.append(el("path", { class: "gap", d: `M ${l[0]} ${yTop} L ${r[0]} ${yTop}` }));
  g.append(txt((l[0] + r[0]) / 2, yTop - 7, `\u00d7${hidden}`,
    { class: "gap-n", "text-anchor": "middle" }));
  return { left: [l[0] - 6, l[1] + 10], right: [r[0] + 6, r[1] + 10] };
}

/** First, marker, last: the whole elision, as one call. */
export function elision(g, { x = 0, y = 0, z = 0, hidden = 34, w = 74, d = 46, h = 60 } = {}) {
  box(g, { x: x - 86, y, z, w, d, h, color: KINDS.stack });
  box(g, { x: x + 34, y, z, w, d, h, color: KINDS.stack });
  gapMark(g, { x: x + 12, y, z, w: 78, d, h, hidden });
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
export function residual(g, { x = 0, y = 0, z = 0, w = 74, d = 46 } = {}) {
  // A short stub leaving the block's right corner at half height, with the add
  // at its end. Drawn across the block's base it crossed its own front face,
  // because the base is a diamond and a horizontal line through its middle
  // runs along the face; drawn below the block it floated free of it. A stub
  // says "output, plus the bypass" and touches nothing it should not.
  const corner = iso(x + w, y + d / 2, z);
  const y0 = corner[1];
  const x0 = corner[0] + 4, x1 = corner[0] + 34;
  g.append(el("path", { class: "residual",
    d: `M ${x0} ${y0} L ${x1} ${y0}` }));
  g.append(el("circle", { class: "sum", cx: x1, cy: y0, r: 7 }));
  g.append(txt(x1, y0 + 4, "+", { class: "sum-t", "text-anchor": "middle" }));
}

/**
 * A connection.
 *
 * Orthogonal by construction, and routed in screen space. The bends are
 * right angles because the path is assembled from axis-aligned segments, not
 * because the endpoints happen to line up -- the first version ended `L p1.x
 * p1.y` from a mid point whose y had been nudged by -10, and that last leg
 * moved on both axes at once, so every "rectangular" route finished on a
 * diagonal.
 *
 * The tensor it carries is written along the arrow's underside, the way
 * vecArrow puts its xlabel on the arrow rather than beside it.
 */
/**
 * A connection.
 *
 * Orthogonal by construction, and routed in screen space. The bends are
 * right angles because the path is assembled from axis-aligned segments, not
 * because the endpoints happen to line up -- the first version ended `L p1.x
 * p1.y` from a mid point whose y had been nudged by -10, and that last leg
 * moved on both axes at once, so every "rectangular" route finished on a
 * diagonal.
 *
 * The tensor it carries is written along the arrow's underside, the way
 * vecArrow puts its xlabel on the arrow rather than beside it.
 */
export function wire(g, { p0, p1, shape = null, head = "url(#head)" } = {}) {
  const tip = [p1[0] - 9, p1[1]];
  let d, mid;
  if (Math.abs(p0[1] - p1[1]) < 1.5) {
    d = `M ${p0[0]} ${p0[1]} L ${tip[0]} ${tip[1]}`;
    mid = [(p0[0] + tip[0]) / 2, p0[1] + 13];
  } else {
    const mx = (p0[0] + tip[0]) / 2;
    d = `M ${p0[0]} ${p0[1]} L ${mx} ${p0[1]} L ${mx} ${tip[1]} L ${tip[0]} ${tip[1]}`;
    mid = [mx, tip[1] + 13];
  }
  g.append(el("path", { class: "edge", d, "marker-end": head }));
  if (shape) {
    g.append(txt(mid[0], mid[1], shape.join("\u00d7"),
      { class: "shape", "text-anchor": "middle" }));
  }
  return tip;
}


/** Dimensions on the three edges they measure, per Box.sty. */
export function edges(g, { x = 0, y = 0, z = 0, w = 74, d = 46, h = 60,
  width = 4096, params = "622.3M", depth = null } = {}) {
  box(g, { x, y, z, w, d, h, color: KINDS.embed });
  const a = iso(x, y + d, z), b = iso(x + w, y + d, z);
  g.append(txt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 12, String(width),
    { class: "dim", "text-anchor": "middle" }));
  // the leftmost corner of the footprint -- iso(x, y+d/2, z) sits *inside* the
  // block's silhouette, so the label was cut in half by the face
  const c = iso(x, y + d, z), e = iso(x, y + d, z + h);
  g.append(txt(c[0] - 8, (c[1] + e[1]) / 2 + 3, params,
    { class: "dim", "text-anchor": "end" }));
  if (depth != null) {
    const f = iso(x + w, y + d, z), g2 = iso(x + w, y, z);
    const mx = (f[0] + g2[0]) / 2 + 12, my = (f[1] + g2[1]) / 2 + 4;
    g.append(txt(mx, my, String(depth), {
      class: "dim", "text-anchor": "middle", transform: `rotate(-30 ${mx} ${my})`,
    }));
  }
}
