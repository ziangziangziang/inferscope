/**
 * Drawing the graph, in the same isometric camera the main app uses.
 *
 * The camera is `iso(x, y, z)` — a world point becomes a screen point — and
 * every block, the ground and the captions all go through it, so they are
 * literally the same perspective as the architecture view in the app rather
 * than a flat diagram that happens to be laid out left to right.
 *
 * Three things carry meaning, and each one was wrong the first time:
 *
 *  1. Columns are stages and rows are branches. A component sits one column
 *     right of whatever feeds it, so the vision lane and the language lane stay
 *     column-aligned and the join between them is visible.
 *  2. A long stack is its first block, a marker for the ones between, and its
 *     last. Sixty-five identical blocks say "there are sixty-five" and take
 *     more room than everything else on the page.
 *  3. Height and depth scale with the parameters a block holds, on a log
 *     scale — the counts span six orders of magnitude, so a linear one would
 *     make every block except the embedding a speck.
 */
import { fmt } from "./hub.js";

const COS30 = Math.cos(Math.PI / 6);
const SIN30 = 0.5;

const iso = (x, y, z) => [(x - y) * COS30, (x + y) * SIN30 - z];
const poly = (pts) => pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");

const COL = 152;      // world-x spacing between stages
const LANE = 172;     // world-y spacing between branches; at 108 the
                       // front lane landed on top of the back one
const INK = "#2f3a2c";

/** block face colours, one per kind */
const KINDS = {
  tensor: "#a9b4a0",
  embed: "#7cb98f",
  stack: "#4c9e69",
  attention: "#6d9fd2",
  norm: "#aab59f",
  mlp: "#b2bd7d",
  projector: "#9a7ec1",
  moe: "#c99962",
};

/**
 * A patch on a block's top face, addressed in face units (u, v) rather than
 * screen units, so it lies in the block's own perspective. This is what puts a
 * decoder block's four sub-layers on its lid, the way the main app draws the
 * cells on its parts.
 */
function topPatch(x, y, w, d, z1, u, v, uw, vh) {
  const p00 = iso(x, y, z1), p10 = iso(x + w, y, z1), p01 = iso(x, y + d, z1);
  const at = (uu, vv) => [
    p00[0] + uu * (p10[0] - p00[0]) + vv * (p01[0] - p00[0]),
    p00[1] + uu * (p10[1] - p00[1]) + vv * (p01[1] - p00[1]),
  ];
  return poly([at(u, v), at(u + uw, v), at(u + uw, v + vh), at(u, v + vh)]);
}

const el = (tag, attrs, kids) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  for (const k of [].concat(kids || [])) if (k) e.append(k);
  return e;
};
const txt = (x, y, s, attrs = {}) =>
  el("text", { x, y, ...attrs }, document.createTextNode(s));

/** how big a block looks, from the parameters it holds */
function bulk(params, max) {
  if (!params || !max) return 0;
  return Math.max(0, Math.min(1, Math.log10(params + 1) / Math.log10(max + 1)));
}

/** Stage per node, from the edges: one column right of whatever feeds it. */
function stages(g) {
  const st = new Map();
  for (const nd of g.nodes) {
    let s = 0;
    for (const e of g.edges) {
      if (e.to === nd.id && st.has(e.from)) s = Math.max(s, st.get(e.from) + 1);
    }
    st.set(nd.id, s);
  }
  return st;
}

function layout(g) {
  const st = stages(g);
  const max = Math.max(...g.nodes.map((n) => n.params || 0));
  const lanes = new Map();
  for (const nd of g.nodes) {
    const name = nd.branch || "text";
    if (!lanes.has(name)) lanes.set(name, []);
    lanes.get(name).push(nd);
  }
  // vision at the back, language in front: painter's order matches reading order
  const order = [...lanes.keys()].sort((a, b) =>
    (a === "vision" ? -1 : b === "vision" ? 1 : 0));

  const items = [];
  order.forEach((name, li) => {
    const list = lanes.get(name).slice().sort((a, b) => st.get(a.id) - st.get(b.id));
    for (const nd of list) {
      const t = bulk(nd.params, max);
      const x = st.get(nd.id) * COL;
      const y = li * LANE;
      const w = 74, d = 26 + t * 30, h = 20 + t * 52;
      const repeat = nd.repeat || 0;
      if (repeat > 4) {
        items.push({ nd, i: 0, x: x - 52, y, z: 0, w, d, h, lane: li });
        items.push({ nd, i: -1, hidden: repeat - 2, x: x - 6, y, z: 0, w: 40, d, h, lane: li });
        items.push({ nd, i: repeat - 1, x: x + 26, y, z: 0, w, d, h, lane: li });
      } else {
        items.push({ nd, i: -1, x, y, z: 0, w, d, h, lane: li });
      }
    }
  });
  return items;
}

export function render(svg, g, onPick) {
  svg.replaceChildren();
  const items = layout(g);
  const boxes = new Map();          // node id -> the block edges should leave from
  for (const it of items) {
    if (it.nd.repeat) {
      if (it.i === it.nd.repeat - 1 || !boxes.has(it.nd.id)) boxes.set(it.nd.id, it);
    } else boxes.set(it.nd.id, it);
  }

  // ---- one root group; pan/zoom only ever touches this ----
  const root = el("g", {});
  svg.append(root);

  // ---- bounds first, so the viewBox fits the drawing with no scrolling ----
  const bb = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  const add = (x, y) => {
    bb.x0 = Math.min(bb.x0, x); bb.x1 = Math.max(bb.x1, x);
    bb.y0 = Math.min(bb.y0, y); bb.y1 = Math.max(bb.y1, y);
  };
  for (const it of items) {
    for (const p of [iso(it.x, it.y, 0), iso(it.x + it.w, it.y + it.d, it.h),
      iso(it.x, it.y + it.d, 0), iso(it.x + it.w, it.y, 0)]) add(p[0], p[1]);
    const [cx, cy] = iso(it.x + it.w / 2, it.y + it.d, 0);
    add(cx - 60, cy + 30); add(cx + 60, cy + 30);     // room for the caption
    add(cx - 60, cy - 14); add(cx + 60, cy - 14);
  }
  // The viewBox is set at the very end, not here: the group and branch
  // captions also reserve room, and they are placed further down. Setting it
  // now meant "LANGUAGE STACK" was clipped to "GUAGE STACK".
  const pad = 46;
  const setView = () => svg.setAttribute("viewBox",
    `${(bb.x0 - pad).toFixed(0)} ${(bb.y0 - pad).toFixed(0)} ` +
    `${(bb.x1 - bb.x0 + pad * 2).toFixed(0)} ${(bb.y1 - bb.y0 + pad * 2).toFixed(0)}`);

  // ---- ground ----
  const gx0 = -18, gy0 = -18;
  const gx1 = Math.max(...items.map((i) => i.x + i.w)) + 18;
  const gy1 = Math.max(...items.map((i) => i.y + i.d)) + 18;
  root.append(el("polygon", {
    points: poly([iso(gx1, gy0, 0), iso(gx1, gy1, 0), iso(gx1, gy1, -6), iso(gx1, gy0, -6)]),
    fill: "#b0c8a9", stroke: "#7d9c78", "stroke-width": 1,
  }));
  root.append(el("polygon", {
    points: poly([iso(gx0, gy1, 0), iso(gx1, gy1, 0), iso(gx1, gy1, -6), iso(gx0, gy1, -6)]),
    fill: "#c2d8bb", stroke: "#7d9c78", "stroke-width": 1,
  }));
  root.append(el("polygon", {
    points: poly([iso(gx0, gy0, 0), iso(gx1, gy0, 0), iso(gx1, gy1, 0), iso(gx0, gy1, 0)]),
    fill: "#e9f1e4", stroke: "#88a882", "stroke-width": 1.2,
  }));

  // ---- arrows ----
  // Two rules, both of which the first attempt broke:
  //  - every turn is a right angle. The old path ended `L p1.x p1.y` from a
  //    mid point whose y had been nudged by -10, so the last leg came out as a
  //    diagonal, which is what made these look like scribble rather than wiring.
  //  - where the route bends, it bends once. A straight run between two blocks
  //    at the same height stays a straight line.
  for (const e of g.edges) {
    const a = boxes.get(e.from), b = boxes.get(e.to);
    if (!a || !b) continue;
    // join the blocks at half height, so an arrow meets a face rather than
    // skimming along the plate underneath
    const p0 = iso(a.x + a.w, a.y + a.d / 2, a.h / 2);
    const p1 = iso(b.x, b.y + b.d / 2, b.h / 2);
    const tip = [p1[0] - 9, p1[1]];
    let d;
    if (Math.abs(p0[1] - p1[1]) < 1.5) {
      d = `M ${p0[0]} ${p0[1]} L ${tip[0]} ${tip[1]}`;
    } else {
      const mx = (p0[0] + tip[0]) / 2;
      d = `M ${p0[0]} ${p0[1]} L ${mx} ${p0[1]} L ${mx} ${tip[1]} L ${tip[0]} ${tip[1]}`;
    }
    root.append(el("path", { class: "edge", d, "marker-end": "url(#head)" }));
  }

  // ---- blocks, back lane first (painter's algorithm: x+y is nearer) ----
  const sorted = items.slice().sort((a, b) => (a.x + a.y) - (b.x + b.y));
  for (const it of sorted) {
    const nd = it.nd;
    const color = KINDS[nd.kind] || KINDS.tensor;
    if (it.i === -1 && nd.repeat) {                 // the elision marker
      const p0 = iso(it.x + 4, it.y + it.d / 2, it.h / 2);
      const p1 = iso(it.x + it.w - 4, it.y + it.d / 2, it.h / 2);
      root.append(el("line", { class: "gap", x1: p0[0], y1: p0[1], x2: p1[0], y2: p1[1] }));
      const [cx, cy] = iso(it.x + it.w / 2, it.y + it.d * 0.86, it.h);
      root.append(txt(cx, cy - 4, `×${it.hidden}`, {
        class: "gap-n", "text-anchor": "middle",
      }));
      continue;
    }

    const { x, y, z, w, d, h } = it;
    const x1 = x + w, y1 = y + d, z1 = z + h;
    const gEl = el("g", { class: "node", tabindex: "0", role: "button",
      "aria-label": nd.label });
    gEl.append(el("polygon", {
      points: poly([iso(x, y1, z), iso(x, y1, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
      fill: color, "fill-opacity": 0.22, stroke: INK, "stroke-width": 0.9,
    }));
    gEl.append(el("polygon", {
      points: poly([iso(x1, y, z), iso(x1, y, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
      fill: color, "fill-opacity": 0.14, stroke: INK, "stroke-width": 0.9,
    }));
    gEl.append(el("polygon", {
      points: poly([iso(x, y, z1), iso(x1, y, z1), iso(x1, y1, z1), iso(x, y1, z1)]),
      fill: "#ffffff", "fill-opacity": 0.62, stroke: INK, "stroke-width": 0.9,
    }));
    // what is inside one instance, painted on the lid the way the main app
    // draws the cells on its parts
    if (nd.template?.length) {
      const n = nd.template.length;
      const gap = 0.025;
      const uw = (1 - gap * (n + 1)) / n;
      nd.template.forEach((sub, i) => {
        gEl.append(el("polygon", {
          points: topPatch(x, y, w, d, z1, gap + i * (uw + gap), 0.14, uw, 0.48),
          fill: KINDS[sub.kind] || KINDS.tensor, "fill-opacity": 0.8,
          stroke: INK, "stroke-width": 0.55,
        }));
      });
    }
    const act = () => onPick && onPick(nd, it);
    gEl.addEventListener("click", act);
    gEl.addEventListener("keydown", (ev) => { if (ev.key === "Enter") act(); });
    root.append(gEl);

    // caption under the front face; the block's own height already says how
    // much it weighs, so the caption carries only what height cannot
    const [cx, cy] = iso(x + w / 2, y + d, 0);
    if (!nd.repeat) {
      const room = Math.max(6, Math.floor(w / 3.1));
      const title = nd.label.length > room ? nd.label.slice(0, room - 1) + "…" : nd.label;
      // Below the front face is where the *next* lane's blocks are, so the back
      // lane's captions go above instead. Same reason the main app does this.
      const above = it.lane === 0;
      // a caption is wider than its block, so the front one is pushed off the
      // face rather than laid across it
      const [tx, ty] = above ? iso(x + w / 2, y, it.h) : iso(x + w / 2, y + d + 18, 0);
      root.append(txt(tx, ty + (above ? -10 : 4), title,
        { class: "cap", "text-anchor": "middle" }));
    }
  }

  // ---- one caption per repeated group, over the pair rather than each half ----
  for (const nd of g.nodes) {
    if (!nd.repeat || nd.repeat <= 4) continue;
    const own = items.filter((i) => i.nd.id === nd.id);
    const x0 = Math.min(...own.map((i) => i.x));
    const x1 = Math.max(...own.map((i) => i.x + i.w));
    const back = own[0].y;
    const above = own[0].lane === 0;
    const at = above ? iso((x0 + x1) / 2, back, own[0].h)
      : iso((x0 + x1) / 2, own[0].y + own[0].d + 46, 0);
    // clear of the xN marker, which sits on the top face at the same height
    const dy = above ? -54 : 16;
    root.append(txt(at[0], at[1] + dy, `${nd.label}`,
      { class: "cap", "text-anchor": "middle" }));
    root.append(txt(at[0], at[1] + dy + 12,
      `${fmt(nd.params / nd.repeat)} each · ${fmt(nd.params)} total`,
      { class: "cap-sub", "text-anchor": "middle" }));
    add(at[0] - 100, at[1] + dy - 12); add(at[0] + 100, at[1] + dy + 24);
  }

  // ---- branch captions, above the back lane, left of the front one ----
  for (const [name, title] of [["vision", "vision branch"], ["text", "language stack"]]) {
    const own = items.filter((i) => (i.nd.branch || "text") === name);
    if (!own.length) continue;
    const back = Math.min(...own.map((i) => i.y));
    const x0 = Math.min(...own.map((i) => i.x));
    const x1 = Math.max(...own.map((i) => i.x + i.w));
    const at = iso(x0 - 14, back + 34, 0);
    root.append(txt(at[0], at[1], title.toUpperCase(),
      { class: "branch", "text-anchor": "end" }));
    add(at[0] - 150, at[1] - 12); add(at[0] + 6, at[1] + 12);
  }

  setView();
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  installPanZoom(svg, root);
  return { head: describe(g) };
}

/**
 * Pan and zoom.
 *
 * A view that moves on its own is worse than one that will not move at all, so
 * this only pans while a button is actually down. Previously any pointer move
 * past a few pixels started a drag, which meant simply moving the mouse across
 * the drawing dragged it away.
 */
function installPanZoom(svg, root) {
  let k = 1, tx = 0, ty = 0;
  const apply = () => root.setAttribute("transform", `translate(${tx},${ty}) scale(${k})`);
  let down = false, moved = false, sx = 0, sy = 0, ox = 0, oy = 0;

  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    k = Math.min(3.5, Math.max(0.4, k * (e.deltaY < 0 ? 1.12 : 0.89)));
    apply();
  }, { passive: false });

  svg.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    down = true; moved = false;
    sx = e.clientX; sy = e.clientY; ox = tx; oy = ty;
  });
  svg.addEventListener("pointermove", (e) => {
    // the guard that matters: hover is not a drag
    if (!down || !(e.buttons & 1)) return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (!moved && Math.hypot(dx, dy) < 4) return;
    if (!moved) {
      moved = true;
      svg.classList.add("drag");
      svg.setPointerCapture(e.pointerId);
    }
    tx = ox + dx; ty = oy + dy;
    apply();
  });
  const end = (e) => {
    if (!down) return;
    down = false;
    svg.classList.remove("drag");
    if (e && e.pointerId != null && svg.hasPointerCapture(e.pointerId)) {
      svg.releasePointerCapture(e.pointerId);
    }
    if (moved) {                       // a drag must not also count as a click
      moved = false;
      const swallow = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
      svg.addEventListener("click", swallow, { capture: true, once: true });
      setTimeout(() => svg.removeEventListener("click", swallow, true), 0);
    }
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("pointerleave", end);
}

/** One sentence about the whole thing, shown in the header. */
export function describe(g) {
  const m = g.meta;
  const bits = [
    `${m.layers} layers × ${m.hidden} wide`,
    `${m.heads} heads over ${m.kv_heads} kv`,
    m.moe ? `${m.moe.experts} experts, top-${m.moe.top_k}` : null,
    m.vision ? `vision ${m.vision.layers}×${m.vision.hidden}` : null,
    `${fmt(g.params)} parameters, derived from config`,
  ].filter(Boolean);
  return `${g.family} · ${bits.join(" · ")}`;
}