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
const DROP = 26;      // altitude lost per stage -- the U-Net profile: the
                       // pipeline descends as it goes. It is also what lets a
                       // residual run straight, because a block and its own
                       // bypass share one altitude and need no turn.
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

/**
 * A plate: a run of thin sheets, one per element of the leading axis, with one
 * sheet pulled forward as the vector being read right now.
 *
 * This is the difference between "Token ids" as a box and tokenised text as
 * what it is. A box says there is a tensor here; a plate says there are B of
 * them, they are stacked, and this one is the current step -- which is the only
 * reason the rest of the diagram has a single highlighted path through it.
 *
 * The sheet count is representative, not the config's own batch size: "B" is
 * symbolic, and drawing 32 sheets to stand for it would be a lie about scale.
 * The axis label carries the real shape instead.
 */
function drawPlate(root, nd, it, hit) {
  const { count, shape, width, unit } = nd.plate;
  const { x, y, z } = it;
  const w = 68, d = 48, t = 6;
  const step = d / count;
  const focus = Math.min(count - 1, 3);

  for (let i = 0; i < count; i++) {
    const on = i === focus;
    // the focused sheet comes forward and left, so it is unmistakably the one
    const sy = y + (on ? -step * 0.55 : i * step);
    const sx = x + (on ? -9 : 0);
    const c = on ? "#7cb98f" : "#b9c6b2";
    const x1 = sx + w, y1 = sy + step * 0.82, z1 = z + t + (on ? 5 : 0);
    const g = el("g", { class: on ? "node sheet-on" : "node" });
    g.append(el("polygon", {
      points: poly([iso(sx, y1, z), iso(sx, y1, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
      fill: c, "fill-opacity": on ? 0.34 : 0.2, stroke: INK, "stroke-width": 0.7,
    }));
    g.append(el("polygon", {
      points: poly([iso(x1, sy, z), iso(x1, sy, z1), iso(x1, y1, z1), iso(x1, y1, z)]),
      fill: c, "fill-opacity": on ? 0.24 : 0.14, stroke: INK, "stroke-width": 0.7,
    }));
    g.append(el("polygon", {
      points: poly([iso(sx, sy, z1), iso(x1, sy, z1), iso(x1, y1, z1), iso(sx, y1, z1)]),
      fill: on ? "#ffffff" : "#f4f8f1", "fill-opacity": on ? 0.92 : 0.7,
      stroke: on ? "#2e8b4c" : INK, "stroke-width": on ? 1.5 : 0.7,
    }));
    const act = () => hit && hit(nd);
    g.addEventListener("click", act);
    g.append(el("title", {}, document.createTextNode(
      `${nd.label} — one sheet per ${unit}`)));
    root.append(g);
  }

  // the vector itself, and what it is
  const [lx, ly] = iso(x - 9, y - step * 0.55, z + t + 5);
  root.append(txt(lx, ly - 8, `t₀`, { class: "sheet-t", "text-anchor": "middle" }));
  const [ax, ay] = iso(x + w / 2, y + d, z);
  root.append(txt(ax, ay + 14, shape.join("×"), { class: "dim", "text-anchor": "middle" }));
  if (width) {
    const [vx, vy] = iso(x - 9, y - step * 0.55, z + t);
    root.append(txt(vx - 8, vy + 3, String(width),
      { class: "dim", "text-anchor": "end" }));
  }
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

/**
 * Split a block's height between its sub-layers, so a decoder block is drawn as
 * a stack of slabs rather than one solid lump. Heights follow the parameters
 * each sub-layer holds, with a floor so a 4K norm is still visible.
 */
function slabPlan(nd, h) {
  const parts = nd.template;
  const tot = parts.reduce((a, q) => a + (q.params || 1), 0);
  const room = h - (parts.length - 1) * 1.5;
  let z = 0;
  return parts.map((q) => {
    const sh = Math.max(5, (q.params || 1) / tot * room);
    const out = { sub: q, dz: z, h: sh };
    z += sh + 1.5;
    return out;
  });
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
      const stage = st.get(nd.id);
      const x = stage * COL;
      const y = li * LANE;
      const z = -stage * DROP;                 // altitude encodes position
      const w = 74, d = 26 + t * 26;
      const repeat = nd.repeat || 0;
      if (repeat > 4) {
        const h = 52 + t * 40;
        items.push({ nd, i: 0, x: x - 52, y, z, w, d, h, lane: li });
        items.push({ nd, i: -1, hidden: repeat - 2, x: x - 6, y, z, w: 40, d, h, lane: li });
        items.push({ nd, i: repeat - 1, x: x + 26, y, z, w, d, h, lane: li });
      } else {
        items.push({ nd, i: -1, x, y, z, w, d, h: 18 + t * 46, lane: li });
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
    for (const p of [iso(it.x, it.y, it.z), iso(it.x + it.w, it.y + it.d, it.z + it.h),
      iso(it.x, it.y + it.d, it.z), iso(it.x + it.w, it.y, it.z)]) add(p[0], p[1]);
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

  // No ground plate. The plate in the workbench is honest there -- those parts
  // physically sit on a host -- but a flat plane contradicts the one thing
  // altitude is now for: the pipeline descending through space. Blocks at
  // stage 6 sank straight through it.
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
    const p0 = a.nd.plate
      ? iso(a.x + 57, a.y - 2, a.z + 11)
      : iso(a.x + a.w, a.y + a.d / 2, a.z + a.h / 2);
    const p1 = iso(b.x, b.y + b.d / 2, b.z + b.h / 2);
    const tip = [p1[0] - 9, p1[1]];
    let d;
    if (Math.abs(p0[1] - p1[1]) < 1.5) {
      d = `M ${p0[0]} ${p0[1]} L ${tip[0]} ${tip[1]}`;
    } else {
      const mx = (p0[0] + tip[0]) / 2;
      d = `M ${p0[0]} ${p0[1]} L ${mx} ${p0[1]} L ${mx} ${tip[1]} L ${tip[0]} ${tip[1]}`;
    }
    root.append(el("path", { class: "edge", d, "marker-end": "url(#head)" }));
    // vecArrow writes the tensor it carries along the arrow's underside rather
    // than beside it; same reason, so the two diagrams read alike.
    if (e.shape) {
      const mid = p0[0] < tip[0] ? [(p0[0] + tip[0]) / 2, p0[1]]
        : [(tip[0] + p0[0]) / 2, tip[1]];
      root.append(txt(mid[0], mid[1] + 13, e.shape.join("×"),
        { class: "shape", "text-anchor": "middle" }));
    }
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
      const [cx, cy] = iso(it.x + it.w / 2, it.y + it.d * 0.78, it.z + it.h);
      root.append(txt(cx, cy - 4, `×${it.hidden}`, {
        class: "gap-n", "text-anchor": "middle",
      }));
      continue;
    }

    const { x, y, z, w, d, h } = it;
    const x1 = x + w, y1 = y + d, z1 = z + h;
    if (nd.plate) { drawPlate(root, nd, it, onPick); }
    const gEl = el("g", { class: "node", tabindex: "0", role: "button",
      "aria-label": nd.label });
    if (!nd.plate) {

    // The residual, drawn first so the slabs sit on top of it. Its input and
    // output are at the same altitude by construction, so it is one straight
    // line with no bend at all -- which is the whole reason the blocks are
    // separated in height.
    if (nd.repeat) {
      const a0 = iso(x - 16, y + d / 2, z);
      const a1 = iso(x1 + 18, y + d / 2, z);
      root.append(el("path", { class: "residual",
        d: `M ${a0[0]} ${a0[1]} L ${a1[0]} ${a1[1]}` }));
      // the add, drawn as the operator it is: a disc on the wire, the way
      // \elementsum puts a green circle with a + mid-connection
      const sum = iso(x1 + 18, y + d / 2, z);
      root.append(el("circle", { class: "sum", cx: sum[0], cy: sum[1], r: 7 }));
      root.append(txt(sum[0], sum[1] + 4, "+", {
        class: "sum-t", "text-anchor": "middle" }));
    }
    const shell = (bx, by, bz, bw, bd, bh, c) => {
      const bx1 = bx + bw, by1 = by + bd, bz1 = bz + bh;
      gEl.append(el("polygon", {
        points: poly([iso(bx, by1, bz), iso(bx, by1, bz1), iso(bx1, by1, bz1), iso(bx1, by1, bz)]),
        fill: c, "fill-opacity": 0.22, stroke: INK, "stroke-width": 0.8,
      }));
      gEl.append(el("polygon", {
        points: poly([iso(bx1, by, bz), iso(bx1, by, bz1), iso(bx1, by1, bz1), iso(bx1, by1, bz)]),
        fill: c, "fill-opacity": 0.14, stroke: INK, "stroke-width": 0.8,
      }));
      gEl.append(el("polygon", {
        points: poly([iso(bx, by, bz1), iso(bx1, by, bz1), iso(bx1, by1, bz1), iso(bx, by1, bz1)]),
        fill: "#ffffff", "fill-opacity": 0.62, stroke: INK, "stroke-width": 0.8,
      }));
    };
    if (nd.template?.length) {
      for (const sl of slabPlan(nd, h)) {
        shell(x, y, z + sl.dz, w, d, sl.h, KINDS[sl.sub.kind] || KINDS.tensor);
      }
    } else {
      shell(x, y, z, w, d, h, color);
    }
    }
    const act = () => onPick && onPick(nd, it);
    gEl.addEventListener("click", act);
    gEl.addEventListener("keydown", (ev) => { if (ev.key === "Enter") act(); });
    root.append(gEl);

    // caption under the front face; the block's own height already says how
    // much it weighs, so the caption carries only what height cannot
    const [cx, cy] = iso(x + w / 2, y + d, z);
    const above = it.lane === 0;
    if (!nd.repeat && !nd.plate) {
      // Dimensions labelled the way Box.sty does it: each number on the edge
      // it measures -- width under the front-bottom edge, height up the left
      // edge, depth sloped along the edge that recedes. Three numbers on three
      // axes, rather than a list floating beside the block.
      const wLab = nd.dims && (nd.dims.width || nd.dims.out);
      if (wLab) {
        const p = iso(x, y + d, z), q = iso(x + w, y + d, z);
        root.append(txt((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 12, String(wLab),
          { class: "dim", "text-anchor": "middle" }));
      }
      // Only real, declared quantities go on an edge. This used to print the
      // block's own height and depth in world units -- numbers like "42" that
      // read as dimensions but are just where the layout happened to put it.
      if (nd.params) {
        const p = iso(x, y + d / 2, z), q = iso(x, y + d / 2, z + h);
        root.append(txt(p[0] - 7, (p[1] + q[1]) / 2 + 3, `${fmt(nd.params)} params`,
          { class: "dim", "text-anchor": "end" }));
      }
      const dLab = nd.dims && (nd.dims.vocab || nd.dims.patch);
      if (dLab) {
        const rp = iso(x + w, y + d, z), rq = iso(x + w, y, z);
        const mx = (rp[0] + rq[0]) / 2 + 12, my = (rp[1] + rq[1]) / 2 + 4;
        root.append(txt(mx, my, String(dLab), {
          class: "dim", "text-anchor": "middle", transform: `rotate(-30 ${mx} ${my})`,
        }));
      }
      // caption on the face itself, where Box.sty puts it
      const room = Math.max(6, Math.floor(w / 3.1));
      const title = nd.label.length > room ? nd.label.slice(0, room - 1) + "…" : nd.label;
      const [tx, ty] = above ? iso(x + w / 2, y, z + h) : iso(x + w / 2, y + d + 18, z);
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
    const at = above ? iso((x0 + x1) / 2, back, own[0].z + own[0].h)
      : iso((x0 + x1) / 2, own[0].y + own[0].d + 46, own[0].z);
    // clear of the xN marker, which sits on the top face at the same height
    const dy = above ? -72 : 16;
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