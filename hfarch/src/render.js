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
import { Bounds, KINDS, box, el, iso, txt } from "./iso.js";
import { edges, gapMark, plate, residual, tower, wire } from "./modules.js";

const COL = 176;      // world-x spacing between stages; the blocks got
                       // wider so their captions fit on the face
const LANE = 208;     // world-y spacing between branches. At 108 the
                       // front lane landed on the back one; at 172 the two
                       // towers themselves overlapped. A tower is much wider
                       // than a plain block, so the lane has to clear the widest
                       // thing in it, not the average one.
const DROP = 26;      // altitude lost per stage -- the U-Net profile: the
                       // pipeline descends as it goes. It is also what lets a
                       // residual run straight, because a block and its own
                       // bypass share one altitude and need no turn.
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


/**
 * A label placer.
 *
 * Text in an isometric drawing does not get a free position. Every caption
 * wants the same place -- under the block, centred -- and the lanes are close
 * enough that they land on each other's bodies. Handing every label its
 * preferred spot is what produced the mess: captions crossing blocks, "622.3M"
 * bisected by a face, "S×1152" written through a wire.
 *
 * So labels are requests, not placements. Each one offers several candidate
 * anchors in order of preference, the biggest and most important labels are
 * placed first, and a candidate is accepted only if its rectangle is clear of
 * everything already down. Labels that genuinely have nowhere to go keep their
 * preferred spot rather than being dropped -- losing a caption is worse than an
 * overlapping one.
 */
class Placer {
  constructor() { this.taken = []; }

  /** Rough width of a string at a given size. 0.56em is close enough. */
  static width(s, size) { return String(s).length * size * 0.56; }

  /** Free if nothing already placed overlaps this rectangle. */
  fits(r) {
    for (const t of this.taken) {
      if (r.x < t.x + t.w && r.x + r.w > t.x && r.y < t.y + t.h && r.y + r.h > t.y) {
        return false;
      }
    }
    return true;
  }

  /**
   * candidates: [{ at:[x,y], anchor, dy, size }]. Returns the first that fits,
   * else the first one offered.
   */
  place(text, size, candidates) {
    const w = Placer.width(text, size);
    for (const [i, c] of candidates.entries()) {
      const x = c.anchor === "middle" ? c.at[0] - w / 2
        : c.anchor === "end" ? c.at[0] - w : c.at[0];
      const r = { x: x - 3, y: c.at[1] + c.dy - size, w: w + 6, h: size + 4 };
      if (this.fits(r)) { this.taken.push(r); return { ...c, box: r }; }
      if (i === candidates.length - 1) { this.taken.push(r); return { ...c, box: r }; }
    }
    return candidates[0];
  }
}

/** How big a block looks, from the parameters it holds. Log, because the
 *  counts span six orders of magnitude. */
function bulk(params, max) {
  if (!params || !max) return 0;
  return Math.max(0, Math.min(1, Math.log10(params + 1) / Math.log10(max + 1)));
}

/** Split a repeated block's height between its sub-layers. */
function slabPlan(nd, h) {
  const parts = nd.template;
  const FLOOR = 0.07;
  const tot = parts.reduce((a, q) => a + (q.params || 1), 0);
  const free = Math.max(0, 1 - FLOOR * parts.length);
  let dz = 0;
  return parts.map((q) => {
    const sh = (FLOOR + ((q.params || 1) / tot) * free) * h;
    const out = { sub: q, dz, h: sh };
    dz += sh;
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

/**
 * The layout: what goes where, and in what order it is drawn.
 *
 * Three axes, each with one meaning:
 *   x  stage  -- a component sits one column right of whatever feeds it
 *   y  lane   -- vision behind, language in front, so the two stay aligned
 *   z  depth  -- each stage drops, the U-Net profile, so the pipeline descends
 *
 * Layer order is explicit and is the thing that decides whether a figure reads
 * or turns to soup: connectors, then residuals, then solids back to front, then
 * every piece of text above all of it.
 */
function layout(g) {
  const st = stages(g);
  const max = Math.max(...g.nodes.map((n) => n.params || 0));
  const lanes = new Map();
  for (const nd of g.nodes) {
    const name = nd.branch || "text";
    if (!lanes.has(name)) lanes.set(name, []);
    lanes.get(name).push(nd);
  }
  const order = [...lanes.keys()].sort((a, b) =>
    (a === "vision" ? -1 : b === "vision" ? 1 : 0));

  const items = [];
  order.forEach((name, li) => {
    const list = lanes.get(name).slice().sort((a, b) => st.get(a.id) - st.get(b.id));
    for (const nd of list) {
      const t = bulk(nd.params, max);
      const stage = st.get(nd.id);
      const x = stage * COL, y = li * LANE, z = -stage * DROP;
      const w = 92, d = 28 + t * 26;
      const repeat = nd.repeat || 0;
      if (repeat > 4) {
        const h = 52 + t * 40;
        // Each half is narrower than a plain block: the pair is two blocks
        // wide and has to fit inside one stage, or the first and last overlap
        // each other -- which is exactly what happened when blocks were widened.
        const tw = 76;
        items.push({ nd, i: 0, x: x - 88, y, z, w: tw, d, h, lane: li });
        items.push({ nd, i: -1, hidden: repeat - 2, x: x - 20, y, z, w: 40, d, h, lane: li });
        items.push({ nd, i: repeat - 1, x: x + 20, y, z, w: tw, d, h, lane: li });
      } else {
        // floor of 30 so even a parameterless block has a face to be
        // named on -- logits and sampling carry no weights at all
        items.push({ nd, i: -1, x, y, z, w, d, h: 30 + t * 40, lane: li });
      }
    }
  });
  return items;
}

/**
 * Compose the drawing.
 *
 * Layering is the whole job here, and it is why this is explicit and named
 * rather than an accident of loop order. Four layers, always in this sequence:
 *
 *   1. connectors   behind everything; they are about what feeds what
 *   2. residuals    also behind -- a bypass belongs to its block, not on top
 *   3. solids       back to front by x+y, the painter's order this camera needs
 *   4. labels       above all of it, because text that anything can cover is
 *                   text nobody can read
 *
 * Two of these used to be decided by where a loop happened to sit, and it cost
 * real bugs: a residual drawn inside the solid loop was painted over by the
 * next block along, and captions sat below blocks that were then drawn after
 * them.
 */
export function render(svg, g, onPick) {
  svg.replaceChildren();
  const items = layout(g);

  // which block a connection should leave from / arrive at
  const anchor = new Map();
  for (const it of items) {
    if (it.nd.repeat) {
      if (it.i === it.nd.repeat - 1 || !anchor.has(it.nd.id)) anchor.set(it.nd.id, it);
    } else anchor.set(it.nd.id, it);
  }

  const root = el("g", {});
  svg.append(root);
  const L = {};
  // residual after solid: it runs along the block's own base, so a solid
  // drawn over it hid all but the stub that poked out past the edge
  for (const name of ["wire", "solid", "residual", "label"]) {
    L[name] = el("g", { class: `layer-${name}` });
    root.append(L[name]);
  }

  const bb = new Bounds();
  const keep = (p, r = 0) => bb.addP(p, r);

  for (const it of items) bb.addBox(it.x, it.y, it.z, it.w, it.d, it.h);

  // ---- 1. connectors ----
  for (const e of g.edges) {
    const a = anchor.get(e.from), b = anchor.get(e.to);
    if (!a || !b) continue;
    // join at half height so a wire meets a face rather than skimming the floor
    const p0 = iso(a.x + a.w, a.y + a.d / 2, a.z + a.h / 2);
    const p1 = iso(b.x, b.y + b.d / 2, b.z + b.h / 2);
    wire(L.wire, { p0, p1, shape: e.shape || null });
  }

  // ---- 2. residuals ----
  // one per repeated group, not per drawn half: the bypass belongs to the run.
  for (const nd of g.nodes) {
    if (!nd.repeat || nd.repeat <= 4) continue;
    const own = items.filter((i) => i.nd.id === nd.id);
    const last = own[own.length - 1];
    residual(L.residual, { x: last.x, y: last.y, z: last.z, w: last.w, d: last.d });
  }

  // Dimensions are shown on demand rather than always.
  //
  // This was the densest thing in the drawing: a width on the front edge, a
  // parameter count up the left edge and a depth sloped along the receding
  // edge, for twelve blocks. They overprinted each other, the blocks and the
  // wires, and none could be read. Hovering a block, or selecting one, shows
  // its three; the panel always has the full set.
  const dimsLayer = el("g", { class: "layer-dims" });
  L.label.append(dimsLayer);
  const showDims = (gEl, it, on) => {
    if (!it || it.nd.repeat || it.nd.plate) return;
    const dims = it.nd.dims || {};
    const width = dims.width || dims.out;
    if (!width && !it.nd.params) return;
    let box = gEl.__dims;
    if (!box) {
      box = el("g", { class: "dims", style: "display:none" });
      dimsLayer.append(box);
      gEl.__dims = box;
      edges(box, {
        x: it.x, y: it.y, z: it.z, w: it.w, d: it.d, h: it.h,
        width: width || "", params: it.nd.params ? fmt(it.nd.params) : "",
        depth: dims.vocab || dims.patch || null,
      });
    }
    box.style.display = on ? "" : "none";
  };

  // ---- 3. solids, back to front ----
  const sorted = items.slice().sort((a, b) => (a.x + a.y) - (b.x + b.y));
  for (const it of sorted) {
    const nd = it.nd;
    const { x, y, z, w, d, h } = it;

    if (nd.repeat && it.i === -1) {
      gapMark(L.solid, { x: x + w / 2, y, z, w, d, h, hidden: it.hidden });
      continue;
    }

    const gEl = el("g", { class: "node", tabindex: "0", role: "button",
      "aria-label": nd.label });

    if (nd.plate) {
      plate(gEl, { ...nd.plate, x, y, z,
        label: `${nd.label} — one sheet per ${nd.plate.unit}` });
    } else if (nd.template?.length) {
      tower(gEl, {
        x, y, z, w, d, h,
        parts: nd.template.map((q) => q.label),
        weights: nd.template.map((q) => q.params || 1),
        kinds: nd.template.map((q) => q.kind),
      });
    } else {
      box(gEl, { x, y, z, w, d, h, color: KINDS[nd.kind] || KINDS.tensor });
    }

    const act = () => onPick && onPick(nd, it);
    gEl.addEventListener("click", act);
    gEl.addEventListener("keydown", (ev) => { if (ev.key === "Enter") act(); });
    gEl.addEventListener("mouseenter", () => showDims(gEl, it, true));
    gEl.addEventListener("mouseleave", () => showDims(gEl, it, false));
    gEl.addEventListener("focus", () => showDims(gEl, it, true));
    gEl.addEventListener("blur", () => showDims(gEl, it, false));
    L.solid.append(gEl);
  }

  // ---- 4. labels ----
  //
  // Captions go *on the block's own face*, not beside it.
  //
  // Every other arrangement was tried and each one collided, because the blocks
  // sit on a diagonal with no free space beside any of them: place below and
  // you land on the next lane's block, place beside and you land on this one's
  // own face. A caption inside its own block cannot collide with anything,
  // because the only thing that can cover it is that block, drawn earlier.
  //
  // It is also what the reference does -- `\\node at (\\x/2, 0, \\z/2) {\\caption}`.
  const labels = el("g", { class: "layer-text" });
  L.label.append(labels);

  /** Shorten to what actually fits across a face. */
  const fit = (s2, px, size) => {
    const room = Math.max(4, Math.floor(px / (size * 0.56)));
    if (s2.length <= room) return s2;
    // cut at the last space that fits: "Patch embed\u2026" reads as an
    // abbreviation, "Patch embeddi\u2026" reads as a typo
    const cut = s2.slice(0, room);
    const sp = cut.lastIndexOf(" ");
    return (sp > room * 0.5 ? cut.slice(0, sp) : cut) + "\u2026";
  };

  for (const it of items) {
    const nd = it.nd;
    if (nd.repeat || nd.plate) continue;          // those label themselves
    const { x, y, z, w, d, h } = it;
    if (h < 26) continue;                         // no room for a face caption
    // the baseline sits above the text's own centre, so nudge down
    const at = iso(x + w / 2, y + d, z + h * 0.5 - 4);
    labels.append(txt(at[0], at[1], fit(nd.label, w, 11), {
      class: "face-cap", "text-anchor": "middle",
    }));
    if (h >= 44) {
      labels.append(txt(at[0], at[1] + 13, fmt(nd.params || 0), {
        class: "face-sub", "text-anchor": "middle",
      }));
    }
  }

  // one caption per repeated group, on the face of its last block
  for (const nd of g.nodes) {
    if (!nd.repeat || nd.repeat <= 4) continue;
    const own = items.filter((i) => i.nd.id === nd.id);
    const last = own[own.length - 1];
    const at = iso(last.x + last.w / 2, last.y + last.d, last.z + last.h * 0.5 - 4);
    labels.append(txt(at[0], at[1], fit(`${nd.label}`, last.w, 11), {
      class: "face-cap", "text-anchor": "middle",
    }));
    labels.append(txt(at[0], at[1] + 13,
      `${fmt(nd.params / nd.repeat)} each \u00d7 ${nd.repeat}`, {
        class: "face-sub", "text-anchor": "middle",
      }));

  }

  // branch captions: the only text that needs free space, and there is some,
  // out past the start of each lane
  for (const [name, title] of [["vision", "vision branch"], ["text", "language stack"]]) {
    const own = items.filter((i) => (i.nd.branch || "text") === name);
    if (!own.length) continue;
    const back = Math.min(...own.map((i) => i.y));
    const x0 = Math.min(...own.map((i) => i.x));
    const at = iso(x0 - 26, back, 0);
    labels.append(txt(at[0], at[1], title.toUpperCase(),
      { class: "branch", "text-anchor": "end" }));
    keep(at, 90);
  }

  // The viewBox is set last, after every layer has reserved its room. Setting
  // it earlier meant "LANGUAGE STACK" clipped to "GUAGE STACK", because the
  // bounds added after that point were thrown away.
  svg.setAttribute("viewBox", bb.viewBox(46));
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