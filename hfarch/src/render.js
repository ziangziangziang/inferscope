/**
 * Lay the model out on a strict grid, then draw it.
 *
 * Three rules, and they are the reason this looks like a diagram rather than a
 * sketch:
 *
 *   1. A component's column is its stage -- one column right of whatever feeds
 *      it -- and every component in a stage shares a column centre, whatever
 *      its size. That is what makes "Token embed" and the blocks below it line
 *      up on both edges instead of merely being near each other.
 *   2. A component's row is its lane. Two lanes, never more, so the eye can
 *      find the vision tower and the language stack without hunting.
 *   3. Every card is the same size. Size carried meaning in the isometric
 *      version -- it encoded parameter count -- and it cost more than it gave:
 *      labels of different lengths on different-sized boxes, and nothing lining
 *      up. Parameter count is on the card now, where it can be read.
 */
import { fmt } from "./hub.js";
import { Bounds, KINDS, el, skipWire, txt, wrapWire, dropWire, marginDrop } from "./prim.js";
import { block, defs, isoCard, isoPlate, isoResidual, plate, residual, tower, wire }
  from "./modules.js";
import { iso } from "./prim.js";

const COL = 262;        // column pitch. It has to be wide enough for the
                         // connector between two cards to hold the tensor it
                         // carries -- at 232 the gap was 52px and the label
                         // ran under the cards on both sides.
const ROW = 262;        // pitch between the rows a lane wraps into. It used to
                         // be 596, propping single-lane diagrams open to fake
                         // some height; wrapping uses the height honestly, so
                         // the rows sit at reading distance again.
const LANE_GAP = 110;    // extra space where one lane ends and the next begins:
                         // room for the lane caption, and for the join wire's jog
const CARD_W = 194;
const COL3 = 232;        // 3D: world-x spacing between stages
const LANE3 = 228;       // 3D: world-y spacing between bands; at 150 the
                         // front lane's blocks sat on the back lane's
const CARD_H = 126;
const X0 = 40;
const Y0 = 74;
const TARGET_ASPECT = 1.62;   // the shape of the window it is read in
const MAX_ASPECT = 3.1;       // wider than this and it letterboxes

/** A readable name for a lane. Calling an encoder lane "language stack" is
 *  the kind of default that survives long after it stops being true. */
const LANE_NAMES = {
  vision: "vision branch", text: "language stack", encoder: "encoder",
  decoder: "decoder", unet: "U-Net", out: "output", main: "pipeline",
  dit: "patch transformer", enc: "text encoder",
  "image-enc": "image encoder", "text-enc": "text encoder",
};

/** Stage per node, from the edges. */
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
 * Everything drawn gets a home on the grid.
 *
 * A component's row is its lane and its column is its position *within that
 * lane*, so each lane reads left to right as a clean sequence regardless of how
 * many nodes it has.
 *
 * The lanes are then aligned to each other at the point where one joins the
 * other. That matters more than it sounds: the vision branch has five
 * components and joins the decoder stack, which is the third component of the
 * language lane. Left to its own devices the join ran *backwards* -- visual
 * tokens sat two columns to the right of the block they feed, so the wire went
 * backwards across the diagram and read as a mistake. Aligning the vision lane
 * so its last component sits directly above what it feeds turns that into a
 * single vertical arrow.
 */
function layout(g) {
  const lanes = new Map();
  for (const nd of g.nodes) {
    const name = nd.branch || "text";
    if (!lanes.has(name)) lanes.set(name, []);
    lanes.get(name).push(nd);
  }
  const names = [...lanes.keys()].sort((a, b) =>
    (a === "vision" ? -1 : b === "vision" ? 1 : 0));

  // order within each lane, by stage
  const st = stages(g);
  for (const list of lanes.values()) list.sort((a, b) => st.get(a.id) - st.get(b.id));
  const index = new Map();
  for (const list of lanes.values()) list.forEach((nd, i) => index.set(nd.id, i));

  // offset each lane so any cross-lane join lands in the same column
  const base = new Map(names.map((n) => [n, 0]));
  for (const e of g.edges) {
    const from = g.nodes.find((n) => n.id === e.from);
    const to = g.nodes.find((n) => n.id === e.to);
    if (!from || !to) continue;
    const a = from.branch || "text", b = to.branch || "text";
    if (a === b) continue;                       // within a lane: already ordered
    base.set(a, (base.get(b) || 0) + (index.get(to.id) - index.get(from.id)));
  }

  // ---- wrapping ----
  // A single lane of seven cards is 1,900px wide and 220px tall: an 8:1 band
  // with empty margins above and below. The sequence is flowed into rows of a
  // fixed width instead, chosen so the whole diagram lands near the shape of
  // the window it is read in. The column pitch never changes, so column n of
  // every row still lines up on both edges, and reading order is untouched:
  // each row reads left to right, rows read top to bottom, and the wrap
  // connector takes the same path a line of text takes at the page edge.
  const raw = new Map();                        // node id -> column, before the shift
  for (const [name, list] of lanes) {
    for (const nd of list) raw.set(nd.id, index.get(nd.id) + (base.get(name) || 0));
  }
  const shift = -Math.min(...raw.values());     // nothing wraps from a negative column
  const spans = new Map();                      // lane -> [min col, max col], shifted
  for (const [name, list] of lanes) {
    const cols = list.map((nd) => raw.get(nd.id) + shift);
    spans.set(name, [Math.min(...cols), Math.max(...cols)]);
  }
  const total = Math.max(...[...spans.values()].map(([, hi]) => hi)) + 1;
  const perRow = chooseWidth(spans, total);

  const bandStart = new Map();                  // lane -> first band index
  let band = 0;
  for (const name of names) {
    const [lo, hi] = spans.get(name);
    bandStart.set(name, band);
    band += Math.floor(hi / perRow) - Math.floor(lo / perRow) + 1;
  }

  const items = [];
  names.forEach((name, li) => {
    for (const nd of lanes.get(name)) {
      const col = raw.get(nd.id) + shift;
      const row = bandStart.get(name) + Math.floor(col / perRow);
      items.push({
        nd, lane: li, col: col % perRow, row, perRow,
        x: (col % perRow) * COL + (COL - CARD_W) / 2,
        y: Y0 + row * ROW + li * LANE_GAP,
        w: CARD_W, h: CARD_H,
      });
    }
  });

  const cols = total;
  return { items, byId: new Map(items.map((i) => [i.nd.id, i])), cols, perRow, lanes: names };
}

/**
 * How many columns a row holds.
 *
 * Unwrapped if that already fits the window; otherwise the width whose aspect
 * lands nearest the window's, over every width the lanes can wrap to. Columns
 * are never fewer than three: two cards and a connector is not a row, it is a
 * list drawn sideways.
 */
function chooseWidth(spans, total) {
  // Height, estimated the way the drawing will actually come out: one caption
  // above, one row of cards, and a gutter of wire below each row. It does not
  // have to be exact -- it has to rank widths the way the window will see them.
  const aspectAt = (g) => {
    const bands = [...spans.values()]
      .reduce((s, [lo, hi]) => s + (Math.floor(hi / g) - Math.floor(lo / g) + 1), 0);
    const w = g * COL;
    const h = (bands - 1) * ROW + CARD_H + 170;
    return w / h;
  };
  if (aspectAt(total) <= MAX_ASPECT) return total;
  let best = total, score = Infinity;
  for (let g = 3; g <= total; g++) {
    const s = Math.abs(Math.log(aspectAt(g) / TARGET_ASPECT));
    if (s < score) { score = s; best = g; }
  }
  return best;
}

/** The lanes this graph actually has, in drawing order. */
function laneList(items) {
  const seen = [];
  for (const it of items) {
    const b = it.nd.branch || "text";
    if (!seen.includes(b)) seen.push(b);
  }
  return seen.map((b) => [b, LANE_NAMES[b] || b.replace(/^./, (c) => c.toUpperCase())]);
}

export function render(svg, g, onPick, mode = "2d") {
  if (mode === "3d") return render3d(svg, g, onPick);
  svg.replaceChildren();
  svg.append(defs());
  const { items, byId, cols, perRow } = layout(g);

  const root = el("g", {});
  svg.append(root);
  const layer = (n) => { const e = el("g", { class: `layer-${n}` }); root.append(e); return e; };
  const L = { wire: layer("wire"), solid: layer("solid"), label: layer("label") };

  const bb = new Bounds();

  // ---- 1. connections, behind ----
  // Three routes, chosen by where the wire is going. Within a row: the plain
  // orthogonal run. Into the next row of the same lane: the wrap, out by the
  // right margin and back along the gutter. Into a later row of another lane:
  // a drop into the target's top edge. Which route a wire takes is decided by
  // the grid, never hand-picked, so a family cannot draw its joins differently
  // from another family's.
  for (const e of g.edges) {
    const a = byId.get(e.from), b = byId.get(e.to);
    if (!a || !b) continue;
    const shape = e.shape ? e.shape.join("×") : null;
    if (e.skip) {
      skipWire(L.wire, {
        p0: [a.x + a.w / 2, a.y + a.h], p1: [b.x + b.w / 2, b.y + b.h],
        drop: 26 + Math.abs(a.x - b.x) / 40, a, b,
      });
      const deep = a.row === b.row
        ? Math.max(a.y, b.y) + a.h + 26 + Math.abs(a.x - b.x) / 40
        : a.y + a.h + 30;
      bb.add(Math.min(a.x, b.x), Math.min(a.y, b.y)).add(Math.max(a.x, b.x), deep + 24);
      continue;
    }
    if (a.lane === b.lane && b.row > a.row) {
      // the wrap: the end of one row into the start of the next
      wrapWire(L.wire, { p0: [a.x + a.w, a.y + a.h / 2], p1: [b.x, b.y + b.h / 2], shape });
      bb.add(a.x, a.y).add(Math.max(a.x + a.w, b.x) + 46, b.y + b.h);
      continue;
    }
    if (b.row > a.row && b.x <= a.x + a.w && b.x + b.w >= a.x) {
      // a lane join: the target sits below, so drop into its top edge -- via
      // the margin when lanes intervene, straight down when they do not
      if (b.row - a.row === 1) dropWire(L.wire, { a, b, shape });
      else marginDrop(L.wire, { a, b, shape });
      bb.add(Math.min(a.x, b.x), a.y).add(Math.max(a.x + a.w, b.x + b.w) + 46, b.y);
      continue;
    }
    const p0 = [a.x + a.w, a.y + a.h / 2];
    const p1 = [b.x, b.y + b.h / 2];
    wire(L.wire, { p0, p1, shape });
  }

  // ---- 2. residuals, above the block they wrap ----
  for (const nd of g.nodes) {
    if (!nd.repeat || nd.repeat <= 4) continue;
    const it = byId.get(nd.id);
    if (it) residual(L.solid, it);
  }

  // ---- 3. cards ----
  for (const it of items) {
    const nd = it.nd;
    const gEl = el("g", { class: "node", tabindex: "0", role: "button",
      "aria-label": nd.label });

    if (nd.plate) {
      plate(gEl, { ...it, label: nd.plate.title || nd.label,
        shape: nd.plate.shape.join("×"), count: 7 });
    } else if (nd.repeat) {
      const per = nd.params / nd.repeat;
      tower(gEl, {
        ...it,
        label: nd.label,
        sub: `${fmt(per)} each`,
        count: nd.repeat,
        parts: (nd.template || []).map((t) => ({ label: t.label, kind: t.kind })),
        weights: (nd.template || []).map((t) => t.params || 1),
      });
    } else {
      block(gEl, {
        ...it,
        label: nd.label,
        sub: nd.params ? `${fmt(nd.params)} parameters` : null,
        kind: nd.kind,
        dims: nd.dims,
      });
    }
    const act = () => onPick && onPick(nd, it);
    gEl.addEventListener("click", act);
    gEl.addEventListener("keydown", (ev) => { if (ev.key === "Enter") act(); });
    L.solid.append(gEl);
    bb.addBox(it.x, it.y, it.w, it.h);
  }

  // ---- 4. lane captions, in their own margin ----
  for (const [name, title] of laneList(items)) {
    const own = items.filter((i) => (i.nd.branch || "text") === name);
    if (!own.length) continue;
    const first = own[0];
    const y = first.y + CARD_H / 2;
    const lx = Math.min(...own.map((i) => i.x)) - 22;
    L.label.append(txt(lx, y, title.toUpperCase(),
      { class: "lane-t", "text-anchor": "end" }));
    bb.add(lx - 150, y - 12).add(lx, y + 12);
  }

  // column count is worth stating: it is the model's depth
  const wrapNote = perRow < cols ? ` · wrapped every ${perRow}` : "";
  L.label.append(txt(Math.min(...items.map((i) => i.x)), Y0 - 30,
    `${cols} steps in the longest chain${wrapNote} · every count derived from config.json`,
    { class: "cap" }));
  bb.add(0, Y0 - 46).add(760, Y0 - 18);

  svg.setAttribute("viewBox", bb.viewBox(30));
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  installPanZoom(svg, root);
  return { head: describe(g) };
}

/** Pan and zoom. Only while a button is actually held. */
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
    if (!down || !(e.buttons & 1)) return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (!moved && Math.hypot(dx, dy) < 4) return;
    if (!moved) { moved = true; svg.classList.add("drag"); svg.setPointerCapture(e.pointerId); }
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
    if (moved) {
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

/**
 * One sentence about the whole model, for the header.
 *
 * It has to fit the family. "9 layers × 320 wide · 0 heads over 0 kv" is what a
 * transformer template produces when handed a U-Net: a head count of zero and a
 * "layer" that is actually a resolution level.
 */
export function describe(g) {
  const m = g.meta;
  const n = fmt(g.params);
  if (g.family === "diffusion") {
    return `diffusion · ${m.levels.length} levels · ${m.levels.join(" → ")} channels`
      + ` · ${n} derived; the feed-forward inside each block is not declared in the config`;
  }
  if (g.family === "dit") {
    return `dit · ${m.layers} blocks × ${m.hidden} wide · patches of ${m.patch}px`
      + ` · ${m.conditioning ? `cross-attn from width ${m.conditioning}, ` : ""}${n} derived`
      + `; the feed-forward width is not declared in the config, so it is not counted`;
  }
  if (g.family === "encoder") {
    return `encoder · ${m.layers} layers × ${m.hidden} wide · no output head, nothing`
      + ` to sample · ${n} derived`;
  }
  if (g.family === "contrastive") {
    return `contrastive · text ${m.text_layers}×${m.text_hidden}, vision `
      + `${m.vision_layers}×${m.vision_hidden} · both project to width ${m.projection}`
      + ` · ${n} derived`;
  }
  if (g.family === "vae") {
    return `vae · ${m.levels.length} levels · ${m.levels.join(" → ")} channels · ${n} derived`;
  }
  if (g.family === "speech") {
    return `speech · ${m.encoder_layers} encoder + ${m.decoder_layers} decoder blocks`
      + ` · ${m.mel_bins || "?"} mel bins, width ${m.hidden} · ${n} derived`;
  }
  const bits = [
    `${m.layers} layers × ${m.hidden} wide`,
    `${m.heads} heads over ${m.kv_heads} kv`,
    m.moe ? `${m.moe.experts} experts, top-${m.moe.top_k}` : null,
    m.vision ? `vision ${m.vision.layers}×${m.vision.hidden}` : null,
    `${n} parameters, derived from config`,
  ].filter(Boolean);
  return `${g.family} · ${bits.join(" · ")}`;
}

void KINDS;
/**
 * The same diagram, in 3D.
 *
 * Same layout function, same ordering, same columns and lanes -- only the
 * drawing differs. That is the point of sharing `layout()`: the two modes
 * cannot drift into disagreeing about what feeds what, because neither of them
 * decides that.
 *
 * Every block sits at z = 0, so the input and the output are at the same
 * altitude and you can read straight across the model from one to the other.
 * Altitude that drifts with stage was tried and is worse than useless here: it
 * puts the two ends of the same flow as far apart as the page allows and makes
 * the reader think the output is downstream in space as well as in time.
 */
function render3d(svg, g, onPick) {
  svg.replaceChildren();
  svg.append(defs());
  const { items, byId, cols, perRow } = layout(g);

  const root = el("g", {});
  svg.append(root);
  const layer = (n) => { const e = el("g", { class: `layer-${n}` }); root.append(e); return e; };
  const L = { wire: layer("wire"), solid: layer("solid"), label: layer("label") };

  const bb = new Bounds();
  const H = 62;                       // a plain block's height
  const HT = 120;                     // a repeated stack's: it holds more
  const D = 62;

  // grid cell -> world cell. Same column, same band, nothing else. The band is
  // what wrapping produced; a lane's later rows simply sit deeper in the scene,
  // so both modes read the same order out of the same layout.
  const world = new Map();
  for (const it of items) {
    const tall = !!(it.nd.repeat || it.nd.plate);
    world.set(it.nd.id, {
      x: it.col * COL3, y: it.row * LANE3, z: 0,       // z = 0: input and output level
      w: COL3 * 0.60, d: D, h: tall ? HT : H,
    });
  }

  const anchor = new Map();
  for (const it of items) anchor.set(it.nd.id, it);

  /** A node's whole screen extent, for routing and for the viewBox. */
  const screenBox = (geo) => {
    const lx = (geo.x - geo.y - geo.d) * Math.cos(Math.PI / 6);
    const rx = (geo.x + geo.w - geo.y) * Math.cos(Math.PI / 6);
    const ty = (geo.x + geo.w + geo.y) * 0.5 - geo.z - geo.h - 20;
    const by = (geo.x + geo.w + geo.y + geo.d) * 0.5 - geo.z + 26;
    return { x: lx, y: ty, w: rx - lx, h: by - ty };
  };

  // ---- connectors, behind ----
  for (const e of g.edges) {
    const a = world.get(e.from), b = world.get(e.to);
    const A = anchor.get(e.from), B = anchor.get(e.to);
    if (!a || !b || !A || !B) continue;
    const shape = e.shape ? e.shape.join("×") : null;
    const sa = { ...screenBox(a), row: A.row, lane: A.lane };
    const sb = { ...screenBox(b), row: B.row, lane: B.lane };
    if (e.skip) {
      skipWire(L.wire, {
        p0: iso(a.x + a.w / 2, a.y + a.d, a.z),
        p1: iso(b.x + b.w / 2, b.y + b.d, b.z),
        drop: 40 + Math.abs(a.x - b.x) / 30, a: sa, b: sb,
      });
      const deep = A.row === B.row
        ? Math.max(sa.y + sa.h, sb.y + sb.h) + 40 + Math.abs(a.x - b.x) / 30
        : sa.y + sa.h + 40;
      bb.add(Math.min(sa.x, sb.x), Math.min(sa.y, sb.y)).add(Math.max(sa.x, sb.x), deep);
      continue;
    }
    if (A.lane === B.lane && B.row > A.row) {
      wrapWire(L.wire, {
        p0: [sa.x + sa.w, sa.y + sa.h / 2], p1: [sb.x, sb.y + sb.h / 2], shape,
      });
      bb.add(sa.x, sa.y).add(Math.max(sa.x + sa.w, sb.x) + 46, sb.y + sb.h);
      continue;
    }
    if (B.row > A.row && sb.x <= sa.x + sa.w && sb.x + sb.w >= sa.x) {
      if (B.row - A.row === 1) dropWire(L.wire, { a: sa, b: sb, shape });
      else marginDrop(L.wire, { a: sa, b: sb, shape });
      bb.add(Math.min(sa.x, sb.x), sa.y).add(Math.max(sa.x + sa.w, sb.x + sb.w) + 46, sb.y + sb.h);
      continue;
    }
    const p0 = iso(a.x + a.w, a.y + a.d / 2, a.z + a.h / 2);
    const p1 = iso(b.x, b.y + b.d / 2, b.z + b.h / 2);
    wire(L.wire, { p0, p1, shape });
  }

  // ---- residuals ----
  for (const it of items) {
    if (!it.nd.repeat || it.nd.repeat <= 4) continue;
    isoResidual(L.solid, world.get(it.nd.id));
  }

  // ---- solids, back to front ----
  const sorted = items.slice().sort((a, b) => {
    const wa = world.get(a.nd.id), wb = world.get(b.nd.id);
    return (wa.x + wa.y) - (wb.x + wb.y);
  });
  for (const it of sorted) {
    const nd = it.nd;
    const geo = world.get(nd.id);
    const gEl = el("g", { class: "node", tabindex: "0", role: "button",
      "aria-label": nd.label });
    if (nd.plate) {
      isoPlate(gEl, { ...geo, label: nd.plate.title || nd.label,
        shape: nd.plate.shape.join("×") });
    } else if (nd.repeat) {
      isoCard(gEl, {
        ...geo, label: nd.label,
        sub: `${fmt(nd.params / nd.repeat)} each`,
        count: nd.repeat, kind: "stack",
        parts: (nd.template || []).map((t) => ({ label: t.label, kind: t.kind })),
        weights: (nd.template || []).map((t) => t.params || 1),
      });
    } else {
      isoCard(gEl, { ...geo, label: nd.label,
        sub: nd.params ? `${fmt(nd.params)} params` : null, kind: nd.kind });
    }
    const act = () => onPick && onPick(nd, it);
    gEl.addEventListener("click", act);
    gEl.addEventListener("keydown", (ev) => { if (ev.key === "Enter") act(); });
    L.solid.append(gEl);
    // the block's whole screen extent, not one corner: sampling a single
    // point clipped the last block in the run off the bottom-right
    const s = screenBox(geo);
    bb.add(s.x - 30, s.y).add(s.x + s.w + 40, s.y + s.h);
  }

  // ---- lane captions ----
  for (const [name, title] of laneList(items)) {
    const own = items.filter((i) => (i.nd.branch || "text") === name);
    if (!own.length) continue;
    const w0 = world.get(own[0].nd.id);
    const at = iso(w0.x - 64, w0.y + w0.d / 2, 0);
    L.label.append(txt(at[0], at[1], title.toUpperCase(),
      { class: "lane-t", "text-anchor": "end" }));
    bb.add(at[0] - 150, at[1] - 12).add(at[0], at[1] + 12);
  }
  const first = world.get(items[0].nd.id);
  const h0 = iso(first.x, first.y + first.d, 0);
  const wrapNote = perRow < cols ? ` · wrapped every ${perRow}` : "";
  L.label.append(txt(h0[0], h0[1] - 70,
    `${cols} stages deep${wrapNote} · one plane, input and output at the same altitude`,
    { class: "cap" }));
  bb.add(h0[0], h0[1] - 86).add(h0[0] + 560, h0[1] - 58);

  svg.setAttribute("viewBox", bb.viewBox(40));
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  installPanZoom(svg, root);
  return { head: describe(g) };
}
