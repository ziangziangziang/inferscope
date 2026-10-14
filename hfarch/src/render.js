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
import { Bounds, KINDS, el, txt } from "./prim.js";
import { block, defs, plate, residual, tower, wire } from "./modules.js";

const COL = 262;        // column pitch. It has to be wide enough for the
                         // connector between two cards to hold the tensor it
                         // carries -- at 232 the gap was 52px and the label
                         // ran under the cards on both sides.
const ROW = 596;        // lane pitch; the diagram is two lanes deep and
                         // very wide and only two lanes deep, so a tight row
                         // left it a thin band letterboxed in the middle of the
                         // canvas. Opening the lanes out uses the height and puts
                         // the vision-to-language join in clear space
const CARD_W = 194;
const CARD_H = 126;
const X0 = 40;
const Y0 = 74;

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

  const items = [];
  names.forEach((name, li) => {
    for (const nd of lanes.get(name)) {
      const col = index.get(nd.id) + (base.get(name) || 0);
      items.push({
        nd, lane: li, col,
        x: col * COL + (COL - CARD_W) / 2,
        y: Y0 + li * ROW,
        w: CARD_W, h: CARD_H,
      });
    }
  });

  // shift so nothing sits at a negative column
  const minCol = Math.min(...items.map((i) => i.col));
  for (const it of items) it.x += -minCol * COL + (COL - CARD_W) / 2;
  const cols = Math.max(...items.map((i) => i.col)) + 1;
  return { items, byId: new Map(items.map((i) => [i.nd.id, i])), cols };
}

export function render(svg, g, onPick) {
  svg.replaceChildren();
  svg.append(defs());
  const { items, byId, cols, lanes } = layout(g);

  const root = el("g", {});
  svg.append(root);
  const layer = (n) => { const e = el("g", { class: `layer-${n}` }); root.append(e); return e; };
  const L = { wire: layer("wire"), solid: layer("solid"), label: layer("label") };

  const bb = new Bounds();

  // ---- 1. connections, behind ----
  for (const e of g.edges) {
    const a = byId.get(e.from), b = byId.get(e.to);
    if (!a || !b) continue;
    const p0 = [a.x + a.w, a.y + a.h / 2];
    const p1 = [b.x, b.y + b.h / 2];
    wire(L.wire, { p0, p1, shape: e.shape ? e.shape.join("×") : null });
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
  for (const [name, title] of [["vision", "vision branch"], ["text", "language stack"]]) {
    const own = items.filter((i) => (i.nd.branch || "text") === name);
    if (!own.length) continue;
    const li = own[0].lane;
    const y = Y0 + li * ROW + CARD_H / 2;
    const lx = Math.min(...own.map((i) => i.x)) - 22;
    L.label.append(txt(lx, y, title.toUpperCase(),
      { class: "lane-t", "text-anchor": "end" }));
    bb.add(lx - 150, y - 12).add(lx, y + 12);
  }

  // column count is worth stating: it is the model's depth
  L.label.append(txt(Math.min(...items.map((i) => i.x)), Y0 - 30,
    `${cols} stages deep · every count derived from config.json`, { class: "cap" }));
  bb.add(0, Y0 - 46).add(700, Y0 - 18);

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

void KINDS;