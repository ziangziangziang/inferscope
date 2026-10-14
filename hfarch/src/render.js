/**
 * Drawing the graph.
 *
 * Three things this has to get right, in order of how much they were getting
 * wrong before:
 *
 *  1. A long stack is drawn as first, a marker, last. The count stays in the
 *     picture; sixty-five boxes do not.
 *  2. Arrows, because boxes say what exists and not what feeds what.
 *  3. Boxes sized by the parameters they hold, so the shape of a model is
 *     visible before a single number is read — but with a floor and a ceiling,
 *     because one 389M embedding beside a 5K norm should not be a rectangle and
 *     a speck.
 */
import { fmt } from "./hub.js";

const BOX = { w: 172, h: 56 };
const GAP = { x: 74, y: 22 };
const COL = { stem: 0.92 };          // box width per layer of depth

const KINDS = {
  tensor: ["#eceee6", "#6d7565"],
  embed: ["#d7e6da", "#2c5c41"],
  stack: ["#cfe2d5", "#22503a"],
  attention: ["#d3e0ee", "#2b5273"],
  norm: ["#e4e8dd", "#5b6353"],
  mlp: ["#e0e4d0", "#55603c"],
  projector: ["#e2d8ea", "#5f4470"],
  moe: ["#eadfd2", "#7a5636"],
};

const size = (params, max) => {
  if (!params) return 30;
  const t = Math.max(0.06, Math.min(1, Math.log10(params + 1) / Math.log10(max + 1)));
  return 30 + t * 66;
};

/**
 * Lay the graph out.
 *
 * Columns are *stages*, taken from the edges: a component sits one column
 * right of whatever feeds it. Rows are lanes, so the vision tower runs in its
 * own lane and the language stack in another, and the two stay column-aligned.
 * That is what makes the join visible — visual tokens land in the same column
 * the text tokens left, which is where the two stacks actually meet.
 *
 * The earlier version walked the node list and advanced on both axes, which
 * produced a diagonal staircase that ran off the side of the page.
 */
function layout(g) {
  const stage = new Map();
  const order = g.nodes.map((n) => n.id);
  for (const id of order) {
    let s = 0;
    for (const e of g.edges) {
      if (e.to === id && stage.has(e.from)) s = Math.max(s, stage.get(e.from) + 1);
    }
    stage.set(id, s);
  }

  const lanes = [];
  for (const nd of g.nodes) {
    const name = nd.branch || "text";
    let lane = lanes.find((l) => l.name === name);
    if (!lane) lanes.push((lane = { name, items: [] }));
    lane.items.push({ nd, stage: stage.get(nd.id) });
  }
  lanes.sort((a, b) => (a.name === "vision" ? -1 : b.name === "vision" ? 1 : 0));

  const max = Math.max(...g.nodes.map((n) => n.params || 0));
  const placed = [];
  const LANE_W = 60;
  let y0 = 40;

  lanes.forEach((lane, li) => {
    // sort by stage so a lane reads left to right; ties keep insertion order
    lane.items.forEach((it, i) => (it.i = i));
    lane.items.sort((a, b) => a.stage - b.stage || a.i - b.i);
    let laneTop = y0;
    for (const it of lane.items) {
      const { nd } = it;
      const repeat = nd.repeat || 0;
      const d = Math.max(44, Math.min(96, size(nd.params, max)));
      const x = 40 + it.stage * (BOX.w + GAP.x) + li * LANE_W;
      if (repeat > 4) {
        const s = Math.max(d, 60);
        const w = Math.min(BOX.w, Math.max(110, s * 1.4));
        const cx = x + w / 2;
        placed.push({ nd, i: 0, x: cx - s - 40, y: laneTop, w: s, d: s, kind: "box" });
        placed.push({ nd, i: -1, hidden: repeat - 2, x: cx - 30, y: laneTop, w: 60, d: s, kind: "gap" });
        placed.push({ nd, i: repeat - 1, x: cx + 40, y: laneTop, w, d: s, kind: "box" });
        laneTop += s + GAP.y;
      } else {
        placed.push({ nd, i: -1, x, y: laneTop, w: BOX.w, d, kind: "box" });
        laneTop += d + GAP.y;
      }
    }
    lane.top = y0;
    y0 = laneTop + 54;          // the next lane starts clear of this one
  });
  return placed;
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

const el = (tag, attrs, kids) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const k of [].concat(kids || [])) if (k) e.append(k);
  return e;
};

export function render(svg, g, onPick) {
  svg.replaceChildren();
  const P = layout(g);
  const byId = new Map();
  for (const p of P) {
    if (p.kind === "box" && p.i >= 0) {
      // an edge leaving a repeated stack leaves its last member
      const prev = byId.get(p.nd.id);
      if (!prev || p.i > prev.i) byId.set(p.nd.id, p);
    } else if (!p.nd.repeat) byId.set(p.nd.id, p);
  }

  // --- pan/zoom, so a wide model is explorable ---
  const root = el("g", {});
  svg.append(root);
  const apply = (k, tx, ty) => {
    root.setAttribute("transform", `translate(${tx},${ty}) scale(${k})`);
  };
  apply(1, 0, 0);

  let k = 1, tx = 0, ty = 0, dragging = false, moved = false, sx = 0, sy = 0, ox = 0, oy = 0;
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    k = Math.min(3, Math.max(0.3, k * (e.deltaY < 0 ? 1.12 : 0.89)));
    apply(k, tx, ty);
  }, { passive: false });
  // Panning must not eat clicks. Capturing the pointer on pointerdown made
  // every click land on the <svg> instead of the box under it, so the panel
  // never opened; capture only once the pointer has actually travelled.
  svg.addEventListener("pointerdown", (e) => {
    sx = e.clientX; sy = e.clientY; ox = tx; oy = ty;
    moved = false;
  });
  svg.addEventListener("pointermove", (e) => {
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (!moved && Math.hypot(dx, dy) < 4) return;
    if (!dragging) {
      dragging = true; moved = true;
      svg.classList.add("drag");
      svg.setPointerCapture(e.pointerId);
    }
    tx = ox + dx; ty = oy + dy;
    apply(k, tx, ty);
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    svg.classList.remove("drag");
    if (e && e.pointerId != null && svg.hasPointerCapture(e.pointerId)) {
      svg.releasePointerCapture(e.pointerId);
    }
    // a real drag must not also register as a click on whatever it started over
    if (moved) {
      moved = false;
      const swallow = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
      svg.addEventListener("click", swallow, { capture: true, once: true });
      setTimeout(() => svg.removeEventListener("click", swallow, true), 0);
    }
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  // --- arrows first, so boxes sit on top of them ---
  for (const e of g.edges) {
    const a = byId.get(e.from), b = byId.get(e.to);
    if (!a || !b) continue;
    const x1 = a.x + a.w + 8, y1 = a.y + a.d / 2;
    const x2 = b.x - 8, y2 = b.y + b.d / 2;
    const mx = (x1 + x2) / 2;
    root.append(el("path", {
      class: "edge",
      d: `M ${x1} ${y1} L ${mx} ${y1} L ${mx} ${y2} L ${x2} ${y2}`,
      "marker-end": "url(#head)",
    }));
    if (e.shape && x2 - x1 > 70) {
      root.append(el("text", {
        class: "gap-n", x: mx, y: (y1 + y2) / 2 - 6, "text-anchor": "middle",
      }, document.createTextNode(e.shape.map((s) => String(s)).join("×"))));
    }
  }

  // --- boxes ---
  for (const p of P) {
    const nd = p.nd;
    const [fill, ink] = KINDS[nd.kind] || KINDS.tensor;
    if (p.kind === "gap") {
      root.append(el("path", {
        class: "gap", d: `M ${p.x} ${p.y + p.d / 2} L ${p.x + p.w} ${p.y + p.d / 2}`,
      }));
      root.append(el("text", {
        class: "gap-n", x: p.x + p.w / 2, y: p.y + p.d / 2 - 7, "text-anchor": "middle",
      }, document.createTextNode(`×${p.hidden}`)));
      continue;
    }
    const gEl = el("g", { class: "node", tabindex: "0", role: "button" });
    gEl.append(el("rect", {
      x: p.x, y: p.y, width: p.w, height: p.d, rx: 8,
      fill, stroke: ink, "stroke-width": 1.2,
    }));
    const room = Math.max(6, Math.floor((p.w - 18) / 6.6));
    const label = nd.label.length > room ? nd.label.slice(0, room - 1) + "…" : nd.label;
    gEl.append(el("text", {
      x: p.x + 11, y: p.y + 23, "font-size": 13, fill: ink, "font-weight": 600,
    }, document.createTextNode(label)));
    if (nd.params) {
      gEl.append(el("text", {
        x: p.x + 11, y: p.y + 41, "font-size": 11, fill: ink, opacity: .75,
      }, document.createTextNode(`${fmt(nd.params)} params`)));
    }
    const act = () => onPick && onPick(nd, p);
    gEl.addEventListener("click", act);
    gEl.addEventListener("keydown", (e) => { if (e.key === "Enter") act(); });
    root.append(gEl);
  }

  // --- branch captions ---
  for (const [name, label] of [["vision", "vision branch"], ["text", "language stack"]]) {
    if (!g.nodes.some((n) => n.branch === name)) continue;
    const items = P.filter((p) => p.nd.branch === name);
    const x = Math.min(...items.map((p) => p.x));
    const y = Math.max(56, Math.min(...items.map((p) => p.y)) - 30);
    root.append(el("text", { class: "branch", x, y }, document.createTextNode(label)));
  }

  return { placed: P, head: describe(g) };
}