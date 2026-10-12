import { iso } from "./iso";

/**
 * Layout model for the model-architecture scene.
 *
 * Pure geometry and sizing, kept out of the renderer so it can be reasoned about
 * (and tested) on its own.
 *
 * Two rules drive everything here:
 *
 *  1. **A block's size is its real size.** A component's footprint is drawn from
 *     its analytic parameter count, so the one visual comparison in the diagram
 *     is a true comparison. Raw counts span 6 orders of magnitude here
 *     (a 1.27B embedding against a 5K RMSNorm), so edges scale as `params^p`
 *     with `p < 0.5` — monotonic and honest about *ordering* and *ratio*, while
 *     keeping the small parts legible. Every block is labelled with its actual
 *     count, so the number is never lost to the scaling.
 *
 *  2. **Repetitive units wrap.** 64 identical transformer blocks drawn as 64
 *     separate boxes would be an unreadable smear, so a repeat group becomes a
 *     near-square grid of unit boxes, each still sized by its own real count.
 *     The group's *footprint* then reflects the count, which is the honest
 *     thing for a grid to encode.
 */

import type { ArchNode } from "./archTypes";

/** Edge-length exponent. 0.5 would make footprint linear in params but buries
 *  the small parts; 0.36 keeps a visible, meaningful ratio across the range. */
export const SIZE_EXP = 0.36;
export const SIZE_MIN = 11;
export const SIZE_MAX = 42;

export const ARCH_COLORS: Record<string, string> = {
  input: "#9aa79b",
  tensor: "#9aa79b",
  embed: "#3f8fa8",
  matmul: "#7b5ea7",
  norm: "#6f8a6a",
  attention: "#4f9d6d",
  positional: "#b08a4a",
  repeat_group: "#2f7d4f",
  patchify: "#5b8fa8",
  projector: "#5b8fa8",
  sample: "#b5823a",
  add: "#98a89b",
};

export function colorOf(n: ArchNode): string {
  if (n.kind === "repeat_group") return ARCH_COLORS.repeat_group ?? "#2f7d4f";
  return ARCH_COLORS[n.kind] ?? "#7d8f80";
}

/** Top-face artwork per kind, in unit (u,v) so it scales with the box. */
export function artOf(n: ArchNode): { cells: { u: number; v: number; w: number; h: number }[]; stripe?: { v: number; h: number } } {
  const grid = (r: number, c: number, u0 = 0.1, v0 = 0.1, w = 0.8, h = 0.72) => ({
    cells: Array.from({ length: r * c }, (_, i) => ({
      u: u0 + (i % c) * (w / c), v: v0 + Math.floor(i / c) * (h / r),
      w: w / c * 0.84, h: h / r * 0.84,
    })),
    stripe: { v: 0.88, h: 0.07 },
  });
  switch (n.kind) {
    case "embed": return grid(3, 4);
    case "matmul": return grid(4, 4);
    case "repeat_group": return grid(3, 3);
    case "norm": return { cells: [{ u: 0.14, v: 0.3, w: 0.72, h: 0.3 }] };
    case "attention": return grid(2, 3);
    case "patchify": return grid(2, 5, 0.08, 0.2, 0.84, 0.5);
    case "projector": return grid(2, 2);
    case "sample": return { cells: [{ u: 0.2, v: 0.25, w: 0.22, h: 0.4 }, { u: 0.52, v: 0.32, w: 0.22, h: 0.3 }] };
    case "positional": return { cells: Array.from({ length: 7 }, (_, i) => ({ u: 0.1 + i * 0.12, v: 0.3, w: 0.06, h: 0.3 })) };
    default: return { cells: [{ u: 0.3, v: 0.3, w: 0.4, h: 0.4 }] };
  }
}

/** Near-square grid shape for a repeat count. */
export function gridShape(n: number): { cols: number; rows: number } {
  if (n <= 1) return { cols: 1, rows: 1 };
  const cols = Math.max(1, Math.ceil(Math.sqrt(n)));
  return { cols, rows: Math.ceil(n / cols) };
}

export interface SizeCtx { scale: number }

export function makeSizer(nodes: ArchNode[]): (params: number) => number {
  // Scale so the largest *single* component lands on SIZE_MAX, where
  // `single` means per-unit for repeat groups (a grid of 64 is not one box).
  let biggest = 0;
  for (const n of nodes) {
    const per = n.repeat?.count ? n.params / Math.max(1, n.repeat.count) : n.params;
    if (per > biggest) biggest = per;
  }
  const scale = biggest > 0 ? SIZE_MAX / Math.pow(biggest, SIZE_EXP) : 1;
  return (params: number) => {
    if (!(params > 0)) return SIZE_MIN;
    return Math.max(SIZE_MIN, Math.min(SIZE_MAX, Math.pow(params, SIZE_EXP) * scale));
  };
}

export interface Placed {
  id: string;
  node: ArchNode;
  x: number; y: number; z: number;
  w: number; d: number; h: number;
  /** index within its repeat group, for units */
  unitIndex?: number;
  groupId?: string;
  groupCount?: number;
  /** template nodes, present on repeat-group units */
  sub?: ArchNode[];
  column: number;
  order: number;
}

export interface Cap {
  key: string;
  /** projected anchor point */
  at: [number, number];
  anchor: "start" | "middle" | "end";
  title: string;
  sub?: string;
  hint?: string;
  color: string;
  cls: "grp-t" | "br-t";
  subCls: "grp-s" | "br-s";
  /** full text width, for bounds (the anchor decides which side it uses) */
  halfW: number;
  /** extra lines below the title, for bounds */
  lines: number;
}

/**
 * Work out where every caption in the architecture tier goes — in one place.
 *
 * This used to be computed twice, once for rendering and once for the viewBox,
 * and the two drifted apart: captions were drawn at one height while the
 * bounds reserved another, which put labels a long way from their boxes. There
 * is now exactly one answer.
 *
 * A caption is placed to the side of its box when the box sits at either end of
 * its row, because a box-sized caption under a box in the middle of a wide row
 * lands on top of its neighbours.
 */
/** Full width a caption needs, from the monospace advance of its longest line.
 *
 *  This has to be the *full* width, not the half: an `end`-anchored caption
 *  extends its whole length to the left of its anchor point, so reserving half
 *  clipped "reads the image · 27 blocks" off the edge of the viewBox. */
export function capW(title: string, sub?: string, hint?: string): number {
  const titleW = title.length * 7.0;        // 11.5px mono advance
  const subW = (sub?.length ?? 0) * 6.0;    // 9.5px
  const hintW = (hint?.length ?? 0) * 5.3;  // 9px italic
  return Math.max(48, titleW, subW, hintW) + 14;
}

export function layoutCaptions(layout: ArchLayout): Cap[] {
  const out: Cap[] = [];
  const groups = new Map<string, Placed[]>();
  for (const it of layout.items) {
    const k = it.groupId ?? it.id;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(it);
  }

  // branch captions first, so a component caption can be nudged off them
  const branchOf = new Map<string, number>();
  // Rows are laid out back to front; the back row's caption goes *above* it,
  // because below is exactly where the front row's first component is.
  const byDepth = (c: number) => Math.min(...layout.items.filter((i) => i.column === c).map((i) => i.y));
  const cols = [...new Set(layout.items.map((i) => i.column))].sort((a, b) => byDepth(a) - byDepth(b));
  const backCol = cols[0];
  for (const col of cols) {
    const row = layout.items.filter((i) => i.column === col);
    const back = Math.min(...row.map((i) => i.y));
    const front = Math.max(...row.map((i) => i.y + i.d));
    // Above its own row, centred: anchored at the left end it lands in the
    // same screen patch as the front row's first component and the two
    // overprint each other.
    const at = col === backCol
      ? iso((Math.min(...row.map((i) => i.x)) + Math.max(...row.map((i) => i.x + i.w))) / 2,
        back - 34, row[0].z)
      : iso(Math.min(...row.map((i) => i.x)) - 4, front + 26, row[0].z);
    const isVision = col === 1;
    const vcount = row.find((i) => i.unitIndex != null)?.groupCount;
    out.push({
      key: `branch-${col}`, at, anchor: col === backCol ? "middle" : "end",
      title: isVision ? "vision branch" : "text branch",
      sub: `${isVision ? "reads the image" : "reads the words"}${vcount ? ` · ${vcount} blocks` : ""}`,
      color: "#3d4a3e", cls: "br-t", subCls: "br-s",
      halfW: capW(isVision ? "vision branch" : "text branch",
        `${isVision ? "reads the image" : "reads the words"}${vcount ? ` · ${vcount} blocks` : ""}`), lines: 1,
    });
    for (const it of row) branchOf.set(it.id, col);
  }

  const biggest = Math.max(...[...groups.values()].map((v) => v.length));
  for (const [gid, items] of groups) {
    const node = items[0].node;
    const cx = items.reduce((a, i) => a + i.x + i.w / 2, 0) / items.length;
    const back = Math.min(...items.map((i) => i.y));
    const front = Math.max(...items.map((i) => i.y + i.d));
    const col = branchOf.get(items[0].id) ?? 0;
    const row = layout.items.filter((i) => i.column === col);
    const rx0 = Math.min(...row.map((i) => i.x));
    const rx1 = Math.max(...row.map((i) => i.x + i.w));
    const rel = (cx - rx0) / Math.max(1, rx1 - rx0);
    const at = items[0].z;

    if (items.length > 1) {
      const per = node.params / items.length;
      out.push({
        key: gid, at: iso(cx, front + 8, at), anchor: "middle",
        title: node.label,
        sub: `${fmtParams(per)} each · ${fmtParams(node.params)} total`,
        hint: items.length === biggest ? "click any box for its sub-layers" : undefined,
        color: "#2f7d4f", cls: "grp-t", subCls: "grp-s",
        halfW: capW(node.label,
          `${fmtParams(per)} each · ${fmtParams(node.params)} total`,
          "click any box for its sub-layers"), lines: 2,
      });
      continue;
    }

    const p = items[0];
    if (p.w < 20) continue;   // too small to letter; tooltip + panel cover it
    // At either end of a row there is empty space beside the box, which is
    // where a wide caption belongs.
    const side = rel < 0.24 ? "left" : rel > 0.76 ? "right" : "below";
    const pos = side === "left" ? iso(p.x - 6, p.y + p.d / 2, at)
      : side === "right" ? iso(p.x + p.w + 6, p.y + p.d / 2, at)
        : iso(cx, front + 8, at);
    out.push({
      key: gid, at: pos,
      anchor: side === "left" ? "end" : side === "right" ? "start" : "middle",
      title: p.node.label,
      sub: p.node.params ? fmtParams(p.node.params) : undefined,
      color: "#4a5560", cls: "grp-t", subCls: "grp-s",
      halfW: capW(p.node.label, p.node.params ? fmtParams(p.node.params) : undefined),
      lines: 1,
    });
  }

  // Branch captions first so component captions win any residual overlap. This
  // used to sort by width, which had the opposite effect: it silently painted
  // one caption over another instead of letting the collision be visible.
  return out.sort((a, b) => (a.cls === "br-t" ? -1 : 1) - (b.cls === "br-t" ? -1 : 1));
}

export interface ArchLayout {
  items: Placed[];
  /** total world extents of the layout, for the bounds */
  x0: number; y0: number; x1: number; y1: number;
  size: (params: number) => number;
  totalParams: number;
}

export function fmtParams(p: number): string {
  if (!p) return "0";
  if (p >= 1e9) return (p / 1e9).toFixed(2) + "B";
  if (p >= 1e6) return (p / 1e6).toFixed(1) + "M";
  if (p >= 1e3) return (p / 1e3).toFixed(1) + "K";
  return String(p);
}

export interface LayoutOpts {
  /** gap between neighbouring boxes along a branch */
  gap: number;
  /** gap between branch rows */
  rowGap: number;
  /** height of every architecture box */
  h: number;
}

const BRANCH_MIN = 92;

/**
 * Lay the architecture out as parallel branches, each read left to right.
 *
 * Two passes, because the obvious one-liner gets it wrong: a branch's *depth*
 * is only known once its repeat grid has been placed, and with a 64-block grid
 * that depth is over 300 world units. Fixing the row spacing up front makes the
 * second row land on top of the first.
 *
 * Rows are then stacked at `rowGap` apart and the whole tier is centred on the
 * plate's screen axis, so it grows straight up out of the model instead of
 * drifting off to one side.
 */
export function layoutArch(nodes: ArchNode[], opts: LayoutOpts): ArchLayout {
  const size = makeSizer(nodes);
  const { gap, rowGap, h } = opts;
  const items: Placed[] = [];

  const branches = new Map<number, ArchNode[]>();
  for (const n of nodes) {
    const c = n.column ?? 0;
    if (!branches.has(c)) branches.set(c, []);
    branches.get(c)!.push(n);
  }
  // Descending: column 1 (vision) is prepended by the backend, so it is the
  // back row and column 0 (text) reads in front of it.
  const cols = [...branches.keys()].sort((a, b) => b - a);

  // --- pass 1: each branch into its own strip, at y = 0 ---
  const strips: { col: number; d: number }[] = [];
  cols.forEach((col) => {
    const list = (branches.get(col) ?? []).slice()
      .sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    let x = 0;
    let deepest = 0;
    for (const n of list) {
      const count = n.repeat?.count ?? 0;
      if (count > 1) {
        const { cols: gc, rows: gr } = gridShape(count);
        const us = size(n.params / count);
        const g = Math.max(2, us * 0.1);
        const sub = n.repeat?.template?.nodes ?? [];
        for (let i = 0; i < count; i++) {
          items.push({
            id: `${n.id}#${i}`, node: n, groupId: n.id, groupCount: count, unitIndex: i,
            sub, column: col, order: n.order ?? 0,
            x: x + (i % gc) * (us + g), y: Math.floor(i / gc) * (us + g), z: 0,
            w: us, d: us, h,
          });
        }
        deepest = Math.max(deepest, gr * us + (gr - 1) * g);
        x += gc * us + (gc - 1) * g + gap;
      } else {
        const s = size(n.params);
        items.push({
          id: n.id, node: n, column: col, order: n.order ?? 0,
          x, y: 0, z: 0, w: s, d: s, h,
        });
        deepest = Math.max(deepest, s);
        x += s + gap;
      }
    }
    strips.push({ col, d: Math.max(deepest, BRANCH_MIN) });
  });

  // --- pass 2: stack the strips, then centre the tier on the plate axis ---
  let y = 0;
  for (const st of strips) {
    for (const it of items) if (it.column === st.col) it.y += y;
    y += st.d + rowGap;
  }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const it of items) {
    x0 = Math.min(x0, it.x); y0 = Math.min(y0, it.y);
    x1 = Math.max(x1, it.x + it.w); y1 = Math.max(y1, it.y + it.d);
  }
  // Centring on the screen axis: screen x is (x - y) * cos30, so equal x and y
  // means centred. Only y moves, which keeps the tier horizontal.
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const dy = cx - cy;
  for (const it of items) it.y += dy;
  y0 += dy; y1 += dy;

  const totalParams = nodes.reduce((a, n) => a + (n.params ?? 0), 0);
  return { items, x0, y0, x1, y1, size, totalParams };
}