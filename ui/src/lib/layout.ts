// Deterministic layered layout for the architecture graph. Nodes carry a column
// (0 = main stream, 1 = vision branch) and an order (vertical position); we turn
// that into x/y boxes. Kept pure so it is unit-testable without a browser.
import type { GraphEdge, GraphNode } from "./api";

export const NODE_W = 200;
export const NODE_H = 60;
export const COL_GAP = 150;
export const ROW_GAP = 30;
export const PAD = 40;

export interface Pos { x: number; y: number; w: number; h: number; }
export interface Layout {
  pos: Record<string, Pos>;
  width: number;
  height: number;
}

export function layoutNodes(nodes: GraphNode[]): Layout {
  const byColumn = new Map<number, GraphNode[]>();
  for (const n of nodes) {
    const c = n.column ?? 0;
    if (!byColumn.has(c)) byColumn.set(c, []);
    byColumn.get(c)!.push(n);
  }
  const cols = [...byColumn.keys()].sort((a, b) => a - b);
  const pos: Record<string, Pos> = {};
  let maxRows = 0;
  cols.forEach((c, ci) => {
    const list = byColumn.get(c)!.slice().sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    maxRows = Math.max(maxRows, list.length);
    const x = PAD + ci * (NODE_W + COL_GAP);
    list.forEach((n, ri) => {
      pos[n.id] = { x, y: PAD + ri * (NODE_H + ROW_GAP), w: NODE_W, h: NODE_H };
    });
  });
  const width = PAD * 2 + cols.length * (NODE_W + COL_GAP);
  const height = PAD * 2 + maxRows * (NODE_H + ROW_GAP);
  return { pos, width, height };
}

// A vertical bezier between two boxes; residual edges get a class in the caller.
export function edgePath(a: Pos, b: Pos): string {
  const x1 = a.x + a.w / 2, y1 = a.y + a.h;
  const x2 = b.x + b.w / 2, y2 = b.y;
  // same column, downward
  if (Math.abs(x1 - x2) < 1 && y2 > y1) {
    return `M ${x1} ${y1} L ${x2} ${y2}`;
  }
  const my = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;
}

export function edgeEndpoints(a: Pos, b: Pos): { x: number; y: number } {
  return { x: (a.x + a.w / 2 + b.x + b.w / 2) / 2, y: (a.y + a.h + b.y) / 2 };
}

export function nodeKindClass(kind: string): string {
  if (kind === "matmul" || kind === "head" || kind === "projector" || kind === "vae") return "matmul";
  if (kind === "attention") return "attention";
  if (kind === "add") return "add";
  if (kind === "norm") return "norm";
  return "";
}

export function summarizeDims(dims?: Record<string, unknown>): string {
  if (!dims) return "";
  const parts: string[] = [];
  for (const [k, v] of Object.entries(dims)) {
    if (Array.isArray(v)) parts.push(`${k}:[${v.join(",")}]`);
    else if (typeof v === "number") parts.push(`${k}:${v}`);
  }
  return parts.slice(0, 3).join(" ");
}
