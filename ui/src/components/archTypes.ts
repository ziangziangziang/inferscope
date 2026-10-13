/**
 * Wire types for the architecture graph served by `/api/models/{id}/architecture`.
 *
 * The scene reuses the app's existing `Graph` shape and only normalises it:
 * `params` becomes required (0 for data-flow markers that hold no weights) and
 * `notes` is allowed to contain the nulls the backend emits.
 */

import type { Graph, GraphEdge, GraphNode } from "../lib/api";

export interface ArchNode extends Omit<GraphNode, "params" | "notes" | "repeat"> {
  /** analytic parameter count; 0 for data-flow markers with no weights */
  params: number;
  /** what this node belongs to: "encoder", "decoder", "front end"...
   *  Absent means fall back to the old column-based naming. */
  branch?: string;
  notes?: (string | null)[] | null;
  repeat?: {
    count: number;
    template: { nodes: ArchNode[]; edges: GraphEdge[] };
  } | null;
}

export interface ArchGraph extends Omit<Graph, "nodes"> {
  nodes: ArchNode[];
  /** whole-model parameter total, when the builder could derive one */
  params_total?: number;
}

export function toArch(g: Graph | null | undefined): ArchGraph | null {
  if (!g || !Array.isArray(g.nodes)) return null;
  // Spelled out field by field rather than spread: `GraphNode.repeat` holds
  // un-normalised template nodes, and spreading it in would smuggle the
  // optional `params` back into the required one.
  const norm = (n: GraphNode): ArchNode => ({
    id: n.id, label: n.label, kind: n.kind, branch: n.branch,
    column: n.column, order: n.order, group: n.group, provenance: n.provenance,
    dims: n.dims, op: n.op, flops: n.flops,
    param_share: n.param_share, flop_share: n.flop_share,
    params: n.params ?? 0,
    notes: n.notes ?? null,
    repeat: null,
  });
  return {
    ...g,
    nodes: g.nodes.map((n) => {
      const base = norm(n);
      return n.repeat
        ? { ...base, repeat: {
            count: n.repeat.count,
            template: {
              nodes: n.repeat.template.nodes.map(norm),
              edges: n.repeat.template.edges,
            },
          } }
        : base;
    }),
  };
}

/** What the user has clicked in the scene. Drives both the detail panel and the
 *  dimming, so the two can never disagree about what is selected. */
export type Sel =
  | { kind: "host" }
  | { kind: "engine" }
  | { kind: "model" }
  | { kind: "node"; id: string }
  | { kind: "unit"; id: string; index: number }
  | null;