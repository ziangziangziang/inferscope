import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GraphEdge, GraphNode } from "../lib/api";
import {
  edgeEndpoints, edgePath, layoutNodes, nodeKindClass, PAD, summarizeDims,
} from "../lib/layout";
import { gflops, params } from "../lib/format";

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedId?: string;
  onSelectNode: (n: GraphNode) => void;
  onSelectEdge: (e: GraphEdge) => void;
  onExpand: (n: GraphNode) => void;
}

// Zoom/pan transform applied to the whole <g>. Wheel zooms toward the cursor;
// drag pans. Kept in one place so the graph is explorable at any size.
export function Graph({ nodes, edges, selectedId, onSelectNode, onSelectEdge, onExpand }: Props) {
  const layout = layoutNodes(nodes);
  // Box area encodes parameter share: a bar rather than a resized box, so the
  // layered layout stays aligned and labels stay readable.
  const maxParams = useMemo(
    () => Math.max(1, ...nodes.map((n) => n.params ?? 0)), [nodes]);
  const [t, setT] = useState({ x: 0, y: 0, k: 1 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const fit = useCallback(() => {
    const el = svgRef.current;
    if (!el) return;
    const { clientWidth, clientHeight } = el;
    const k = Math.min(1.2, Math.min(clientWidth / layout.width, clientHeight / layout.height) * 0.95);
    setT({ x: (clientWidth - layout.width * k) / 2, y: 12, k: Math.max(0.3, k) });
  }, [layout.width, layout.height]);

  useEffect(() => { fit(); }, [fit]);

  const onWheel = (e: React.WheelEvent) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const dk = e.deltaY < 0 ? 1.12 : 0.89;
    setT((p) => {
      const k = Math.min(3, Math.max(0.2, p.k * dk));
      return { k, x: mx - ((mx - p.x) / p.k) * k, y: my - ((my - p.y) / p.k) * k };
    });
  };
  const onDown = (e: React.MouseEvent) => { drag.current = { x: e.clientX - t.x, y: e.clientY - t.y }; };
  const onMove = (e: React.MouseEvent) => {
    if (!drag.current) return;
    setT((p) => ({ ...p, x: e.clientX - drag.current!.x, y: e.clientY - drag.current!.y }));
  };
  const onUp = () => { drag.current = null; };

  const zoom = (mult: number) => setT((p) => ({ ...p, k: Math.min(3, Math.max(0.2, p.k * mult)) }));

  return (
    <div className="graph-wrap" style={{ height: "100%", minHeight: 420 }}>
      <div className="graph-toolbar">
        <button title="zoom in" onClick={() => zoom(1.2)}>+</button>
        <button title="zoom out" onClick={() => zoom(0.83)}>−</button>
        <button title="fit" onClick={fit}>⤢</button>
      </div>
      <svg ref={svgRef} className="graph" onWheel={onWheel} onMouseDown={onDown}
        onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--line2)" />
          </marker>
        </defs>
        <g transform={`translate(${t.x},${t.y}) scale(${t.k})`}>
          {edges.map((e, i) => {
            const a = layout.pos[e.from], b = layout.pos[e.to];
            if (!a || !b) return null;
            const isRes = e.label === "residual";
            const mid = edgeEndpoints(a, b);
            return (
              <g key={i} onClick={(ev) => { ev.stopPropagation(); onSelectEdge(e); }} style={{ cursor: "pointer" }}>
                <path className={`edge ${isRes ? "residual" : ""}`} d={edgePath(a, b)} />
                {e.label && !isRes && (
                  <text className="edge-label" x={mid.x + 4} y={mid.y}>{e.label}</text>
                )}
              </g>
            );
          })}
          {nodes.map((n) => {
            const p = layout.pos[n.id];
            if (!p) return null;
            const sel = n.id === selectedId;
            const sub = summarizeDims(n.dims);
            const hasCost = n.params != null;
            // "tied" only makes sense for a weight-shared output projection;
            // a zero count elsewhere just means "no learned weights".
            const costLabel = n.params === 0
              ? (n.id === "head" ? "tied weights" : "no weights")
              : params(n.params);
            const flopsLabel = n.flops ? ` · ${gflops(n.flops)}` : "";
            const barMax = n.repeat ? p.w - 74 : p.w - 20;
            const barW = hasCost
              ? Math.max(2, Math.round(((n.params ?? 0) / maxParams) * barMax))
              : 0;
            return (
              <g key={n.id} className={`node ${nodeKindClass(n.kind)} ${sel ? "selected" : ""}`}
                transform={`translate(${p.x},${p.y})`} onClick={(ev) => { ev.stopPropagation(); onSelectNode(n); }}>
                <rect width={p.w} height={p.h} rx={6} />
                <text x={10} y={17}>{n.label.length > 23 ? n.label.slice(0, 22) + "…" : n.label}</text>
                {sub && <text className="sub" x={10} y={30}>{sub.slice(0, 26)}</text>}
                {hasCost && (
                  <>
                    <text className="sub" x={10} y={43} style={{ fontSize: 9.5, fill: "var(--fg2)" }}>
                      {costLabel}{flopsLabel}
                    </text>
                    <rect x={10} y={46} width={barMax} height={3} rx={1.5} fill="var(--bg3)" />
                    <rect x={10} y={46} width={barW} height={3} rx={1.5}
                      fill="var(--accent)" opacity={0.8} />
                  </>
                )}
                <circle cx={p.w - 9} cy={9} r={3.5} fill={`var(--prov-${n.provenance})`} />
                {n.repeat && (
                  <g onClick={(ev) => { ev.stopPropagation(); onExpand(n); }} style={{ cursor: "pointer" }}>
                    <rect x={p.w - 48} y={p.h - 17} width={42} height={14} rx={3} fill="var(--bg3)" stroke="var(--line2)" />
                    <text className="repeat-tag" x={p.w - 44} y={p.h - 6} style={{ fontSize: 9 }}>expand ▸</text>
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
