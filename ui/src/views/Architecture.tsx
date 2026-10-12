import { useEffect, useState } from "react";
import { useApp } from "../state";
import { api } from "../lib/api";
import type { GraphNode, TraceEvent } from "../lib/api";
import { useArch } from "../lib/useArch";
import { Graph as GraphCanvas } from "../components/Graph";
import { Inspector } from "../components/Inspector";
import { Empty, Notice, Prov } from "../components/common";
import { gflops, params } from "../lib/format";
import { Info } from "../components/Info";

interface Frame { title: string; nodes: GraphNode[]; edges: import("../lib/api").Graph["edges"]; }

export function Architecture() {
  const { sessionId, select, sel, isDemo } = useApp();
  const [model, setModel] = useState("");
  const { graph, models } = useArch(sessionId, model);
  const [stack, setStack] = useState<Frame[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);

  useEffect(() => { setStack([]); }, [graph]);
  useEffect(() => {
    if (sessionId) api.events(sessionId, "kernel", 4000).then((r) => setEvents(r.events)).catch(() => {});
  }, [sessionId]);

  const view = stack.length ? stack[stack.length - 1]
    : (graph ? { title: graph.model_id, nodes: graph.nodes, edges: graph.edges } : null);
  const totals = graph?.costs?.totals;

  const expand = (n: GraphNode) => {
    if (!n.repeat) return;
    setStack((s) => [...s, { title: n.label, nodes: n.repeat!.template.nodes, edges: n.repeat!.template.edges }]);
    select({ node: n });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="view-head">
        <h1>Architecture</h1>
        <Info text="The model's computational graph, built from its config file. Click a block to inspect it, or expand a × N group to look inside one layer." />
        <div style={{ flex: 1 }} />
        <select value={model} onChange={(e) => setModel(e.target.value)} style={{ width: 280 }}>
          <option value="">— {graph ? `session model (${graph.family})` : "loading…"} —</option>
          {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.model_id} ({m.family})</option>)}
        </select>
      </div>

      {isDemo && !model && <Notice kind="warn">Estimated graph — pick a real checkpoint above.</Notice>}

      {!view ? <Empty>{graph ? "loading…" : "no model configs found on this host"}</Empty> : (
        <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>
          <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
            <div className="row mb" style={{ gap: 8 }}>
              <span className="pill" style={{ cursor: "pointer" }} onClick={() => setStack([])}>{graph!.model_id}</span>
              {stack.map((f, i) => (
                <span key={i} className="pill" style={{ cursor: "pointer" }}
                  onClick={() => setStack((s) => s.slice(0, i + 1))}>▸ {f.title}</span>
              ))}
              <Prov p={stack.length ? "derived" : graph!.provenance} />
              <span className="faint" style={{ fontSize: 11 }}>family: {graph!.family} · {view.nodes.length} blocks</span>
            </div>
            {!stack.length && totals && (
              <div className="row mb" style={{ gap: 8 }}>
                <span className="pill" title="analytic parameter count from the config">
                  {params(totals.params)} params
                </span>
                <span className="pill" title="parameters in one transformer block">
                  {params(totals.params_per_layer)} / block
                </span>
                <span className="pill" title={`FLOPs per token at ${totals.ctx} tokens of context`}>
                  {gflops(totals.flops_per_token)} / token
                </span>
                {totals.vision_params != null && (
                  <span className="pill" title="parameters in the vision tower">
                    vision {params(totals.vision_params)}
                  </span>
                )}
              </div>
            )}
            <div style={{ flex: 1, minHeight: 0 }}>
              <GraphCanvas nodes={view.nodes} edges={view.edges}
                selectedId={sel.node?.id}
                onSelectNode={(n) => select({ node: n })}
                onSelectEdge={(e) => select({ edge: e })}
                onExpand={expand} />
            </div>
            <div className="legend mt">
              <span className="item"><span className="sw" style={{ background: "var(--op-gemm)" }} />matmul/proj</span>
              <span className="item"><span className="sw" style={{ background: "var(--op-attn-full)" }} />attention</span>
              <span className="item"><span className="sw" style={{ background: "#24503c" }} />residual</span>
              <span className="item"><span className="sw" style={{ background: "var(--bg2)", border: "1px solid var(--line2)" }} />norm/other</span>
              <span className="item faint"><span className="sw" style={{ background: "var(--accent)", opacity: 0.8, height: 3 }} />bar = parameters (relative)</span>
              <span className="item faint">● = provenance colour</span>
            </div>
          </div>
          <Inspector events={events} />
        </div>
      )}
    </div>
  );
}
