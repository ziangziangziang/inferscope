import { useMemo } from "react";
import { useApp } from "../state";
import { Prov } from "./common";
import { bytes, gflops, num, params, us } from "../lib/format";
import type { GraphNode, TraceEvent } from "../lib/api";
import { Info } from "./Info";

function fmtShape(shape?: (number | string)[]): string {
  if (!shape) return "—";
  return "[" + shape.map((s) => (typeof s === "number" ? s.toLocaleString() : s)).join(", ") + "]";
}

export function Inspector({ events }: { events: TraceEvent[] }) {
  const { sel, clear } = useApp();
  const node = sel.node, edge = sel.edge;

  // Correlate execution events to the selected node by component/op name. Only
  // claim a correlation when names actually match; otherwise say "no trace link".
  const related = useMemo(() => {
    if (!node) return [];
    const key = node.label.toLowerCase();
    const op = (node.op || node.id).toLowerCase();
    return events.filter((e) => {
      const c = (e.component || e.op_id || "").toLowerCase();
      return c && (c.includes(op) || op.includes(c) || key.includes(c));
    }).slice(0, 8);
  }, [node, events]);

  if (!node && !edge) {
    return (
      <div className="inspector">
        <h2>Inspector</h2>
        <div className="faint" style={{ fontSize: 11.5 }}>
          Click a block or a line to see its numbers.
          <Info text="Shows tensor sizes, parameter counts and any matching execution events. To look inside a repeated group, click its expand ▸ tag." />
        </div>
      </div>
    );
  }

  return (
    <div className="inspector">
      <span className="close" onClick={clear}>✕</span>
      <h2>{node ? node.label : `${edge!.from} → ${edge!.to}`}</h2>
      <div className="row mb" style={{ gap: 6 }}>
        <span className="pill">{node ? node.kind : "edge"}</span>
        <Prov p={node ? node.provenance : edge!.provenance} />
      </div>

      {node?.dims && (
        <table className="kv-table"><tbody>
          {Object.entries(node.dims).map(([k, v]) => (
            <tr key={k}><td className="k">{k}</td>
              <td className="v">{Array.isArray(v) ? `[${(v as (number | string)[]).join(", ")}]` : String(v)}</td></tr>
          ))}
        </tbody></table>
      )}

      {node?.params != null && (
        <table className="kv-table"><tbody>
          <tr><td className="k">parameters</td><td className="v">{params(node.params)}</td></tr>
          {node.param_share != null && (
            <tr><td className="k">share of model</td>
              <td className="v">{node.param_share >= 0.01
                ? `${node.param_share.toFixed(2)}%`
                : `${node.param_share.toFixed(4)}%`}</td></tr>
          )}
          {node.flops ? (
            <>
              <tr><td className="k">compute</td>
                <td className="v">{gflops(node.flops)} / token</td></tr>
              {node.flop_share != null && (
                <tr><td className="k">share of total</td>
                  <td className="v">{node.flop_share.toFixed(2)}%</td></tr>
              )}
            </>
          ) : (
            <tr><td className="k">compute</td><td className="v">unavailable</td></tr>
          )}
        </tbody></table>
      )}

      {edge?.tensor && (
        <table className="kv-table"><tbody>
          <tr><td className="k">shape</td><td className="v">{fmtShape(edge.tensor.shape)}</td></tr>
          <tr><td className="k">dtype</td><td className="v">{edge.tensor.dtype ?? "—"}</td></tr>
          <tr><td className="k">size</td>
            <td className="v">{edge.tensor.bytes != null ? bytes(edge.tensor.bytes) : "symbolic dims"}</td></tr>
        </tbody></table>
      )}

      {node?.notes && node.notes.length > 0 && (
        <div className="mt">
          <div className="faint" style={{ fontSize: 10, marginBottom: 3 }}>notes</div>
          <ul style={{ margin: 0, paddingLeft: 16, fontSize: 11.5 }}>
            {node.notes.filter(Boolean).map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </div>
      )}

      {node && (
        <div className="mt">
          <div className="faint" style={{ fontSize: 10, marginBottom: 4 }}>correlated execution</div>
          {related.length ? (
            <table className="kv-table"><tbody>
              {related.map((e, i) => (
                <tr key={i}>
                  <td className="k">{e.kind}{e.step_idx != null ? ` #${e.step_idx}` : ""}</td>
                  <td className="v">{us(e.duration_us ?? null)}</td>
                </tr>
              ))}
            </tbody></table>
          ) : (
            <div className="faint" style={{ fontSize: 11 }}>
              No kernel events match this block by name. Architecture is config-derived; a runtime
              correlation needs a profiling session with layer attribution (see Timeline).
            </div>
          )}
        </div>
      )}

      <div className="faint mt" style={{ fontSize: 10.5 }}>
        <Info text={node
          ? "These numbers come from the model config file, not from this run."
          : "Size is the tensor's element count multiplied by its dtype width."} />
      </div>
    </div>
  );
}
