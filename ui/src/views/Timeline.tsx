import { useEffect, useMemo, useState } from "react";
import { useApp } from "../state";
import { api } from "../lib/api";
import type { TraceEvent } from "../lib/api";
import { Empty, Notice, Prov } from "../components/common";
import { clockUs, num, OP_COLOR, us } from "../lib/format";
import { Reveal } from "../components/Info";

const LANE_ORDER = ["request", "prefill", "decode_step", "kernel", "memcpy", "sync", "alloc"];
const LANE_LABEL: Record<string, string> = {
  request: "requests", prefill: "prefill", decode_step: "decode steps",
  kernel: "kernels", memcpy: "memory copy", sync: "sync", alloc: "alloc",
};

function evColor(e: TraceEvent): string {
  const op = (e.extra as any)?.op_class as string | undefined;
  if (op && OP_COLOR[op]) return OP_COLOR[op];
  if (e.kind === "memcpy") return "var(--accent2)";
  if (e.kind === "prefill") return "var(--warn)";
  if (e.kind === "decode_step") return "var(--good)";
  if (e.kind === "request") return "var(--dim)";
  return "var(--op-other)";
}

interface Props {
  /** Embedded in Analyze: the parent owns the heading and the empty state. */
  embedded?: boolean;
  onSelect?: (e: TraceEvent) => void;
}

export function Timeline({ embedded = false, onSelect }: Props = {}) {
  const { sessionId, select, sel, isDemo } = useApp();
  const [lanes, setLanes] = useState<Record<string, TraceEvent[]>>({});
  const [total, setTotal] = useState(0);
  const [downsampled, setDownsampled] = useState(false);
  const [range, setRange] = useState<[number, number] | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    api.timeline(sessionId).then((r) => { setLanes(r.lanes); setTotal(r.total); setDownsampled(!!r.downsampled); setRange(null); }).catch(() => {});
  }, [sessionId]);

  const all = useMemo(() => Object.values(lanes).flat(), [lanes]);
  const t0 = all.length ? Math.min(...all.map((e) => e.ts_wall_ns)) : 0;
  const t1 = all.length ? Math.max(...all.map((e) => e.ts_wall_ns + (e.duration_us ?? 0) * 1000)) : 1;
  const span = (range ? (range[1] - range[0]) : (t1 - t0)) || 1;
  const base = range ? range[0] : t0;

  const xOf = (e: TraceEvent) => ((e.ts_wall_ns - base) / span) * 100;
  const wOf = (e: TraceEvent) => Math.max(0.15, ((e.duration_us ?? 0) * 1000 / span) * 100);

  const present = LANE_ORDER.filter((k) => lanes[k]?.length);
  if (!all.length && embedded) return null;
  if (!all.length) {
    return (
      <div>
        <div className="view-head"><h1>Execution timeline</h1>
          <span className="desc">request / step / kernel events over time</span></div>
        <Empty>
          <div style={{ textAlign: "center", maxWidth: 520 }}>
            <div style={{ fontSize: 14, color: "var(--fg)", marginBottom: 8 }}>No execution events for this session</div>
            <div>
              A <b>live</b> session records continuous telemetry (GPU/CPU/RAM) but produces
              request/step/kernel <b>events</b> only when an engine is connected or a profiling
              capture is running. This session has neither.
            </div>
            <div className="mt" style={{ color: "var(--dim)" }}>
              Switch the session (top-right) to a <b>demo</b> or <b>profile</b> session to see a
              populated timeline, or start an engine with metrics.
            </div>
          </div>
        </Empty>
      </div>
    );
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => ({
    x: f * 100, label: clockUs(base + f * span),
  }));

  const zoomTo = (e: TraceEvent) => {
    const pad = span * 0.08;
    setRange([e.ts_wall_ns - pad, e.ts_wall_ns + (e.duration_us ?? 0) * 1000 + pad]);
    select({ event: e, stepIdx: e.step_idx ?? undefined });
    onSelect?.(e);
  };

  return (
    <div style={embedded
      ? { display: "flex", flexDirection: "column", minHeight: 0 }
      : undefined}>
      {!embedded && (
        <div className="view-head">
          <h1>Execution timeline</h1>
          <span className="desc">{num(total, 0)} events · {present.length} lanes · absolute wall-clock{downsampled ? " · downsampled view" : ""}</span>
          <div className="spacer" style={{ flex: 1 }} />
          {range && <button className="btn" onClick={() => setRange(null)}>reset zoom</button>}
        </div>
      )}
      {embedded && (
        <div className="row mb" style={{ gap: 10, flexWrap: "wrap" }}>
          <span className="faint" style={{ fontSize: 10.5 }}>execution over time</span>
          <span className="faint" style={{ fontSize: 10.5 }}>
            {num(total, 0)} events · {present.length} lanes{downsampled ? " · downsampled" : ""}
          </span>
          <div style={{ flex: 1 }} />
          {range && <button className="btn" onClick={() => setRange(null)}>reset zoom</button>}
        </div>
      )}

      {isDemo && <Notice kind="warn">Synthetic timeline — the shape is illustrative, the timings are <b>simulated</b>, not measured.</Notice>}

      <div className={`card${embedded ? " grow" : ""}`}>
        <div className="row mb" style={{ gap: 8, alignItems: "center" }}>
          <Reveal label="colours">
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {["gemm", "attn_full", "attn_linear", "elementwise"].map((k) => (
                <span key={k} className="legend">
                  <span className="item"><span className="sw" style={{ background: OP_COLOR[k] }} />{k}</span>
                </span>
              ))}
              <span className="legend"><span className="item"><span className="sw" style={{ background: "var(--accent2)" }} />memcpy</span></span>
              <span className="legend"><span className="item"><span className="sw" style={{ background: "var(--good)" }} />decode</span></span>
            </div>
          </Reveal>
        </div>

        <div className={`tl${embedded ? " grow" : ""}`}>
          {present.map((lane) => (
            <div className="tl-lane" key={lane}>
              <div className="name">{LANE_LABEL[lane] ?? lane}</div>
              <div className="tl-track">
                {(lanes[lane] || []).map((e, i) => (
                  <div key={i}
                    className={`tl-ev ${sel.event === e ? "selected" : ""}`}
                    style={{ left: `${xOf(e)}%`, width: `${wOf(e)}%`, background: evColor(e) }}
                    title={`${e.kind} ${e.component || e.op_id || ""} ${us(e.duration_us ?? null)}`}
                    onClick={() => zoomTo(e)}
                  />
                ))}
              </div>
            </div>
          ))}
          <div className="tl-axis">
            <div />
            <div className="ticks">
              {ticks.map((t, i) => <span key={i} className="tick" style={{ left: `${t.x}%` }}>{t.label}</span>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
