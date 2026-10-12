import { useEffect, useMemo, useState } from "react";
import { useApp } from "../state";
import { api, type TraceEvent } from "../lib/api";
import { Empty, Notice, Prov } from "../components/common";
import { Info, Reveal } from "../components/Info";
import { ExperimentRecorder } from "../components/ExperimentRecorder";
import { Timeline } from "./Timeline";
import { num, us } from "../lib/format";

interface OpRow { op_class: string; launches: number; total_us: number; pct: number; }

const OP_HELP: Record<string, string> = {
  gemm: "matrix multiply — the big matrix maths a transformer spends most of its time on",
  elementwise: "tiny maths applied to each element on its own (shapes, activations)",
  attn_full: "full attention, where every token looks at every other token",
  attn_linear: "linear attention — a cheap running summary instead of full attention",
  memcpy: "copying bytes between memory and the GPU",
  other: "anything else",
};

const OP_COLOR: Record<string, string> = {
  gemm: "var(--op-gemm)", attn_full: "var(--op-attn-full)",
  attn_linear: "var(--op-attn-linear)", elementwise: "var(--op-elementwise)",
  other: "var(--op-other)",
};

/**
 * Where did the time go, for the selected session.
 *
 * The op-class breakdown and the timeline are the same events in two forms, so
 * they live in one view: the aggregate answers "which op dominates" in one
 * glance, the timeline answers "when and in what order" for that same set. They
 * used to be separate tabs, which made a user hunt for the other half.
 */
export function Analyze() {
  const { sessionId, sessions, isDemo, select } = useApp();
  const [ops, setOps] = useState<OpRow[] | null>(null);
  const [totalUs, setTotalUs] = useState(0);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [selected, setSelected] = useState<TraceEvent | null>(null);

  const session = useMemo(
    () => sessions.find((s) => s.session_id === sessionId), [sessions, sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    setOps(null);
    setSelected(null);
    api.opclasses(sessionId)
      .then((r) => { setOps(r.by_class); setTotalUs(r.total_us); })
      .catch(() => { setOps([]); setTotalUs(0); });
    api.events(sessionId, undefined, 3000)
      .then((r) => setEvents(r.events))
      .catch(() => setEvents([]));
  }, [sessionId]);

  const kernels = session?.kernel_events ?? 0;

  if (session && !kernels && !events.length) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
        <Head session={session} />
        <Empty>
          This session has telemetry but no execution events yet — nothing has been
          profiled in it. Send a prompt under <b>Run</b> (with the kernel trace on)
          to capture one, or pick a session that already has a capture.
        </Empty>
      </div>
    );
  }

  const top = ops?.[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
      <Head session={session} />

      {/* Summary first: the aggregate is the answer most people came for. */}
      <div className="row mb" style={{ gap: 8, flexWrap: "wrap" }}>
        <span className="pill">{num(kernels, 0)} kernels</span>
        <span className="pill">{(totalUs / 1000).toFixed(0)} ms GPU time</span>
        {top && <span className="pill" title="dominant op class by total GPU time">
          {top.op_class} {top.pct}%
        </span>}
        {session?.top_op && <Prov p="measured" />}
      </div>

      <div style={{ display: "flex", gap: 12, minHeight: 260 }}>
        <div style={{ flex: "0 0 300px", minWidth: 0, display: "flex", flexDirection: "column" }}>
          <div className="faint" style={{ fontSize: 10, marginBottom: 4 }}
            title="Kinds of GPU work. gemm = matrix multiply (the bulk of a transformer), elementwise = tiny per-element maths, attn = attention, memcpy = moving bytes around">
            time by kind of work
          </div>
          {ops && ops.length ? (
            <table className="kv-table">
              <tbody>
                {ops.map((r) => (
                  <tr key={r.op_class} style={{ cursor: "pointer" }}
                    onClick={() => select({ op: r.op_class })}>
                    <td className="k">
                      <span className="sw" style={{
                        background: OP_COLOR[r.op_class] ?? "var(--op-other)",
                        display: "inline-block", width: 9, height: 9,
                        borderRadius: 2, marginRight: 6,
                      }} />
                      <span title={OP_HELP[r.op_class] ?? "other GPU work"}>{r.op_class}</span>
                    </td>
                    <td className="v">{r.pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div className="faint" style={{ fontSize: 11 }}>no kernels in this session</div>}

          {ops && ops.length > 0 && (
            <Reveal label="launch counts">
              <div style={{ fontSize: 10.5, color: "var(--faint)" }}>
                {ops.slice(0, 6).map((r) => (
                  <div key={r.op_class}>{r.op_class}: {num(r.launches, 0)}</div>
                ))}
              </div>
            </Reveal>
          )}

          {selected && (
            <div className="mt">
              <div className="faint" style={{ fontSize: 10, marginBottom: 3 }}>selected</div>
              <div style={{ fontSize: 11 }}>{us(selected.duration_us)}</div>
              <div className="faint" style={{ fontSize: 10 }}>{selected.component ?? selected.kind}</div>
            </div>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
          <Timeline embedded onSelect={setSelected} />
        </div>
      </div>

      {isDemo && (
        <div className="mt"><Notice kind="warn">
          Synthetic session — the breakdown above is <b>simulated</b>, not measured.
        </Notice></div>
      )}

      <div className="mt">
        <Reveal label="Record an experiment">
          <ExperimentRecorder sessions={sessions} />
        </Reveal>
      </div>
    </div>
  );
}

function Head({ session }: { session?: { session_id: string; kind: string; model_id: string } }) {
  return (
    <div className="view-head">
      <h1>Analyze</h1>
      <Info text="Where the time went. The breakdown answers which kind of work dominated; the timeline below answers when, and in what order." />
      <div style={{ flex: 1 }} />
      <span className="faint" style={{ fontSize: 11 }}>
        {session?.session_id}
      </span>
    </div>
  );
}