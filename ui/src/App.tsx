import { useState } from "react";
import { useApp } from "./state";
import { PROV_LABEL } from "./lib/format";
import { Info } from "./components/Info";

const PROV_HELP: Record<string, string> = {
  measured: "read straight from the hardware or engine",
  derived: "computed from measured values",
  estimated: "worked out from a formula or config",
  simulated: "fake data from the demo",
  unavailable: "this platform does not report it",
};
import { Analyze } from "./views/Analyze";
import { Machine } from "./views/Machine";
import { ModelSection, useModelTab } from "./views/ModelSection";
import { Chat } from "./views/Chat";
import { Index } from "./views/Index";

/**
 * Four sections, ordered as the workflow actually runs:
 *   Run     — produce data
 *   Analyze — read what was measured
 *   Model   — reference facts about the model
 *   Machine — live state of the box
 *
 * This replaces seven equal tabs, which hid the fact that only one of them
 * (Run) produces anything and that two of them were the same data twice.
 */
type View = "index" | "run" | "analyze" | "arch" | "memory" | "mm" | "machine";

const MODEL_TABS = new Set<View>(["arch", "memory", "mm"]);

export function App() {
  // Land on the index: it explains what this is before asking anything of you.
  const [view, setView] = useState<View>("index");
  const [modelTab, setModelTab] = useModelTab();
  const { health, sessions, sessionId, setSessionId, isDemo, error } = useApp();

  const gpu = health?.collectors?.gpu;
  const go = (v: View) => {
    if (MODEL_TABS.has(v)) {
      setModelTab(v === "arch" ? "arch" : v === "memory" ? "memory" : "mm");
    }
    setView(v);
  };

  return (
    <div className="app">
      <div className="topbar">
        <span className="brand" title={`InferScope — inference observability workbench, v${health?.version}`}>
          INFER<b>SCOPE</b></span>
        <div className="spacer" />
        {health?.public_url && (
          <a className="pill" href={health.public_url} target="_blank" rel="noreferrer"
            title="This app is served through the tunnel — that URL is the canonical host">
            ◉ tunnel host
          </a>
        )}
        {isDemo && <span className="badge simulated"><span className="dot" /> demo data</span>}
        <span className="badge" title={gpu?.available ? `GPU via ${gpu.backend}` : "no GPU collector"}>
          <span className="dot" style={{ background: gpu?.available ? "var(--good)" : "var(--bad)" }} />
          {gpu?.available ? gpu.backend : "no gpu"}
        </span>
        <label className="row" style={{ gap: 6 }}>
          <span className="faint" style={{ fontSize: 11 }}>session</span>
          <select value={sessionId} onChange={(e) => setSessionId(e.target.value)}
            style={{ width: 260 }}
            title={sessions.find((s) => s.session_id === sessionId)?.session_id}>
            {sessions.map((s) => (
              <option key={s.session_id} value={s.session_id}>
                {sessionLabel(s.session_id, s.kind, s.kernel_events, s.event_count, s.top_op)}
              </option>
            ))}
          </select>
        </label>
      </div>

      <nav className="nav">
        <NavBtn id="index" view={view} go={go} icon="◍" label="Start here" />

        <Section label="produce" />
        <NavBtn id="run" view={view} go={go} icon="✦" label="Run & capture" />

        <Section label="understand" />
        <NavBtn id="analyze" view={view} go={go} icon="◧" label="Analyze" />

        <Section label="reference" />
        <NavBtn id="arch" view={view} go={go} icon="⌗" label="Architecture" sub />
        <NavBtn id="memory" view={view} go={go} icon="▤" label="Memory & KV" sub />
        <NavBtn id="mm" view={view} go={go} icon="◈" label="Multimodal" sub />

        <Section label="machine" />
        <NavBtn id="machine" view={view} go={go} icon="◉" label="Status" />

        <div className="legend-inline">
          <Info text={
            <span style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {(["measured", "derived", "estimated", "simulated", "unavailable"] as const).map((p) => (
                <span key={p}><b>{PROV_LABEL[p]}</b> — {PROV_HELP[p]}</span>
              ))}
            </span>
          } />
          <span className="faint">numbers</span>
        </div>
      </nav>

      <main className="main">
        {error && <div className="notice bad mb">backend unreachable: {error}</div>}
        {view === "index" && <Index onOpen={(v) => go(v)} />}
        {view === "run" && <Chat />}
        {view === "analyze" && <Analyze />}
        {MODEL_TABS.has(view) && (
          <ModelSection tab={modelTab} setTab={(t) => go(t === "arch" ? "arch" : t === "memory" ? "memory" : "mm")} />
        )}
        {view === "machine" && <Machine />}
      </main>
    </div>
  );
}

function Section({ label }: { label: string }) {
  return <div className="section" style={{ marginTop: 12 }}>{label}</div>;
}

function NavBtn({ id, view, go, icon, label, sub }: {
  id: View; view: View; go: (v: View) => void;
  icon: string; label: string; sub?: boolean;
}) {
  return (
    <button
      className={`${view === id ? "active" : ""} ${sub ? "subitem" : ""}`}
      onClick={() => go(id)}
      style={sub ? { paddingLeft: 26 } : undefined}
    >
      <span className="ic">{icon}</span>{label}
    </button>
  );
}

/** Session picker: say what is *in* the session, not just its id. */
function sessionLabel(id: string, kind: string, kernels: number, events: number, topOp: string | null) {
  const what = kernels > 0
    ? `${kernels.toLocaleString()} kernels${topOp ? ` · ${topOp}` : ""}`
    : events > 0
      ? `${events.toLocaleString()} events`
      : "empty";
  return `${id} — ${what}`;
}