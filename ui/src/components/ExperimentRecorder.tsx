import { useState } from "react";
import { api } from "../lib/api";

interface Exp { name: string; kind: string; baseline: string; candidate: string; note: string; }

/**
 * Records a baseline-vs-candidate comparison.
 *
 * This used to live in the Bottlenecks tab. When Timeline and Bottlenecks were
 * merged into Analyze it would have been dropped, so it is preserved here: an
 * experiment is recorded *after* you have read the analysis, which is exactly
 * where Analyze sits.
 */
export function ExperimentRecorder({ sessions }: { sessions: { session_id: string }[] }) {
  const [exp, setExp] = useState<Exp>({
    name: "", kind: "quantisation", baseline: "", candidate: "", note: "" });
  const [saved, setSaved] = useState<string | null>(null);

  async function save() {
    if (!exp.name) { setSaved("name is required"); return; }
    try {
      await api.createExperiment(exp);
      setSaved("recorded");
      setExp({ name: "", kind: exp.kind, baseline: "", candidate: "", note: "" });
      setTimeout(() => setSaved(null), 2500);
    } catch (e) { setSaved(String(e)); }
  }

  return (
    <div className="card mt">
      <h3>Record an experiment</h3>
      <div className="faint" style={{ fontSize: 11, marginBottom: 8 }}>
        Baseline vs candidate. Recording stores the intent only — an improvement is
        never claimed without a comparable measurement between the two sessions.
      </div>
      <div className="grid cols-4">
        <label className="field"><span>name</span>
          <input value={exp.name} placeholder="fp8 KV cache"
            onChange={(e) => setExp({ ...exp, name: e.target.value })} /></label>
        <label className="field"><span>technique</span>
          <select value={exp.kind} onChange={(e) => setExp({ ...exp, kind: e.target.value })}>
            {["quantisation", "continuous batching", "prefix caching", "kv-cache",
              "speculative decoding", "kernel fusion", "graph capture", "offloading"]
              .map((k) => <option key={k}>{k}</option>)}
          </select></label>
        <label className="field"><span>baseline session</span>
          <input list="exp-sessions" value={exp.baseline} placeholder="baseline id"
            onChange={(e) => setExp({ ...exp, baseline: e.target.value })} /></label>
        <label className="field"><span>candidate session</span>
          <input list="exp-sessions" value={exp.candidate} placeholder="candidate id"
            onChange={(e) => setExp({ ...exp, candidate: e.target.value })} /></label>
      </div>
      <datalist id="exp-sessions">
        {sessions.map((s) => <option key={s.session_id} value={s.session_id} />)}
      </datalist>
      <div className="row mt" style={{ gap: 10 }}>
        <button className="btn primary" onClick={save}>record</button>
        {saved && <span className="faint">{saved}</span>}
      </div>
    </div>
  );
}
