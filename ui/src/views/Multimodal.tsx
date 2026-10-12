import { useMemo, useState } from "react";
import { useApp } from "../state";
import { useArch } from "../lib/useArch";
import { Graph as GraphCanvas } from "../components/Graph";
import { Empty, Notice, Prov } from "../components/common";
import { Info, Reveal } from "../components/Info";
import { num } from "../lib/format";

// Vision-language: how an image's pixel budget becomes visual tokens.
function visualTokens(w: number, h: number, patch: number, merge: number, frames = 1) {
  const pw = Math.floor(w / patch), ph = Math.floor(h / patch);
  const raw = pw * ph * frames;
  return { raw, merged: Math.floor(raw / (merge * merge)), pw, ph };
}

export function Multimodal() {
  const { sessionId, select, sel } = useApp();
  const { graph, loading } = useArch(sessionId, "");
  const [res, setRes] = useState({ w: 1024, h: 1024, patch: 16, merge: 2, frames: 1 });

  const vt = useMemo(() => visualTokens(res.w, res.h, res.patch, res.merge, res.frames), [res]);
  const family = graph?.family ?? "text";

  if (!graph) return <div><div className="view-head"><h1>Multimodal lab</h1></div>
    <Empty>{loading ? "loading…" : "no model configs found on this host"}</Empty></div>;

  return (
    <div>
      <div className="view-head"><h1>Multimodal</h1>
        <Info text={`Model family: ${family}. These tabs show how an image or video is turned into tokens the language model can read, and how the pipeline repeats.`} />
      </div>

      {family === "text" && <Notice>Model family is <b>text</b>. Load a vision-language or generation checkpoint (Architecture ▸ model selector) to see the multimodal views.</Notice>}

      <div className="grid cols-2">
        <div className="card" style={{ minHeight: 380 }}>
          <h3>Pipeline graph <Prov p={graph.provenance} /></h3>
          <div style={{ height: 340 }}>
            <GraphCanvas nodes={graph.nodes} edges={graph.edges}
              selectedId={sel.node?.id} onSelectNode={(n) => select({ node: n })}
              onSelectEdge={(e) => select({ edge: e })}
              onExpand={(n) => select({ node: n })} />
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {(family === "vlm") && (
            <div className="card">
              <h3>Image → visual tokens <Prov p="estimated" /></h3>
              <p className="faint" style={{ fontSize: 11.5 }}>
                A frame is split into <code>{res.patch}×{res.patch}</code> patches; the merger collapses each
                <code> {res.merge}×{res.merge}</code> neighbourhood into one LM token. Token count grows with the
                <b> square</b> of resolution — the main lever on VLM prefill cost.
              </p>
              <div className="grid cols-2">
                <label className="field"><span>width px</span><input type="number" value={res.w} onChange={(e) => setRes({ ...res, w: +e.target.value })} /></label>
                <label className="field"><span>height px</span><input type="number" value={res.h} onChange={(e) => setRes({ ...res, h: +e.target.value })} /></label>
                <label className="field"><span>frames</span><input type="number" value={res.frames} onChange={(e) => setRes({ ...res, frames: +e.target.value })} /></label>
              </div>
              <table className="kv-table"><tbody>
                <tr><td className="k">patch grid</td><td className="v">{vt.pw} × {vt.ph}</td></tr>
                <tr><td className="k">raw patches</td><td className="v">{num(vt.raw, 0)}</td></tr>
                <tr><td className="k">LM visual tokens</td><td className="v">{num(vt.merged, 0)}</td></tr>
              </tbody></table>
            </div>
          )}

          {(family === "image" || family === "video") && (
            <div className="card">
              <h3>Denoising loop <Prov p="estimated" /></h3>
              {graph.nodes.filter((n) => n.repeat).map((n) => (
                <div key={n.id} className="row" style={{ justifyContent: "space-between", padding: "4px 0" }}>
                  <span>{n.label}</span>
                  <span className="pill">repeat × {n.repeat!.count}</span>
                </div>
              ))}
              <div className="faint mt" style={{ fontSize: 11 }}>
                Per-step latency and peak memory are only shown once a Diffusers instrumented run
                (see the diffusers adapter) records <code>denoise_step</code> events into a session.
                Architecture alone cannot measure them.
              </div>
            </div>
          )}

          <div className="card">
            <Reveal label="How to record an image/video generation">
              <p className="faint" style={{ fontSize: 11.5 }}>
                Wrap the pipeline rather than patching it globally:
                <code> from inferscope.diffusers import instrument </code> then wrap a pipeline to emit
                pipeline / component / denoise-step events and memory readings.
                See <code>docs/</code> and <code>inferscope/diffusers.py</code>.
              </p>
              <div className="row"><Prov p="measured" /><span className="faint" style={{ fontSize: 11 }}>records real timings when you run a generation</span></div>
            </Reveal>
          </div>
        </div>
      </div>
    </div>
  );
}
