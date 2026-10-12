import { useEffect, useMemo, useState } from "react";
import { useApp } from "../state";
import { api } from "../lib/api";
import type { KVResult } from "../lib/api";
import { Bar, Prov } from "../components/common";
import { bytes, num, pct } from "../lib/format";

const DTYPES = ["bfloat16", "float16", "float32", "int8", "fp8", "q4_0"];

// Rough transformer weight count from config — clearly an estimate.
function estParams(m: Record<string, unknown>): number | null {
  const L = Number(m.n_layers) || 0, H = Number(m.hidden) || 0;
  const I = Number(m.intermediate) || 0, V = Number(m.vocab) || 0;
  const nh = Number(m.n_heads) || 0, kvh = Number(m.kv_heads) || nh, hd = Number(m.head_dim) || (H / (nh || 1));
  if (!L || !H) return null;
  const attn = H * (nh * hd) + 2 * H * (kvh * hd) + (nh * hd) * H;
  const mlp = 3 * H * I;
  return L * (attn + mlp + 2 * H) + 2 * V * H;
}

export function Memory() {
  const { sessionId, live, isDemo } = useApp();
  const [arch, setArch] = useState<Record<string, unknown>>({});
  const [form, setForm] = useState({
    n_layers: 64, context: 4096, batch: 8, kv_heads: 4, head_dim: 256,
    dtype: "bfloat16", sliding_window: 0, swa_fraction: 1.0, shared_prefix: 0,
  });
  const [kv, setKV] = useState<KVResult | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    api.sessionArch(sessionId).then((g) => {
      const meta = (g.meta || {}) as Record<string, unknown>;
      setArch(meta);
      if (meta.n_layers) {
        setForm((f) => ({
          ...f,
          n_layers: Number(meta.n_layers) || f.n_layers,
          kv_heads: Number(meta.kv_heads) || f.kv_heads,
          head_dim: Number(meta.head_dim) || f.head_dim,
        }));
      }
    }).catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    const p: Record<string, string | number> = { ...form };
    if (!form.sliding_window) delete (p as any).sliding_window;
    if (!form.shared_prefix) delete (p as any).shared_prefix;
    api.kv(p).then(setKV).catch(() => {});
  }, [form]);

  const vramUsed = live?.vram_used_bytes ?? null;
  const vramTotal = live?.vram_total_bytes ?? null;
  const params = useMemo(() => estParams(arch), [arch]);
  const weightBytes = params ? params * 2 : null; // bf16 assumption, estimated
  const headroom = vramUsed != null && vramTotal ? vramTotal - vramUsed : null;

  const set = (k: string, v: number | string) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <div>
      <div className="view-head"><h1>Memory & KV cache</h1>
        <span className="desc">measured occupancy vs model-derived estimates — kept visibly separate</span></div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Measured VRAM <Prov p={vramUsed != null ? "measured" : "unavailable"} /></h3>
          {vramUsed != null ? (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="mono" style={{ fontSize: 22 }}>{bytes(vramUsed)}</span>
                <span className="faint">of {bytes(vramTotal, 0)}</span>
              </div>
              <div className="mt"><Bar frac={vramTotal ? vramUsed / vramTotal : null} color="var(--accent2)" /></div>
              <table className="kv-table mt"><tbody>
                <tr><td className="k">occupancy</td><td className="v">{pct(live?.mem_used_pct)}</td></tr>
                <tr><td className="k">headroom</td><td className="v">{headroom != null ? bytes(headroom) : "—"}</td></tr>
                <tr><td className="k">process RSS</td><td className="v">{bytes(live?.proc_rss_bytes)}</td></tr>
              </tbody></table>
              <div className="faint mt" style={{ fontSize: 11 }}>
                Headroom is against reported capacity, not nominal VRAM — the engine and driver reserve a slice.
              </div>
            </>
          ) : <div className="faint">No live VRAM reading. Open a live session (or connect an engine) to measure.</div>}
        </div>

        <div className="card">
          <h3>Estimated breakdown <Prov p="estimated" /></h3>
          <table className="kv-table"><tbody>
            <tr><td className="k">weights ({params ? num(params / 1e9, 1) + "B params" : "config n/a"})</td>
              <td className="v">{weightBytes != null ? bytes(weightBytes, 0) : "—"}</td></tr>
            <tr><td className="k">KV cache @ ctx {num(form.context, 0)}×b{form.batch}</td>
              <td className="v">{kv ? bytes(kv.bytes, 2) : "—"}</td></tr>
            <tr><td className="k">weights + KV</td>
              <td className="v">{weightBytes != null && kv ? bytes(weightBytes + kv.bytes, 0) : "—"}</td></tr>
            <tr><td className="k">measured total</td><td className="v">{vramUsed != null ? bytes(vramUsed, 0) : "—"}</td></tr>
            <tr><td className="k">unattributed (measured − known)</td>
              <td className="v">{vramUsed != null && weightBytes != null && kv ? bytes(Math.max(0, vramUsed - weightBytes - kv.bytes), 0) : "—"}</td></tr>
          </tbody></table>
          <div className="faint mt" style={{ fontSize: 11 }}>
            Unattributed = allocator overhead, activations, CUDA/HIP graphs, engine buffers. We show the gap
            rather than pretending the estimate is the whole truth.
          </div>
        </div>
      </div>

      <div className="card mt">
        <h3>KV-cache calculator <Prov p="estimated" /></h3>
        <div className="grid cols-4">
          <label className="field"><span>layers</span><input type="number" value={form.n_layers} onChange={(e) => set("n_layers", +e.target.value)} /></label>
          <label className="field"><span>context</span><input type="number" value={form.context} onChange={(e) => set("context", +e.target.value)} /></label>
          <label className="field"><span>batch</span><input type="number" value={form.batch} onChange={(e) => set("batch", +e.target.value)} /></label>
          <label className="field"><span>KV heads</span><input type="number" value={form.kv_heads} onChange={(e) => set("kv_heads", +e.target.value)} /></label>
          <label className="field"><span>head dim</span><input type="number" value={form.head_dim} onChange={(e) => set("head_dim", +e.target.value)} /></label>
          <label className="field"><span>cache dtype</span>
            <select value={form.dtype} onChange={(e) => set("dtype", e.target.value)}>{DTYPES.map((d) => <option key={d}>{d}</option>)}</select></label>
          <label className="field"><span>sliding window (0=off)</span><input type="number" value={form.sliding_window} onChange={(e) => set("sliding_window", +e.target.value)} /></label>
          <label className="field"><span>SWA layer fraction</span><input type="number" step="0.1" min="0" max="1" value={form.swa_fraction} onChange={(e) => set("swa_fraction", +e.target.value)} /></label>
        </div>
        {kv && (
          <div className="mt">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="mono" style={{ fontSize: 20 }}>{num(kv.gib, 2)} GiB</span>
              <span className="faint mono">{kv.formula}</span>
            </div>
            <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 11.5 }}>
              {kv.notes.map((n, i) => <li key={i} className="dim">{n}</li>)}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
