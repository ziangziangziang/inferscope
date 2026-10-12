import { useEffect, useMemo, useState } from "react";
import { useApp } from "../state";
import { api, type SampleRow, type Session } from "../lib/api";
import { Notice, Prov, Tile } from "../components/common";
import { Info, Reveal } from "../components/Info";

/**
 * Landing view: is anything running, and what did the last run cost?
 *
 * Opening the app used to land on a grid of idle GPU readings, which reads as
 * "nothing happened". The explanations that used to sit under every number are
 * now behind info glyphs — the page should be scannable in one glance, with the
 * reasoning a hover away rather than read inline.
 */
export function Machine() {
  const { sessions, health, sessionId, setSessionId } = useApp();
  const [chat, setChat] = useState<{ ready: boolean; server: { running?: boolean; model?: string; uptime_s?: number | null } } | null>(null);
  // Telemetry belongs to the box, not to whichever session is selected. A
  // profiled session holds kernels but no samples, so reading it would make a
  // healthy idle GPU look broken.
  const [live, setLive] = useState<SampleRow | null>(null);

  useEffect(() => {
    const tick = () => api.chatState().then(setChat).catch(() => setChat(null));
    tick();
    const id = setInterval(tick, 8000);
    return () => clearInterval(id);
  }, []);

  const liveSid = health?.live_session;
  useEffect(() => {
    if (!liveSid) { setLive(null); return; }
    const tick = () => api.overview(liveSid).then((o) => setLive(o.latest)).catch(() => setLive(null));
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, [liveSid]);

  const best = useMemo<Session | null>(() => {
    const withKernels = sessions.filter((s) => s.kernel_events > 0);
    if (!withKernels.length) return null;
    return withKernels.reduce((a, b) => (b.kernel_events > a.kernel_events ? b : a));
  }, [sessions]);

  const current = useMemo(
    () => sessions.find((s) => s.session_id === sessionId), [sessions, sessionId]);

  const engineUp = !!chat?.ready;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0, overflowY: "auto" }}>
      <div className="view-head">
        <h1>Status</h1>
        <Info text="What is running right now, and what your most recent capture found. This is the landing page — start from here." />
      </div>

      <div className="row" style={{ gap: 10, flexWrap: "wrap", marginBottom: 12 }}>
        <span className={`badge ${engineUp ? "measured" : "unavailable"}`}>
          <span className="dot" />{engineUp ? "engine running" : "engine off"}
        </span>
        {engineUp && chat?.server?.model && <span className="pill">{chat.server.model}</span>}
        {engineUp && chat?.server?.uptime_s != null && (
          <span className="pill">{Math.round(chat.server.uptime_s)}s up</span>
        )}
        {engineUp ? (
          <Info text="A model server is loaded and answering requests. Telemetry below reflects real work." />
        ) : (
          <Info text="No model server is attached, so the GPU sits idle at 0%. That is expected — it is not a sign that measurement failed. Start one under Run & capture." />
        )}
      </div>

      <div className="card">
        <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
          <h3 style={{ margin: 0 }}>Last capture</h3>
          <Info text="The session holding the most measured kernel evidence. Selecting it makes Analyze show that capture." />
        </div>
        {best ? (
          <div className="row" style={{ gap: 8, flexWrap: "wrap", marginTop: 8 }}>
            <button className="pill" style={{ cursor: "pointer" }}
              onClick={() => setSessionId(best.session_id)}>
              {best.kernel_events.toLocaleString()} kernels ▸
            </button>
            <span className="pill">{Math.round(best.gpu_ms)} ms GPU time</span>
            {best.top_op && <span className="pill">{best.top_op}</span>}
            <Prov p="measured" />
            <span className="faint" style={{ fontSize: 10.5 }}>{best.session_id}</span>
          </div>
        ) : (
          <div className="faint" style={{ fontSize: 11.5, marginTop: 6 }}>
            None yet — send a prompt under <b>Run &amp; capture</b>.
          </div>
        )}
      </div>

      <div className="row" style={{ gap: 8, margin: "12px 0 6px" }}>
        <span className="faint" style={{ fontSize: 10.5 }}>live telemetry</span>
        <Info text={`Measured from the running box (${liveSid ?? "no live session"}), independent of which session you are analyzing.`} />
      </div>
      <div className="grid cols-4">
        <Tile k="GPU busy" v={`${(live?.gpu_util_pct ?? 0).toFixed(0)}%`} prov="measured" />
        <Tile k="VRAM" v={live?.vram_used_bytes != null ? vram(live.vram_used_bytes) : "unavailable"}
          prov={live?.vram_used_bytes != null ? "measured" : "unavailable"} />
        <Tile k="GPU temp" v={live?.temp_c != null ? `${live.temp_c.toFixed(0)} °C` : "unavailable"}
          prov={live?.temp_c != null ? "measured" : "unavailable"} />
        <Tile k="CPU busy" v={`${(live?.cpu_util_pct ?? 0).toFixed(0)}%`} prov="measured" />
      </div>

      {current && current.kernel_events === 0 && current.sample_count > 0 && (
        <div className="mt">
          <Notice kind="info">
            {current.session_id} has telemetry but no captured kernels.
            {best && <> Try <button className="link" onClick={() => setSessionId(best.session_id)}>{best.session_id}</button>.</>}
          </Notice>
        </div>
      )}

      <CollectorHealth />
    </div>
  );
}

/** Pick a unit that does not collapse a 27 MiB reading to "0.0". */
function vram(b: number): string {
  if (b >= 2 ** 30) return `${(b / 2 ** 30).toFixed(1)} GiB`;
  return `${(b / 2 ** 20).toFixed(0)} MiB`;
}

function CollectorHealth() {
  const { health } = useApp();
  const caps = useMemo(
    () => Object.entries(health?.capabilities ?? {}).map(([k, v]) => [k, v.available] as [string, boolean]),
    [health]);
  const sys = health?.collectors.system.available;
  const gpu = health?.collectors.gpu;
  const up = caps.filter(([, ok]) => ok).length;

  return (
    <div className="mt">
      <Reveal label={`Collectors · ${sys && gpu?.available ? "ok" : "check"} (${up}/${caps.length})`}>
        <table className="kv-table"><tbody>
          <tr><td className="k">system</td><td className="v">{sys ? "up" : "down"}</td></tr>
          <tr><td className="k">gpu</td>
            <td className="v">{gpu?.available ? `up (${gpu.backend})` : "down"}</td></tr>
          {caps.map(([k, ok]) => (
            <tr key={k}><td className="k">{k}</td>
              <td className="v" style={{ color: ok ? "var(--good)" : "var(--dim)" }}>{ok ? "up" : "down"}</td></tr>
          ))}
        </tbody></table>
        <div className="faint" style={{ fontSize: 10.5, marginTop: 6 }}>
          gfx1100: <code>rocm-smi --showuse</code> reports 0% GPU use on this
          platform; InferScope reads <code>amdsmi</code> instead.
        </div>
      </Reveal>
    </div>
  );
}