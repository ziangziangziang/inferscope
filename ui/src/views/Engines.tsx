/**
 * Engine load / offload.
 *
 * One GPU holds one heavy engine at a time, so this screen is a console rather
 * than a list of checkboxes: you pick which model you want resident, and the
 * app tells you what it had to evict to make room. That eviction is the thing
 * a user cannot guess from VRAM numbers alone, so it is stated, not implied.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { engines, engineLoad, engineOffload, type EngineStatus, type EngineInfo } from "../lib/api";
import { api } from "../lib/api";
import { useApp } from "../state";

const FAMILY_LABEL: Record<string, string> = {
  speech: "speech", vision: "vision", language: "language",
};

function uptime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

function EngineCard({ e, status, busy, onLoad, onOffload }: {
  e: EngineInfo; status: EngineStatus; busy: string | null;
  onLoad: (k: string) => void; onOffload: () => void;
}) {
  const resident = status.loaded === e.key;
  const state = resident && e.ready ? "resident" : e.ready ? "answering" : e.configured ? "ready" : "unavailable";
  return (
    <div className={`eng-card ${resident ? "on" : ""}`} data-engine={e.key}>
      <div className="eng-head">
        <span className="eng-name">{e.label || e.key}</span>
        <span className={`eng-state ${state}`}>{state}</span>
      </div>
      <p className="eng-purpose">{e.purpose}</p>
      <dl className="eng-kv">
        <dt>family</dt><dd>{FAMILY_LABEL[e.family] ?? e.family}</dd>
        <dt>model</dt><dd title={e.model}>{e.model ? shortPath(e.model) : "—"}</dd>
        <dt>binary</dt><dd title={e.binary}>{e.binary ? shortPath(e.binary) : "—"}</dd>
        <dt>port</dt><dd>{e.port}</dd>
        {resident ? (<><dt>pid</dt><dd>{status.pid}</dd></>) : null}
      </dl>
      {!e.configured ? (
        <p className="eng-problem">not available — {e.problem}</p>
      ) : null}
      <div className="eng-actions">
        {resident ? (
          <button className="btn" onClick={onOffload} disabled={busy === e.key}>
            {busy === e.key ? "offloading…" : "offload"}
          </button>
        ) : (
          <button className="btn primary" onClick={() => onLoad(e.key)}
            disabled={!e.configured || busy === e.key}>
            {busy === e.key ? "loading…" : "load"}
          </button>
        )}
      </div>
    </div>
  );
}

function shortPath(p: string): string {
  const parts = p.split("/");
  return parts.length <= 2 ? p : "…/" + parts.slice(-2).join("/");
}

export function Engines() {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const { health } = useApp();
  // VRAM lives on the live sample, not on /api/health (which only reports
  // whether a GPU collector exists).
  const [vram, setVram] = useState<{ used?: number | null; total?: number | null }>({});
  const sid = health?.live_session;
  useEffect(() => {
    if (!sid) return;
    const tick = () => api.overview(sid)
      .then((o) => setVram({
        used: o.latest?.vram_used_bytes ?? null,
        total: o.latest?.vram_total_bytes ?? null,
      }))
      .catch(() => {});
    tick();
    const id = setInterval(tick, 2000);
    return () => clearInterval(id);
  }, [sid]);
  const poll = useRef<number | null>(null);

  const refresh = useCallback(() => {
    engines().then(setStatus).catch((e) => setErr(String(e?.message ?? e)));
  }, []);

  useEffect(() => {
    refresh();
    poll.current = window.setInterval(refresh, 4000);
    return () => { if (poll.current) clearInterval(poll.current); };
  }, [refresh]);

  const load = async (key: string) => {
    setBusy(key); setErr(null); setNote(null);
    try {
      const r = await engineLoad(key);
      if (!r.ok) setErr(r.error || "could not load");
      else if (r.evicted) setNote(`${r.evicted} was offloaded to make room for ${key}.`);
      else if (r.already) setNote(`${key} was already loaded.`);
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally { setBusy(null); refresh(); }
  };

  const offload = async () => {
    setBusy("__off"); setErr(null); setNote(null);
    try {
      const r = await engineOffload();
      if (!r.stopped) setNote(r.detail || "nothing was loaded");
      else setNote(`${r.engine} offloaded.`);
    } catch (e) { setErr(String((e as Error)?.message ?? e)); }
    finally { setBusy(null); refresh(); }
  };

  return (
    <div className="view">
      <div className="view-head">
        <h1>Engines</h1>
        <p>
          This machine has one GPU, so only one heavy engine can be resident at a
          time. Load the one you want; whatever was holding the card is offloaded
          first, and we say so rather than leaving you to work it out from the
          memory numbers.
        </p>
      </div>

      <div className="eng-resident">
        <div className="eng-resident-l">
          <b>resident</b>
          <span className={status?.loaded ? "yes" : "no"}>
            {status?.loaded ? `${label(status.engines, status.loaded)}` : "none — the GPU is free"}
          </span>
          {status?.loaded ? <span className="faint">pid {status.pid} · up {uptime(status.uptime_s)}</span> : null}
        </div>
        <div className="eng-resident-r">
          {status?.loaded ? (
            <button className="btn" onClick={offload} disabled={busy === "__off"}>
              {busy === "__off" ? "offloading…" : "offload everything"}
            </button>
          ) : null}
        </div>
      </div>

      {vram.total ? (
        <div className="eng-vram">
          <span>VRAM in use</span>
          <div className="eng-bar">
            <div style={{ width: `${Math.min(100, ((vram.used ?? 0) / vram.total) * 100)}%` }} />
          </div>
          <span className="mono">
            {((vram.used ?? 0) / 2 ** 30).toFixed(1)} / {(vram.total / 2 ** 30).toFixed(1)} GiB
          </span>
          <span className="faint">measured</span>
        </div>
      ) : null}

      {err ? <div className="notice bad">{err}</div> : null}
      {note ? <div className="notice">{note}</div> : null}

      <div className="eng-grid">
        {(status?.engines ?? []).map((e) => (
          <EngineCard key={e.key} e={e} status={status!} busy={busy}
            onLoad={load} onOffload={offload} />
        ))}
      </div>
    </div>
  );
}

function label(engines: EngineInfo[], key: string): string {
  const e = engines.find((x) => x.key === key);
  return e ? `${e.label} · ${key}` : key;
}
