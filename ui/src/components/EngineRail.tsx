/**
 * The engine rail.
 *
 * One GPU holds one heavy engine, so this is a rail rather than a list: the
 * resident one is obvious, and loading another says which one it displaced.
 * That eviction is the part a user cannot infer from a VRAM number, so it is
 * stated rather than left to be discovered when something stops working.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  engines, engineLoad, engineOffload, api, type EngineStatus, type EngineInfo,
} from "../lib/api";
import { useApp } from "../state";

export function useEngineStatus(pollMs = 4000) {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const refresh = useCallback(() => {
    engines().then(setStatus).catch(() => {});
  }, []);
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, pollMs);
    return () => clearInterval(id);
  }, [refresh, pollMs]);
  return { status, refresh };
}

const FAMILY_ICON: Record<string, string> = {
  speech: "◑", vision: "◫", language: "✦",
};

function uptime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

export function EngineRail({ status, onChanged }:
  { status: EngineStatus | null; onChanged: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const { health } = useApp();
  const [vram, setVram] = useState<{ used?: number | null; total?: number | null }>({});
  const sid = health?.live_session;

  useEffect(() => {
    if (!sid) return;
    const tick = () => api.overview(sid)
      .then((o) => setVram({ used: o.latest?.vram_used_bytes ?? null, total: o.latest?.vram_total_bytes ?? null }))
      .catch(() => {});
    tick();
    const id = setInterval(tick, 2500);
    return () => clearInterval(id);
  }, [sid]);

  const load = async (key: string) => {
    setBusy(key); setErr(null); setNote(null);
    try {
      const r = await engineLoad(key);
      if (!r.ok) setErr(r.error || "could not load");
      else if (r.evicted) setNote(`${r.evicted} was offloaded to make room for ${key}.`);
      else if (r.already) setNote(`${key} was already loaded.`);
    } catch (e) { setErr(String((e as Error)?.message ?? e)); }
    finally { setBusy(null); onChanged(); }
  };

  const offload = async (key: string) => {
    setBusy(key); setErr(null); setNote(null);
    try {
      const r = await engineOffload();
      setNote(r.stopped ? `${r.engine} offloaded — the GPU is free.` : (r.detail || "nothing was loaded"));
    } catch (e) { setErr(String((e as Error)?.message ?? e)); }
    finally { setBusy(null); onChanged(); }
  };

  const engines_ = status?.engines ?? [];

  return (
    <section className="rail" data-testid="engine-rail">
      <div className="rail-head">
        <h3>Engines</h3>
        <span className="faint">
          one GPU — loading an engine offloads whatever held it
        </span>
        <div className="rail-vram">
          {vram.total ? (
            <>
              <div className="eng-bar">
                <div style={{ width: `${Math.min(100, ((vram.used ?? 0) / vram.total) * 100)}%` }} />
              </div>
              <span className="mono">
                {((vram.used ?? 0) / 2 ** 30).toFixed(1)} / {(vram.total / 2 ** 30).toFixed(1)} GiB
              </span>
            </>
          ) : null}
        </div>
      </div>

      <div className="rail-chips">
        {engines_.map((e: EngineInfo) => {
          const resident = status?.loaded === e.key;
          const st = resident ? "resident" : e.configured ? "ready" : "unavailable";
          return (
            <div key={e.key} className={`rail-chip ${resident ? "on" : ""}`} data-engine={e.key}>
              <span className="rail-ico" aria-hidden>{FAMILY_ICON[e.family] ?? "◧"}</span>
              <div className="rail-meta">
                <b>{e.label || e.key}</b>
                <span className="rail-state" data-state={st}>
                  {resident ? `resident · up ${uptime(status!.uptime_s)}` : e.configured ? "not loaded" : e.problem}
                </span>
              </div>
              {resident ? (
                <button className="btn sm" onClick={() => offload(e.key)} disabled={busy === e.key}>
                  {busy === e.key ? "…" : "offload"}
                </button>
              ) : (
                <button className="btn sm primary" onClick={() => load(e.key)}
                  disabled={!e.configured || busy === e.key}
                  title={e.configured ? `load ${e.label} (offloads the current one)` : e.problem}>
                  {busy === e.key ? "…" : "load"}
                </button>
              )}
            </div>
          );
        })}
      </div>

      {err ? <div className="notice bad">{err}</div> : null}
      {note ? <div className="notice">{note}</div> : null}
    </section>
  );
}