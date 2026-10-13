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
  engines, engineOffload, api, type EngineStatus,
} from "../lib/api";
import type { Unit } from "./unitModel";
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

/**
 * Resident status: what is on the GPU, how much of it, and a way to let go.
 *
 * Loading moved to the unit picker above, which can also choose *which* model.
 * Keeping a second set of load buttons here meant two controls for one action,
 * so this is status and release only.
 */
export function ResidentStrip({ status, unit, llmRunning, onChanged, onOffloaded }:
  { status: EngineStatus | null; unit: Unit | null; llmRunning: boolean;
    onChanged: () => void; onOffloaded?: () => void }) {
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

  const offload = async (key: string) => {
    setBusy(key); setErr(null); setNote(null);
    try {
      const r = await engineOffload();
      setNote(r.stopped ? `${r.engine} offloaded — the GPU is free.` : (r.detail || "nothing was loaded"));
      // Nothing is on the GPU any more, so the page must stop claiming a unit
      // is chosen — otherwise the scene keeps drawing a stack for an engine
      // that has been switched off.
      onOffloaded?.();
    } catch (e) { setErr(String((e as Error)?.message ?? e)); }
    finally { setBusy(null); onChanged(); }
  };

  const spec = (status?.engines ?? []).find((e) => e.key === status?.loaded);

  return (
    <section className="rail" data-testid="resident-strip">
      <div className="rail-head">
        <h3>On the GPU</h3>
        {status?.loaded ? (
          <>
            <b className="resident-name">{unit?.model_label ?? status.loaded}</b>
            <span className="faint">{spec?.label ?? status.loaded} · pid {status.pid} · up {uptime(status.uptime_s)}</span>
          </>
        ) : (
          <span className="faint">nothing resident — the card is free</span>
        )}
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
          {status?.loaded ? (
            <button className="btn sm" onClick={() => offload(status.loaded!)} disabled={busy === status.loaded}>
              {busy === status.loaded ? "…" : "offload"}
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}

