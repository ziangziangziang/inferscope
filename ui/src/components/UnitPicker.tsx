/**
 * Choose which engine and model should be on the GPU.
 *
 * The scene's blocks are a unit, so this is the control that changes them. A
 * swap goes through *evict -> load* rather than a cross-fade: two towers briefly
 * sharing the GPU would be a state this machine can never actually be in.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  units, engineLoad, engineOffload,
  type UnitCatalogue, type UnitInfo, type LoadResult,
} from "../lib/api";
import { fmtBytes, fmtParams, EVICT_MS, LOAD_MS } from "./unitModel";

export type Swap = { phase: "idle" | "evicting" | "loading"; key: string | null };

/** Drives evict-then-load for a user-initiated swap. */
export function useSwapEngine() {
  const [swap, setSwap] = useState<Swap>({ phase: "idle", key: null });
  const timers = useRef<number[]>([]);
  const clear = () => { timers.current.forEach(clearTimeout); timers.current = []; };
  useEffect(() => clear, []);

  const swapTo = useCallback(async (u: UnitInfo) => {
    clear();
    const current = await currentEngine();
    if (current && current !== u.engine) {
      setSwap({ phase: "evicting", key: current });
      timers.current.push(window.setTimeout(() => void engineOffload(), 0));
      timers.current.push(window.setTimeout(() => setSwap({ phase: "loading", key: u.engine }), EVICT_MS));
      timers.current.push(window.setTimeout(() => {
        setSwap({ phase: "idle", key: u.engine });
      }, EVICT_MS + LOAD_MS));
    }
    // The language engine is not in the engine registry — chat.py owns that
    // process — so asking enginectl to load it just returns "unknown engine".
    const r: LoadResult = u.managed
      ? await engineLoad(u.engine)
      : { ok: true, error: null, evicted: null };
    if (!r.ok) {
      setSwap({ phase: "idle", key: null });
      throw new Error(r.error || "could not load");
    }
    if (r.evicted) {
      setSwap((p) => ({ ...p, phase: "evicting", key: null }));
      timers.current.push(window.setTimeout(
        () => setSwap({ phase: "loading", key: u.engine }), EVICT_MS));
    }
  }, []);

  return { swap, swapTo, reset: () => setSwap({ phase: "idle", key: null }) };
}

async function currentEngine(): Promise<string | null> {
  try {
    const c = await units();
    return c.resident;
  } catch { return null; }
}

export function UnitPicker({ catalogue, resident, onPick, swap, disabled }:
  { catalogue: UnitInfo[]; resident: string | null; onPick: (u: UnitInfo) => void;
    swap: Swap; disabled?: boolean }) {
  const byEngine = useMemo(() => {
    const m = new Map<string, UnitInfo[]>();
    for (const u of catalogue) {
      if (!m.has(u.engine)) m.set(u.engine, []);
      m.get(u.engine)!.push(u);
    }
    return m;
  }, [catalogue]);

  return (
    <div className="picker" data-testid="unit-picker">
      {[...byEngine.entries()].map(([engine, us]) => {
        return (
          <div key={engine} className="picker-grp">
            <div className="picker-grp-h">
              <b>{us[0]?.engine_label ?? engine}</b>
              {us.length > 1 ? <span className="faint">{us.length} models</span> : null}
            </div>
            <div className="picker-models">
              {us.map((u) => {
                const on = resident === u.engine;
                return (
                  <button key={u.key}
                    className={`model-chip ${on ? "on" : ""} ${u.configured ? "" : "off"}`}
                    data-unit={u.key}
                    disabled={disabled || !u.configured}
                    title={u.configured ? u.model_path : u.problem}
                    onClick={() => onPick(u)}>
                    <b>{u.model_label}</b>
                    <span className="faint">
                      {u.params ? fmtParams(u.params)
                        : u.size_bytes ? fmtBytes(u.size_bytes) : u.problem || "—"}
                    </span>
                    {u.managed ? null : <span className="faint tag">chat</span>}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
