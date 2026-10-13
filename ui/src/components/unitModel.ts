/**
 * The unit model: what a loaded block in the scene actually *is*.
 *
 * The scene used to draw a hardcoded llama.cpp slab with a hardcoded model on
 * top, so nothing could be swapped and nothing was loaded by default. Now a
 * block is a `Unit` — one engine paired with one checkpoint — and the layout
 * below turns that into geometry and a load/evict choreography.
 *
 * Geometry rules, unchanged from when the tower was two fixed slabs:
 *
 *   * the engine block is the size of the host plate and drawn as an outline,
 *     because software has no size of its own — it borrows the hardware;
 *   * the model sits directly on it with the same footprint, subdivided into one
 *     block per component when the model has parts, each block's width being
 *     its real share of the weights.
 *
 * So a single-file model is one solid block, and a diffusion model is several
 * side by side whose relative widths are the honest ones.
 */

import { useEffect, useRef, useState } from "react";

export interface UnitPart { key: string; label: string; weight: number }

export interface Unit {
  key: string; engine: string; engine_label: string; model_label: string;
  model_path: string; port: number; binary: string;
  size_bytes: number; params: number;
  parts: UnitPart[]; note: string; problem: string;
  /** one phrase describing what this engine does, for the block caption */
  engine_role?: string;
  configured: boolean; managed: boolean;
}

export interface UnitCatalogue { units: Unit[]; resident: string | null }

/** What one block on the stack is. */
export interface StackBlock {
  key: string;
  label: string;
  sub: string;
  /** engine blocks are outlines; weights are solid */
  ghost: boolean;
  /** width as a fraction of the plate footprint; sums to 1 within a tier */
  frac: number;
  /** x offset as a fraction of the plate footprint */
  offset: number;
  /** which engine this block belongs to, for grouping and colour */
  engine: string;
  engine_label: string;
  model_label: string;
  /** stagger order for the load animation */
  order: number;
}

export function fmtBytes(n: number): string {
  if (!n) return "—";
  if (n < 1024) return `${n} B`;
  // u[0] is KiB, so the exponent index is one less than the power of 1024 —
  // without the -1 every size was one unit too large, and a 141 MiB model
  // reported as "0.1 MiB".
  const u = [1024, 2 ** 20, 2 ** 30, 2 ** 40];
  const i = Math.min(u.length - 1, Math.max(0, Math.floor(Math.log(n) / Math.log(1024)) - 1));
  const v = n / u[i];
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${["KiB", "MiB", "GiB", "TiB"][i]}`;
}

export function fmtParams(p: number): string {
  if (!p) return "";
  if (p >= 1e9) return `${(p / 1e9).toFixed(2)}B parameters`;
  if (p >= 1e6) return `${(p / 1e6).toFixed(1)}M parameters`;
  if (p >= 1e3) return `${(p / 1e3).toFixed(1)}K parameters`;
  return `${p} parameters`;
}

/**
 * Lay a unit out as a stack of blocks.
 *
 * Returns the engine tier first (always one plate-sized outline), then the
 * model tier, whose blocks are the model's parts — or a single block when it
 * has none. Nothing here knows about llama.cpp, whisper or diffusers.
 */
export function layoutUnit(u: Unit | null): StackBlock[] {
  if (!u) return [];
  const out: StackBlock[] = [{
    key: `${u.key}#engine`,
    label: u.engine_label,
    // trim the sentence stop only: splitting on commas chopped
    // "reads words, predicts the next one" down to "reads words"
    sub: (u.engine_role || "engine").replace(/[.\s]+$/, ""),
    ghost: true,
    frac: 1, offset: 0,
    engine: u.engine, engine_label: u.engine_label,
    model_label: u.model_label,
    order: 0,
  }];

  const parts = (u.parts ?? []).filter((p) => p && p.weight > 0);
  if (!parts.length) {
    out.push({
      key: `${u.key}#model`,
      label: u.model_label,
      sub: u.params ? fmtParams(u.params) : fmtBytes(u.size_bytes),
      ghost: false, frac: 1, offset: 0,
      engine: u.engine, engine_label: u.engine_label,
      model_label: u.model_label,
      order: 1,
    });
    return out;
  }

  const total = parts.reduce((a, p) => a + p.weight, 0) || 1;
  // A sliver is unreadable; give every block a floor and take the difference
  // back from the largest, so widths stay proportional in aggregate.
  const MIN = 0.16;
  const raw = parts.map((p) => Math.max(MIN, p.weight / total));
  const scale = 1 / raw.reduce((a, w) => a + w, 0);
  let x = 0;
  parts.forEach((p, i) => {
    const frac = raw[i] * scale;
    out.push({
      key: `${u.key}#${p.key}`,
      label: p.label,
      sub: "",
      ghost: false, frac, offset: x,
      engine: u.engine, engine_label: u.engine_label,
      model_label: u.model_label,
      order: 1 + i,
    });
    x += frac;
  });
  return out;
}

/* ---- animation ---- */

export type LoadPhase = "empty" | "loading" | "loaded" | "evicting";

/** ms; the whole swap is roughly this long end to end */
export const EVICT_MS = 380;
export const LOAD_MS = 520;
export const STAGGER_MS = 90;

export interface StackAnimation {
  phase: LoadPhase;
  /** per-block animation-delay, ms */
  delay: (order: number) => number;
  /** per-block animation class */
  cls: (order: number) => string;
}

/**
 * Drive the load/evict/swap choreography from the resident unit.
 *
 * A swap does not cross-fade two towers: it sinks the old one into the plate,
 * and only once it is gone does the new one rise out of it. Overlapping them
 * looked like two models briefly sharing the GPU, which is exactly the state
 * this machine can never actually be in.
 */
export function useStackAnimation(unit: Unit | null): StackAnimation {
  const key = unit?.key ?? null;
  const [phase, setPhase] = useStackPhase(key);
  const blocks = layoutUnit(unit);

  return {
    phase,
    delay: (order: number) =>
      phase === "evicting" ? (blocks.length - 1 - order) * STAGGER_MS
        : order * STAGGER_MS + (phase === "loading" ? 120 : 0),
    cls: (order: number) => {
      if (phase === "evicting") return "blk-evict";
      return phase === "loading" ? "blk-load" : "blk-settled";
    },
  };
}

function useStackPhase(key: string | null): [LoadPhase, (p: LoadPhase) => void] {
  const [phase, setPhase] = useState<LoadPhase>(key ? "loading" : "empty");
  const prev = useRef<string | null>(key);
  useEffect(() => {
    if (prev.current === key) return;
    prev.current = key;
    if (!key) { setPhase("empty"); return; }
    setPhase("loading");
    const t = window.setTimeout(() => setPhase("loaded"), LOAD_MS);
    return () => clearTimeout(t);
  }, [key]);
  return [phase, setPhase];
}

/** Imperative hook for a user-initiated load: evict the old stack first. */
export function useSwap() {
  const [phase, setPhase] = useState<LoadPhase>("loaded");
  const timers = useRef<number[]>([]);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);
  return { phase, setPhase, timers };
}