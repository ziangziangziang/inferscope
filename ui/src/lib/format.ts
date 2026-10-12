// Formatting + provenance presentation. Provenance colour/label is defined once
// here so every view renders it identically.
import type { Provenance } from "./api";

export const PROV_LABEL: Record<Provenance, string> = {
  measured: "measured",
  derived: "derived",
  estimated: "estimated",
  simulated: "simulated",
  unavailable: "unavailable",
};

export const PROV_HINT: Record<Provenance, string> = {
  measured: "read directly from hardware/engine telemetry",
  derived: "computed from measured inputs",
  estimated: "model/config arithmetic, not observed at runtime",
  simulated: "synthetic demo data — not a live reading",
  unavailable: "this platform or engine does not expose it",
};

export function asProv(p: string | undefined | null): Provenance {
  if (p === "measured" || p === "derived" || p === "estimated" || p === "simulated") return p;
  return "unavailable";
}

export function bytes(n: number | null | undefined, dp = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let v = n, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i === 0 ? 0 : dp)} ${u[i]}`;
}

export function us(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  if (n < 1000) return `${n.toFixed(0)} µs`;
  if (n < 1e6) return `${(n / 1000).toFixed(2)} ms`;
  return `${(n / 1e6).toFixed(2)} s`;
}

export function num(n: number | null | undefined, dp = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: dp });
}

export function pct(n: number | null | undefined, dp = 0): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return `${n.toFixed(dp)}%`;
}

export function clockUs(tsNs: number | null | undefined): string {
  if (!tsNs) return "—";
  const ms = tsNs / 1e6;
  const d = new Date(ms);
  return d.toLocaleTimeString(undefined, { hour12: false }) + "." +
    String(Math.floor(ms % 1000)).padStart(3, "0");
}

export const OP_COLOR: Record<string, string> = {
  gemm: "var(--op-gemm)",
  attn_full: "var(--op-attn-full)",
  attn_linear: "var(--op-attn-linear)",
  elementwise: "var(--op-elementwise)",
  other: "var(--op-other)",
};

// Parameter counts span <1 K (a bias vector) to >10 B (a full model), so the
// unit is chosen per value to keep node labels short and comparable.
export function params(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  if (n === 0) return "0";
  if (n < 1e3) return `${n}`;
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 1e4 ? 2 : 1)}K`;
  if (n < 1e9) return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
  return `${(n / 1e9).toFixed(2)}B`;
}

// Per-token compute. Below 1 MFLOP the number is noise, so it is shown as 0.
export function gflops(flops: number | null | undefined): string {
  if (flops === null || flops === undefined || !Number.isFinite(flops)) return "—";
  if (flops < 1e6) return "0";
  if (flops < 1e9) return `${(flops / 1e6).toFixed(1)} MF`;
  return `${(flops / 1e9).toFixed(2)} GF`;
}
