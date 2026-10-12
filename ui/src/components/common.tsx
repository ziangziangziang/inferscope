// Small shared presentational primitives. Provenance badges are rendered here so
// every view labels a value's origin identically.
import type { ReactNode } from "react";
import { asProv, PROV_HINT, PROV_LABEL } from "../lib/format";

export function Prov({ p, hint }: { p?: string | null; hint?: string }) {
  const prov = asProv(p);
  return (
    <span className={`badge ${prov}`} title={hint || PROV_HINT[prov]}>
      <span className="dot" />{PROV_LABEL[prov]}
    </span>
  );
}

export function Tile({ k, v, u, prov, warn }: {
  k: string; v: ReactNode; u?: string; prov?: string | null; warn?: boolean;
}) {
  return (
    <div className="card tile">
      <div className="k">{k}</div>
      <div className="v" style={warn ? { color: "var(--warn)" } : undefined}>{v}
        {u && <span className="u"> {u}</span>}
      </div>
      {prov && <div style={{ marginTop: 6 }}><Prov p={prov} /></div>}
    </div>
  );
}

export function Sparkline({ values, color = "var(--accent)", w = 160, h = 34 }: {
  values: (number | null)[]; color?: string; w?: number; h?: number;
}) {
  const vals = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (vals.length < 2) {
    return <div className="faint" style={{ fontSize: 10, height: h, display: "flex", alignItems: "center" }}>insufficient samples</div>;
  }
  const min = Math.min(...vals), max = Math.max(...vals);
  const span = max - min || 1;
  const step = w / (vals.length - 1);
  const pts = vals.map((v, i) => `${(i * step).toFixed(1)},${(h - 2 - ((v - min) / span) * (h - 4)).toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.4} />
    </svg>
  );
}

export function Bar({ frac, color = "var(--accent)" }: { frac: number | null; color?: string }) {
  if (frac === null || !Number.isFinite(frac)) return <div className="faint" style={{ fontSize: 10 }}>—</div>;
  return <div className="bar"><i style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%`, background: color }} /></div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Notice({ kind = "info", children }: { kind?: "info" | "warn" | "bad"; children: ReactNode }) {
  const cls = kind === "warn" ? "notice warn" : kind === "bad" ? "notice bad" : "notice";
  return <div className={cls}>{children}</div>;
}
