/**
 * The inference scene.
 *
 * One isometric camera for everything, built from the primitives in `iso.tsx`.
 * Reading the diagram from the bottom up:
 *
 *   HOST plate ─ four hardware cubes (CPU / GPU / RAM / VRAM), fill = how busy
 *        │
 *        ├─ engine block   (llama.cpp)  ─┐ plate-sized, hanging clear above,
 *        │                                │  dashed because it is software
 *        └─ model block    (weights)     ─┘ the model block is the *same size*
 *                                             as the engine block: same
 *                                             footprint, same height. They are
 *                                             deliberately not merged, because
 *                                             the engine and the weights are
 *                                             different things that happen to
 *                                             live in the same place.
 *
 * Clicking the model unfolds the architecture above it: one box per component,
 * sized by that component's real parameter count, with repetitive units wrapped
 * into near-square grids so 64 transformer blocks stay readable.
 *
 * The tower hangs close enough to overlap the hardware cubes — that overlap is
 * the point, it says "this is running on this machine" — and lifting on hover
 * (macOS-dock style) pulls it clear when you want to see what is underneath.
 */

import { useEffect, useRef, useState } from "react";
import { Box, Ground, Bounds, iso, depth } from "./iso";
import { fmtParams } from "./sceneModel";
import type { ArchGraph, ArchNode, Sel } from "./archTypes";

const COS30 = Math.cos(Math.PI / 6);

/* ---- hardware geometry (unchanged from the accepted design) ---- */
const C = 56;
const G = 36;
const S = C + G;
const M = 34;
const PX0 = -M, PY0 = -M, PX1 = S + C + M, PY1 = S + C + M;
const PW = PX1 - PX0, PD = PY1 - PY0;

/* ---- the hanging engine -> model tower ---- */
const TOWER_Z = C + 56;          // base of the engine block
const TOWER_H = 58;              // both blocks: identical dimension
const LIFT = 86;                 // hover rise, in world units of z

/* ---- architecture tier ---- */

/** how far the tower rises on hover, in *screen* px (1 world z unit = 1 px) */
export const TOWER_LIFT = LIFT;

type Art = { cells: { u: number; v: number; w: number; h: number }[]; stripe?: { v: number; h: number } };

const ART: Record<string, Art> = {
  cpu: {
    cells: [
      { u: 0.14, v: 0.16, w: 0.32, h: 0.30 },
      { u: 0.54, v: 0.16, w: 0.32, h: 0.30 },
      { u: 0.14, v: 0.56, w: 0.32, h: 0.26 },
      { u: 0.54, v: 0.56, w: 0.32, h: 0.26 },
    ],
    stripe: { v: 0.86, h: 0.08 },
  },
  gpu: {
    cells: Array.from({ length: 24 }, (_, i) => ({
      u: 0.07 + (i % 8) * 0.113, v: 0.13 + Math.floor(i / 8) * 0.175, w: 0.092, h: 0.14,
    })),
    stripe: { v: 0.88, h: 0.07 },
  },
  ram: {
    cells: Array.from({ length: 8 }, (_, i) => ({
      u: 0.06 + i * 0.113, v: 0.30, w: 0.085, h: 0.42,
    })),
    stripe: { v: 0.80, h: 0.10 },
  },
  vram: {
    cells: Array.from({ length: 6 }, (_, i) => ({
      u: 0.07 + i * 0.152, v: 0.26, w: 0.115, h: 0.50,
    })),
    stripe: { v: 0.84, h: 0.10 },
  },
};

export interface Node {
  key: string;
  label: string;
  caption: string;
  color: string;
  /** 0-100 */
  value: number;
}

export interface EngineState {
  running: boolean;
  engine: string;
  model: string;
  tokens: number;
}

/** Strip quantisation / precision markers from a GGUF name: the face of the box
 *  is not wide enough for "Qwen3.8-27B-UD-Q4_K_M", and the markers are shown in
 *  the detail panel where there is room. */
const QUANT = /-(UD|Q\d(_[A-Z0-9]+)*|IQ\d[A-Z0-9_]*|BF16|F16|F32|F64|INT\d+|GPTQ[A-Za-z0-9]*)$/i;

function shortModel(name: string): string {
  let base = name.replace(/\.gguf$/i, "");
  // peel repeatedly: "Q-UD-Q4_K_M" needs two passes
  for (let i = 0; i < 4; i++) {
    const next = base.replace(QUANT, "");
    if (next === base) break;
    base = next;
  }
  return base.length > 24 ? base.slice(0, 23) + "…" : base;
}

function totalParams(arch: ArchGraph | null): number {
  if (!arch) return 0;
  return arch.nodes.reduce((a, n) => a + (n.params ?? 0), 0);
}

/* ------------------------------------------------------------------ */
/* hardware                                                            */
/* ------------------------------------------------------------------ */

type Anchor = "above" | "right" | "left" | "below";

/** Where each cube's caption hangs. Kept out of the cube itself because the
 *  captions are drawn *after* the tower — the tower deliberately hangs over the
 *  back of the host, and a caption underneath it would simply vanish. */
function captionAt(anchor: Anchor, x0: number, y0: number): [number, number] {
  return {
    above: iso(x0 + C / 2, y0 + C / 2, C),
    right: iso(x0 + C + 8, y0 - 4, C / 2),
    left: iso(x0 - 8, y0 + C + 4, C / 2),
    below: iso(x0 + C / 2, y0 + C + 34, 0),
  }[anchor];
}

function HardwareCube({ node, x0, y0, dim }:
  { node: Node; x0: number; y0: number; dim: boolean }) {
  const art = ART[node.key];
  return (
    <g className={dim ? "dimmed" : undefined}>
      <Box x={x0} y={y0} z={0} w={C} d={C} h={C}
        color={node.color} level={node.value} cells={art?.cells} stripe={art?.stripe}
        tip={`${node.label} — ${node.caption}`} />
    </g>
  );
}

/** Caption with a paper-coloured halo, so it stays readable over the tower. */
function HardwareCaption({ node, at, dim }:
  { node: Node; at: [number, number]; dim: boolean }) {
  return (
    <g className={"hw-cap" + (dim ? " dimmed" : "")}>
      <text x={at[0]} y={at[1] - 14} textAnchor="middle" className="cap-t halo"
        fill={node.color}>{node.label}</text>
      <text x={at[0]} y={at[1]} textAnchor="middle" className="cap-s halo">{node.caption}</text>
    </g>
  );
}

/* ------------------------------------------------------------------ */
/* the hanging tower                                                   */
/* ------------------------------------------------------------------ */

function Tower({ engine, running, modelName, params, sel, onSelect }:
  { engine: EngineState; running: boolean; modelName: string; params: number;
    sel: Sel; onSelect: (s: NonNullable<Sel>) => void }) {
  const zE = TOWER_Z, zM = TOWER_Z + TOWER_H;
  const dimEng = sel != null && sel.kind !== "engine";
  // The model slab stays lit while the selection is anywhere inside it — it is
  // the thing you are inside of, so dimming it would lose the "where am I".
  const insideModel = sel?.kind === "model" || sel?.kind === "node" || sel?.kind === "unit";
  const dimMod = sel != null && sel.kind !== "model" && !insideModel;

  // plate-sized weight-matrix artwork for the model slab
  const wCells = Array.from({ length: 6 * 4 }, (_, i) => ({
    u: 0.07 + (i % 6) * 0.148, v: 0.10 + Math.floor(i / 6) * 0.20, w: 0.108, h: 0.15,
  }));
  const eCells = Array.from({ length: 4 * 3 }, (_, i) => ({
    u: 0.10 + (i % 4) * 0.215, v: 0.14 + Math.floor(i / 4) * 0.25, w: 0.165, h: 0.18,
  }));

  // Anchored just off the tower's right vertex, clear of both slabs.
  const labelAt = (z: number): [number, number] => {
    const [x, y] = iso(PX1, PY0, z);
    return [x + 14, y];
  };

  return (
    <g className="tower">
      {/* the engine: software, so it is drawn as an outline */}
      <g className={dimEng ? "dimmed" : undefined}>
        <Box x={PX0} y={PY0} z={zE} w={PW} d={PD} h={TOWER_H}
          color="#4f8f6a" ghost top={0.10} left={0.07} right={0.04} cellOpacity={0.30}
          cells={running ? eCells : undefined} className="tier-engine"
          tip={running
            ? `${engine.engine} — the inference engine, running`
            : `${engine.engine} — the inference engine`}
          onClick={() => onSelect({ kind: "engine" })}
          selected={sel?.kind === "engine"} />
        <text x={labelAt(zE + TOWER_H / 2)[0]}
          y={labelAt(zE + TOWER_H / 2)[1] - 2} textAnchor="start"
          className="tier-name">llama.cpp</text>
        <text x={labelAt(zE + TOWER_H / 2)[0]}
          y={labelAt(zE + TOWER_H / 2)[1] + 13} textAnchor="start"
          className="tier-role">{running ? "engine · running" : "engine · idle"}</text>
      </g>

      {/* the model: the weights. Same box size as the engine, stacked on it. */}
      <g className={dimMod ? "dimmed" : undefined}>
        <Box x={PX0} y={PY0} z={zM} w={PW} d={PD} h={TOWER_H}
          color="#2f7d96" top={0.13} left={0.09} right={0.05} cellOpacity={0.34}
          cells={wCells} stroke="#26505e" className="tier-model"
          tip={`${shortModel(modelName)} — the model weights. Click to open its architecture.`}
          onClick={() => onSelect({ kind: "model" })}
          selected={sel?.kind === "model"} />
        <text x={labelAt(zM + TOWER_H / 2)[0]}
          y={labelAt(zM + TOWER_H / 2)[1] - 2} textAnchor="start"
          className="tier-name model">{shortModel(modelName)}</text>
        <text x={labelAt(zM + TOWER_H / 2)[0]}
          y={labelAt(zM + TOWER_H / 2)[1] + 13} textAnchor="start"
          className="tier-role model">
          {params > 0 ? `${fmtParams(params)} parameters` : "weights"}
        </text>
        <text x={labelAt(zM + TOWER_H / 2)[0]}
          y={labelAt(zM + TOWER_H / 2)[1] + 27} textAnchor="start"
          className="tier-hint">click to open ↓</text>
      </g>
    </g>
  );
}

/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* the scene                                                           */
/* ------------------------------------------------------------------ */

export interface HostSceneProps {
  hardware: Node[];
  engine: EngineState;
  arch: ArchGraph | null;
  sel: Sel;
  onSelect: (s: NonNullable<Sel>) => void;
  onBackground: () => void;
  hostTitle?: string;
}

export function HostScene(props: HostSceneProps) {
  const { hardware, engine, arch, sel, onSelect, hostTitle = "HOST" } = props;

  const cells: { key: string; x: number; y: number; anchor: Anchor }[] = [
    { key: "cpu", x: 0, y: 0, anchor: "above" },
    { key: "gpu", x: S, y: 0, anchor: "right" },
    { key: "ram", x: 0, y: S, anchor: "left" },
    { key: "vram", x: S, y: S, anchor: "below" },
  ];

  const params = totalParams(arch);
  const modelName = engine.running ? (engine.model || "model") : (arch?.model_id ?? "model");

  // Reserve the space the tower occupies both at rest and lifted, so hovering
  // never resizes the diagram.
  const b = new Bounds();
  for (const zz of [0, C]) {
    b.addP(iso(PX0, PY0, zz)); b.addP(iso(PX1, PY0, zz));
    b.addP(iso(PX1, PY1, zz)); b.addP(iso(PX0, PY1, zz));
  }
  for (const c of cells) {
    b.label(captionAt(c.anchor, c.x, c.y), 86, c.anchor === "below" ? 10 : 40, 20);
  }
  for (const zz of [TOWER_Z, TOWER_Z + 2 * TOWER_H + LIFT]) {
    b.addP(iso(PX0, PY0, zz)); b.addP(iso(PX1, PY0, zz));
    b.addP(iso(PX1, PY1, zz)); b.addP(iso(PX0, PY1, zz));
  }
  for (const zz of [TOWER_Z + TOWER_H / 2, TOWER_Z + 1.5 * TOWER_H]) {
    const at = iso(PX1, PY0, zz);
    b.add(at[0] + 14, at[1] - 16);
    b.add(at[0] + 14 + 200, at[1] + 22);
  }
  const hostLabel = iso(PX1, PY1, 0);
  b.label(hostLabel, 60, 0, 34);

  // Anything selected inside the model still belongs to the model, so the model
  // slab stays lit and everything else recedes.
  const inModel = sel?.kind === "model" || sel?.kind === "node" || sel?.kind === "unit";
  const hwDim = sel != null && sel.kind !== "host" && !inModel;


  // Hover the tower and it rises clear of the hardware, so the parts underneath
  // are readable without losing the sense that the software is sitting on them.
  //
  // The un-lift is delayed: the tower moves *away* from the pointer as it rises,
  // so without hysteresis a cursor near the lower edge knocks it out of its own
  // hover target, which drops it again, which picks it up again — a visible
  // flicker that also makes the element impossible to click.
  const [lift, setLift] = useState(false);
  const dropTimer = useRef<number | null>(null);
  const enter = () => {
    if (dropTimer.current !== null) { clearTimeout(dropTimer.current); dropTimer.current = null; }
    setLift(true);
  };
  const leave = () => {
    if (dropTimer.current !== null) clearTimeout(dropTimer.current);
    dropTimer.current = window.setTimeout(() => { setLift(false); dropTimer.current = null; }, 200);
  };
  useEffect(() => () => {
    if (dropTimer.current !== null) clearTimeout(dropTimer.current);
  }, []);

  return (
    <svg className="tree-svg scene-svg" viewBox={b.viewBox(16)} role="img"
      aria-label="Host hardware with the inference engine and the model weights above it"
      onClick={(e) => { if (e.target === e.currentTarget) props.onBackground(); }}>
      <Ground x0={PX0} y0={PY0} x1={PX1} y1={PY1} />

      {cells
        .slice()
        .sort((a, b2) => depth(a.x, a.y) - depth(b2.x, b2.y))
        .map((c) => (
          <HardwareCube key={c.key}
            node={hardware.find((n) => n.key === c.key) ??
              { key: c.key, label: c.key.toUpperCase(), caption: "—", color: "#999", value: 0 }}
            x0={c.x} y0={c.y} dim={hwDim} />
        ))}

      <text x={hostLabel[0]} y={hostLabel[1] + 30} textAnchor="middle" className="host-title">
        {hostTitle}
      </text>

      <g className={lift ? "upper lifted" : "upper"}
        onMouseEnter={enter} onMouseLeave={leave}>
        <Tower engine={engine} running={engine.running} modelName={modelName}
          params={params} sel={sel} onSelect={onSelect} />

      </g>

      {/* captions last, so the hanging tower never hides which part is which */}
      {cells.map((c) => {
        const n = hardware.find((h) => h.key === c.key) ??
          { key: c.key, label: c.key.toUpperCase(), caption: "—", color: "#999", value: 0 };
        return <HardwareCaption key={c.key} node={n} at={captionAt(c.anchor, c.x, c.y)}
          dim={hwDim} />;
      })}
    </svg>
  );
}