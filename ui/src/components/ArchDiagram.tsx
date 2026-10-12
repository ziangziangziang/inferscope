/**
 * The model's architecture, drawn as its own diagram.
 *
 * This lives in the detail panel rather than in the scene, because the scene's
 * job is the *overview* — what machine, what engine, what model — and a 101-box
 * unfold on top of it buried the host and still had to be scaled down to fit.
 * Here it gets half the screen to itself and the host stays legible.
 *
 * Geometry and sizing come from `sceneModel.ts`, which knows nothing about the
 * scene or the panel; this file only draws it.
 */

import { useMemo } from "react";
import { Box, Bounds, iso, depth } from "./iso";
import {
  layoutArch, layoutCaptions, artOf, colorOf, fmtParams,
  type Placed, type ArchLayout,
} from "./sceneModel";
import type { ArchGraph, Sel } from "./archTypes";

const H = 24;          // box height
const GAP = 12;        // gap between boxes along a branch
const ROW_GAP = 28;    // gap between branch rows

export function ArchDiagram({ arch, sel, onSelect }:
  { arch: ArchGraph; sel: Sel; onSelect: (s: NonNullable<Sel>) => void }) {
  const layout: ArchLayout = useMemo(
    () => layoutArch(arch.nodes, { gap: GAP, rowGap: ROW_GAP, h: H }),
    [arch]);

  const captions = useMemo(() => layoutCaptions(layout), [layout]);

  // viewBox from the real geometry plus every caption's real extent
  const b = useMemo(() => {
    const bb = new Bounds();
    for (const it of layout.items) {
      bb.addP(iso(it.x, it.y, it.z + it.h));
      bb.addP(iso(it.x + it.w, it.y + it.d, it.z + it.h));
      bb.addP(iso(it.x + it.w, it.y, it.z));
      bb.addP(iso(it.x, it.y + it.d, it.z));
    }
    for (const c of captions) {
      const bottom = 20 + c.lines * 14 + (c.hint ? 14 : 0);
      // `halfW` is the caption's full width; which side of the anchor it
      // occupies depends on the anchor.
      if (c.anchor === "start") {
        bb.add(c.at[0], c.at[1] - 16); bb.add(c.at[0] + c.halfW, c.at[1] + bottom);
      } else if (c.anchor === "end") {
        bb.add(c.at[0] - c.halfW, c.at[1] - 16); bb.add(c.at[0], c.at[1] + bottom);
      } else {
        bb.label(c.at, c.halfW / 2, 16, bottom);
      }
    }
    return bb.viewBox(14);
  }, [layout, captions]);

  // Drill into a unit dims its 63 siblings; a single component dims the rest.
  const drilled = sel?.kind === "unit";
  const picked = sel?.kind === "node" || sel?.kind === "unit";

  return (
    <svg className="arch-svg" viewBox={b} role="img"
      aria-label="The model's architecture: one box per component, sized by the parameters it holds">
      <g className="arch-boxes">
        {layout.items
          .slice()
          .sort((x, y) => depth(x.x, x.y) - depth(y.x, y.y))
          .map((p) => (
            <g key={p.id} data-id={p.id}>
              <ArchBox p={p} sel={sel} onSelect={onSelect}
                dim={picked && !isActive(p, sel)} />
            </g>
          ))}
      </g>

      {/* The vision branch feeds the text stream: show the join, or the two
          parallel rows read as unrelated. */}
      {(() => {
        const from = layout.items.filter((i) => i.column === 1);
        const to = layout.items.filter((i) => i.column === 0);
        if (!from.length || !to.length) return null;
        const last = from.reduce((a, i) => (i.order > a.order ? i : a));
        const first = to.reduce((a, i) => (i.order < a.order ? i : a));
        const p1 = iso(last.x + last.w, last.y + last.d / 2, last.z + last.h / 2);
        const p2 = iso(first.x, first.y + first.d / 2, first.z + first.h / 2);
        const mx = (p1[0] + p2[0]) / 2, my = (p1[1] + p2[1]) / 2 - 46;
        return (
          <path className="arch-feed" fill="none"
            d={`M ${p1[0].toFixed(1)} ${p1[1].toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`} />
        );
      })()}

      {/* captions on top of every box */}
      <g className="arch-caps">
        {captions.map((c) => (
          <g key={c.key}>
            <text x={c.at[0]} y={c.at[1] + 12} textAnchor={c.anchor} className={c.cls}
              fill={c.color}>{c.title}</text>
            {c.sub ? (
              <text x={c.at[0]} y={c.at[1] + 26} textAnchor={c.anchor}
                className={c.subCls}>{c.sub}</text>
            ) : null}
            {c.hint ? (
              <text x={c.at[0]} y={c.at[1] + 40} textAnchor={c.anchor}
                className="grp-hint">{c.hint}</text>
            ) : null}
          </g>
        ))}
      </g>

      {!drilled && picked ? (
        <text className="arch-tap-hint" x={0} y={0} aria-hidden />
      ) : null}
    </svg>
  );
}

function isActive(p: Placed, sel: Sel): boolean {
  if (!sel) return false;
  if (p.unitIndex != null) {
    return sel.kind === "unit" && sel.id === p.groupId && sel.index === p.unitIndex;
  }
  return sel.kind === "node" && sel.id === p.id;
}

function ArchBox({ p, sel, onSelect, dim }:
  { p: Placed; sel: Sel; onSelect: (s: NonNullable<Sel>) => void; dim: boolean }) {
  const isUnit = p.unitIndex != null;
  const color = colorOf(p.node);
  const art = artOf(p.node);

  const tip = isUnit
    ? `${p.node.label} — unit ${(p.unitIndex ?? 0) + 1} of ${p.groupCount}\n` +
      `${fmtParams(p.node.params / (p.groupCount ?? 1))} parameters each\nclick to open its sub-layers`
    : `${p.node.label}\n${p.node.params ? `${p.node.params.toLocaleString()} parameters` : "no weights"}` +
      (p.node.notes?.length ? `\n${p.node.notes.filter(Boolean).join(" · ")}` : "");

  return (
    <Box x={p.x} y={p.y} z={p.z} w={p.w} d={p.d} h={p.h}
      color={color} cells={art.cells} stripe={art.stripe}
      top={0.74} left={0.26} right={0.16} cellOpacity={0.46}
      className={dim ? "dimmed" : undefined}
      selected={isActive(p, sel)}
      tip={tip}
      onClick={() => onSelect(isUnit
        ? { kind: "unit", id: p.groupId!, index: p.unitIndex! }
        : { kind: "node", id: p.id })} />
  );
}