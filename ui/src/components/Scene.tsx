/**
 * The inference scene.
 *
 * One isometric ground plane, one camera, and six cubes that are all the same
 * size: `width = height = depth`, semi-transparent faces with a hairline
 * border — the `BaseBox` convention from the reference schematic figures. The
 * earlier version shared the viewing *angle* but not the box *convention* (a
 * wide slab for the host, squat boxes for the parts), which is what made the
 * perspective read wrong.
 *
 * The host sits behind the group, so it appears higher on screen. The four
 * components occupy a real 2x2 in world space, which projects as a diamond —
 * that is the honest consequence of one perspective, and the same thing the
 * reference figures do.
 *
 * Each cube carries artwork for the thing it actually is (see `ART`), and a
 * fill on its two visible faces showing measured utilisation.
 */

const COS30 = Math.cos(Math.PI / 6);
const SIN30 = 0.5;

const C = 56;                 // cube side: width = height = depth
const G = 36;                 // gap between cubes
const S = C + G;

function iso(x: number, y: number, z: number): [number, number] {
  return [(x - y) * COS30, (x + y) * SIN30 - z];
}

function poly(pts: [number, number][]): string {
  return pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
}

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

interface P { x: number; y: number }

type Anchor = "above" | "right" | "left" | "below" | "none";

/** One cube of side C, based at world (x0,y0), with artwork and a fill. */
function Cube({ node, x0, y0, anchor = "none" }:
  { node: Node; x0: number; y0: number; anchor?: Anchor }) {
  const x1 = x0 + C, y1 = y0 + C;
  const h = (Math.max(0, Math.min(100, node.value)) / 100) * C;

  const top = [iso(x0, y0, C), iso(x1, y0, C), iso(x1, y1, C), iso(x0, y1, C)];
  const left = [iso(x0, y1, 0), iso(x0, y1, C), iso(x1, y1, C), iso(x1, y1, 0)];
  const right = [iso(x1, y0, 0), iso(x1, y0, C), iso(x1, y1, C), iso(x1, y1, 0)];

  // artwork in unit (u,v) on the top face
  const p00 = iso(x0, y0, C), p10 = iso(x1, y0, C), p01 = iso(x0, y1, C);
  const uv = (u: number, v: number): P => ({
    x: p00[0] + u * (p10[0] - p00[0]) + v * (p01[0] - p00[0]),
    y: p00[1] + u * (p10[1] - p00[1]) + v * (p01[1] - p00[1]),
  });
  const quad = (u: number, v: number, w: number, hh: number) => {
    const a = uv(u, v), b = uv(u + w, v), c = uv(u + w, v + hh), d = uv(u, v + hh);
    return `${a.x.toFixed(1)},${a.y.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)} ` +
      `${c.x.toFixed(1)},${c.y.toFixed(1)} ${d.x.toFixed(1)},${d.y.toFixed(1)}`;
  };

  const art = ART[node.key];
  const fillLeft = [iso(x0, y1, 0), iso(x1, y1, 0), iso(x1, y1, h), iso(x0, y1, h)];
  const fillRight = [iso(x1, y0, 0), iso(x1, y1, 0), iso(x1, y1, h), iso(x1, y0, h)];

  // Each cube in the 2x2 diamond gets its caption on its *outer* side, so the
  // back cube's label never lands on the front cube's body.
  let lx = 0, ly = 0;
  let anchor_: "start" | "middle" | "end" = "start";
  if (anchor === "above") {
    [lx, ly] = iso(x0 + C / 2, y0 + C / 2, C);
    ly -= C * 0.5 + 16; anchor_ = "middle";
  } else if (anchor === "right") {
    [lx, ly] = iso(x1 + 8, y0 - 4, C / 2); anchor_ = "start";
  } else if (anchor === "left") {
    [lx, ly] = iso(x0 - 8, y1 + 4, C / 2); anchor_ = "end";
  } else if (anchor === "below") {
    [lx, ly] = iso(x0 + C / 2, y1 + 14, 0); anchor_ = "middle";
  }

  return (
    <g>
      <polygon points={poly(left)} fill={node.color} fillOpacity="0.22"
        stroke="#2f3a2c" strokeWidth="0.9" />
      <polygon points={poly(right)} fill={node.color} fillOpacity="0.14"
        stroke="#2f3a2c" strokeWidth="0.9" />
      {h > 0.5 && (
        <>
          <polygon points={poly(fillLeft)} fill={node.color} fillOpacity="0.72" stroke="none" />
          <polygon points={poly(fillRight)} fill={node.color} fillOpacity="0.55" stroke="none" />
        </>
      )}
      <polygon points={poly(top)} fill="#ffffff" fillOpacity="0.62"
        stroke="#2f3a2c" strokeWidth="0.9" />
      {art?.cells.map((c, i) => (
        <polygon key={i} points={quad(c.u, c.v, c.w, c.h)} fill={node.color}
          fillOpacity="0.62" stroke="#2f3a2c" strokeWidth="0.6" />
      ))}
      {art?.stripe && (
        <polygon points={quad(0.04, art.stripe.v, 0.92, art.stripe.h)} fill={node.color}
          fillOpacity="0.45" stroke="#2f3a2c" strokeWidth="0.6" />
      )}
      {anchor !== "none" && (
        <>
          <text x={lx} y={ly} textAnchor={anchor_} className="cube-cap"
            fill={node.color}>{node.label}</text>
          <text x={lx} y={ly + 13} textAnchor={anchor_} className="cube-cap2">{node.caption}</text>
        </>
      )}
    </g>
  );
}

/** The single ground plane every cube stands on. */
function Ground({ x0, y0, x1, y1 }: { x0: number; y0: number; x1: number; y1: number }) {
  const T = 8;
  return (
    <g>
      <polygon points={poly([iso(x1, y0, 0), iso(x1, y1, 0), iso(x1, y1, -T), iso(x1, y0, -T)])}
        fill="#b0c8a9" stroke="#7d9c78" strokeWidth="1" />
      <polygon points={poly([iso(x0, y1, 0), iso(x1, y1, 0), iso(x1, y1, -T), iso(x0, y1, -T)])}
        fill="#c2d8bb" stroke="#7d9c78" strokeWidth="1" />
      <polygon points={poly([iso(x0, y0, 0), iso(x1, y0, 0), iso(x1, y1, 0), iso(x0, y1, 0)])}
        fill="#e9f1e4" stroke="#88a882" strokeWidth="1.2" />
    </g>
  );
}

export function HostScene({ hardware, hostTitle = "HOST" }:
  { hardware: Node[]; hostTitle?: string }) {
  // Host sits behind the group; the four parts take a real 2x2 in world space.
  const cells: { key: string; x: number; y: number; anchor: Anchor }[] = [
    { key: "cpu", x: 0, y: 0, anchor: "above" },
    { key: "gpu", x: S, y: 0, anchor: "right" },
    { key: "ram", x: 0, y: S, anchor: "left" },
    { key: "vram", x: S, y: S, anchor: "below" },
  ];

  // The host is the plate itself: one slab covering the 2x2 with a margin.
  const M = 34;
  const gx0 = -M, gy0 = -M, gx1 = S + C + M, gy1 = S + C + M;
  const hx0 = 0, hy0 = (gx1 + gy1) * 0.5;      // front vertex of the plate

  // viewBox from the real projected bounds of ground + cubes + captions
  const b = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  const acc = (p: [number, number]) => {
    b.x0 = Math.min(b.x0, p[0]); b.x1 = Math.max(b.x1, p[0]);
    b.y0 = Math.min(b.y0, p[1]); b.y1 = Math.max(b.y1, p[1]);
  };
  for (const zz of [0, C]) {
    acc(iso(gx0, gy0, zz)); acc(iso(gx1, gy0, zz));
    acc(iso(gx1, gy1, zz)); acc(iso(gx0, gy1, zz));
  }
  for (const [cx0, cy0] of cells.map((c) => [c.x, c.y] as [number, number])) {
    for (const zz of [0, C]) {
      acc(iso(cx0, cy0, zz)); acc(iso(cx0 + C, cy0, zz));
      acc(iso(cx0 + C, cy0 + C, zz)); acc(iso(cx0, cy0 + C, zz));
    }
    const [lx, ly] = iso(cx0 + C / 2, cy0 + C / 2, C);
    acc([lx - 130, ly - C]); acc([lx + 150, ly + 40]);
  }
  // the host label, drawn under the plate's front edge
  acc([hx0 - 90, hy0 + 8]); acc([hx0 + 90, hy0 + 34]);

  const P = 14;
  const vb = `${(b.x0 - P).toFixed(1)} ${(b.y0 - P).toFixed(1)} ${(b.x1 - b.x0 + P * 2).toFixed(1)} ${(b.y1 - b.y0 + P * 2).toFixed(1)}`;

  // painter's order: further from the viewer first
  const drawables = [
    ...cells.map((c) => ({
      d: c.x + c.y,
      el: <Cube key={c.key} node={hardware.find((n) => n.key === c.key) ?? {
        key: c.key, label: c.key.toUpperCase(), caption: "—", color: "#999", value: 0,
      }} x0={c.x} y0={c.y} anchor={c.anchor} />,
    })),
  ].sort((a, b2) => a.d - b2.d);

  return (
    <svg className="tree-svg" viewBox={vb} role="img"
      aria-label="The host: CPU, GPU, RAM and VRAM cubes standing on its plate">
      <Ground x0={gx0} y0={gy0} x1={gx1} y1={gy1} />
      {/* the host label sits on the plate's front edge */}
      <text x={hx0} y={hy0 + 24} textAnchor="middle" className="host-title">{hostTitle}</text>
      {drawables.map((d, i) => <g key={i}>{d.el}</g>)}

    </svg>
  );
}