/**
 * Isometric primitives.
 *
 * One camera for the whole diagram: `iso(x, y, z)` projects a world point to the
 * screen, and every box in the scene — the host plate, the hardware cubes, the
 * engine slab, the model slab and each model component — is drawn with this one
 * function, so they are always literally the same perspective.
 *
 * A `Box` is deliberately *not* forced to be a cube. In the reference figures
 * every part happened to be the same size; here a component's dimensions carry
 * meaning (a box's footprint is proportional to the weights it holds), so the
 * primitive takes explicit world width/depth/height.
 */

const COS30 = Math.cos(Math.PI / 6);
const SIN30 = 0.5;

export type P = [number, number];

export function iso(x: number, y: number, z: number): P {
  return [(x - y) * COS30, (x + y) * SIN30 - z];
}

export function poly(pts: P[]): string {
  return pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
}

/** Painter's-algorithm depth: larger x+y is nearer the viewer. */
export function depth(x: number, y: number): number {
  return x + y;
}

/** Accumulator for computing a viewBox from real projected geometry. */
export class Bounds {
  x0 = Infinity; y0 = Infinity; x1 = -Infinity; y1 = -Infinity;
  add(x: number, y: number): this {
    this.x0 = Math.min(this.x0, x); this.x1 = Math.max(this.x1, x);
    this.y0 = Math.min(this.y0, y); this.y1 = Math.max(this.y1, y);
    return this;
  }
  addP(p: P): this { return this.add(p[0], p[1]); }
  /** Reserve screen space for a label anchored near a projected point. */
  label(p: P, halfW: number, top: number, bottom: number): this {
    return this.add(p[0] - halfW, p[1] - top).add(p[0] + halfW, p[1] + bottom);
  }
  viewBox(pad = 14): string {
    return `${(this.x0 - pad).toFixed(1)} ${(this.y0 - pad).toFixed(1)} ` +
      `${(this.x1 - this.x0 + pad * 2).toFixed(1)} ${(this.y1 - this.y0 + pad * 2).toFixed(1)}`;
  }
}

export interface Cell { u: number; v: number; w: number; h: number }

export interface BoxProps {
  /** world min corner and base height */
  x: number; y: number; z: number;
  w: number; d: number; h: number;
  color: string;
  /** 0-100 fill level painted up the two side faces (hardware busy-ness) */
  level?: number;
  /** dashed outline = software, not hardware */
  ghost?: boolean;
  /** opacity per face; defaults suit a solid translucent box */
  top?: number; left?: number; right?: number;
  cells?: Cell[];
  /** opacity of the top-face artwork (0 = no artwork) */
  cellOpacity?: number;
  stripe?: { v: number; h: number };
  stroke?: string;
  strokeWidth?: number;
  onClick?: () => void;
  selected?: boolean;
  /** native tooltip */
  tip?: string;
  className?: string;
  children?: React.ReactNode;
}

/**
 * One box. Draws the three faces this camera can see (top, front-left, front-right)
 * with the base `BaseBox` convention: translucent fills, hairline borders, and
 * artwork on the top face.
 */
export function Box(props: BoxProps) {
  const {
    x, y, z, w, d, h, color, level = 0, ghost = false,
    top = 0.62, left = 0.22, right = 0.14,
    cells, cellOpacity = 0.62, stripe, stroke = "#2f3a2c", selected = false,
    onClick, tip, className, children,
  } = props;
  const sw = props.strokeWidth ?? Math.max(0.6, Math.min(1.5, Math.min(w, d, h) * 0.035));
  const x1 = x + w, y1 = y + d, z1 = z + h;

  const topPts = [iso(x, y, z1), iso(x1, y, z1), iso(x1, y1, z1), iso(x, y1, z1)];
  const leftPts = [iso(x, y1, z), iso(x, y1, z1), iso(x1, y1, z1), iso(x1, y1, z)];
  const rightPts = [iso(x1, y, z), iso(x1, y, z1), iso(x1, y1, z1), iso(x1, y1, z)];

  // unit (u,v) -> point on the top face
  const p00 = topPts[0], p10 = topPts[1], p01 = topPts[3];
  const uv = (u: number, v: number): P => [
    p00[0] + u * (p10[0] - p00[0]) + v * (p01[0] - p00[0]),
    p00[1] + u * (p10[1] - p00[1]) + v * (p01[1] - p00[1]),
  ];
  const quad = (u: number, v: number, qw: number, qh: number): string =>
    poly([uv(u, v), uv(u + qw, v), uv(u + qw, v + qh), uv(u, v + qh)]);

  const lh = Math.max(0, Math.min(100, level)) / 100 * h;
  const fillL = lh > 0.4
    ? [iso(x, y1, z), iso(x1, y1, z), iso(x1, y1, z + lh), iso(x, y1, z + lh)] : null;
  const fillR = lh > 0.4
    ? [iso(x1, y, z), iso(x1, y1, z), iso(x1, y1, z + lh), iso(x1, y, z + lh)] : null;

  const dash = ghost ? `${(sw * 3.2).toFixed(1)} ${(sw * 2).toFixed(1)}` : undefined;
  const cls = ["bx", onClick && "bx-hit", selected && "bx-sel", className]
    .filter(Boolean).join(" ");

  return (
    <g className={cls} onClick={onClick} tabIndex={onClick ? 0 : undefined}
      role={onClick ? "button" : undefined}
      aria-label={tip}
      onKeyDown={onClick ? (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); }
      } : undefined}>
      {tip ? <title>{tip}</title> : null}
      <polygon className="bx-l" points={poly(leftPts)} fill={color} fillOpacity={left}
        stroke={stroke} strokeWidth={selected ? sw * 1.9 : sw} strokeDasharray={dash} />
      <polygon className="bx-r" points={poly(rightPts)} fill={color} fillOpacity={right}
        stroke={stroke} strokeWidth={selected ? sw * 1.9 : sw} strokeDasharray={dash} />
      {fillL && <polygon points={poly(fillL)} fill={color} fillOpacity="0.72" stroke="none" />}
      {fillR && <polygon points={poly(fillR)} fill={color} fillOpacity="0.55" stroke="none" />}
      <polygon className="bx-t" points={poly(topPts)} fill="#ffffff" fillOpacity={top}
        stroke={stroke} strokeWidth={selected ? sw * 1.9 : sw} strokeDasharray={dash} />
      {cells?.map((c, i) => (
        <polygon key={i} points={quad(c.u, c.v, c.w, c.h)} fill={color}
          fillOpacity={cellOpacity} stroke={stroke} strokeWidth={sw * 0.7} />
      ))}
      {stripe && (
        <polygon points={quad(0.04, stripe.v, 0.92, stripe.h)} fill={color}
          fillOpacity="0.45" stroke={stroke} strokeWidth={sw * 0.7} />
      )}
      {children}
    </g>
  );
}

/** The ground plane: a thin slab, drawn as top + the two visible rim faces. */
export function Ground({ x0, y0, x1, y1, t = 8 }:
  { x0: number; y0: number; x1: number; y1: number; t?: number }) {
  return (
    <g className="ground">
      <polygon points={poly([iso(x1, y0, 0), iso(x1, y1, 0), iso(x1, y1, -t), iso(x1, y0, -t)])}
        fill="#b0c8a9" stroke="#7d9c78" strokeWidth="1" />
      <polygon points={poly([iso(x0, y1, 0), iso(x1, y1, 0), iso(x1, y1, -t), iso(x0, y1, -t)])}
        fill="#c2d8bb" stroke="#7d9c78" strokeWidth="1" />
      <polygon points={poly([iso(x0, y0, 0), iso(x1, y0, 0), iso(x1, y1, 0), iso(x0, y1, 0)])}
        fill="#e9f1e4" stroke="#88a882" strokeWidth="1.2" />
    </g>
  );
}

/**
 * A caption anchored to a projected point. `where` picks the side so labels of
 * neighbouring boxes never land on each other's bodies.
 */
export function Label({ at, title, sub, where = "below", titleClass = "cap-t", subClass = "cap-s",
  color, dy = 0 }:
  { at: P; title: string; sub?: string; where?: "above" | "below" | "right" | "left";
    titleClass?: string; subClass?: string; color?: string; dy?: number }) {
  const anchor = where === "right" ? "start" : where === "left" ? "end" : "middle";
  const y0 = where === "above" ? dy - 14 : where === "below" ? dy + 16 : dy + 4;
  return (
    <g className="cap">
      <text x={at[0]} y={y0} textAnchor={anchor} className={titleClass}
        fill={color}>{title}</text>
      {sub ? <text x={at[0]} y={y0 + 12} textAnchor={anchor} className={subClass}>{sub}</text> : null}
    </g>
  );
}

/** Reserve room for a caption in a Bounds. */
export function boundLabel(b: Bounds, at: P, halfW: number, above = false) {
  b.label(at, halfW, above ? 40 : 34, 22);
}