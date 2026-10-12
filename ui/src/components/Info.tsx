import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

/**
 * A small "info" affordance for text that would otherwise sit on the page.
 *
 * Explanations are what make a tool feel heavy; moving them behind a single
 * glyph keeps the page scannable while leaving the reasoning one hover, tap or
 * focus away. Nothing is hidden from assistive tech — the trigger is a real
 * button whose accessible name is the explanation.
 *
 * The panel is rendered in a portal and positioned against the button's
 * viewport rect, so it is never clipped by a scrolling ancestor (the sidebar is
 * `overflow: auto`, which would otherwise cut a tooltip in half).
 */
export function Info({ text, side = "right" }: { text: ReactNode; side?: "right" | "left" }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const btn = useRef<HTMLButtonElement | null>(null);
  const panel = useRef<HTMLSpanElement | null>(null);
  const id = useId();

  useEffect(() => {
    if (!open || !btn.current) { setPos(null); return; }
    const place = () => {
      const r = btn.current!.getBoundingClientRect();
      const w = 300;
      // Prefer opening away from the nearest screen edge.
      let left = side === "left" ? r.right - w : r.left;
      left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
      const below = r.bottom + 8;
      const fitsBelow = below + 180 < window.innerHeight;
      setPos({ top: fitsBelow ? below : Math.max(8, r.top - 188), left });
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, side]);

  return (
    <>
      <span className="info"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}>
        <button
          ref={btn}
          type="button"
          className="info-btn"
          aria-label={typeof text === "string" ? text : "more information"}
          aria-expanded={open}
          aria-describedby={open ? id : undefined}
          onClick={() => setOpen((v) => !v)}
          onFocus={() => setOpen(true)}
          onBlur={(e) => {
            if (!panel.current?.contains(e.relatedTarget as Node)) setOpen(false);
          }}
        >i</button>
      </span>
      {open && pos && createPortal(
        <span ref={panel} className="info-tip" id={id} role="tooltip"
          style={{ top: pos.top, left: pos.left }}>{text}</span>,
        document.body,
      )}
    </>
  );
}

/** Secondary detail, revealed only when asked for. */
export function Reveal({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="reveal">
      <button type="button" className="reveal-btn"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}>
        <span className={`chev ${open ? "open" : ""}`}>›</span> {label}
      </button>
      {open && <div className="reveal-body">{children}</div>}
    </div>
  );
}