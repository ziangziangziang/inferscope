// Global app state: which session is shown, the live sample stream, and a single
// linked "selection" object. Selecting a graph node, a timeline event or a decode
// step all write here, and every view reads from here — that is what makes the
// three perspectives move together.
import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import { api } from "./lib/api";
import type { GraphEdge, GraphNode, Health, SampleRow, Session, TraceEvent } from "./lib/api";

export interface Selection {
  node?: GraphNode;
  edge?: GraphEdge;
  event?: TraceEvent;
  stepIdx?: number;
  layerIdx?: number;
  op?: string;
}

interface AppState {
  health: Health | null;
  sessions: Session[];
  sessionId: string;
  setSessionId: (id: string) => void;
  live: SampleRow | null;
  sel: Selection;
  select: (s: Selection) => void;
  clear: () => void;
  error: string | null;
  isDemo: boolean;
  refresh: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside provider");
  return v;
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [live, setLive] = useState<SampleRow | null>(null);
  const [sel, setSel] = useState<Selection>({});
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const load = useCallback(async () => {
    try {
      const [h, s] = await Promise.all([api.health(), api.sessions()]);
      setHealth(h);
      setSessions(s.sessions);
      // Land on the session holding the most measured kernel evidence. The
      // newest live session usually has telemetry but no events, so defaulting
      // to it shows an empty Timeline and looks like nothing was ever measured.
      // Fall back to live, then demo, then anything — never blank.
      setSessionId((cur) => cur || s.best_session || h.live_session
        || h.demo_session || s.sessions[0]?.session_id || "");
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  // SSE: subscribe to the live sample stream for the current session.
  useEffect(() => {
    if (!sessionId) return;
    const es = new EventSource("/api/stream");
    esRef.current = es;
    es.addEventListener("sample", (ev) => {
      try {
        const msg = JSON.parse((ev as MessageEvent).data);
        if (msg.session_id === sessionId) setLive(msg.data);
      } catch { /* ignore malformed frame */ }
    });
    es.onerror = () => { /* browser auto-reconnects */ };
    return () => { es.close(); esRef.current = null; };
  }, [sessionId]);

  // Pull an initial latest so the overview is populated before the first SSE tick.
  useEffect(() => {
    if (!sessionId) return;
    api.overview(sessionId).then((o) => o.latest && setLive(o.latest)).catch(() => {});
  }, [sessionId]);

  const value = useMemo<AppState>(() => ({
    health, sessions, sessionId, setSessionId, live,
    sel, select: (s) => setSel((p) => ({ ...p, ...s })), clear: () => setSel({}),
    error, isDemo: sessionId.startsWith("demo") || health?.demo_session === sessionId,
    refresh: () => void load(),
  }), [health, sessions, sessionId, live, sel, error, load]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
