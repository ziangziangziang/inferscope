// What is on the GPU, and the means to change it.
//
// This used to live inside the Start-here view, which meant the engine and
// model picker only existed on that one page. It belongs to the frame instead:
// it decides what the machine is doing, so it should be reachable from
// anywhere, and the scene on the workbench should simply read the same state
// rather than keep a private copy.
import {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from "react";
import { api } from "./lib/api";
import { units as fetchUnits, engineOffload, type UnitInfo } from "./lib/api";
import { useEngineStatus } from "./components/EngineRail";
import { useSwapEngine, type Swap } from "./components/UnitPicker";

export interface GpuState {
  // catalogue + choice
  catalogue: UnitInfo[];
  chosen: UnitInfo | null;
  setChosen: (u: UnitInfo | null) => void;
  pickUnit: (u: UnitInfo) => void;
  swap: Swap;

  // residency, from both owners
  status: ReturnType<typeof useEngineStatus>["status"];
  resident: string | null;
  refreshEngines: () => void;
  shownEngine: string | null;

  // derived
  /** chosen, else whatever is actually resident. No parameter fixups — the
   *  workbench adds those, since only it knows the loaded architecture. */
  activeUnit: UnitInfo | null;
  llmEngine: boolean;
  inputHint: string;

  // the language engine, which chat.py owns rather than enginectl
  chat: Awaited<ReturnType<typeof api.chatState>> | null;
  running: boolean;
  server: { model?: string } | undefined;
  starting: boolean;
  toggleEngine: () => void;

  err: string | null;
}

const Ctx = createContext<GpuState | null>(null);

export function useGpu(): GpuState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useGpu outside provider");
  return v;
}

export function GpuProvider({ children }: { children: React.ReactNode }) {
  const [catalogue, setCatalogue] = useState<UnitInfo[]>([]);
  const { status, refresh: refreshEngines } = useEngineStatus();
  const resident = status?.loaded ?? null;
  const { swap, swapTo } = useSwapEngine();

  const [chat, setChat] = useState<GpuState["chat"]>(null);
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const running = !!chat?.ready;
  const server = chat?.server;

  useEffect(() => {
    let dead = false;
    const tick = () => fetchUnits()
      .then((c) => { if (!dead) setCatalogue(c.units); })
      .catch(() => {});
    tick();
    const id = setInterval(tick, 6000);
    return () => { dead = true; clearInterval(id); };
  }, []);

  useEffect(() => {
    const tick = () => api.chatState().then(setChat).catch(() => {});
    tick();
    const id = setInterval(tick, 5000);
    return () => clearInterval(id);
  }, []);

  // Residency has two owners: enginectl owns everything it starts, and chat.py
  // owns the language-model server. The scene has to see both, or the plate
  // shows nothing while llama.cpp is happily running.
  const effectiveResident = resident ?? (running ? "llm" : null);

  const [chosen, setChosen] = useState<UnitInfo | null>(null);

  const pickUnit = useCallback(async (u: UnitInfo) => {
    setErr(null);
    setChosen(u);
    try {
      if (u.managed) {
        await swapTo(u);
      } else {
        // The language engine lives outside enginectl, so free the card and
        // start it through its own endpoint — otherwise picking the model the
        // whole page is about would fail with "unknown engine: llm".
        await engineOffload();
        setStarting(true);
        try {
          // the unit key, not the default: picking a vision checkpoint has to
          // bring its projector with it
          await api.chatStartServer(false, u.key);
        } finally {
          setStarting(false);
        }
        api.chatState().then(setChat).catch(() => {});
      }
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
      setChosen(null);
    } finally {
      refreshEngines();
    }
  }, [swapTo, refreshEngines]);

  const toggleEngine = useCallback(async () => {
    setErr(null);
    if (running) {
      await api.chatStopServer().catch((e) => setErr(String(e)));
      return;
    }
    setStarting(true);
    try {
      await api.chatStartServer(false);
    } catch (e) {
      setErr(String(e));
    } finally {
      setStarting(false);
    }
    api.chatState().then(setChat).catch(() => {});
  }, [running]);

  // once the engine we asked for is actually up, stop tracking the intent
  useEffect(() => {
    if (chosen && resident === chosen.engine) setChosen(null);
  }, [resident, chosen]);

  const activeUnit = chosen
    ?? (effectiveResident ? catalogue.find((x) => x.engine === effectiveResident) ?? null : null);
  const llmEngine = activeUnit?.engine === "llm";
  const shownEngine = effectiveResident ?? chosen?.engine ?? null;

  const inputHint = llmEngine ? "words in, tokens out"
    : shownEngine === "asr" ? "audio in, text out"
    : shownEngine ? "not wired up yet"
    : "pick an engine above";

  const value = useMemo<GpuState>(() => ({
    catalogue, chosen, setChosen, pickUnit, swap,
    status, resident, refreshEngines, shownEngine,
    activeUnit, llmEngine, inputHint,
    chat, running, server, starting, toggleEngine,
    err,
  }), [catalogue, chosen, pickUnit, swap, status, resident, refreshEngines,
    shownEngine, activeUnit, llmEngine, inputHint, chat, running, server,
    starting, toggleEngine, err]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}