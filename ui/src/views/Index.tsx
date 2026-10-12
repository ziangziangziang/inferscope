import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../state";
import { api, chatSend, type ChatState, type SampleRow } from "../lib/api";
import { HostScene, type Node } from "../components/Scene";
import { Info } from "../components/Info";

/**
 * The front door.
 *
 * Everything else assumes you already know what an inference engine is. This
 * page shows the whole path in one diagram, top to bottom: the four pieces of
 * hardware doing the work, the engine driving them, the model it is running,
 * and the conversation that falls out — which grows a leaf per token.
 *
 * Starting the engine and talking to it both happen here, so the diagram fills
 * in without anyone having to go looking for another page.
 */
export function Index({ onOpen }: { onOpen: (view: "run" | "analyze" | "arch") => void }) {
  const { health } = useApp();
  const [live, setLive] = useState<SampleRow | null>(null);
  const [chat, setChat] = useState<ChatState | null>(null);
  const [starting, setStarting] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [leaves, setLeaves] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [think, setThink] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  // Counted in a ref: the catch handler closes over the state from the render
  // that started the turn, so reading state there would always report 0.
  const got = useRef(0);

  const liveSid = health?.live_session;
  useEffect(() => {
    const tick = () => liveSid
      ? api.overview(liveSid).then((o) => setLive(o.latest)).catch(() => {})
      : Promise.resolve();
    tick();
    const id = setInterval(tick, 1200);
    return () => clearInterval(id);
  }, [liveSid]);

  useEffect(() => {
    const tick = () => api.chatState().then(setChat).catch(() => {});
    tick();
    const id = setInterval(tick, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => () => abort.current?.abort(), []);

  const running = !!chat?.ready;
  const server = chat?.server;

  const pct = (v?: number | null, d = 0) => (v == null ? 0 : Math.round(v));

  const hardware: Node[] = [
    { key: "cpu", label: "CPU", color: "#2e8b4c", value: pct(live?.cpu_util_pct),
      caption: live?.cpu_util_pct != null ? `${pct(live.cpu_util_pct)}% busy` : "idle" },
    { key: "ram", label: "RAM", color: "#2563c9", value: pct(live?.ram_used_bytes != null && live.ram_total_bytes
      ? 100 * live.ram_used_bytes / live.ram_total_bytes : null),
      caption: live?.ram_used_bytes != null
        ? `${(live.ram_used_bytes / 2 ** 30).toFixed(0)}/${(live.ram_total_bytes! / 2 ** 30).toFixed(0)} GiB`
        : "idle" },
    { key: "gpu", label: "GPU", color: "#c77b12", value: pct(live?.gpu_util_pct),
      caption: live?.gpu_util_pct != null ? `${pct(live.gpu_util_pct)}% busy` : "idle" },
    { key: "vram", label: "VRAM", color: "#7c3aed", value: pct(live?.vram_total_bytes
      ? 100 * (live.vram_used_bytes ?? 0) / live.vram_total_bytes : null),
      caption: live?.vram_used_bytes != null
        ? `${(live.vram_used_bytes / 2 ** 30).toFixed(1)} GiB used` : "idle" },
  ];

  async function toggleEngine() {
    setErr(null);
    if (running) {
      await api.chatStopServer().catch((e) => setErr(String(e)));
      setLeaves([]); setText("");
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
  }

  async function grow() {
    const q = prompt.trim() || "Tell me what a GPU is.";
    setPrompt(""); setBusy(true); setText(""); setThink(""); setLeaves([]);
    got.current = 0;
    const ac = new AbortController();
    abort.current = ac;
    await chatSend(
      { message: q, history: [], profile: false },
      (kind, data) => {
        const d = data as Record<string, unknown>;
        if (kind === "delta") {
          const s = String(d);
          setText((t) => t + s);
          if (s.trim()) { got.current += 1; setLeaves((l) => [...l, s]); }
        } else if (kind === "reasoning") {
          // A reasoning model thinks before it answers. Those are real tokens
          // doing real work, so they grow the tree too.
          const s = String(d);
          setThink((t) => t + s);
          if (s.trim()) { got.current += 1; setLeaves((l) => [...l, s]); }
        } else if (kind === "error") setErr(String(d.message ?? ""));
      },
      ac.signal,
    ).catch((e) => {
      if ((e as Error)?.name !== "AbortError") {
        // A long stream can be cut by the tunnel mid-answer. Whatever arrived is
        // already on screen, so say that rather than implying nothing happened.
        const partial = got.current;
        setErr(partial > 0
          ? `connection dropped after ${partial} tokens — showing what arrived`
          : `chat failed: ${(e as Error)?.message ?? e}`);
      }
    }).finally(() => { abort.current = null; setBusy(false); });
  }

  return (
    <div className="index">
      <div className="index-hero">
        <h1>Watch a model think.</h1>
        <p>
          A language model reads your words one at a time and predicts the next one,
          over and over, until it has an answer. That is called <b>inference</b>.
          Everything below is the path those words travel — read it from the top.
        </p>
      </div>

      <div className="scene-wrap">
        <div className="scene-title">
          one ground plane: the host, and the four parts that make it
          <span className="scene-key">fill = how busy · artwork = what the part really is</span>
        </div>
        <HostScene hardware={hardware} hostTitle="HOST" />
      </div>

      <div className="rig">
        <span className={`rig-led ${running ? "" : "off"}`} />
        <b>llama.cpp</b>
        <span style={{ color: "var(--dim)" }}>
          {running
            ? `${server?.model ?? "model loaded"} · ${leaves.length} tokens`
            : "not running — start it to chat"}
        </span>
        <div style={{ flex: 1 }} />
        <button className="btn primary" onClick={toggleEngine} disabled={starting}>
          {starting ? "loading…" : running ? "stop engine" : "start engine"}
        </button>
      </div>

      <div className="row" style={{ gap: 8, margin: "12px 0 8px" }}>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !busy) grow(); }}
          placeholder={running ? "Say something and watch the leaves appear…" : "Start the engine to chat"}
          disabled={!running || busy}
          style={{ flex: 1, padding: "9px 11px" }} />
        <button className="btn primary" onClick={grow} disabled={!running || busy}>
          {busy ? "growing…" : "send"}
        </button>
      </div>

      {err && <div className="notice bad mb">{err}</div>}
      {think && <div className="index-think">
        <b>thinking</b> {think}
      </div>}
      {text && <div className="index-reply">{text}</div>}

      <div className="index-next">
        <div className="index-card" onClick={() => onOpen("analyze")}>
          <b>Where did the time go?</b>
          <span>See which kind of GPU work dominated a captured run.</span>
        </div>
        <div className="index-card" onClick={() => onOpen("run")}>
          <b>Run and capture</b>
          <span>Record every kernel the GPU executes while answering.</span>
        </div>
        <div className="index-card" onClick={() => onOpen("arch")}>
          <b>Inside the model</b>
          <span>Its layers, their sizes, and how a picture becomes tokens.</span>
        </div>
      </div>
    </div>
  );
}
