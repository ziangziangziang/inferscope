import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "../state";
import { api, chatSend, type ChatState, type TurnResult } from "../lib/api";
import { Notice, Prov, Tile } from "../components/common";
import { Info } from "../components/Info";

interface Msg {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  turn?: TurnResult;
  pending?: boolean;
  error?: string;
}

export function Chat() {
  const { sessionId, health } = useApp();
  const [st, setSt] = useState<ChatState | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [profile] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  const refresh = useCallback(() => {
    api.chatState().then(setSt).catch(() => setSt(null));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const startServer = async (withTrace: boolean) => {
    setStarting(true);
    setNotice(null);
    try {
      const r = await api.chatStartServer(withTrace);
      if (r.error) setNotice(`server start failed: ${r.error}`);
      refresh();
    } catch (e) {
      setNotice(String(e));
    } finally {
      setStarting(false);
    }
  };

  const stopServer = async () => {
    setNotice(null);
    try {
      const r = await api.chatStopServer();
      const n = (r as unknown as { ingested?: { events?: number; error?: string } }).ingested;
      if (n?.events) {
        setNotice(`server stopped · ingested ${n.events.toLocaleString()} kernel events into this session`);
      } else if (n?.error) {
        setNotice(`server stopped · trace not ingested: ${n.error}`);
      }
    } catch { /* stop is best-effort */ }
    refresh();
  };

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    const history = msgs
      .filter((m) => !m.error && m.content)
      .map((m) => ({ role: m.role, content: m.content }));
    setInput("");
    setBusy(true);
    setNotice(null);
    setMsgs((p) => [
      ...p,
      { role: "user", content: text },
      { role: "assistant", content: "", pending: true },
    ]);
    const ac = new AbortController();
    abortRef.current = ac;

    const patch = (fn: (m: Msg) => Msg) =>
      setMsgs((p) => {
        const copy = p.slice();
        const i = copy.length - 1;
        if (i >= 0 && copy[i].role === "assistant") copy[i] = fn(copy[i]);
        return copy;
      });

    await chatSend(
      { message: text, history, profile, session_id: sessionId || undefined },
      (kind, data) => {
        const d = data as Record<string, unknown>;
        if (kind === "delta") {
          patch((m) => ({ ...m, content: m.content + String(d) }));
        } else if (kind === "reasoning") {
          // Reasoning models emit thinking before the answer; show it rather
          // than rendering an empty bubble.
          patch((m) => ({
            ...m,
            reasoning: (m.reasoning || "") + String(d),
            content: m.content || String(d),
          }));
        } else if (kind === "notice") {
          setNotice(String(d.message ?? ""));
        } else if (kind === "done") {
          const t = d as unknown as TurnResult;
          patch((m) => ({
            ...m, pending: false, content: m.content || t.text || "",
            turn: t, error: t.error ?? undefined,
          }));
          refresh();
        } else if (kind === "error") {
          patch((m) => ({ ...m, pending: false, error: String(d.message ?? "chat failed") }));
        }
      },
      ac.signal,
    ).catch(() => {
      patch((m) => ({ ...m, pending: false, error: "request aborted" }));
    }).finally(() => {
      abortRef.current = null;
      setBusy(false);
    });
  };

  const ready = !!st?.ready;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div className="view-head">
        <h1>Run &amp; capture</h1>
        <Info text="Talk to a local model. Every turn is timed and counted, and the whole session can be kernel-traced while the server runs." />
        <div style={{ flex: 1 }} />
        {st?.managed_mode && (
          ready ? (
            <button className="pill" onClick={stopServer} style={{ cursor: "pointer" }}>■ stop server</button>
          ) : (
            <>
              <button className="pill" onClick={() => startServer(false)} disabled={starting}
                style={{ cursor: "pointer", opacity: starting ? 0.6 : 1 }}>
                {starting ? "starting…" : "▶ start llama-server"}
              </button>
              {st.profiling_available && (
                <button className="pill" onClick={() => startServer(true)} disabled={starting}
                  style={{ cursor: "pointer", opacity: starting ? 0.6 : 1 }}
                  title="Runs the server under rocprofv3: the trace covers the whole session">
                  ⏺ start + kernel trace
                </button>
              )}
            </>
          )
        )}
      </div>

      {notice && <Notice kind="warn">{notice}</Notice>}

      {!ready && (
        <div className="mb">
          <Notice kind={st?.managed_mode ? "info" : "warn"}>
            {st?.managed_mode ? (
              <>No model server yet. Pick one above to load a model onto the GPU.
                <Info text={`This loads ${st.model_path ?? "the configured model"} into VRAM, which takes a little while. "start + kernel trace" runs it under the profiler so the capture covers the whole session.`} /></>
            ) : (
              <>No chat endpoint.
                <Info text="Set INFERSCOPE_CHAT_BASE_URL to an OpenAI-compatible server, or turn managed mode on so InferScope starts one." /></>
            )}
          </Notice>
        </div>
      )}

      <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
          <div className="chat-log" style={{ flex: 1, minHeight: 0 }}>
            {msgs.length === 0 && (
              <div className="faint" style={{ fontSize: 12, padding: "18px 4px" }}>
                Send a prompt. The reply streams in and the turn is recorded as a
                <code> request </code> event in the selected session, so its cost
                shows up under <b>Analyze</b>. Start the server with the kernel trace
                enabled and the same turn also yields a session-wide kernel capture.
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={`chat-msg ${m.role}`}>
                <div className="chat-role">{m.role}</div>
                <div className="chat-body">
                  {m.content || (m.pending ? <span className="faint">thinking…</span> : null)}
                  {m.error && <div className="notice bad" style={{ marginTop: 6 }}>{m.error}</div>}
                  {m.turn && <TurnStats t={m.turn} />}
                </div>
              </div>
            ))}
            <div ref={endRef} />
          </div>

          <div className="row mt" style={{ gap: 8, alignItems: "flex-end" }}>
            <textarea
              className="chat-input"
              value={input}
              placeholder={ready ? "Ask the model something…" : "Start the server first"}
              disabled={!ready || busy}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              rows={2}
            />
            <button className="primary" onClick={send} disabled={!ready || busy || !input.trim()}>
              {busy ? "streaming…" : "send"}
            </button>
          </div>
          <div className="row mt" style={{ gap: 10, flexWrap: "wrap" }}>
            {st?.server?.profiling ? (
              <span className="pill" title={st.server.trace_out_dir ?? ""}>
                ⏺ kernel trace active (whole session)
              </span>
            ) : (
              <span className="faint" style={{ fontSize: 11 }}>
                per-turn kernel trace: unavailable
                <Info text={st?.per_turn_trace_reason ?? "Requires CAP_SYS_PTRACE."} />
              </span>
            )}
            <Prov p="measured" hint="TTFT and wall time are timed client-side; token counts are deltas of the engine's own counters" />
          </div>
        </div>

        <div className="inspector" style={{ flex: "0 0 280px" }}>
          <h2>Session</h2>
          <div className="faint" style={{ fontSize: 11, marginBottom: 8 }}>
            Turns are recorded into the session selected in the top bar.
          </div>
          <Tile k="session" v={sessionId || "—"} />
          <div className="mt">
            <div className="faint" style={{ fontSize: 10, marginBottom: 4 }}>server</div>
            <table className="kv-table"><tbody>
              <tr><td className="k">mode</td><td className="v">{st?.managed_mode ? "managed" : "external"}</td></tr>
              <tr><td className="k">endpoint</td><td className="v">{st?.base_url ?? "—"}</td></tr>
              <tr><td className="k">model</td><td className="v">{st?.server?.model ?? "—"}</td></tr>
              <tr><td className="k">context</td><td className="v">{st?.server?.ctx ?? "—"}</td></tr>
              <tr><td className="k">pid</td><td className="v">{st?.server?.pid ?? "—"}</td></tr>
              <tr><td className="k">uptime</td><td className="v">{st?.server?.uptime_s != null ? `${st.server.uptime_s}s` : "—"}</td></tr>
              <tr><td className="k">kernel trace</td><td className="v">{st?.server?.profiling ? "on (session)" : "off"}</td></tr>
              <tr><td className="k">per-turn trace</td><td className="v">{st?.per_turn_trace ? "available" : "unavailable"}</td></tr>
            </tbody></table>
          </div>
          <div className="faint mt" style={{ fontSize: 10.5 }}>
            <Info text={st?.note ?? ""} />
          </div>
        </div>
      </div>
    </div>
  );
}

function TurnStats({ t }: { t: TurnResult }) {
  const bad = !!t.error;
  return (
    <div className="row mt" style={{ gap: 6, flexWrap: "wrap", fontSize: 11 }}>
      {t.ttft_ms != null && (
        <span className="pill" title="Time to first token — how long before the first word appeared">ttft {Math.round(t.ttft_ms)} ms</span>
      )}
      <span className="pill" title="Total time for the whole reply">total {Math.round(t.total_ms)} ms</span>
      {t.prompt_tokens != null && (
        <span className="pill" title="How many words went in">{t.prompt_tokens} in</span>
      )}
      {t.completion_tokens != null && (
        <span className="pill" title="How many words came out">{t.completion_tokens} out</span>
      )}
      {t.tok_per_s != null && (
        <span className="pill" title="Words per second it generated">{t.tok_per_s.toFixed(1)} words/s</span>
      )}
      {t.kernel_events > 0 && (
        <span className="pill" title="kernel dispatches captured for this turn">
          {t.kernel_events.toLocaleString()} kernels
        </span>
      )}
      {bad && <span className="pill" style={{ color: "var(--bad)" }}>{t.error}</span>}
    </div>
  );
}