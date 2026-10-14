import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../state";
import { useGpu } from "../gpuState";
import { InputBox, describeAccepts } from "../components/InputBox";
import { ResultOut } from "../components/ResultOut";
import { ttsSynthesize, imageGenerate, videoGenerate,
         type TtsResult, type ImageResult, type VideoResult } from "../lib/api";
import { api, chatSend, type ChatState, type SampleRow } from "../lib/api";
import { HostScene, type Node } from "../components/Scene";
import { SceneDetail } from "../components/SceneDetail";
import { toArch, type Sel } from "../components/archTypes";
import { Info } from "../components/Info";
import { AsrTask } from "../components/tasks/AsrTask";

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
export function Index({ onOpen, onOpenDetails }: {
  onOpen: (view: "run" | "analyze" | "arch") => void;
  /** Clicking a result opens the details panel, which is where the rest of
   *  the app's sections now live. */
  onOpenDetails?: () => void;
}) {
  const { health } = useApp();
  const [live, setLive] = useState<SampleRow | null>(null);
  const [prompt, setPrompt] = useState("");
  const [pendingImage, setPendingImage] = useState<string | null>(null);
  const [audio, setAudio] = useState<TtsResult | null>(null);
  const [picture, setPicture] = useState<ImageResult | null>(null);
  const [clip, setClip] = useState<VideoResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [leaves, setLeaves] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [think, setThink] = useState("");
  const [turnErr, setTurnErr] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  /* --- model architecture, for the scene --------------------------- */
  const [sel, setSel] = useState<Sel>(null);
  const [fetchedArch, setArch] = useState<ReturnType<typeof toArch>>(null);

  // Engine, model and residency are owned by the frame now (the left rail's
  // configurable panel sets them), so the scene just reads the same state.
  const g = useGpu();
  const {
    chosen, swap, resident, chat, running, server, starting, err,
    shownEngine, setChosen, toggleEngine,
  } = g;

  // The scene draws the chosen unit, or whatever is resident if nothing is
  // chosen. Parameter counts for the language model come from the loaded
  // architecture, which only this view fetches.
  // whisper.cpp ships no config, so the catalogue builds its graph from the
  // engine's own load report and hands it over inline. Prefer that over
  // anything fetched, and let it supply the parameter total the block is sized
  // by — otherwise a 141 MiB whisper would be drawn as an unknown slab.
  const unitArch = (g.activeUnit?.arch ?? null) as ReturnType<typeof toArch>;
  const arch = unitArch ?? fetchedArch;

  const activeUnit = useMemo(() => {
    const u = g.activeUnit;
    if (!u) return null;
    const total = u.engine === "llm"
      ? (fetchedArch?.nodes ?? []).reduce((a, n) => a + (n.params ?? 0), 0)
      : u.params;
    const fromArch = unitArch?.params_total;
    return { ...u, params: total || fromArch || u.params };
  }, [g.activeUnit, fetchedArch, unitArch]);

  const llmUnit = activeUnit?.engine === "llm" ? activeUnit : null;

  // What this box takes, from the unit's own catalogue entry. A unit with no
  // engine wired up still declares it, so the box can say so rather than vanish.
  const accepts = (activeUnit?.accepts ?? []) as string[];
  const inputHint = describeAccepts(accepts.length ? accepts : ["text"]);
  const notWired = !!activeUnit && !activeUnit.configured;
  const inputReason = !activeUnit
    ? "No engine is loaded, so nothing can run. Pick one in the rail."
    : notWired
      ? (activeUnit.problem || "this engine is not wired up yet")
      : llmUnit && !running
        ? "start the engine above to use it"
        : null;
  const inputDisabled = !activeUnit || notWired || (!!llmUnit && !running);


  // Whether the detail panel is showing the model itself. The scene stays the
  // overview either way; the architecture lives in the panel.
  const inModel = sel?.kind === "model" || sel?.kind === "node" || sel?.kind === "unit";

  // One submit, routed by what the model accepts. The box is uniform; what
  // happens when you press the button is the engine's business.
  const runInput = useCallback(async () => {
    const engine = activeUnit?.engine;
    if (!engine || inputDisabled) return;
    setTurnErr(null);
    if (engine === "tts") {
      setBusy(true);
      try {
        const r = await ttsSynthesize(prompt);
        setAudio((prev) => { if (prev) URL.revokeObjectURL(prev.url); return r; });
        setPicture(null);
      } catch (e) { setTurnErr(String((e as Error)?.message ?? e)); }
      finally { setBusy(false); setPrompt(""); }
      return;
    }
    if (engine === "video") {
      setBusy(true);
      try {
        const r = await videoGenerate(prompt, { frames: 33, steps: 8 });
        setClip((prev) => { if (prev) URL.revokeObjectURL(prev.url); return r; });
        setPicture(null);
      } catch (e) { setTurnErr(String((e as Error)?.message ?? e)); }
      finally { setBusy(false); }
      return;
    }
    if (engine === "image") {
      setBusy(true);
      try {
        const r = await imageGenerate(prompt);
        setPicture(r);
      } catch (e) { setTurnErr(String((e as Error)?.message ?? e)); }
      finally { setBusy(false); }
      return;
    }
    if (engine === "asr") return;            // the audio slot owns its own action
    await grow();
  }, [activeUnit, inputDisabled, prompt, grow]);

  const pick = (s: NonNullable<Sel>) => {
    onOpenDetails?.();
    setSel((cur) => {
      // clicking the model a second time folds the architecture back away
      if (s.kind === "model" && cur?.kind === "model") return null;
      return s;
    });
  };
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

  useEffect(() => () => abort.current?.abort(), []);

  // `api.models()` lists the GGUF files that exist *on disk*, which is not the
  // same as the model in play: with whisper.cpp loaded it still happily returned
  // the Qwen gguf, and the detail panel went on describing Qwen's layers. Only
  // ask for an architecture when a language model is actually the chosen unit,
  // and drop it the moment it is not.
  // Nothing to fetch when the unit brought its own architecture.
  const llmKey = unitArch ? null
    : llmUnit?.key ?? (activeUnit == null && running ? "llm:running" : null);
  useEffect(() => {
    let cancelled = false;
    if (!llmKey) { setArch(null); return; }
    api.models()
      .then((m) => {
        if (cancelled || !m.models?.length) return null;
        // Only an actual match will do. Falling back to "the last model in the
        // registry" described the wrong checkpoint with total confidence — a
        // vision model was being shown a text model's layers — so an unmatched
        // name yields no architecture rather than a borrowed one.
        const want = (server?.model ?? "").replace(/\.gguf$/i, "");
        const found = m.models.find((x) => x.model_id === want)
          ?? m.models.find((x) => want && (want.startsWith(x.model_id)
                                           || x.model_id.startsWith(want)));
        if (!found) {
          if (!cancelled) setArch(null);
          return null;
        }
        return api.modelArch(found.model_id)
          .then((g) => { if (!cancelled) setArch(toArch(g)); })
          .catch(() => {});
      })
      .catch(() => {});
    return () => { cancelled = true; };
    // `server` is derived from `chat`; depending on the model string is enough.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [llmKey, server?.model, running]);

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


  async function grow() {
    const q = prompt.trim() || "Tell me what a GPU is.";
    setPrompt(""); setBusy(true); setText(""); setThink(""); setLeaves([]);
    got.current = 0;
    const ac = new AbortController();
    abort.current = ac;
    await chatSend(
      { message: q, history: [], profile: false, images: pendingImage ? [pendingImage] : [] },
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
        } else if (kind === "error") setTurnErr(String(d.message ?? ""));
      },
      ac.signal,
    ).catch((e) => {
      if ((e as Error)?.name !== "AbortError") {
        // A long stream can be cut by the tunnel mid-answer. Whatever arrived is
        // already on screen, so say that rather than implying nothing happened.
        const partial = got.current;
        setTurnErr(partial > 0
          ? `connection dropped after ${partial} tokens — showing what arrived`
          : `chat failed: ${(e as Error)?.message ?? e}`);
      }
    }).finally(() => { abort.current = null; setBusy(false); setPendingImage(null); });
  }

  return (
    <div className={"index" + (sel ? " wide" : "")}>
      <div className="index-hero">
        <h1>Watch a model think.</h1>
        <p>
          A language model reads your words one at a time and predicts the next one,
          over and over, until it has an answer. That is called <b>inference</b>.
          Everything below is the path those words travel — read it from the top.
        </p>
      </div>

      <div className={"scene-wrap" + (sel ? " side-open" : "")}>
        <div className="scene-split" style={{ width: sel ? "100%" : undefined }}>
          <div className="scene-main">
            <div className="scene-title">
              {inModel
                ? "the host, with the engine and the model running on it"
                : "the host: four parts on the plate, with the engine and the model above"}
              <span className="scene-key">
                fill = how busy · hover the tower to lift it clear · click a slab to open it
              </span>
            </div>
            <HostScene hardware={hardware} hostTitle="HOST"
              arch={arch} sel={sel} unit={swap.phase === "evicting" ? null : activeUnit}
              onSelect={pick} onBackground={() => setSel(null)} />
          </div>
          <div className="scene-side">
            {sel ? (
              <SceneDetail sel={sel} arch={arch}
                modelName={llmUnit
                  ? (running ? (server?.model ?? arch?.model_id ?? "") : (arch?.model_id ?? ""))
                  // not a language model: name the unit that is actually up,
                  // or the panel titles whisper's weights with Qwen's filename
                  : (activeUnit?.model_label ?? "")}
                engineLabel={activeUnit?.engine_label}
                engineIsLlm={!!llmUnit}
                running={running} tokens={leaves.length}
                port={(() => {
                  const e = (g.status?.engines ?? []).find((x) => x.key === activeUnit?.engine);
                  return e?.port ? `127.0.0.1:${e.port}` : undefined;
                })()}
                onSelect={pick} onClose={() => setSel(null)} />
            ) : null}
          </div>
        </div>
      </div>

      {/* The engine and model picker used to sit here. It is in the left rail
       * now, so it stays reachable from every page instead of only this one. */}

      {/* Every model needs an input. What that input *is* depends on the
       * engine — words for a language model, audio for speech recognition, a
       * prompt for the image and video engines — so the row is always here
       * rather than appearing only for llama.cpp. It just says what it is
       * waiting for. */}
      <div className="pick-wrap">
        <div className="scene-title">
          input — every model needs something to run on
          <span className="scene-key">{inputHint}</span>
        </div>

        {llmUnit ? (
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
        ) : null}

        <InputBox
          accepts={accepts}
          engine={activeUnit?.engine ?? ""}
          disabled={inputDisabled}
          reason={inputReason}
          value={prompt}
          onChange={setPrompt}
          onSubmit={() => void runInput()}
          busy={busy}
          image={pendingImage}
          onImage={setPendingImage}
          audioSlot={<AsrTask loaded={resident === "asr"} embedded />}
        />

        <ResultOut audio={audio} image={picture} video={clip} />
      </div>

      {turnErr && <div className="notice bad mb">{turnErr}</div>}
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
