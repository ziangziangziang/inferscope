/**
 * Speech: transcribe an audio file.
 *
 * The engine is never started from here. Loading one evicts whatever else is
 * on the GPU, which is a consequence the user should choose deliberately from
 * the Engines screen — so this view tells them when it is not loaded and
 * points at the button that changes it.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { engines, asrTranscribe, type EngineStatus, type AsrResult } from "../lib/api";

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 2 ** 20) return `${(n / 1024).toFixed(0)} KiB`;
  return `${(n / 2 ** 20).toFixed(1)} MiB`;
}

export function Speech() {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [res, setRes] = useState<AsrResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const input = useRef<HTMLInputElement | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);

  const refresh = useCallback(() => {
    engines().then(setStatus).catch(() => {});
  }, []);
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [refresh]);

  const asr = status?.engines.find((e) => e.key === "asr");
  const loaded = status?.loaded === "asr";

  const run = async () => {
    if (!file) return;
    setBusy(true); setErr(null); setRes(null);
    try {
      setRes(await asrTranscribe(file));
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally { setBusy(false); }
  };

  const rtf = res && res.audio_s && res.audio_s > 0 ? res.audio_s / res.seconds : null;

  return (
    <div className="view">
      <div className="view-head">
        <h1>Speech to text</h1>
        <p>
          Whisper running on the GPU. Upload a clip and it comes back as text,
          with how long the audio was and how long the transcription took — so
          you can see the engine's real speed rather than a round-trip time.
        </p>
      </div>

      {!loaded ? (
        <div className="notice warn">
          The ASR engine is not loaded. Loading it offloads whatever is on the
          GPU right now — do that from <b>Engines</b>.
        </div>
      ) : null}

      <div className="sp-row">
        <input ref={input} type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
          onChange={(e) => { setFile(e.target.files?.[0] ?? null); setRes(null); setErr(null); }}
          aria-label="audio file" />
        <button className="btn primary" onClick={run} disabled={!file || busy || !loaded}>
          {busy ? "transcribing…" : "transcribe"}
        </button>
        {res ? <button className="btn" onClick={() => { setRes(null); setFile(null); if (input.current) input.current.value = ""; }}>clear</button> : null}
      </div>

      {file ? (
        <div className="sp-file">
          <b>{file.name}</b>
          <span className="faint">{fmtBytes(file.size)}</span>
          <audio ref={audio} controls src={URL.createObjectURL(file)} />
        </div>
      ) : null}

      {err ? <div className="notice bad">{err}</div> : null}

      {res ? (
        <div className="sp-result" data-testid="asr-result">
          <div className="sp-metrics">
            <span><b>{res.audio_s ?? "—"}</b>s audio</span>
            <span><b>{res.seconds.toFixed(3)}</b>s engine</span>
            {rtf ? <span><b>{rtf.toFixed(1)}×</b> realtime</span> : null}
            <span className="faint">{res.engine}</span>
          </div>
          <p className="sp-text">{res.text || <i className="faint">(nothing recognised)</i>}</p>
          <p className="faint" style={{ fontSize: 11 }}>
            timings measured on the host · {res.provenance}
          </p>
        </div>
      ) : null}

      {asr ? (
        <div className="sp-foot">
          <span className="faint">engine</span> <span className="mono">{asr.label}</span>
          <span className="faint">model</span> <span className="mono">{asr.model}</span>
        </div>
      ) : null}
    </div>
  );
}
