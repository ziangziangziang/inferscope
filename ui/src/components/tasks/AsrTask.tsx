/**
 * Speech-to-text task: a file, or the microphone.
 *
 * Recorded audio is resampled to 16 kHz mono WAV in the browser (see
 * `useRecorder`) so the engine gets what it can actually read, rather than
 * adding a server-side ffmpeg dependency for a container format the GPU path
 * does not need.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { asrTranscribe, type AsrResult } from "../../lib/api";
import { useRecorder } from "../useRecorder";

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 2 ** 20) return `${(n / 1024).toFixed(0)} KiB`;
  return `${(n / 2 ** 20).toFixed(1)} MiB`;
}

export function AsrTask({ loaded }: { loaded: boolean }) {
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState<"file" | "mic">("file");
  const [res, setRes] = useState<AsrResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const input = useRef<HTMLInputElement | null>(null);
  const rec = useRecorder();

  const run = useCallback(async (f: File) => {
    setBusy(true); setErr(null); setRes(null);
    try {
      setRes(await asrTranscribe(f));
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    } finally { setBusy(false); }
  }, []);

  // recording stops -> transcribe straight away
  const stopAndRun = useCallback(async () => {
    const r = await rec.stop();
    if (r) {
      setSource("mic");
      setFile(r.file);
      await run(r.file);
    }
  }, [rec, run]);

  // Release the microphone if the page goes away mid-take. This must key on
  // rec.stop (stable) and not on rec.state or rec: the hook returns a fresh
  // object every render, and the seconds ticker re-renders twice a second, so a
  // cleanup keyed on either of those tore the recording down ~200ms in — which
  // is what made it look like the microphone quit immediately.
  useEffect(() => () => { void rec.stop(); }, [rec.stop]);

  const rtf = res && res.audio_s && res.audio_s > 0 ? res.audio_s / res.seconds : null;

  return (
    <div className="task" data-testid="asr-task">
      <div className="task-head">
        <div>
          <h3>Speech to text</h3>
          <p className="task-sub">Whisper on the GPU. Transcribes speech to text.</p>
        </div>
        <div className="task-sources" role="tablist" aria-label="audio source">
          <button role="tab" aria-selected={source === "file"}
            className={"chip" + (source === "file" ? " on" : "")}
            onClick={() => setSource("file")}>file</button>
          <button role="tab" aria-selected={source === "mic"}
            className={"chip" + (source === "mic" ? " on" : "")}
            onClick={() => setSource("mic")}>microphone</button>
        </div>
      </div>

      {!loaded ? (
        <div className="notice warn">Load the ASR engine to use this — it offloads whatever else is on the GPU.</div>
      ) : null}

      {source === "file" ? (
        <div className="sp-row">
          <input ref={input} type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
            aria-label="audio file" disabled={!loaded}
            onChange={(e) => { setFile(e.target.files?.[0] ?? null); setRes(null); setErr(null); }} />
          <button className="btn primary" onClick={() => file && run(file)}
            disabled={!file || busy || !loaded}>
            {busy ? "transcribing…" : "transcribe"}
          </button>
        </div>
      ) : (
        <div className="sp-mic">
          {rec.state !== "recording" ? (
            <button className="btn primary" onClick={() => void rec.start()} disabled={!loaded}>
              ● record
            </button>
          ) : (
            <>
              <span className="rec-dot" aria-hidden>●</span>
              <b className="mono">{rec.seconds.toFixed(1)}s</b>
              <button className="btn" onClick={() => void stopAndRun()}>■ stop and transcribe</button>
            </>
          )}
          {!rec.supported ? (
            <span className="faint">this browser has no microphone API</span>
          ) : null}
        </div>
      )}

      {rec.message ? <p className="sp-hint warn">{rec.message}</p> : null}

      {file ? (
        <div className="sp-file">
          <b>{file.name}</b>
          <span className="faint">{fmtBytes(file.size)}</span>
          {source === "file" ? (
            <audio controls src={URL.createObjectURL(file)} />
          ) : (
            <span className="faint">recorded in the browser, resampled to 16 kHz mono</span>
          )}
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
            {source === "mic" ? <span className="faint">from microphone</span> : null}
          </div>
          <p className="sp-text">{res.text || <i className="faint">(nothing recognised)</i>}</p>
          <p className="faint" style={{ fontSize: 11 }}>timings measured on the host · {res.provenance}</p>
        </div>
      ) : null}
    </div>
  );
}