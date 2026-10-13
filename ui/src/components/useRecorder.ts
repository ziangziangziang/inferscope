/**
 * Microphone capture, straight to the 16 kHz mono WAV the engine wants.
 *
 * The browser gives us Opus-in-WebM from MediaRecorder, and the engine speaks
 * WAV. Rather than add an ffmpeg dependency to the host for a step the browser
 * can already do, this decodes with the Web Audio API and resamples through an
 * OfflineAudioContext, then writes the WAV header by hand.
 *
 * Raw PCM is captured with a ScriptProcessor rather than MediaRecorder so the
 * codec is never in the path — what we upload is exactly what we captured.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const TARGET_RATE = 16000;

export type RecState = "idle" | "arming" | "recording" | "denied" | "unsupported" | "error";

export interface Recording {
  file: File;
  seconds: number;
  samples: number;
}

/** Encode mono float samples as a 16-bit PCM WAV. */
export function encodeWav(samples: Float32Array, rate: number): Blob {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const str = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  dv.setUint32(16, 16, true);          // PCM chunk size
  dv.setUint16(20, 1, true);           // format = PCM
  dv.setUint16(22, 1, true);           // channels
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * 2, true);    // byte rate
  dv.setUint16(32, 2, true);           // block align
  dv.setUint16(34, 16, true);          // bits per sample
  str(36, "data");
  dv.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    // clamp before scaling, or a hot sample wraps to a loud click
    const s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

/** Resample mono float audio to `TARGET_RATE` via the Web Audio API. */
async function resample(input: Float32Array, rate: number): Promise<Float32Array> {
  if (rate === TARGET_RATE) return input;
  const frames = Math.max(1, Math.round((input.length / rate) * TARGET_RATE));
  const Ctor: typeof OfflineAudioContext =
    (window as unknown as { OfflineAudioContext: typeof OfflineAudioContext }).OfflineAudioContext
    || (window as unknown as { webkitOfflineAudioContext: typeof OfflineAudioContext }).webkitOfflineAudioContext;
  if (!Ctor) return input;                       // give up quietly rather than throw
  const off = new Ctor(1, frames, TARGET_RATE);
  const buf = off.createBuffer(1, input.length, rate);
  // copyToChannel wants a plain-ArrayBuffer view; a generic Float32Array can be
  // backed by a SharedArrayBuffer, which it rejects.
  buf.copyToChannel(new Float32Array(input) as Float32Array<ArrayBuffer>, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

export function useRecorder() {
  const [state, setState] = useState<RecState>("idle");
  const [seconds, setSeconds] = useState(0);
  const [message, setMessage] = useState<string>("");
  const stream = useRef<MediaStream | null>(null);
  const ctx = useRef<AudioContext | null>(null);
  const chunks = useRef<Float32Array[]>([]);
  const ticker = useRef<number | null>(null);
  const startedAt = useRef(0);
  const stateRef = useRef<RecState>("idle");
  /* The rate the mic actually opened at. It has to be read while the context
   * is still open — stop() closes the context, and reading it afterwards fell
   * back to 16 kHz, which quietly mislabels a 48 kHz recording and plays it
   * back about three times too fast. */
  const rateRef = useRef(TARGET_RATE);

  const supported = typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof AudioContext !== "undefined";

  const cleanup = useCallback(() => {
    if (ticker.current) { clearInterval(ticker.current); ticker.current = null; }
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    void ctx.current?.close().catch(() => {});
    ctx.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);
  useEffect(() => { stateRef.current = state; }, [state]);

  const start = useCallback(async () => {
    setMessage("");
    if (!supported) { setState("unsupported"); setMessage("this browser cannot record audio"); return; }
    setState("arming");
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      stream.current = s;
      const ac = new AudioContext();
      ctx.current = ac;
      rateRef.current = ac.sampleRate;
      // A context created outside a user gesture starts suspended, and a
      // suspended context never fires onaudioprocess — the recording would
      // look like it ran and come back with no samples at all. resume() can
      // itself hang when the browser refuses, so never block the take on it.
      if (ac.state === "suspended") {
        await Promise.race([ac.resume(), new Promise((r) => setTimeout(r, 300))]);
      }
      chunks.current = [];
      const src = ac.createMediaStreamSource(s);
      // ScriptProcessor is deprecated but is the only node that hands back raw
      // samples synchronously in every browser we care about here; AudioWorklet
      // needs a separate module file this app does not ship.
      const proc = ac.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (e) => {
        chunks.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      src.connect(proc);
      proc.connect(ac.destination);
      startedAt.current = Date.now();
      setSeconds(0);
      ticker.current = window.setInterval(
        () => setSeconds((Date.now() - startedAt.current) / 1000), 200);
      setState("recording");
    } catch (e) {
      const err = e as Error;
      // Every one of these is a different thing a user can fix, so they get
      // different sentences. The default branch used to surface the raw
      // browser message, which on this box was literally "Not supported".
      const why: Record<string, [RecState, string]> = {
        NotAllowedError: ["denied", "microphone permission was refused"],
        PermissionDeniedError: ["denied", "microphone permission was refused"],
        SecurityError: ["denied", "microphone permission was refused"],
        NotFoundError: ["error", "no microphone was found"],
        DevicesNotFoundError: ["error", "no microphone was found"],
        NotReadableError: ["error", "the microphone is already in use elsewhere"],
        TrackStartError: ["error", "the microphone is already in use elsewhere"],
        NotSupportedError: ["unsupported", "this browser cannot record audio"],
        AbortError: ["error", "the microphone stopped unexpectedly"],
      };
      const hit = why[err?.name ?? ""];
      if (hit) {
        setState(hit[0]);
        setMessage(hit[1]);
      } else {
        setState("error");
        setMessage(`could not open the microphone (${err?.name || err?.message || "unknown"})`);
      }
      cleanup();
    }
  }, [supported, cleanup]);

  const stop = useCallback(async (): Promise<Recording | null> => {
    if (stateRef.current !== "recording") return null;
    const rate = rateRef.current;
    cleanup();
    setState("idle");
    const total = chunks.current.reduce((a, c) => a + c.length, 0);
    if (!total) { setMessage("nothing was captured"); return null; }
    const flat = new Float32Array(total);
    let at = 0;
    for (const c of chunks.current) { flat.set(c, at); at += c.length; }
    try {
      const out = await resample(flat, rate);
      const secs = out.length / TARGET_RATE;
      if (secs < 0.25) { setMessage("that was too short to transcribe"); return null; }
      return { file: new File([encodeWav(out, TARGET_RATE)], "recording.wav", { type: "audio/wav" }),
               seconds: secs, samples: out.length };
    } catch (e) {
      setMessage(`could not prepare the audio: ${(e as Error)?.message ?? e}`);
      return null;
    }
  }, [cleanup]);

  return { state, seconds, message, supported, start, stop, reset: () => { setMessage(""); setState("idle"); setSeconds(0); } };
}