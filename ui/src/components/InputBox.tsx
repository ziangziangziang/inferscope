/**
 * One input box, whatever the model takes.
 *
 * Every model needs an input, so the box is always in the same place with the
 * same shape. What changes is the *format* it accepts, taken from the unit's
 * own catalogue entry rather than from the engine's name:
 *
 *   text        a prompt box
 *   text+image  the same box, plus a picture attach — only offered when a
 *               vision projector is actually installed beside the weights
 *   audio       a file picker or the microphone
 *
 * A model whose engine is not wired up yet still shows the box, disabled, with
 * the reason. Silently hiding it is what made this page feel broken before.
 */
import { useRef, useState } from "react";

export type Accepts = string[];

/** One-line description of what this box is waiting for. */
export function describeAccepts(a: Accepts): string {
  if (a.includes("image")) return "words and a picture";
  if (a.includes("audio")) return "audio";
  return "words";
}

/** The verb on the submit button, so it is not always "send". */
export function actionLabel(engine: string): string {
  switch (engine) {
    case "asr": return "transcribe";
    case "tts": return "speak";
    case "image": return "draw";
    case "video": return "animate";
    default: return "send";
  }
}

/** Tiny inline preview, so an attached picture is visibly the thing you picked. */
function Thumb({ url, onClear }: { url: string; onClear: () => void }) {
  return (
    <span className="in-thumb">
      <img src={url} alt="attached" />
      <button className="in-thumb-x" onClick={onClear} title="remove">×</button>
    </span>
  );
}

export interface InputBoxProps {
  accepts: Accepts;
  engine: string;
  disabled?: boolean;
  /** why it is disabled, shown under the box */
  reason?: string | null;
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  busy?: boolean;
  submitLabel?: string;
  placeholder?: string;
  /** rendered inside the box when the accepted format is audio */
  audioSlot?: React.ReactNode;
  /** the attached picture, held by the caller so it can be sent with the turn */
  image?: string | null;
  onImage?: (v: string | null) => void;
}

const MAX_IMAGE_BYTES = 8 << 20;

export function InputBox({
  accepts, engine, disabled, reason, value, onChange, onSubmit, busy,
  submitLabel, placeholder, audioSlot, image = null, onImage,
}: InputBoxProps) {
  const [problem, setProblem] = useState<string | null>(null);
  const setImage = onImage ?? (() => {});
  const file = useRef<HTMLInputElement>(null);
  const takesImage = accepts.includes("image");
  const takesAudio = accepts.includes("audio");
  const canSubmit = !disabled && !busy && (
    takesAudio ? Boolean(audioSlot) : (value.trim().length > 0 || (takesImage && image))
  );

  const pickImage = async (f: File | null | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) { setProblem("that is not an image"); return; }
    if (f.size > MAX_IMAGE_BYTES) { setProblem("image is larger than 8 MB"); return; }
    const reader = new FileReader();
    reader.onload = () => { setImage(String(reader.result)); setProblem(null); };
    reader.readAsDataURL(f);
  };

  return (
    <div className="input-box">
      {takesImage && image ? <Thumb url={image} onClear={() => setImage(null)} /> : null}

      {takesAudio ? (
        <div className="in-audio">{audioSlot}</div>
      ) : (
        <input
          className="in-text"
          value={value}
          onChange={(e) => { onChange(e.target.value); if (problem) setProblem(null); }}
          onKeyDown={(e) => { if (e.key === "Enter" && canSubmit) onSubmit(); }}
          placeholder={placeholder ?? "type here"}
          disabled={disabled}
          aria-label="model input"
        />
      )}

      {takesImage ? (
        <>
          <input ref={file} type="file" accept="image/*" hidden
            onChange={(e) => { void pickImage(e.target.files?.[0]); e.target.value = ""; }} />
          <button className="btn" title="attach a picture"
            onClick={() => file.current?.click()} disabled={disabled}>
            ▣ picture
          </button>
        </>
      ) : null}

      {/* For audio the action lives with the audio control itself (pick a file,
          or record, then transcribe) — two submit buttons would be a trap. */}
      {takesAudio ? null : (
        <button className="btn primary" onClick={onSubmit} disabled={!canSubmit}>
          {busy ? "working…" : submitLabel ?? actionLabel(engine)}
        </button>
      )}

      {problem ? <p className="sp-hint warn in-problem">{problem}</p> : null}
      {reason && !problem ? <p className="sp-hint in-reason">{reason}</p> : null}
    </div>
  );
}