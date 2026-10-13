/**
 * What the model just produced.
 *
 * Every model answers in a different shape — words, a waveform, a picture — so
 * this renders whichever came back instead of each engine growing its own
 * result panel. Timings are labelled by where they were measured: the browser's
 * own numbers are ours, the engine's come back over the wire.
 */
import type { ImageResult, TtsResult, VideoResult } from "../lib/api";

function secs(v: number) {
  return v >= 1 ? `${v.toFixed(1)}s` : `${Math.round(v * 1000)}ms`;
}

export function ResultOut({ audio, image, video, rtfNote }:
  { audio?: TtsResult | null; image?: ImageResult | null;
    video?: VideoResult | null; rtfNote?: string }) {
  if (video) {
    return (
      <div className="out" data-testid="video-result">
        <img src={video.url} alt="generated clip" className="out-clip" />
        <p className="out-meta">
          {video.frames} frames{" · "}{video.size}
          {" · "}{(video.bytes / 1024 / 1024).toFixed(1)} MiB
          {" · "}{secs(video.engine_s)} in the engine
        </p>
      </div>
    );
  }
  if (audio) {
    return (
      <div className="out" data-testid="tts-result">
        <audio controls src={audio.url} />
        <p className="out-meta">
          {secs(audio.bytes / 2 / (audio.sample_rate || 22050))} of speech
          {" · "}{(audio.bytes / 1024).toFixed(0)} KiB
          {" · "}{secs(audio.engine_s)} in the engine
          {rtfNote ? <>{" · "}{rtfNote}</> : null}
        </p>
      </div>
    );
  }
  if (image) {
    return (
      <div className="out" data-testid="image-result">
        <img src={image.image} alt={image.prompt} />
        <p className="out-meta">
          {image.steps} steps{" · "}{secs(image.engine_s)} in the engine
        </p>
      </div>
    );
  }
  return null;
}