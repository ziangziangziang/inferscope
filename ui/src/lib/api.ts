// Typed client for the InferScope API. Mirrors the backend responses so the
// compiler catches drift between the two.

export type Provenance =
  | "measured" | "derived" | "estimated" | "simulated" | "unavailable";

export interface Capability {
  name: string; available: boolean; detail: string; notes: string[];
  supports: Record<string, boolean>;
}

export interface Health {
  status: string; version: string; schema_version: number; bind: string;
  public_url?: string | null;
  live_session: string | null; demo_session: string | null;
  collectors: { system: { available: boolean }; gpu: { backend: string; available: boolean } };
  adapters: Record<string, { available: boolean; detail?: string; state?: string; metrics?: Record<string, unknown> }>;
  capabilities: Record<string, Capability>;
}

export interface Access { public_url: string | null; reachable: boolean; how: string; source?: string; }

export interface Session {
  session_id: string; kind: string; engine: string; model_id: string;
  started_ns: number; ended_ns: number | null; source: string; notes: string;
  // What the session actually holds — so the picker can tell a profiled run
  // from an empty one instead of showing a column of identical ids.
  event_count: number; kernel_events: number; sample_count: number;
  gpu_ms: number; span_s: number | null; top_op: string | null; has_events: boolean;
}

export interface SampleRow {
  session_id: string; ts_wall_ns: number; ts_mono_ns: number;
  provenance?: string; source?: string;
  gpu_util_pct?: number | null; mem_used_pct?: number | null;
  vram_used_bytes?: number | null; vram_total_bytes?: number | null;
  power_w?: number | null; temp_c?: number | null;
  sclk_mhz?: number | null; mclk_mhz?: number | null;
  cpu_util_pct?: number | null; load1?: number | null; load5?: number | null; load15?: number | null;
  ram_used_bytes?: number | null; ram_total_bytes?: number | null; proc_rss_bytes?: number | null;
  active_requests?: number | null; queue_depth?: number | null; tokens_per_s?: number | null;
  per_core?: number[] | null; reason?: string | null;
  engine_metrics?: Record<string, Record<string, unknown>>;
  [k: string]: unknown;
}

export interface Overview {
  session_id: string; session: Session | null; latest: SampleRow | null;
  sample_count: number; adapter_health: Record<string, unknown>;
}

export interface Tensor { shape?: (number | string)[]; dtype?: string; bytes?: number; provenance?: string; }

export interface GraphNode {
  id: string; label: string; kind: string; column: number; order: number;
  branch?: string;
  group?: string; provenance: string; dims?: Record<string, unknown>;
  notes?: string[] | null; op?: string;
  params?: number; flops?: number;
  param_share?: number; flop_share?: number;
  repeat?: { count: number; template: { nodes: GraphNode[]; edges: GraphEdge[] } };
}
export interface GraphEdge {
  from: string; to: string; provenance: string; label?: string; tensor?: Tensor;
}
export interface Graph {
  model_id: string; family: string; provenance: string; source: string;
  meta: Record<string, unknown>; nodes: GraphNode[]; edges: GraphEdge[];
  costs?: GraphCosts;
  _provenance?: string; _source?: string;
}

export interface GraphCosts {
  ctx?: number;
  per_layer?: { params: number; flops: number };
  totals?: {
    params: number; flops_per_token: number; params_per_layer: number;
    layers: number; ctx: number; attention?: string;
    vision_params?: number; params_per_token?: number;
  };
}

export interface TraceEvent {
  kind: string; source: string; provenance: string; request_id?: string | null;
  step_idx?: number | null; component?: string | null; op_id?: string | null;
  ts_wall_ns: number; duration_us?: number | null; bytes_moved?: number | null;
  extra?: Record<string, unknown>; [k: string]: unknown;
}

export interface KVResult {
  bytes: number; gib: number; baseline_bytes: number; breakdown: Record<string, number>;
  notes: string[]; formula: string; provenance: string;
}

export interface ModelInfo { model_id: string; path: string; family: string; architectures: string[]; }

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${await r.text().catch(() => "")}`.slice(0, 200));
  return r.json() as Promise<T>;
}

export interface ChatServer {
  managed?: boolean; running?: boolean; pid?: number | null;
  base_url?: string; model_path?: string; model?: string; ctx?: number;
  uptime_s?: number | null; log_path?: string; error?: string | null;
  kernel_trace?: string; ready?: boolean; profiling?: boolean;
  trace_out_dir?: string | null;
}
export interface ChatState {
  managed_mode: boolean; base_url: string | null; ready: boolean; detail: string;
  server: ChatServer; profiling_available: boolean; per_turn_trace: boolean;
  per_turn_trace_reason: string | null; profile_default: boolean;
  max_tokens: number; model_path: string | null; server_bin: string | null; note: string;
}
export interface TurnResult {
  text: string; ttft_ms: number | null; total_ms: number;
  prompt_tokens: number | null; completion_tokens: number | null;
  tok_per_s: number | null; model: string; finish_reason: string;
  error: string | null; kernel_events: number; capture: Record<string, unknown>;
  request_id?: string; session_id?: string;
}

export const engines = () => j<EngineStatus>("/api/engines");
const post = <T,>(url: string, body?: unknown) =>
  j<T>(url, { method: "POST",
           ...(body ? { headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body) } : {}) });

/** `model` is the unit key, so an engine with several models starts the one
 *  that was actually picked rather than the configured default. */
export const engineLoad = (key: string, model?: string) =>
  post<LoadResult>(`/api/engines/${encodeURIComponent(key)}/load`,
                   model ? { model } : undefined);
export const engineOffload = () => post<LoadResult>("/api/engines/offload");

export interface TtsResult {
  url: string;           // object URL for the returned WAV
  bytes: number;
  engine_s: number;
  sample_rate: number;
}

/** POST /api/tts/synthesize and hand back something playable. */
export async function ttsSynthesize(text: string): Promise<TtsResult> {
  const r = await fetch("/api/tts/synthesize", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error((await r.text().catch(() => "")).slice(0, 200) || "tts failed");
  const blob = await r.blob();
  return {
    url: URL.createObjectURL(blob),
    bytes: blob.size,
    engine_s: Number(r.headers.get("X-Engine-Seconds") ?? 0),
    sample_rate: Number(r.headers.get("X-Sample-Rate") ?? 0),
  };
}

export interface ImageResult {
  image: string;         // data URL
  engine_s: number;
  steps: number;
  prompt: string;
}

/** POST /api/image/generate and return the rendered picture. */
export async function imageGenerate(prompt: string, steps = 4): Promise<ImageResult> {
  const r = await fetch("/api/image/generate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, steps }),
  });
  if (!r.ok) throw new Error((await r.text().catch(() => "")).slice(0, 200) || "image failed");
  return r.json();
}

export interface VideoResult {
  url: string;           // object URL for the returned GIF
  bytes: number;
  engine_s: number;
  frames: number;
  size: string;          // e.g. "480x320"
}

/** POST /api/video/generate and hand back something playable. */
export async function videoGenerate(prompt: string,
                                    opts: { frames?: number; steps?: number } = {}): Promise<VideoResult> {
  const r = await fetch("/api/video/generate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, ...opts }),
  });
  if (!r.ok) throw new Error((await r.text().catch(() => "")).slice(0, 200) || "video failed");
  const blob = await r.blob();
  return {
    url: URL.createObjectURL(blob),
    bytes: blob.size,
    engine_s: Number(r.headers.get("X-Engine-Seconds") ?? 0),
    frames: Number(r.headers.get("X-Frames") ?? 0),
    size: r.headers.get("X-Size") ?? "",
  };
}

export async function asrTranscribe(file: File): Promise<AsrResult> {
  const fd = new FormData();
  fd.append("file", file, file.name);
  const r = await fetch("/api/asr/transcribe", { method: "POST", body: fd });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body?.detail || `transcription failed (${r.status})`);
  // A 200 that is not a transcript happens when something in front of the app
  // answers instead (a proxy, the SPA fallback, an empty body). Rendering it as
  // a result crashed the view on `seconds.toFixed`; say so instead.
  if (typeof body?.text !== "string" || typeof body?.seconds !== "number") {
    throw new Error("the backend did not return a transcript");
  }
  return body as AsrResult;
}

export const api = {
  health: () => j<Health>("/api/health"),
  access: () => j<Access>("/api/access"),
  capabilities: () => j<Record<string, Capability>>("/api/capabilities"),
  sessions: () => j<{ sessions: Session[]; live_session: string | null; demo_session: string | null; best_session: string | null }>("/api/sessions"),
  overview: (sid: string) => j<Overview>(`/api/sessions/${encodeURIComponent(sid)}/overview`),
  samples: (sid: string, limit = 600) => j<{ samples: SampleRow[] }>(`/api/sessions/${encodeURIComponent(sid)}/samples?limit=${limit}`),
  events: (sid: string, kind?: string, limit = 4000) =>
    j<{ events: TraceEvent[]; count: number }>(`/api/sessions/${encodeURIComponent(sid)}/events?limit=${limit}${kind ? `&kind=${kind}` : ""}`),
  timeline: (sid: string) => j<{ lanes: Record<string, TraceEvent[]>; total: number; returned: number; downsampled?: boolean }>(`/api/sessions/${encodeURIComponent(sid)}/timeline`),
  opclasses: (sid: string) => j<{ by_class: { op_class: string; launches: number; total_us: number; pct: number }[]; total_us: number; provenance: string }>(`/api/sessions/${encodeURIComponent(sid)}/opclasses`),
  sessionArch: (sid: string) => j<Graph>(`/api/sessions/${encodeURIComponent(sid)}/architecture`),
  models: () => j<{ models: ModelInfo[] }>("/api/models"),
  modelArch: (id: string) => j<Graph>(`/api/models/${encodeURIComponent(id)}/architecture`),
  kv: (p: Record<string, string | number>) => j<KVResult>("/api/kv?" + new URLSearchParams(
    Object.fromEntries(Object.entries(p).map(([k, v]) => [k, String(v)]))),
  ),
  experiments: () => j<{ experiments: unknown[] }>("/api/experiments"),
  createExperiment: (p: { name: string; kind: string; baseline: string; candidate: string; note: string }) =>
    j<{ id: number }>("/api/experiments", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p),
    }),
  chatState: () => j<ChatState>("/api/chat/state"),
  chatStartServer: (profile = false, model?: string) => j<ChatServer>("/api/chat/server/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile, ...(model ? { model } : {}) }),
  }),
  chatStopServer: () => j<ChatServer & { stopped: boolean }>("/api/chat/server/stop", { method: "POST" }),
};

/** POST /api/chat/send and read the SSE reply as typed events. */
export async function chatSend(body: {
  message: string; history: { role: string; content: string }[];
  profile: boolean; session_id?: string; max_tokens?: number; temperature?: number;
  /** data: URLs, forwarded to llama.cpp as multimodal content parts */
  images?: string[];
}, onEvent: (kind: string, data: unknown) => void,
  signal?: AbortSignal): Promise<void> {
  const res = await fetch("/api/chat/send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = `chat failed (${res.status})`;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON error */ }
    onEvent("error", { message: detail });
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    // SSE frames are separated by a blank line.
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let kind = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) kind = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      try { onEvent(kind, JSON.parse(dataLines.join("\n"))); } catch { /* partial frame */ }
    }
  }
}

/* ---- engine load/offload (speech, image, video) ---- */

export interface EngineInfo {
  key: string; label: string; family: string; purpose: string;
  port: number; binary: string; model: string;
  configured: boolean; problem: string; ready: boolean;
  exclusive: boolean; extra: Record<string, string>; hint: string;
}

export interface EngineStatus {
  loaded: string | null;
  pid: number | null;
  uptime_s: number;
  log_path: string;
  engines: EngineInfo[];
}

export interface LoadResult {
  ok?: boolean; engine?: string; evicted?: string | null;
  error: string | null; already?: boolean;
  /** offload replies with stopped/detail instead of ok */
  stopped?: boolean; detail?: string;
}

export interface AsrResult {
  text: string; filename: string; bytes: number;
  seconds: number; engine: string; model: string;
  audio_s: number | null; provenance: string;
}

/* ---- the unit catalogue: engine + model pairs ---- */

export interface UnitPart { key: string; label: string; weight: number }

export interface UnitInfo {
  key: string; engine: string; engine_label: string; model_label: string;
  model_path: string; port: number; binary: string;
  size_bytes: number; params: number; parts: UnitPart[];
  note: string; problem: string; configured: boolean; managed: boolean;
  /** what this unit accepts as input: text, image, audio... */
  accepts?: string[];
  /** vision projector beside the weights, when there is one */
  mmproj?: string | null;
  /** a graph built by the backend for engines with no config to read */
  arch?: { nodes: unknown[]; edges: unknown[]; params_total: number } | null;
}

export interface UnitCatalogue { units: UnitInfo[]; resident: string | null }

export const units = () => j<UnitCatalogue>("/api/units");
