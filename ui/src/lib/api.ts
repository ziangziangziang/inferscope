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
  chatStartServer: (profile = false) => j<ChatServer>("/api/chat/server/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile }),
  }),
  chatStopServer: () => j<ChatServer & { stopped: boolean }>("/api/chat/server/stop", { method: "POST" }),
};

/** POST /api/chat/send and read the SSE reply as typed events. */
export async function chatSend(body: {
  message: string; history: { role: string; content: string }[];
  profile: boolean; session_id?: string; max_tokens?: number; temperature?: number;
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
