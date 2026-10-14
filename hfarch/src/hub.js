/**
 * Talking to the Hub, from the browser.
 *
 * A model's shape is declared in `config.json`, so no weights are ever fetched.
 * That is the easy half. The hard half is that "could not read the model" has
 * at least five causes with five different fixes, and the browser reports most
 * of them identically — as a bare `TypeError`, or as nothing at all:
 *
 *   unreachable   the host is not reachable from this network
 *   cors          the host answers, but refuses a cross-origin read
 *   timeout       packets are being dropped, so nothing answers either way
 *   gated         401/403, which is a permission, not a failure
 *   missing       the repo has no config.json where we looked
 *
 * Only by separating them can the app tell the reader something useful. So
 * every attempt is classified and recorded, and the failure carries its own
 * remedies rather than one line of text.
 *
 * Two of those causes are not the reader's fault and have a fix here:
 *
 *   - `hf-mirror.com` answers its CDN redirect with
 *     `Access-Control-Allow-Origin: https://hf-mirror.com`, so a page served
 *     from localhost gets nothing back. A no-cors request proves the network
 *     is fine and the refusal is CORS — that is what `classify()` does.
 *   - `serve.py`, which ships with this app, adds a same-origin route that
 *     reads the same file with urllib, where the network is fine.
 */
const HOSTS = ["https://huggingface.co", "https://hf-mirror.com"];
const FILES = ["config.json", "generation_config.json", "preprocessor_config.json"];

/** `owner/name`, or `owner/name/subfolder` for one component of a pipeline. */
const REPO = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/;

const DEADLINE = 3500;        // per hub request
const ORIGIN_DEADLINE = 20000; // the route does the read server-side

/** Sub-folders a diffusers pipeline is made of, for the "where else to look" hint. */
const COMPONENTS = ["unet", "transformer", "vae", "text_encoder", "text_encoder_2", "scheduler"];

export function splitRepo(repo) {
  const parts = String(repo || "").split("/");
  return { repo: parts.slice(0, 2).join("/"), sub: parts[2] || "" };
}

/**
 * A failure with an explanation attached.
 *
 * `attempts` is what was tried, in order, with the outcome; `fixes` is what
 * would actually get a diagram on screen. Both are shown.
 */
export class HubError extends Error {
  constructor(message, { kind = "network", title = "could not read this model",
    attempts = [], fixes = [] } = {}) {
    super(message);
    this.name = "HubError";
    this.kind = kind; this.title = title; this.attempts = attempts; this.fixes = fixes;
  }
}

async function timed(url, ms, init = {}) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  const t0 = Date.now();
  try {
    return await fetch(url, { redirect: "follow", ...init, signal: ctl.signal });
  } catch (e) {
    // Only two things throw out of fetch, and they mean different things:
    // AbortError is a deadline we set; anything else here is refused or
    // unreachable at the network or CORS layer, which a no-cors probe tells
    // apart afterwards.
    const err = new Error(e instanceof Error ? e.message : String(e));
    err.kind = e && e.name === "AbortError" ? "timeout" : "blocked";
    err.ms = Date.now() - t0;
    throw err;
  } finally {
    clearTimeout(t);
  }
}

/**
 * Why did that fetch fail?
 *
 * `fetch` itself will not say: a CORS refusal, a DNS failure and a dropped
 * connection all arrive as "Failed to fetch". A no-cors request has no CORS
 * check to fail, so if *it* succeeds the network is fine and the original
 * refusal was policy; if it fails too, the host is genuinely unreachable.
 */
async function classify(url) {
  try {
    await timed(url, 3000, { mode: "no-cors" });
    return "cors";
  } catch (e) {
    return e.kind === "timeout" ? "timeout" : "unreachable";
  }
}

const url = (host, repo, file) => {
  const { repo: r, sub } = splitRepo(repo);
  return `${host}/${r}/resolve/main/${sub ? sub + "/" : ""}${file}`;
};

/* ------------------------------------------------------------------ *
 * What this browser has learned about reaching the Hub.
 *
 * The Hub is blocked from here, and every load paying two full timeouts
 * before it discovers that again is most of the wait. One record per host,
 * consulted before anything is tried.
 * ------------------------------------------------------------------ */
const DIAG = "hfarch.reach.v1";

function readDiag() {
  try { return JSON.parse(localStorage.getItem(DIAG) || "{}"); } catch { return {}; }
}
function writeDiag(host, state) {
  try {
    const all = readDiag();
    all[host] = { state, at: Date.now() };
    localStorage.setItem(DIAG, JSON.stringify(all));
  } catch { /* the diagnosis is optional */ }
}

/** Try one Hub host for the whole metadata set. */
async function fromHost(host, repo, notes) {
  const out = { repo, files: {}, meta: null, host, via: "hub" };
  for (const f of FILES) {
    try {
      const t0 = Date.now();
      const r = await timed(url(host, repo, f), DEADLINE);
      if (!r.ok) {
        if (f === "config.json") {
          const body = (await r.text().catch(() => "")).slice(0, 160).trim();
          notes.push({ host, ms: Date.now() - t0, state: `HTTP ${r.status}`,
            note: r.status === 401 || r.status === 403
              ? (body.startsWith("{") ? body.slice(0, 90) : body || "no sign-in here")
              : "" });
          throw Object.assign(new Error(""), {
            kind: r.status === 401 || r.status === 403 ? "gated"
              : r.status === 429 ? "ratelimited" : "missing",
            status: r.status, ms: Date.now() - t0, host, detail: body,
          });
        }
        continue;                                  // the other two are optional
      }
      out.files[f] = await r.json();
    } catch (e) {
      if (e && (e.kind === "gated" || e.kind === "ratelimited" || e.kind === "missing")) throw e;
      if (f !== "config.json") continue;         // optional file, ignore
      const t0 = Date.now();
      const state = await classify(url(host, repo, f));
      notes.push({ host, ms: Date.now() - t0, state });
      throw Object.assign(new Error(""), { kind: state, host, ms: e.ms || Date.now() - t0 });
    }
  }
  if (!out.files["config.json"]) {
    throw Object.assign(new Error(""), { kind: "missing", host, ms: 0 });
  }
  writeDiag(host, "ok");
  try {
    const r = await timed(`${host}/api/models/${splitRepo(repo).repo}`, 2500);
    if (r.ok) out.meta = await r.json();
  } catch { /* metadata is a nicety */ }
  return out;
}

/**
 * Last resort: ask the host we were served from.
 *
 * `serve.py` reads the same three files with urllib, which reaches the Hub
 * from networks where the browser cannot. It is same-origin, so there is no
 * CORS check to fail.
 */
async function viaOrigin(repo, notes) {
  const t0 = Date.now();
  let r;
  try {
    r = await fetch(`/api/hf/config?repo=${encodeURIComponent(repo)}`,
      { signal: AbortSignal.timeout(ORIGIN_DEADLINE) });
  } catch (e) {
    notes.push({ host: "this server", ms: Date.now() - t0,
      state: e && e.name === "TimeoutError" ? "timeout" : "no answer" });
    throw Object.assign(new Error(""), { kind: "noanswer", ms: Date.now() - t0 });
  }
  const ms = Date.now() - t0;
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    if (r.status === 404) {
      notes.push({ host: "this server", ms, state: "no such route" });
      throw Object.assign(new Error(""), { kind: "noroute", ms });
    }
    notes.push({ host: "this server", ms, state: `HTTP ${r.status}` });
    const detail = String((body && body.detail) || "");
    const perHost = JSON.stringify((body && body.detail_by_host) || "");
    throw Object.assign(new Error(""), {
      kind: r.status === 400 ? "badname"
        : /gated/i.test(detail + perHost) ? "gated"
        : /429/.test(detail + perHost) ? "ratelimited" : "failed",
      detail, ms,
    });
  }
  notes.push({ host: "this server", ms, state: "ok", via: "origin" });
  return {
    repo, files: body.files || {}, meta: null,
    host: body.host || "this server", via: "origin",
  };
}

const say = {
  cors: "answered, then refused the cross-origin read",
  unreachable: "not reachable from this browser",
  timeout: "no answer before the deadline",
  missing: "no config.json at that address",
  gated: "repo is gated — sign in to the Hub",
  ratelimited: "rate limited (429)",
  noroute: "served by a plain static server: no /api/hf/config route",
  noanswer: "the server did not answer",
  badname: "that is not a model name",
  failed: "read failed",
};

export async function getModel(repo) {
  repo = String(repo || "").trim()
    .replace(/^https?:\/\/(huggingface\.co|hf-mirror\.com)\//, "")
    .replace(/\/+$/, "");
  if (!REPO.test(repo)) {
    throw new HubError("give a model name like Qwen/Qwen3-4B — or "
      + "owner/name/unet for one component of a pipeline",
      { kind: "badname", title: "that name is not readable",
        attempts: [{ host: "—", ms: 0, state: `does not match owner/name: "${repo}"` }],
        fixes: ["Qwen/Qwen3-4B", "openai/clip-vit-large-patch14",
          "facebook/DiT-XL-2-256/transformer"] });
  }

  const notes = [];
  const diag = readDiag();
  const blocked = HOSTS.every((h) => diag[h] && diag[h].state !== "ok");
  // Known-blocked: ask the same-origin route first, it is the only thing that
  // has ever worked from here. Otherwise the Hub is tried first, because it is
  // the honest source, and this order is remembered per host afterwards.
  const order = blocked ? ["origin", ...HOSTS] : [...HOSTS, "origin"];

  let originTried = false;
  let last = null;
  for (const host of order) {
    if (host === "origin") {
      if (originTried) continue;
      originTried = true;
      if (!location.protocol.startsWith("http")) continue;
      try {
        return await viaOrigin(repo, notes);
      } catch (e) {
        last = e;
        continue;
      }
    }
    try {
      return await fromHost(host, repo, notes);
    } catch (e) {
      last = e;
      writeDiag(host, e.kind);
    }
  }

  const attempts = notes.map((nt) => ({
    host: nt.host, ms: nt.ms,
    state: nt.state === "ok" ? "ok" : (say[nt.state] || nt.state),
    note: nt.note || "",
  }));
  const states = new Set(notes.map((nt) => nt.state).concat(last ? [last.kind] : []));

  if (states.has("gated")) {
    throw new HubError(`${repo} is gated. The Hub answers 401 or 403 for it, `
      + `and this viewer carries no token.`,
      { kind: "gated", title: "this repo is gated", attempts,
        fixes: ["open the repo in a signed-in tab and copy its config.json, then drop it here",
          "or name a repo that is public"] });
  }
  if (states.has("missing") && !states.has("cors") && !states.has("unreachable")) {
    const base = splitRepo(repo).repo;
    throw new HubError(`no config.json at ${repo}.`,
      { kind: "missing", title: "nothing to read at that address", attempts,
        fixes: ["a diffusers pipeline keeps each part in its own folder — try "
          + COMPONENTS.map((c) => `${base}/${c}`).join(", ")] });
  }
  if (states.has("badname")) {
    throw new HubError(last && last.detail ? last.detail : "that name was refused.",
      { kind: "badname", title: "not a readable name", attempts, fixes: [] });
  }

  const fixes = [];
  if (states.has("cors") || states.has("unreachable") || states.has("timeout")) {
    fixes.push("serve this directory with the bundled route — python3 serve.py 8811 "
      + "— which reads config.json server-side, where the network is fine");
  }
  if (states.has("noroute") || states.has("noanswer")) {
    fixes.push("you are on a plain static server; restart it as python3 serve.py 8811");
  }
  fixes.push("or drop this model's config.json anywhere on this page — everything "
    + "below is derived from that one file");
  if (last && last.detail) fixes.push(`detail: ${last.detail}`);

  throw new HubError(
    `${repo}: this browser could not read it. Nothing failed politely, so here is what was tried.`,
    { kind: (last && last.kind) || "network",
      title: "this browser cannot read the Hub", attempts, fixes });
}

/* ------------------------------------------------------------------ *
 * A cache of configs already read.
 *
 * A model's shape does not change, so re-reading config.json for a repo this
 * browser has already seen buys nothing but a wait -- and on a network like
 * this one the wait is the better part of a minute. Cached reads are instant.
 * ------------------------------------------------------------------ */
const KEY = "hfarch.configs.v1";
const LIMIT = 24;

export function cacheGet(repo) {
  try {
    const all = JSON.parse(localStorage.getItem(KEY) || "{}");
    const hit = all[repo];
    if (!hit) return null;
    return hit;                       // {files, host, at}
  } catch {
    return null;                      // private mode, quota, or corrupt JSON
  }
}

export function cachePut(repo, files, host) {
  try {
    const all = JSON.parse(localStorage.getItem(KEY) || "{}");
    all[repo] = { files, host, at: Date.now() };
    // keep the most recent, drop the rest
    const keep = Object.entries(all)
      .sort((a, b) => b[1].at - a[1].at).slice(0, LIMIT);
    localStorage.setItem(KEY, JSON.stringify(Object.fromEntries(keep)));
    return true;
  } catch {
    return false;                     // over quota: the cache is optional
  }
}

/** Repos this browser has drawn, newest first. */
export function recent() {
  try {
    const all = JSON.parse(localStorage.getItem(KEY) || "{}");
    return Object.entries(all).sort((a, b) => b[1].at - a[1].at).map(([k]) => k);
  } catch {
    return [];
  }
}

export function forget(repo) {
  try {
    const all = JSON.parse(localStorage.getItem(KEY) || "{}");
    delete all[repo];
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch { /* nothing to do */ }
}

export const age = (ms) => {
  const s = Math.max(0, (Date.now() - ms) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
};

export const fmt = (v) => {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2).replace(/\.?0+$/, "") + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
  return String(Math.round(v));
};
