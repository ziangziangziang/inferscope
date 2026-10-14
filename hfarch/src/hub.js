/**
 * Talking to the Hub, from the browser.
 *
 * No weights are fetched. A repo's shape is declared in `config.json`, and the
 * Hub serves it with permissive CORS, so the whole thing runs client-side. The
 * mirrors are tried in turn because the CDN that backs a given host is not
 * reachable from everywhere.
 */
const HOSTS = [
  "https://huggingface.co",
  "https://hf-mirror.com",
];

/**
 * Last resort: ask the host we were served from.
 *
 * A browser often cannot reach the Hub at all — behind a corporate proxy, or
 * on a network where the CDN redirect drops its CORS headers — and neither can
 * the server, which needs its own egress proxy. So the deployed copy falls back
 * to a narrow same-origin route that only ever reads public model metadata.
 */
async function viaOrigin(repo) {
  const r = await fetch(`/api/hf/config?repo=${encodeURIComponent(repo)}`,
    { signal: AbortSignal.timeout(25000) });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${repo}: not readable`);
  return { repo, files: body.files, meta: null, host: body.host || "this server" };
}

const FILES = ["config.json", "generation_config.json", "preprocessor_config.json"];

/**
 * Fetch with a deadline. Without one, a host that silently drops packets —
 * which is what a blocked `huggingface.co` does — leaves the drawing waiting
 * on a connection that will never answer, instead of falling through to a
 * mirror that would have worked in half a second.
 */
async function timed(url, ms = 8000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  try {
    return await fetch(url, { redirect: "follow", signal: ctl.signal });
  } finally {
    clearTimeout(t);
  }
}

export async function getModel(repo) {
  repo = String(repo || "").trim().replace(/^https?:\/\/huggingface\.co\//, "")
    .replace(/\/+$/, "");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    throw new Error("give a model name like Qwen/Qwen3-4B");
  }

  let hostErr = null;
  for (const host of HOSTS) {
    try {
      const out = { repo, files: {}, meta: null, host };
      for (const f of FILES) {
        try {
          const r = await timed(`${host}/${repo}/resolve/main/${f}`, 5000);
          if (!r.ok) {
            // a gate or a missing file often explains itself in plain text
            if (f === "config.json" && (r.status === 401 || r.status === 403)) {
              const why = (await r.text()).slice(0, 120).trim();
              if (why && why[0] === "{") out.files[f] = JSON.parse(why);
              else throw new Error(`${repo} is gated — ${why || "sign in to the Hub"}`);
            }
            continue;
          }
          out.files[f] = await r.json();
        } catch (e) {
          if (f === "config.json" && /gated/.test(e.message)) throw e;
        }
      }
      if (!out.files["config.json"]) throw new Error("config.json not readable");
      try {
        const r = await timed(`${host}/api/models/${repo}`, 4000);
        if (r.ok) out.meta = await r.json();
      } catch { /* metadata is a nicety */ }
      return out;
    } catch (e) {
      hostErr = e;
    }
  }
  // no Hub reachable from here — try the origin that served this page
  if (location.protocol.startsWith("http")) {
    try {
      return await viaOrigin(repo);
    } catch (e) {
      hostErr = e;
    }
  }
  throw new Error(`${repo}: ${hostErr ? hostErr.message : "could not be read"}`);
}

/* ------------------------------------------------------------------ *
 * A cache of configs already read.
 *
 * A model's shape does not change, so re-reading config.json for a repo
 * this browser has already seen buys nothing but a wait -- and the wait
 * is not short, because an unreachable Hub costs a full timeout per
 * host before the origin fallback. Cached reads are instant.
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