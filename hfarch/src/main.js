import { HubError, age, cacheGet, cachePut, fmt, getModel, recent } from "./hub.js";
import { build } from "./graph.js";
import { render } from "./render.js";

const MODE_KEY = "hfarch.mode";
const mode = () => localStorage.getItem(MODE_KEY) === "3d" ? "3d" : "2d";

const $ = (id) => document.getElementById(id);

/**
 * The status bar says what is happening, because the wait is not short: the
 * first fetch to an unreachable host costs a full timeout before the mirror is
 * even tried, and with nothing on screen that reads as a broken page rather
 * than a slow one.
 */
const t0 = performance.now();
let step = 0;
function status(state, what, steps = [], meta = "") {
  $("status").dataset.state = state;
  $("stWhat").textContent = what;
  $("stSteps").textContent = steps.map((s, i) =>
    `${s} ${i < step ? "✓" : i === step ? "…" : "·"}`).join("  ");
  $("stMeta").innerHTML = meta;
}
const elapsed = () => `${Math.round(performance.now() - t0)} ms`;
const svg = $("svg"), err = $("err"), side = $("side"), detail = $("detail"),
  hint = $("hint"), why = $("why");

// the arrowhead the renderer points at
svg.insertAdjacentHTML("afterbegin",
  `<defs><marker id="head" viewBox="0 0 10 10" refX="8.6" refY="5"
     markerWidth="4.2" markerHeight="4.2" orient="auto-start-reverse">
     <path d="M 0 2 L 9 5 L 0 8 z" fill="#8fa787"/></marker></defs>`);

function fail(e) {
  err.hidden = false;
  err.textContent = e && e.message ? e.message : String(e);
  setTimeout(() => { err.hidden = true; }, 6000);
}

/**
 * A failure that explains itself.
 *
 * "Failed to fetch" is not an explanation, and the same five words cover a
 * blocked network, a CORS refusal and a typo. So what was tried, in what
 * order, with what outcome, is shown next to what would fix it. The panel
 * stays up: six seconds is not long enough to read a remedy you have to act
 * on.
 */
function explain(e) {
  why.hidden = false;
  hint.hidden = true;
  const title = (e && e.title) || "could not draw this model";
  const attempts = (e && e.attempts) || [];
  const fixes = (e && e.fixes) || [];
  const rows = attempts.length
    ? attempts.map((a) => `<li><b>${a.host}</b><span>${a.state}</span>`
      + `<em>${a.ms ? a.ms + " ms" : ""}</em></li>`).join("")
    : `<li><b>—</b><span>${(e && e.message) || String(e)}</span></li>`;
  why.innerHTML = `
    <h3>${title}</h3>
    ${attempts.length ? `<ul class="why-list">${rows}</ul>` : `<p>${e.message}</p>`}
    ${fixes.length ? `<h4>what would work</h4><ol class="why-fix">`
      + fixes.map((f) => `<li>${f}</li>`).join("") + `</ol>` : ""}`;
  why.scrollTop = 0;
  $("stSteps").textContent = "";
}

function show(node) {
  side.hidden = false;
  const m = last ? last.meta : {};
  const rows = [];
  const kv = (k, v) => rows.push(`<b>${k}</b><span>${v}</span>`);
  if (node.params) kv("parameters", fmt(node.params));
  for (const [k, v] of Object.entries(node.dims || {})) kv(k, String(v));
  if (node.repeat) kv("repeated", `${node.repeat}×`);
  const sub = (node.template || []).map((t) => `
    <div class="sub"><i style="background:${t.kind === "attention" ? "#d3e0ee"
      : t.kind === "mlp" ? "#e0e4d0" : "#e4e8dd"}"></i>${t.label}
      <em>${fmt(t.params)}</em></div>`).join("");

  detail.innerHTML = `
    <h2>${node.label}</h2>
    <div class="role">${node.kind}</div>
    ${node.note ? `<p>${node.note}</p>` : ""}
    ${rows.length ? `<div class="kv">${rows.join("")}</div>` : ""}
    ${sub ? `<div class="role" style="margin-top:16px">one block</div>${sub}` : ""}
    ${m.layers ? `<div class="role" style="margin-top:16px">model</div>
      <div class="kv">
        <b>layers</b><span>${m.layers}</span>
        <b>width</b><span>${m.hidden}</span>
        <b>heads</b><span>${m.heads} q / ${m.kv_heads} kv</span>
        <b>ffn</b><span>${m.ffn}${m.gated_ffn ? " (gated)" : ""}</span>
        <b>vocab</b><span>${m.vocab ?? "—"}</span>
        <b>position</b><span>${m.position || "—"}</span>
      </div>` : ""}
    ${node.dims2 ? `<div class="kv">
      <b>q proj</b><span>${fmt(node.dims2.q)}</span>
      <b>k proj</b><span>${fmt(node.dims2.k)}</span>
      <b>v proj</b><span>${fmt(node.dims2.v)}</span>
      <b>o proj</b><span>${fmt(node.dims2.o)}</span></div>` : ""}`;
}

let last = null;

function fillRecent() {
  const list = recent();
  $("recent").replaceChildren(...list.map((r) => {
    const o = document.createElement("option");
    o.value = r;
    return o;
  }));
  $("drops").textContent = list.length
    ? `${list.length} saved locally` : "";
}

async function load(repo) {
  hint.hidden = true;
  err.hidden = true;
  why.hidden = true;
  step = 0;
  const STEPS = ["config.json", "build graph", "draw"];

  // Already drawn here before: the shape cannot have changed, so skip the
  // network entirely. On a network like this one that is the difference
  // between instant and nothing at all.
  const hit = cacheGet(repo);
  if (hit && hit.files && hit.files["config.json"]) {
    status("working", `${repo} from cache`, STEPS, "");
    paint(hit.files["config.json"], repo, hit.host, `cached ${age(hit.at)}`);
    return;
  }
  status("working", `reading ${repo}`, STEPS, "");
  try {
    const r = await getModel(repo);
    cachePut(repo, r.files, r.host);
    paint(r.files["config.json"], repo, r.host,
      r.via === "origin" ? "via this server's own route" : `via ${r.host.replace("https://", "")}`);
  } catch (e) {
    step = -1;
    status("error", `could not draw ${repo}`, [], `<b>${elapsed()}</b>`);
    if (e instanceof HubError) explain(e);
    else { hint.hidden = false; fail(e); }
  }
}

/** Build and draw, reporting a shape it cannot read rather than throwing. */
function paint(cfg, repo, host, how) {
  try {
    draw(cfg, repo, host, how);
    return true;
  } catch (e) {
    step = -1;
    status("error", `could not draw ${repo}`, [], `<b>${elapsed()}</b>`);
    explain({
      title: "nothing here says what this model is made of",
      message: e.message,
      attempts: [{ host: (host || "config.json"), ms: 0, state: e.message }],
      fixes: ["check it is the model's own config.json — a pipeline's root "
        + "config points at its parts (unet/, transformer/, text_encoder/) rather than describing them",
        "this viewer draws what a config declares; where a key is missing it "
        + "reports it rather than guessing"],
    });
    return false;
  }
}

/**
 * A config.json from this machine.
 *
 * The network is not the only source, and on a network like this one it is
 * not even the fastest: curl writes `config.json` to disk happily. Reading
 * one here is honest — the drawing is derived from exactly the same file.
 */
async function fromFile(file) {
  err.hidden = true;
  why.hidden = true;
  hint.hidden = true;
  let cfg;
  try {
    cfg = JSON.parse(await file.text());
  } catch (e) {
    explain({
      title: "that file is not a config",
      attempts: [{ host: file.name, ms: 0, state: `not JSON — ${e.message}` }],
      fixes: ["drop the model's config.json — the one curl wrote should do"],
    });
    return;
  }
  // not cached: a local file's name is not a repo, and it can change
  paint(cfg, file.name.replace(/\.json$/i, ""), file.name,
    `read from ${file.name} on this machine`);
}

function draw(cfg, repo, host, how) {
  const STEPS = ["config.json", "build graph", "draw"];
  step = 1;
  status("working", `read ${repo}`, STEPS, `${how} · ${elapsed()}`);
  const g = build(cfg);
  step = 2;
  status("working", "drawing", STEPS, `${g.nodes.length} components · ${elapsed()}`);
  last = g; last.cfg = cfg; last.repo = repo; last.host = host;
  why.hidden = true;
  hint.hidden = true;
  $("src").textContent = `${(host || "").replace("https://", "")} · config.json only`;
  const { head } = render(svg, g, (nd) => show(nd), mode());
  $("sum").textContent = head;
  side.hidden = true;
  const named = g.nodes.filter((n) => n.repeat).map((n) => `${n.repeat}×`);
  status("ready", `${repo} drawn`, [],
    `<b>${g.nodes.length}</b> components${named.length ? ` · ${named.join(" · ")}` : ""}`
    + ` · ${how} · <b>${elapsed()}</b>`);
  fillRecent();
}

$("go").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("repo").value.trim();
  if (v) history.replaceState(null, "", "#" + v), load(v);
});
$("close").addEventListener("click", () => { side.hidden = true; });
$("whyclose").addEventListener("click", () => { why.hidden = true; });

/* ---- a config.json from this machine ----
 * Drop Anywhere is forgiving on purpose: a file picker is more discoverable
 * but a drag is what you do when you have just curled a config to disk. */
let dragDepth = 0;
const hasFile = (ev) => [...(ev.dataTransfer ? ev.dataTransfer.types : [])].includes("Files");
const canvas = document.querySelector("main");
canvas.addEventListener("dragenter", (ev) => {
  if (!hasFile(ev)) return;
  ev.preventDefault(); dragDepth++; canvas.classList.add("dropping");
});
canvas.addEventListener("dragover", (ev) => { if (hasFile(ev)) ev.preventDefault(); });
canvas.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; canvas.classList.remove("dropping"); }
});
canvas.addEventListener("drop", (ev) => {
  if (!hasFile(ev)) return;
  ev.preventDefault();
  dragDepth = 0; canvas.classList.remove("dropping");
  const file = ev.dataTransfer.files[0];
  if (file) fromFile(file);
});
$("pick").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (ev) => {
  const file = ev.target.files[0];
  if (file) fromFile(file);
  ev.target.value = "";
});

function syncMode() {
  const b = $("mode");
  const on = mode() === "3d";
  b.textContent = on ? "2D" : "3D";
  b.setAttribute("aria-pressed", String(on));
  b.title = on ? "switch to the flat drawing" : "switch to the 3D drawing";
}
$("mode").addEventListener("click", () => {
  localStorage.setItem(MODE_KEY, mode() === "3d" ? "2d" : "3d");
  syncMode();
  if (last) draw(last.cfg, last.repo, last.host, "re-drawn");
});
syncMode();
fillRecent();

const initial = decodeURIComponent(location.hash.slice(1));
if (initial) { $("repo").value = initial; load(initial); }