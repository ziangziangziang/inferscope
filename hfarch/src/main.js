import { getModel, fmt } from "./hub.js";
import { build } from "./graph.js";
import { render } from "./render.js";

const $ = (id) => document.getElementById(id);
const svg = $("svg"), err = $("err"), side = $("side"), detail = $("detail"), hint = $("hint");

// the arrowhead the renderer points at
svg.insertAdjacentHTML("afterbegin",
  `<defs><marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5"
     markerWidth="5" markerHeight="5" orient="auto-start-reverse">
     <path d="M 0 1 L 9 5 L 0 9 z" fill="#9fb09a"/></marker></defs>`);

function fail(e) {
  err.hidden = false;
  err.textContent = e && e.message ? e.message : String(e);
  setTimeout(() => { err.hidden = true; }, 6000);
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

async function load(repo) {
  hint.hidden = true;
  try {
    const r = await getModel(repo);
    const g = build(r.files["config.json"]);
    last = g;
    $("src").textContent = `${r.host.replace("https://", "")} · config.json only`;
    const { head } = render(svg, g, (nd) => show(nd));
    $("sum").textContent = head;
    side.hidden = true;
  } catch (e) {
    hint.hidden = false;
    fail(e);
  }
}

$("go").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("repo").value.trim();
  if (v) history.replaceState(null, "", "#" + v), load(v);
});
$("close").addEventListener("click", () => { side.hidden = true; });

const initial = decodeURIComponent(location.hash.slice(1));
if (initial) { $("repo").value = initial; load(initial); }