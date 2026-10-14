/**
 * The parameter counts are the part of a diagram that can be quietly wrong, so
 * they get checked against the sizes the model cards publish.
 *
 * Run: node test/graph.test.mjs
 */
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build } from "../src/graph.js";

const here = dirname(fileURLToPath(import.meta.url));
const cfgDir = join(here, "cfg");

let failures = 0;
const ok = (cond, msg) => {
  console.log(`${cond ? "  ok  " : "  FAIL"}  ${msg}`);
  if (!cond) failures++;
};

const B = 1e9;

for (const f of readdirSync(cfgDir).filter((x) => x.endsWith(".json"))) {
  let cfg;
  try { cfg = JSON.parse(readFileSync(join(cfgDir, f), "utf8")); }
  catch { console.log(`  skip  ${f}: not JSON (gated or missing)`); continue; }
  let g;
  try {
    g = build(cfg);
  } catch (e) {
    console.log(`  FAIL  ${f}: ${e.message}`);
    failures++;
    continue;
  }
  const m = g.meta;
  console.log(`\n${f}  [${g.family}]`);
  console.log(`  ${m.layers} layers x ${m.hidden} wide, ${m.heads}q/${m.kv_heads}kv `
    + `heads, ffn ${m.ffn}${m.gated_ffn ? " gated" : ""}, vocab ${m.vocab}`
    + `${m.tied ? ", tied" : ""}${m.moe ? `, ${m.moe.experts} experts top-${m.moe.top_k}` : ""}`
    + `${m.vision ? `, vision ${m.vision.layers}x${m.vision.hidden}` : ""}`);
  console.log(`  computed ${(g.params / B).toFixed(2)}B   ${fmtBreak(g)}`);
  // `params` counts the stack plus embeddings once, with a tied head counted once
  ok(g.params > 0 && Number.isFinite(g.params), "parameter total is a finite number");
  if (f.startsWith("Qwen_Qwen3-4B")) {
    ok(Math.abs(g.params / B - 4.02) < 0.12, "Qwen3-4B within 3% of the published 4.0B");
    ok(m.tied === true, "Qwen3-4B embedding is tied to the output head");
    ok(m.kv_heads === 8 && m.heads === 32, "grouped-query: 32 query over 8 kv heads");
  }
  if (f.startsWith("Qwen_Qwen3-30B")) {
    ok(Math.abs(g.params / B - 30.5) < 0.9, "Qwen3-30B-A3B within 3% of the published 30.5B");
    ok(m.moe && m.moe.experts > 1, "recognised as a mixture of experts");
  }
  if (f.startsWith("Qwen_Qwen3-VL")) {
    ok(g.family === "vision-language", "recognised as vision-language");
    ok(m.vision !== null, "vision tower counted separately");
    ok(Math.abs(g.params / B - 8.2) < 0.5, "Qwen3-VL-8B within ~6% of ~8.2B");
  }
  // Every repeated component, whatever family it is in, must hold exactly
  // one copy's worth of its own sub-layers times the count. This used to be a
  // single check on the language stack, which meant a diffusion U-Net block
  // could be wrong by a factor and nothing noticed.
  const reps = g.nodes.filter((n) => n.repeat);
  ok(reps.length > 0, "the model has at least one repeated component");
  for (const n of reps) {
    const one = n.template.reduce((s, q) => s + q.params, 0);
    ok(n.params === one * n.repeat,
      `${n.label}: params ${n.params} == ${one} x ${n.repeat} (every copy, once)`);
  }

  if (g.family === "language" || g.family === "vision-language") {
    const stack = g.nodes.find((n) => n.repeat === m.layers);
    ok(!!stack, `the ${m.layers}-layer stack is one repeated component, not ${m.layers} boxes`);
  }
  if (g.family === "speech") {
    const enc = g.nodes.find((n) => n.branch === "encoder" && n.repeat);
    const dec = g.nodes.find((n) => n.branch === "decoder" && n.repeat);
    ok(!!enc && !!dec, "speech has both an encoder and a decoder stack");
    ok(dec && dec.template.some((q) => q.label === "Cross-attn"),
      "the decoder stack really contains a cross-attention");
  }
  if (g.family === "diffusion") {
    const downs = g.nodes.filter((n) => /^Down /.test(n.label));
    const ups = g.nodes.filter((n) => /^Up /.test(n.label));
    const skips = g.edges.filter((e) => e.skip);
    ok(downs.length > 0 && downs.length === ups.length,
      `a U-Net: ${downs.length} down levels and ${ups.length} up levels`);
    ok(skips.length === downs.length - 1,
      `the U-Net carries ${downs.length - 1} skip connections, which is what makes it one`);
  }
}

function fmtBreak(g) {
  const stack = g.nodes.reduce((a, n) => a + (n.params || 0), 0);
  const v = g.nodes.find((n) => n.id === "vblocks");
  return `of which vision ${v ? (v.params / B).toFixed(2) + "B" : "0"}`;
}

console.log(`\n${failures ? failures + " FAILED" : "all checks passed"}`);
process.exit(failures ? 1 : 0);