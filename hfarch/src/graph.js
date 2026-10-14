/**
 * Turn a config into a flow graph.
 *
 * The rule throughout: the diagram is arranged in the order data moves, and a
 * repeated component is drawn as its first instance, a marker for the ones in
 * between, and its last. Sixty-five identical boxes say "there are sixty-five"
 * and take more room than everything else on the page; the marker says the same
 * thing in a quarter of the space.
 */
import { attention, family, heads, mlp, moe, norm, pick, position, vision, COMMON }
  from "./blocks.js";

const n = (v, d = 0) => (Number.isFinite(Number(v)) ? Math.trunc(Number(v)) : d);

export function build(cfg) {
  const fam = family(cfg);
  const text = pick(cfg, "text_config") || cfg;      // VLMs nest the LM stack
  const { hidden, h, kv, hd } = heads(text);
  const layers = n(pick(text, "num_hidden_layers", "n_layer", "num_layers"));
  const vocab = n(pick(text, "vocab_size"));
  const attn = attention(text);
  const ff = mlp(text);
  const nrms = norm(text, hidden);
  const mix = moe(text);
  const pos = position(text, hidden);
  const tied = !!pick(text, "tie_word_embeddings");

  if (!hidden || !layers) {
    throw new Error("config declares no hidden size or layer count — nothing to draw");
  }

  // one decoder block, spelled out; this is the template behind the marker
  const inner = [];
  if (nrms.params) inner.push({ id: "n1", label: nrms.kind, kind: "norm", params: nrms.params, note: COMMON.rmsNorm });
  inner.push({
    id: "attn", label: "Attention", kind: "attention", params: attn.params,
    dims: { heads: h, kv: kv, head_dim: hd },
    note: attn.mode === "multi-head" ? COMMON.attention
      : attn.mode === "multi-query" ? COMMON.attention + " " + COMMON.groupedQuery
      : `${attn.mode}: ${h} query heads over ${kv} key/value heads`,
    dims2: { q: attn.q, k: attn.k, v: attn.v, o: attn.o },
  });
  if (nrms.params) inner.push({ id: "n2", label: nrms.kind, kind: "norm", params: nrms.params });
  inner.push(...(mix ? [{
    id: "moe", label: `Experts ×${mix.n}`, kind: "moe", params: mix.params,
    dims: { experts: mix.n, top_k: mix.topk, each: mix.width },
    note: COMMON.moe + (mix.shared ? `, plus ${mix.shared} always-on` : "")
      + (mix.gated ? "" : " — no gate, so a plain pair of projections"),
  }] : [{
    id: "mlp", label: "MLP", kind: "mlp", params: ff.params,
    dims: { width: ff.ff },
    note: ff.gated ? COMMON.gatedMlp : COMMON.mlp,
  }]));
  // experts stand in for the feed-forward block; counting both would be wrong
  const feed = mix ? mix.params : ff.params;
  const per = attn.params + feed + (nrms.params ? 2 * nrms.params : 0);

  const embed = vocab ? vocab * hidden : 0;
  const nodes = [
    {
      id: "ids", label: "Tokens", kind: "tensor", params: 0, branch: "text",
      // drawn as a plate: one thin sheet per token, one of them pulled out as
      // the vector actually being read this step
      plate: { count: 8, unit: "token", title: "Tokens", shape: ["B", "T"], width: hidden,
        note: "one bar per token; the highlighted one is being read" },
      note: "the integers the model reads",
    },
    {
      id: "embed", label: "Token embed", kind: "embed", params: embed, branch: "text",
      dims: { vocab: vocab || "?", width: hidden },
      note: COMMON.tokenEmbedding + (tied ? " " + COMMON.tiedHead : ""),
    },
  ];
  const edges = [
    { from: "ids", to: "embed", shape: ["B", "T"] },
    { from: "embed", to: "blocks", shape: ["B", "T"] },
    { from: "blocks", to: "norm", shape: ["B", "T"] },
    { from: "norm", to: "head", shape: ["B", "T"] },
    { from: "head", to: "logits", shape: ["B", "T"] },
    { from: "logits", to: "sample" },
  ];
  nodes.push(
    {
      id: "blocks", label: "Decoder block", kind: "stack", params: per * layers,
      branch: "text", repeat: layers, template: inner,
      dims: { layers, width: hidden },
      note: pos.kind === "RoPE" ? `every block is identical; positions by ${pos.note}` : `every block is identical`,
    },
    {
      id: "norm", label: `Final ${nrms.kind || "norm"}`, kind: "norm",
      params: nrms.params || hidden, branch: "text",
    },
    {
      id: "head", label: "Output head", kind: "embed", params: tied ? 0 : embed,
      branch: "text", dims: { vocab: vocab || "?", width: hidden },
      note: tied ? COMMON.tiedHead : "a matrix of its own",
    },
    { id: "logits", label: "Logits", kind: "tensor", params: 0, branch: "text" },
    { id: "sample", label: "Sampling", kind: "tensor", params: 0, branch: "text",
      note: "temperature / top-p pick one of those rows" },
  );

  let extra = per * layers + (tied ? embed : 2 * embed) + (nrms.params || hidden);

  // ---- the vision branch, if there is one ----
  const vis = vision(cfg);
  if (vis) {
    nodes.push(
      {
        id: "img", label: "Image", kind: "tensor", params: 0, branch: "vision",
        plate: { count: 8, unit: "patch", title: "Image", shape: ["B", 3, "T", "H", "W"],
          width: vis.hidden, note: `one bar per ${vis.patch}px patch` },
        note: vis.note,
      },
      { id: "patch", label: "Patch embed", kind: "embed", params: vis.merge,
        branch: "vision", dims: { width: vis.hidden } },
      { id: "vblocks", label: "Vision block", kind: "stack",
        params: vis.blocks, branch: "vision", repeat: vis.layers,
        dims: { layers: vis.layers, width: vis.hidden },
        note: "the same shape in every block",
        template: [
          { id: "va", label: "Attention", kind: "attention", params: vis.attn.params },
          { id: "vm", label: "MLP", kind: "mlp", params: vis.mlp.params },
        ] },
      { id: "merger", label: "Merger", kind: "projector", params: vis.merge,
        branch: "vision", dims: { from: vis.hidden, to: hidden },
        note: COMMON.merger },
      { id: "vtok", label: "Visual tokens", kind: "tensor", params: 0, branch: "vision",
        note: "join the text stream here" },
    );
    edges.push(
      { from: "img", to: "patch", shape: ["H", "W", vis.patch] },
      { from: "patch", to: "vblocks", shape: ["S"] },
      { from: "vblocks", to: "merger", shape: ["S"] },
      { from: "merger", to: "vtok", shape: ["S", hidden] },
      { from: "vtok", to: "blocks" },
    );
    extra += vis.blocks + vis.merge * 2;
  }

  return {
    nodes, edges, family: fam, params: extra,
    meta: {
      layers, hidden, heads: h, kv_heads: kv, head_dim: hd,
      ffn: ff.ff, gated_ffn: ff.gated, vocab: vocab || null,
      per_block: per, position: pos.kind, moe: mix ? { experts: mix.n, top_k: mix.topk } : null,
      vision: vis ? { layers: vis.layers, hidden: vis.hidden } : null,
      architectures: pick(cfg, "architectures"),
      tied,
    },
  };
}