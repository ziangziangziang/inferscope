/**
 * The families nothing used to draw.
 *
 * `family()` in blocks.js has recognised diffusion, speech and omni since the
 * beginning and then returned them to the language builder, which threw: "config
 * declares no hidden size or layer count". So a Stable Diffusion UNet and a
 * Whisper model both failed with an error about a transformer, which is a
 * confusing way to be told the viewer does not know your model.
 *
 * Each builder returns the same shape the language builder does, so the
 * renderer, the grid and both draw modes are unchanged. The graph shape differs
 * because the hardware differs: a UNet is not a stack of identical blocks, and
 * pretending otherwise is how a diagram ends up lying.
 *
 * Two rules kept throughout:
 *   - a count comes from the config or it is not printed;
 *   - where a config is silent -- whether a position embedding is learned, how a
 *     VAE scales its latent -- the card says so rather than guessing.
 */
import { COMMON, attention, heads, mlp, norm, pick } from "./blocks.js";

const n = (v, d = 0) => (Number.isFinite(Number(v)) ? Math.trunc(Number(v)) : d);
const lin = (din, dout, bias) => din * dout + (bias ? dout : 0);

/**
 * What a U-Net block holds, from the channels the config declares.
 *
 * A ResNet block is two 3x3 convolutions plus a group norm: 18c^2 plus biases
 * and 2c for the norm. Cross-attention spends q and o at the block's width and
 * k, v at the conditioning width. The feed-forward inside a diffusers
 * Transformer2DModel is *not* in the config, so it is not counted and not
 * guessed -- the card says so, which is why this total reads low against a
 * published figure rather than matching one it has no right to match.
 */
function unetBlock(ch, cross, { resnet = 2, hasAttn = true } = {}) {
  const res = resnet * (9 * ch * ch + ch) + 2 * resnet * ch;
  const attn = hasAttn ? 2 * ch * ch + 2 * ch * cross : 0;
  return {
    params: res + attn,
    template: [
      { id: "r", label: "ResNet", kind: "norm", params: res },
      ...(hasAttn ? [{ id: "a", label: "Cross-attn", kind: "attention", params: attn,
        note: "keys and values come from the text embedding" }] : []),
    ],
  };
}

/** A transformer block template, shared by encoders, decoders and cross-attn. */
function blockTemplate(cfg, { cross = false, depth = 0 } = {}) {
  const a = attention(cfg);
  const m = mlp(cfg);
  const nn = norm(cfg, a.hidden);
  const t = [];
  if (nn.params) t.push({ id: "n1", label: nn.kind, kind: "norm", params: nn.params });
  t.push({ id: "at", label: cross ? "Cross-attn" : "Self-attn", kind: "attention", params: a.params });
  if (cross) {
    // a cross-attention block spends q and o on the same width and k,v on the
    // conditioning width; the extra is one input projection per key and value
    const extra = 2 * (cross.width || 0) * (a.h * a.hd);
    t[t.length - 1].params += extra;
    t[t.length - 1].note = "queries the text, keys and values come from it";
  }
  if (nn.params) t.push({ id: "n2", label: nn.kind, kind: "norm", params: nn.params });
  t.push({ id: "mlp", label: "MLP", kind: "mlp", params: m.params });
  void depth;
  return t;
}

const per = (t) => t.reduce((s, q) => s + q.params, 0);

/* ------------------------------------------------------------------ *
 * Speech: an encoder over a spectrogram, a decoder over text.
 * ------------------------------------------------------------------ */
export function buildSpeech(cfg) {
  const d = n(pick(cfg, "d_model", "hidden_size"));
  const encL = n(pick(cfg, "encoder_layers"));
  const decL = n(pick(cfg, "decoder_layers"));
  const headsEnc = n(pick(cfg, "encoder_attention_heads"));
  const headsDec = n(pick(cfg, "decoder_attention_heads"));
  const ffn = n(pick(cfg, "encoder_ffn_dim"));
  const vocab = n(pick(cfg, "vocab_size"));
  const mel = n(pick(cfg, "num_mel_bins"));
  const frames = n(pick(cfg, "max_source_positions"));
  if (!d || !encL) throw new Error("no d_model or encoder_layers in this config");

  // Whisper's frontend: two stride-2 convolutions over the mel spectrogram.
  const conv = mel * d * 3 + d + d * d * 3 + d;

  const encCfg = {
    hidden_size: d, num_attention_heads: headsEnc, encoder_ffn_dim: ffn,
    ffn_dim: ffn, intermediate_size: ffn, hidden_act: "gelu",
    // Whisper's blocks are LayerNorms: a scale and a shift each. This used to
    // be declared as rms_norm_eps to reuse the formula, which counted half of
    // every norm the model has.
    layer_norm_eps: 1e-5,
  };
  const decCfg = {
    hidden_size: d, num_attention_heads: headsDec, decoder_ffn_dim: ffn,
    ffn_dim: ffn, intermediate_size: ffn, hidden_act: "gelu",
    layer_norm_eps: 1e-5,
  };
  const encT = blockTemplate(encCfg);
  const decT = blockTemplate(decCfg);
  decT.splice(2, 0, { id: "xa", label: "Cross-attn", kind: "attention",
    params: attention({ hidden_size: d, num_attention_heads: headsDec }).params,
    note: COMMON.groupedQuery });

  const encPer = per(encT), decPer = per(decT);
  const nodes = [
    { id: "mel", label: "Mel spectrogram", kind: "tensor", params: 0, branch: "encoder",
      plate: { count: 7, unit: "frame", title: "Audio", shape: [mel || "?", "T"],
        width: mel, note: "one bar per 20ms frame" } },
    { id: "conv", label: "Conv frontend", kind: "mlp", params: conv, branch: "encoder",
      dims: { out: d, width: mel },
      note: "two stride-2 convolutions turn the spectrogram into a sequence" },
    { id: "eenc", label: "Encoder block", kind: "stack", branch: "encoder",
      params: encPer * encL, repeat: encL, template: encT,
      dims: { width: d, heads: headsEnc },
      note: `runs over the audio; ${frames ? `up to ${frames} frames` : "length not declared"}` },
    { id: "ddec", label: "Decoder block", kind: "stack", branch: "decoder",
      params: decPer * decL, repeat: decL, template: decT,
      dims: { width: d, heads: headsDec },
      note: "reads the audio encoding and what has been said so far" },
    { id: "embed", label: "Token embedding", kind: "embed", branch: "decoder",
      params: vocab * d, dims: { vocab: vocab || "?", width: d },
      note: "the half the decoder has already said" },
    { id: "logits", label: "Logits", kind: "tensor", params: 0, branch: "decoder" },
  ];
  const edges = [
    { from: "mel", to: "conv", shape: [mel || "?", frames || "T"] },
    { from: "conv", to: "eenc", shape: ["S", d] },
    { from: "eenc", to: "ddec", shape: ["S", d] },
    { from: "embed", to: "ddec" },
    { from: "ddec", to: "logits", shape: ["T", vocab || "V"] },
  ];
  return {
    nodes, edges, family: "speech",
    params: conv + encPer * encL + decPer * decL + vocab * d,
    meta: {
      layers: encL + decL, hidden: d, heads: headsEnc || headsDec, kv_heads: headsEnc || headsDec,
      ffn, per_block: encPer, position: "sinusoidal", moe: null, vision: null, tied: false,
      vocab, encoder_layers: encL, decoder_layers: decL, mel_bins: mel, frames,
    },
  };
}

/* ------------------------------------------------------------------ *
 * Diffusion: a U-Net, which is not a stack of identical blocks.
 * ------------------------------------------------------------------ */
export function buildDiffusion(cfg) {
  const chans = pick(cfg, "block_out_channels") || [];
  const downs = pick(cfg, "down_block_types") || [];
  const ups = pick(cfg, "up_block_types") || [];
  const perBlock = Math.max(1, n(pick(cfg, "layers_per_block"), 1));
  const cross = n(pick(cfg, "cross_attention_dim"));
  const inCh = n(pick(cfg, "in_channels"), 4);
  const sample = n(pick(cfg, "sample_size"), 64);
  if (!chans.length) throw new Error("no block_out_channels — this is not a U-Net config");

  const nodes = [];
  const edges = [];
  const groups = n(pick(cfg, "norm_num_groups"), 32);

  nodes.push({
    id: "latent", label: "Noisy latent", kind: "tensor", params: 0, branch: "unet",
    plate: { count: 7, unit: "channel", title: "Noisy latent", shape: ["B", inCh, sample, sample],
      width: inCh, note: "one bar per latent channel" },
  });

  let acc = "";
  chans.forEach((c, i) => {
    const res = sample >> i;                       // resolution halves per level
    const crossAt = String(downs[i] || "").includes("CrossAttn");
    const { params: rp, template: t } = unetBlock(c, cross, { hasAttn: crossAt });
    nodes.push({
      id: `d${i}`, label: `Down ${i + 1}`, kind: "stack", branch: "unet",
      params: rp * perBlock, repeat: perBlock, template: t,
      dims: { width: c, groups: groups },
      note: `res ${res}×${res}, ${c} channels; feed-forward is not declared in the config`,
    });
    edges.push({ from: acc || "latent", to: `d${i}`, shape: [res, res] });
    acc = `d${i}`;
  });

  // the bottleneck sits at the deepest level, so both sides meet there
  const deep = chans[chans.length - 1];
  const { params: midP, template: midT } = unetBlock(deep, cross);
  nodes.push({
    id: "mid", label: "Bottleneck", kind: "stack", branch: "unet",
    params: midP * perBlock, repeat: perBlock, template: midT,
    dims: { width: deep, groups }, note: "the deepest point of the U",
  });
  edges.push({ from: acc, to: "mid", shape: [1, 1] });
  acc = "mid";

  const skips = [];
  chans.forEach((c, i) => {
    const j = chans.length - 1 - i;
    const res = sample >> j;
    const crossAt = String(ups[i] || "").includes("CrossAttn");
    const { params: rp, template: t } = unetBlock(c, cross, { hasAttn: crossAt });
    nodes.push({
      id: `u${i}`, label: `Up ${i + 1}`, kind: "stack", branch: "unet",
      params: rp * perBlock, repeat: perBlock, template: t,
      dims: { width: c, groups }, note: `res ${res}×${res}, ${c} channels`,
    });
    edges.push({ from: acc, to: `u${i}`, shape: [res, res] });
    if (i > 0) skips.push({ from: `d${j}`, to: `u${i}`, shape: [res, res], skip: true });
    acc = `u${i}`;
  });

  const unetParams = nodes.filter((v) => v.branch === "unet").reduce((s, v) => s + v.params, 0);
  edges.push(...skips);

  nodes.push(
    { id: "clean", label: "Clean latent", kind: "tensor", params: 0, branch: "unet",
      plate: { count: 7, unit: "channel", title: "Clean latent", shape: ["B", inCh, sample, sample],
        width: inCh, note: "the same shape it started as" } },
    { id: "vae", label: "VAE decoder", kind: "projector", params: 0, branch: "unet",
      note: "not in this config — pass the VAE's own repo to see it" },
  );
  edges.push({ from: acc, to: "clean", shape: [sample, sample] });
  edges.push({ from: "clean", to: "vae", shape: [sample, sample] });

  return {
    nodes, edges, family: "diffusion", params: unetParams,
    meta: {
      layers: chans.length, hidden: chans[0], heads: 0, kv_heads: 0,
      ffn: 0, per_block: midP, position: "none", moe: null, vision: null, tied: false,
      vocab: null, levels: chans, levels_per_block: perBlock,
      deepest: Math.max(...chans), conditioning: cross || null,
    },
  };
}

/* ------------------------------------------------------------------ *
 * VAE, on its own: the decoder half of a latent image pipeline.
 * ------------------------------------------------------------------ */
export function buildVAE(cfg) {
  const chans = pick(cfg, "block_out_channels") || [];
  const latent = n(pick(cfg, "latent_channels"));
  const groups = n(pick(cfg, "block_out_groups", "norm_num_groups"), 32);
  const scale = pick(cfg, "scaling_factor");
  const sample = n(pick(cfg, "sample_size"), 256);
  if (!chans.length) throw new Error("no block_out_channels — this is not a VAE config");

  const { params: p, template: t } = unetBlock(chans[0], 0, { hasAttn: false });
  const params = p * chans.length;
  return {
    nodes: [
      { id: "latent", label: "Latent", kind: "tensor", params: 0,
        plate: { count: 7, unit: "channel", title: "Latent",
          shape: ["B", latent || "?", sample >> 8, sample >> 8], width: latent } },
      { id: "dec", label: "Decoder", kind: "stack", params,
        repeat: chans.length, template: t,
        dims: { width: chans[0], groups },
        note: scale != null ? `latent is scaled by ${scale}` : "scaling factor not declared" },
      { id: "img", label: "Image", kind: "tensor", params: 0,
        plate: { count: 7, unit: "channel", title: "Image",
          shape: ["B", n(pick(cfg, "out_channels"), 3), sample, sample],
          width: n(pick(cfg, "out_channels"), 3) } },
    ],
    edges: [
      { from: "latent", to: "dec", shape: [sample >> 8, sample >> 8] },
      { from: "dec", to: "img", shape: [sample, sample] },
    ],
    family: "vae", params,
    meta: { layers: chans.length, hidden: chans[0], heads: 0, kv_heads: 0, ffn: 0,
      per_block: p, position: "none", moe: null, vision: null, tied: false, vocab: null,
      levels: chans, deepest: Math.max(...chans), conditioning: null },
  };
}

export { blockTemplate, per, heads };
/* ------------------------------------------------------------------ *
 * Encoders on their own: CLIP's text tower, BERT, the T5 encoder.
 *
 * What these do not have matters as much as what they do: no output head over
 * the vocabulary, nothing to sample. The old language builder drew them with
 * a Logits stage and a Sampling stage anyway, which is how a drawing tells
 * you it has exactly one shape for every config.
 * ------------------------------------------------------------------ */

/** One encoder stack: [norm, self-attn, norm, mlp], from the config. */
function encoderStack(cfg, pfx, branch, { label = "Encoder block", note = "" } = {}) {
  const d = n(pick(cfg, "hidden_size"));
  const L = n(pick(cfg, "num_hidden_layers"));
  const t = blockTemplate(cfg);
  const nn = norm(cfg, d);
  return {
    width: d, layers: L, heads: n(pick(cfg, "num_attention_heads")),
    stackParams: per(t) * L, finalNorm: nn.params || d, template: t,
    node: {
      id: `${pfx}stack`, label, kind: "stack", params: per(t) * L,
      branch, repeat: L, template: t,
      dims: { layers: L, width: d, heads: n(pick(cfg, "num_attention_heads")) },
      note,
    },
    finalNode: {
      id: `${pfx}norm`, label: `Final ${nn.kind || "norm"}`, kind: "norm",
      params: nn.params || d, branch,
    },
  };
}

/** The text half of an encoder, as nodes and edges. */
function textEncoder(cfg, pfx, branch, { projDim = 0 } = {}) {
  const d = n(pick(cfg, "hidden_size"));
  const vocab = n(pick(cfg, "vocab_size"));
  const posLen = n(pick(cfg, "max_position_embeddings"));
  const pos = posLen * d;
  const stack = encoderStack(cfg, pfx, branch,
    { note: "every block identical; nothing reads one token ahead of another" });
  let params = vocab * d + pos + stack.stackParams + stack.finalNorm;

  const nodes = [
    { id: `${pfx}tokens`, label: "Tokens", kind: "tensor", params: 0, branch,
      plate: { count: 8, unit: "token", title: "Tokens", shape: ["B", "T"], width: d,
        note: "one bar per token" } },
    { id: `${pfx}embed`, label: "Token embed", kind: "embed", params: vocab * d, branch,
      dims: { vocab: vocab || "?", width: d }, note: COMMON.tokenEmbedding },
  ];
  const edges = [
    { from: `${pfx}tokens`, to: `${pfx}embed`, shape: ["B", "T"] },
  ];
  if (pos) {
    nodes.push({ id: `${pfx}pos`, label: "Position embed", kind: "embed", params: pos,
      branch, dims: { length: posLen, width: d },
      note: "learned positions, one row per slot" });
    edges.push({ from: `${pfx}embed`, to: `${pfx}pos`, shape: ["B", "T"] });
    edges.push({ from: `${pfx}pos`, to: `${pfx}stack`, shape: ["B", "T"] });
  } else {
    edges.push({ from: `${pfx}embed`, to: `${pfx}stack`, shape: ["B", "T"] });
  }
  nodes.push(stack.node, stack.finalNode);
  edges.push({ from: `${pfx}stack`, to: `${pfx}norm`, shape: ["B", "T"] });
  if (projDim) {
    nodes.push({ id: `${pfx}proj`, label: "Projection", kind: "projector",
      params: d * projDim + projDim, branch, dims: { from: d, to: projDim },
      note: "into the space both towers share" });
    edges.push({ from: `${pfx}norm`, to: `${pfx}proj`, shape: [projDim] });
    return { nodes, edges, stack, params: params + d * projDim + projDim };
  }
  return { nodes, edges, stack, params };
}

/** The vision half: patches in, one vector per patch out. */
function visionEncoder(cfg, pfx, branch, { projDim = 0 } = {}) {
  const d = n(pick(cfg, "hidden_size"));
  const patch = n(pick(cfg, "patch_size"), 16);
  const image = n(pick(cfg, "image_size"), 0);
  const layers = n(pick(cfg, "num_hidden_layers", "num_layers", "depth"));
  const stack = encoderStack(cfg, pfx, branch, { label: "Vision block",
    note: "the same shape in every block" });

  // the patch projection is a conv whose kernel is the patch itself, so its
  // input width is in-channels x patch^2. The config names the patch but
  // usually not the channels: three (RGB) is assumed, and the card says so.
  const inCh = n(pick(cfg, "in_channels"), 3);
  const assumedCh = pick(cfg, "in_channels") == null;
  const patchIn = inCh * patch * patch;
  const patchEmbed = patchIn * d;
  const squares = image ? (image / patch) * (image / patch) : null;
  // a class token is prepended -- structural to every ViT, not in the config
  const pos = squares != null ? (squares + 1) * d : 0;

  const nodes = [
    { id: `${pfx}img`, label: "Image", kind: "tensor", params: 0, branch,
      plate: { count: 8, unit: "patch", title: "Image",
        shape: ["B", 3, "T", "H", "W"], width: d,
        note: `one bar per ${patch}px patch` } },
    { id: `${pfx}patch`, label: "Patch embed", kind: "embed", params: patchEmbed,
      branch, dims: { width: d, patch },
      note: `in-channels x patch^2 -> width${assumedCh ? " (three colour channels assumed; the config does not declare them)" : ""}` },
  ];
  const edges = [
    { from: `${pfx}img`, to: `${pfx}patch`, shape: ["H", "W", patch] },
    { from: `${pfx}patch`, to: `${pfx}stack`, shape: ["S"] },
  ];
  if (pos) {
    nodes.push({ id: `${pfx}pos`, label: "Position embed", kind: "embed", params: pos,
      branch, dims: { patches: squares + 1, width: d },
      note: "one row per patch, plus the class token" });
    edges.splice(1, 1,
      { from: `${pfx}patch`, to: `${pfx}pos`, shape: ["S"] },
      { from: `${pfx}pos`, to: `${pfx}stack`, shape: ["S"] });
  }
  nodes.push(stack.node, stack.finalNode);
  edges.push({ from: `${pfx}stack`, to: `${pfx}norm`, shape: ["S"] });
  const params = patchEmbed + pos + stack.stackParams + stack.finalNorm;

  if (projDim) {
    nodes.push({ id: `${pfx}proj`, label: "Projection", kind: "projector",
      params: d * projDim + projDim, branch, dims: { from: d, to: projDim },
      note: "into the space both towers share" });
    edges.push({ from: `${pfx}norm`, to: `${pfx}proj`, shape: [projDim] });
    return { nodes, edges, params: params + d * projDim + projDim, stack };
  }
  edges.push({ from: `${pfx}norm`, to: `${pfx}out`, shape: [d] });
  nodes.push({ id: `${pfx}out`, label: "Emitted vector", kind: "tensor",
    params: 0, branch, dims: { width: d },
    note: "one vector per patch; the picture is pooled from it" });
  return { nodes, edges, params, stack };
}

/** A standalone encoder: config -> graph, with no head and no sampling. */
export function buildEncoder(cfg) {
  if (pick(cfg, "image_size") && pick(cfg, "patch_size") && !pick(cfg, "vocab_size")) {
    const lane = visionEncoder(cfg, "", "image-enc",
      { projDim: n(pick(cfg, "projection_dim")) || 0 });
    return {
      nodes: lane.nodes, edges: lane.edges, family: "encoder", params: lane.params,
      meta: { layers: lane.stack.layers, hidden: lane.stack.width,
        heads: lane.stack.heads, kv_heads: lane.stack.heads, ffn: 0,
        per_block: lane.stack.stackParams / Math.max(1, lane.stack.layers),
        position: "learned", moe: null, vision: null, tied: false, vocab: null },
    };
  }
  const lane = textEncoder(cfg, "", "enc",
    { projDim: /WithProjection/.test(String(pick(cfg, "architectures", [])[0] || ""))
      ? n(pick(cfg, "projection_dim")) || 0 : 0 });
  const nodes = [...lane.nodes, {
    id: "out", label: "Emitted vector", kind: "tensor", params: 0, branch: "enc",
    dims: { width: lane.stack.width },
    note: "one vector per token; the sentence is pooled from it. No output head, nothing to sample.",
  }];
  const edges = [...lane.edges,
    lane.nodes.some((x) => x.id === "proj")
      ? { from: "proj", to: "out", shape: [lane.stack.width] }
      : { from: "norm", to: "out" }];
  return {
    nodes, edges, family: "encoder", params: lane.params,
    meta: {
      layers: lane.stack.layers, hidden: lane.stack.width, heads: lane.stack.heads,
      kv_heads: lane.stack.heads,
      ffn: n(pick(cfg, "intermediate_size")), per_block: lane.stack.stackParams / lane.stack.layers,
      position: n(pick(cfg, "max_position_embeddings")) ? "learned" : "none declared",
      moe: null, vision: null, tied: false, vocab: n(pick(cfg, "vocab_size")) || null,
    },
  };
}

/**
 * Contrastive: two encoders compared in one space.
 *
 * A CLIP config was drawn as a language model with a picture bolted on -- an
 * output head, logits, sampling, none of which it has. What it has is two
 * towers of the same shape, each ending in a projection into a shared width,
 * and one learned number that scales the comparison between them.
 */
export function buildContrastive(cfg) {
  const proj = n(pick(cfg, "projection_dim"));
  const tLane = textEncoder(pick(cfg, "text_config") || cfg, "t", "text-enc",
    { projDim: proj || 0 });
  const vLane = visionEncoder(pick(cfg, "vision_config") || {}, "v", "image-enc",
    { projDim: proj || 0 });
  const logit = pick(cfg, "logit_scale_init_value") != null ? 1 : 0;

  const nodes = [...vLane.nodes, ...tLane.nodes, {
    // its own lane: both towers drop into it, and neither tower's column has
    // to share a band with the other's cards
    id: "joint", label: "Shared space", kind: "tensor", params: logit, branch: "out",
    dims: { width: proj || "?" },
    note: "both towers project here and are compared pairwise; the scale between "
      + "them is one learned number, initialised at "
      + `${pick(cfg, "logit_scale_init_value")} -- the 1 parameter counted above`,
  }];
  const edges = [...vLane.edges, ...tLane.edges,
    { from: "tproj", to: "joint", shape: [proj] },
    { from: "vproj", to: "joint", shape: [proj] },
  ];
  return {
    nodes, edges, family: "contrastive",
    params: tLane.params + vLane.params + logit,
    meta: {
      layers: tLane.stack.layers, hidden: tLane.stack.width, heads: tLane.stack.heads,
      kv_heads: tLane.stack.heads, ffn: n(pick(cfg, "text_config", "intermediate_size")),
      per_block: tLane.stack.stackParams / tLane.stack.layers,
      position: "learned", moe: null,
      vision: { layers: vLane.stack.layers, hidden: vLane.stack.width },
      tied: false, vocab: n(pick(cfg, "text_config", "vocab_size")) || null,
      text_layers: tLane.stack.layers, text_hidden: tLane.stack.width,
      vision_layers: vLane.stack.layers, vision_hidden: vLane.stack.width,
      projection: proj || null,
    },
  };
}

/* ------------------------------------------------------------------ *
 * A transformer over patches: DiT, PixArt.
 *
 * This config used to fall through to the language builder, which threw
 * "config declares no hidden size or layer count" -- true, and useless: the
 * width is declared as num_attention_heads x attention_head_dim, and the
 * layers as num_layers.
 *
 * One deliberate gap, on the card of every block: the feed-forward width is
 * not declared anywhere in these configs, so it is not counted. The total
 * reads about 55% of the published figure, and the card says why.
 * ------------------------------------------------------------------ */
export function buildDiT(cfg) {
  const heads_ = n(pick(cfg, "num_attention_heads"));
  const hd = n(pick(cfg, "attention_head_dim"));
  const d = n(pick(cfg, "hidden_size"), heads_ * hd);
  const layers = n(pick(cfg, "num_layers"));
  const patch = n(pick(cfg, "patch_size"), 1);
  const inCh = n(pick(cfg, "in_channels"), 4);
  const outCh = n(pick(cfg, "out_channels"), inCh);
  const sample = n(pick(cfg, "sample_size"), 64);
  const cross = n(pick(cfg, "cross_attention_dim"), 0);
  const ada = String(pick(cfg, "norm_type") || "");
  const classes = n(pick(cfg, "num_embeds_ada_norm"), 0);
  const captionCh = n(pick(cfg, "caption_channels"), 0);
  const bias = !!pick(cfg, "attention_bias");
  if (!d || !layers) {
    throw new Error("no width or layer count here — not a patch transformer the viewer knows");
  }

  // self-attention over patch tokens: q, k, v from the width, o back to it
  const sa = 3 * lin(d, heads_ * hd, bias) + lin(heads_ * hd, d, bias);
  // cross-attention, when the config declares a conditioning width: queries
  // from the image, keys and values from the text embedding
  const ca = cross ? lin(d, heads_ * hd, bias) + 2 * lin(cross, heads_ * hd, bias)
    + lin(heads_ * hd, d, bias) : 0;
  // adaLN. The config names the norm type; the six modulations (shift, scale
  // and gate for the attention and for the feed-forward) are what that name
  // means. ada_norm_zero produces them with one linear; ada_norm_single keeps
  // them as a plain table.
  const adaP = ada === "ada_norm_zero" ? lin(d, 6 * d, true) : 6 * d;
  const block = adaP + sa + ca;

  const template = [
    { id: "ada", label: "adaLN", kind: "norm", params: adaP,
      note: "timestep and class condition every block through six modulations" },
    { id: "sa", label: "Self-attn", kind: "attention", params: sa },
    ...(ca ? [{ id: "ca", label: "Cross-attn", kind: "attention", params: ca,
      note: "queries the image tokens, keys and values from the caption" }] : []),
  ];

  // the patch projection is a conv whose kernel is the patch, so its input
  // width is in-channels x patch^2
  const patchEmbed = inCh * patch * patch * d;
  // ada_norm_single carries a learned position table (PixArt); ada_norm_zero
  // uses a fixed sin-cos grid (DiT), which holds no parameters
  const pos = ada === "ada_norm_single"
    ? (sample / patch) * (sample / patch) * d : 0;
  const labelEmbed = classes * d;
  const captionProj = captionCh && cross ? lin(captionCh, cross, true) : 0;
  // the output layer: a final norm, its own adaLN modulation, then a linear
  // to patch^2 x out_channels -- the patch shape, un-patchified
  const finalP = 2 * d + (ada === "ada_norm_zero" ? lin(d, 2 * d, true) : 2 * d)
    + lin(d, patch * patch * outCh, true);

  const nodes = [
    { id: "latent", label: "Noisy latent", kind: "tensor", params: 0, branch: "dit",
      plate: { count: 7, unit: "channel", title: "Noisy latent",
        shape: ["B", inCh, sample, sample], width: inCh,
        note: "one bar per latent channel" } },
    { id: "patch", label: "Patch embed", kind: "embed", params: patchEmbed, branch: "dit",
      dims: { width: d, patch },
      note: `in-channels x ${patch}^2 -> width; the image enters as patches` },
    ...(pos ? [{ id: "pos", label: "Position embed", kind: "embed", params: pos,
      branch: "dit", dims: { patches: (sample / patch) ** 2, width: d },
      note: "learned positions, one row per patch" }] : []),
    ...(labelEmbed ? [{ id: "label", label: "Class embed", kind: "embed",
      params: labelEmbed, branch: "dit", dims: { classes, width: d },
      note: "one row per class; conditions every block through adaLN" }] : []),
    ...(captionProj ? [{ id: "caption", label: "Caption proj", kind: "projector",
      params: captionProj, branch: "dit", dims: { from: captionCh, to: cross },
      note: "the text encoder's output, brought to the cross-attention's width" }] : []),
    { id: "blocks", label: "DiT block", kind: "stack", params: block * layers,
      branch: "dit", repeat: layers, template,
      dims: { layers, width: d },
      note: "the feed-forward inside each block has no width declared in this "
        + "config, so it is not counted" },
    { id: "out", label: "Output layer", kind: "projector", params: finalP, branch: "dit",
      dims: { width: d, out: patch * patch * outCh },
      note: "norm, adaLN modulation, and a linear back to patch x patch x channels" },
    { id: "clean", label: "Denoised latent", kind: "tensor", params: 0, branch: "dit",
      plate: { count: 7, unit: "channel", title: "Denoised latent",
        shape: ["B", outCh, sample, sample], width: outCh,
        note: "the noise the block predicts, same shape it started as" } },
  ];
  const chain = ["latent", "patch"];
  if (pos) chain.push("pos");
  if (labelEmbed) chain.push("label");
  if (captionProj) chain.push("caption");
  chain.push("blocks", "out", "clean");
  const edges = [];
  for (let i = 0; i < chain.length - 1; i++) {
    const from = chain[i], to = chain[i + 1];
    const shape = to === "blocks" ? [(sample / patch) ** 2, d]
      : to === "clean" ? [sample, sample] : null;
    edges.push(shape ? { from, to, shape } : { from, to });
  }

  const params = nodes.reduce((s, v) => s + (v.params || 0), 0);
  return {
    nodes, edges, family: "dit", params,
    meta: {
      layers, hidden: d, heads: heads_, kv_heads: heads_, ffn: 0, per_block: block,
      position: pos ? "learned" : "fixed sin-cos (not declared; holds no parameters)",
      moe: null, vision: null, tied: false, vocab: null,
      patch, in_channels: inCh, out_channels: outCh, sample,
      conditioning: cross || null, classes: classes || null,
    },
  };
}
