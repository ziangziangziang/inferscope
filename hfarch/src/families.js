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
    rms_norm_eps: 1e-5, qkv_proj_bias: true,
  };
  const decCfg = {
    hidden_size: d, num_attention_heads: headsDec, decoder_ffn_dim: ffn,
    ffn_dim: ffn, intermediate_size: ffn, hidden_act: "gelu",
    rms_norm_eps: 1e-5, qkv_proj_bias: true,
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