/**
 * The component library.
 *
 * Every transformer is assembled from a handful of the same parts, so the
 * drawing should be assembled from the same vocabulary. Each entry here knows
 * three things about a part: which config keys describe it, how to draw it, and
 * how many parameters it holds.
 *
 * The parameter counts are the point, and they are where a diagram usually
 * goes wrong, so each formula is written out rather than approximated:
 *
 *  - attention is sized by *kv* heads for K and V, not query heads — grouping
 *    them is the norm now and sizing by the query count overstates most models;
 *  - SwiGLU-family feed-forward blocks have three matrices where plain GELU has
 *    two, for the same hidden width;
 *  - biases exist only if the config says so; assuming them inflates every
 *    block of most current models;
 *  - a tied embedding and output head are one matrix, not two.
 */

/** Read a dotted path out of a config, first match wins. */
export const pick = (cfg, ...paths) => {
  for (const p of paths) {
    let n = cfg;
    for (const part of p.split(".")) {
      if (n == null || typeof n !== "object" || !(part in n)) { n = undefined; break; }
      n = n[part];
    }
    if (n != null) return n;
  }
  return undefined;
};
const num = (v, d = 0) => (Number.isFinite(Number(v)) ? Number(v) : d);
const int = (v, d = 0) => Math.trunc(num(v, d));
const lin = (din, dout, bias) => din * dout + (bias ? dout : 0);

/** Heads, and whether they are grouped — the distinction that matters most. */
export function heads(cfg) {
  const hidden = int(pick(cfg, "hidden_size", "n_embd", "d_model", "dim"));
  const h = int(pick(cfg, "num_attention_heads", "n_head", "n_heads"));
  const kv = int(pick(cfg, "num_key_value_heads", "num_kv_heads", "n_kv_heads")) || h;
  const hd = int(pick(cfg, "head_dim")) || (h ? Math.floor(hidden / h) : 0);
  const mode = kv === 0 || kv === h ? "multi-head" : kv === 1 ? "multi-query" : "grouped-query";
  return { hidden, h, kv, hd, mode };
}

/** Attention: the four projections, plus biases only if declared. */
export function attention(cfg) {
  const { hidden, h, kv, hd } = heads(cfg);
  const bias = !!pick(cfg, "attention_bias");
  const q = lin(hidden, h * hd, bias);
  const k = lin(hidden, kv * hd, bias);
  const v = lin(hidden, kv * hd, bias);
  const o = lin(h * hd, hidden, bias);
  return { params: q + k + v + o, q, k, v, o, h, kv, hd, hidden, bias, mode: modeOf(h, kv) };
}
const modeOf = (h, kv) => (kv === h ? "multi-head" : kv === 1 ? "multi-query" : "grouped-query");

/** Feed-forward. SwiGLU/GeGLU gate the up-projection, so three matrices. */
export function mlp(cfg) {
  const { hidden } = heads(cfg);
  const act = String(pick(cfg, "hidden_act", "activation_function") || "").toLowerCase();
  const bias = !!pick(cfg, "mlp_bias");
  const ff = int(pick(cfg, "intermediate_size", "ffn_dim", "n_inner", "d_ff"))
    || hidden * (["silu", "swish", "geglu"].includes(act) ? 8 : 4);
  const gated = ["silu", "swish", "geglu"].includes(act);
  const params = gated
    ? lin(hidden, ff, bias) * 2 + lin(ff, hidden, bias)
    : lin(hidden, ff, bias) + lin(ff, hidden, bias);
  return { params, ff, gated, act: act || "gelu", bias };
}

/** Normalisation, and whether the config even mentions one. */
export function norm(cfg, width) {
  const rms = pick(cfg, "rms_norm_eps") != null;
  const ln = pick(cfg, "layer_norm_eps", "layer_norm_epsilon") != null;
  return { params: rms || ln ? width : 0, kind: rms ? "RMSNorm" : ln ? "LayerNorm" : null };
}

/**
 * Mixture of experts, when the config declares any.
 *
 * Two things are easy to get wrong here and both of them are large:
 *
 *  - Experts are sized by `moe_intermediate_size`, not `intermediate_size`.
 *    Qwen3-30B-A3B declares both: 6144 for the (unused) dense MLP and 768 for
 *    each of its 128 experts. Reading the wrong one overstates the model by 8×.
 *  - Experts *replace* the dense feed-forward block, they do not sit beside
 *    it. Counting both turns a 30B model into a 235B one.
 */
export function moe(cfg) {
  const n = int(pick(cfg, "num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts"));
  if (!n) return null;
  const { hidden } = heads(cfg);
  const topk = int(pick(cfg, "num_experts_per_tok", "moe_top_k", "top_k", "num_selected_experts"), 1);
  const shared = int(pick(cfg, "n_shared_experts", "num_experts_shared"), 0);
  const gated = /silu|swish|geglu/.test(String(pick(cfg, "hidden_act") || "").toLowerCase());
  const bias = !!pick(cfg, "mlp_bias");
  const w = int(pick(cfg, "moe_intermediate_size", "expert_intermediate_size",
    "ffn_dim_experts", "d_ff_expert")) || int(pick(cfg, "intermediate_size"));
  const one = gated
    ? lin(hidden, w, bias) * 2 + lin(w, hidden, bias)
    : lin(hidden, w, bias) + lin(w, hidden, bias);
  const router = lin(hidden, n + shared, false);
  return {
    n, topk, shared, width: w, gated,
    per_expert: one,
    params: one * n + one * shared + router,
  };
}

/** Rotary or learned position, whichever the config actually has. */
export function position(cfg, width) {
  const rope = pick(cfg, "rope_theta", "rope_scaling") != null;
  if (rope) return { kind: "RoPE", params: 0, note: "applied to q/k, no parameters" };
  const learned = int(pick(cfg, "max_position_embeddings")) * width;
  if (learned) return { kind: "learned", params: learned };
  const alibi = pick(cfg, "alibi");
  if (alibi) return { kind: "ALiBi", params: 0, note: "bias added to attention scores" };
  return { kind: "none declared", params: 0 };
}

/** The vision tower, for models that have one. */
export function vision(cfg) {
  const v = pick(cfg, "vision_config");
  if (!v) return null;
  const h = int(pick(v, "hidden_size", "hidden_dim", "d_model"));
  const layers = int(pick(v, "num_hidden_layers", "num_layers", "depth"));
  const patch = int(pick(v, "patch_size", "patch_dim"), 16);
  const image = int(pick(v, "image_size"));
  const attn = attention(v), ml = mlp(v);
  const per = attn.params + ml.params + 2 * h;
  return {
    hidden: h, layers, patch, image,
    block: per, blocks: per * layers,
    merge: h * h,
    attn, mlp: ml,
    note: patch ? `patches of ${patch}px` : "patch size not declared",
  };
}

/** What family is this, decided by what the config contains. */
export function family(cfg) {
  if (vision(cfg)) return "vision-language";
  if (pick(cfg, "audio_config", "speech_config")) return "omni";
  // diffusers writes its class into _class_name, not architectures, so a bare
  // unet/config.json or vae/config.json has neither of the two keys this used
  // to look at and fell through to the language builder
  const arch = ((pick(cfg, "architectures") || [])[0]) || "";
  const cls = String(pick(cfg, "_class_name") || "");
  const t = String(pick(cfg, "model_type") || "");
  // _class_name is the only reliable tell: a VAE has up_block_types too, so
  // keying on its presence sent every AutoencoderKL down the UNet path
  if (/Autoencoder/i.test(cls)) return "vae";
  if (pick(cfg, "block_out_channels")) return "diffusion";
  if ([arch, cls].some((a) => /UNet2DConditionModel|DiTTransformer2DModel/.test(a)) ||
      (t === "unet" && pick(cfg, "in_channels"))) return "diffusion";
  if (pick(cfg, "encoder_layers") && pick(cfg, "decoder_layers")) return "speech";
  if (arch.includes("Whisper") || t === "whisper") return "speech";
  if (/AutoencoderKL|AutoencoderDC/.test(cls)) return "vae";
  return "language";
}

/** A shared component: the same shape wherever it appears. */
export const COMMON = {
  tokenEmbedding: "The ids become vectors, one learned row each.",
  attention: "Every head looks at every position at once, then they are averaged.",
  groupedQuery: "Query heads outnumber key/value heads, so K and V are shared.",
  mlp: "Two matrix multiplies with a nonlinearity between them.",
  gatedMlp: "A gate decides how much of each up-projection to keep.",
  rmsNorm: "Rescale by the root mean square — scale only, no recentring.",
  rope: "Rotary positions: a rotation applied to queries and keys, no weights.",
  moe: "A router sends each token to a few of many feed-forward blocks.",
  tiedHead: "The output head reuses the embedding matrix; it is not a second copy.",
  visionPatch: "The picture is cut into patches and each becomes a vector.",
  merger: "Vision vectors are mapped into the language model's width.",
};