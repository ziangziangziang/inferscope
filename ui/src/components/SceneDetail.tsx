/**
 * The detail panel.
 *
 * Slides in from the right and takes half the screen. Whatever is selected stays
 * highlighted in the scene while everything else dims, so the panel and the
 * diagram are two views of one selection rather than two separate screens.
 */

import { fmtParams } from "./sceneModel";
import type { ArchGraph, ArchNode, Sel } from "./archTypes";

export interface DetailProps {
  sel: Sel;
  arch: ArchGraph | null;
  modelName: string;
  running: boolean;
  tokens: number;
  onSelect: (s: NonNullable<Sel>) => void;
  onClose: () => void;
}

function Row({ label, value, on, onClick }:
  { label: string; value: string; on?: boolean; onClick?: () => void }) {
  const Cmp = onClick ? "button" : "div";
  return (
    <Cmp className={"side-row" + (on ? " on" : "")} onClick={onClick} type={onClick ? "button" : undefined}>
      <b>{label}</b><span>{value}</span>
    </Cmp>
  );
}

/** A sub-layer row with a bar sized by its share of the block's parameters. */
function SubRow({ n, total, on, onClick }:
  { n: ArchNode; total: number; on?: boolean; onClick?: () => void }) {
  const pct = total > 0 ? Math.max(1.5, (n.params / total) * 100) : 1.5;
  return (
    <button className={"side-row sub" + (on ? " on" : "")} onClick={onClick} type="button">
      <span className="sub-bar" style={{ width: `${pct}%` }} aria-hidden />
      <b>{n.label}</b>
      <span>{n.params ? fmtParams(n.params) : "—"}</span>
    </button>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return <><dt>{k}</dt><dd>{v}</dd></>;
}

function nodeById(arch: ArchGraph | null, id: string): ArchNode | undefined {
  return arch?.nodes.find((n) => n.id === id);
}

export function SceneDetail(props: DetailProps) {
  const { sel, arch, modelName, running, tokens, onSelect, onClose } = props;
  const meta = arch?.meta ?? {};
  const total = arch?.nodes.reduce((a, n) => a + n.params, 0) ?? 0;

  let body: React.ReactNode = null;
  let title = "";
  let role = "";

  if (sel?.kind === "engine") {
    title = "llama.cpp";
    role = "the inference engine";
    body = (
      <>
        <p className="side-note">
          The engine is the program that actually runs the model. It reads the
          weights, does the arithmetic, and hands back the next token. It is not
          the model — swap the engine and the same weights still describe the
          same model.
        </p>
        <dl className="side-kv">
          <KV k="state" v={running ? "running" : "idle"} />
          <KV k="tokens" v={String(tokens)} />
          <KV k="loaded" v={modelName || "—"} />
          <KV k="port" v="127.0.0.1:8081" />
        </dl>
        <p className="side-hint">
          The engine box is drawn dashed because it is software: it has no size of
          its own, it borrows the hardware below it.
        </p>
      </>
    );
  } else if (sel?.kind === "model") {
    title = modelName.replace(/\.gguf$/i, "");
    role = "the model weights";
    body = (
      <>
        <p className="side-note">
          The model is just numbers — a few tens of billions of them. The engine
          reads them in order, once per layer, to predict the next token. Every
          box in the diagram above is one group of those numbers, drawn at the
          size of the group it really holds.
        </p>
        <dl className="side-kv">
          <KV k="family" v={String(meta.family ?? arch?.family ?? "—")} />
          <KV k="parameters" v={total ? total.toLocaleString() : "—"} />
          <KV k="layers" v={String(meta.n_layers ?? "—")} />
          <KV k="hidden" v={String(meta.hidden ?? "—")} />
          <KV k="heads" v={`${meta.n_heads ?? "—"} q / ${meta.kv_heads ?? "—"} kv`} />
          <KV k="vocab" v={String(meta.vocab ?? "—")} />
          {meta.vision ? <KV k="vision" v={`${(meta.vision as Record<string, unknown>).depth} blocks`} /> : null}
        </dl>
        <div className="side-rows">
          {arch?.nodes.filter((n) => n.params > 0 || n.repeat).map((n) => (
            <Row key={n.id} label={n.label}
              value={n.params ? fmtParams(n.params) : "no weights"}
              onClick={() => onSelect({ kind: "node", id: n.id })} />
          ))}
        </div>
        <p className="side-hint">
          Click any box in the diagram to move into it.
        </p>
      </>
    );
  } else if (sel?.kind === "unit") {
    const group = nodeById(arch, sel.id);
    const sub = group?.repeat?.template?.nodes ?? [];
    const per = group && sel.index != null
      ? Math.round(group.params / Math.max(1, group.repeat?.count ?? 1)) : 0;
    const ltypes = (meta.layer_types as string[] | undefined) ?? [];
    const lt = ltypes[sel.index ?? -1];
    title = group ? `${group.label}` : "block";
    role = `unit ${(sel.index ?? 0) + 1} of ${group?.repeat?.count ?? 0}`;
    body = (
      <>
        <p className="side-note">
          {group?.repeat?.count ? (
            <>
              This model repeats the same block {group.repeat.count} times. Rather
              than draw {group.repeat.count} separate towers, the diagram wraps
              them into a grid — every box below is one real instance, sized by
              the parameters it holds. You are looking at unit{" "}
              <b>#{(sel.index ?? 0) + 1}</b>.
            </>
          ) : "One unit of the repeated group."}
        </p>
        <dl className="side-kv">
          <KV k="parameters" v={per ? per.toLocaleString() : "—"} />
          <KV k="group total" v={group?.params ? group.params.toLocaleString() : "—"} />
          {lt ? <KV k="attention" v={lt.replace(/_/g, " ")} /> : null}
        </dl>
        {lt === "linear_attention" ? (
          <p className="side-note">
            This unit uses linear attention (a gated deltanet), not full
            attention — the model alternates the two, roughly one full-attention
            layer in every {String(meta.full_interval ?? 4)}.
          </p>
        ) : null}
        {sub.length ? (
          <>
            <div className="side-role" style={{ marginTop: 14 }}>sub-layers inside this block</div>
            <div className="side-rows">
              {sub.map((n) => (
                <SubRow key={n.id} n={n} total={per} />
              ))}
            </div>
          </>
        ) : null}
      </>
    );
  } else if (sel?.kind === "node") {
    const n = nodeById(arch, sel.id);
    title = n?.label ?? sel.id;
    role = n?.kind ?? "component";
    body = (
      <>
        <dl className="side-kv">
          <KV k="kind" v={n?.kind ?? "—"} />
          <KV k="parameters" v={n?.params ? n.params.toLocaleString() : "0"} />
          {n?.repeat ? <KV k="repeated" v={`${n.repeat.count}×`} /> : null}
          {n?.provenance ? <KV k="from" v={n.provenance} /> : null}
        </dl>
        {n?.notes?.filter(Boolean).map((s, i) => (
          <p className="side-note" key={i}>{s}</p>
        ))}
        {n?.dims ? (
          <dl className="side-kv">
            {Object.entries(n.dims).map(([k, v]) => (
              <KV key={k} k={k} v={Array.isArray(v) ? v.join(" × ") : String(v)} />
            ))}
          </dl>
        ) : null}
        {n?.params && total ? (
          <>
            <div className="side-role">share of the model</div>
            <div className="side-rows">
              <Row label={n.label} value={`${((n.params / total) * 100).toFixed(1)}%`} on />
            </div>
          </>
        ) : null}
      </>
    );
  } else {
    title = "the host";
    role = "four parts on one plate";
    body = (
      <p className="side-note">
        A model cannot think on its own. The four boxes on the plate are the
        machine that does the thinking: the two that compute, the two that hold
        things. The fill level on each one is how busy it was during the last run.
      </p>
    );
  }

  return (
    <div className="side-inner" role="region" aria-label={`details for ${title}`}>
      <button className="btn side-close" onClick={onClose} aria-label="close details">close</button>
      <h3>{title}</h3>
      <div className="side-role">{role}</div>
      {body}
    </div>
  );
}