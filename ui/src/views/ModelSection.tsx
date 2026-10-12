import { useState } from "react";
import { Architecture } from "./Architecture";
import { Memory } from "./Memory";
import { Multimodal } from "./Multimodal";
import { Info } from "../components/Info";

/**
 * Static facts about the model, grouped together.
 *
 * These three used to be equal top-level tabs, which implied they were as
 * central as "where did my time go". They are reference material: you consult
 * them while interpreting a capture, not while producing one. Grouping them
 * under one heading keeps them one click away without implying you must visit
 * all three every time.
 */
const TABS = [
  { id: "arch", label: "Architecture" },
  { id: "memory", label: "Memory & KV" },
  { id: "mm", label: "Multimodal" },
] as const;

type TabId = typeof TABS[number]["id"];

export function ModelSection({ tab, setTab }: { tab: TabId; setTab: (t: TabId) => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div className="subtabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""}
            onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
        <span className="faint" style={{ fontSize: 10.5, marginLeft: 8, alignSelf: "center" }}>
          from the model config <Info text="Everything on these tabs is read from the checkpoint's config file. It is derived, not measured from a run." />
        </span>
      </div>
      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        {tab === "arch" && <Architecture />}
        {tab === "memory" && <Memory />}
        {tab === "mm" && <Multimodal />}
      </div>
    </div>
  );
}

export function useModelTab(): [TabId, (t: TabId) => void] {
  const [tab, setTab] = useState<TabId>("arch");
  return [tab, setTab];
}