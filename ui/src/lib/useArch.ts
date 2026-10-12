// Resolve an architecture graph with a fallback chain:
//   explicit model  >  session's stored architecture  >  first discovered model.
// Architecture belongs to the model, so a session without one (no engine running)
// still shows the host's available checkpoint graph rather than an error.
import { useEffect, useState } from "react";
import { api } from "./api";
import type { Graph, ModelInfo } from "./api";

export function useArch(sessionId: string, override: string) {
  const [graph, setGraph] = useState<Graph | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => { api.models().then((r) => setModels(r.models)).catch(() => {}); }, []);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    (async () => {
      try {
        if (override) { setGraph(await api.modelArch(override)); return; }
        if (sessionId) {
          try { setGraph(await api.sessionArch(sessionId)); return; }
          catch { /* fall through to first model */ }
        }
        if (models[0]) setGraph(await api.modelArch(models[0].model_id));
        else setGraph(null);
      } catch { if (alive) setGraph(null); }
      finally { if (alive) setLoading(false); }
    })();
    return () => { alive = false; };
  }, [sessionId, override, models]);

  return { graph, models, loading };
}
