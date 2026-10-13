import { UnitPicker } from "./UnitPicker";
import { ResidentStrip } from "./EngineRail";
import { useGpu } from "../gpuState";

/**
 * The configurable panel: what should be on the GPU.
 *
 * It lives in the left rail rather than on the workbench because it is not
 * part of any one view — it decides what the machine is doing, so it should
 * stay reachable from Analyze, Status or anywhere else.
 */
export function GpuPanel() {
  const g = useGpu();
  return (
    <>
      <div className="section">on the gpu</div>
      <UnitPicker catalogue={g.catalogue} resident={g.shownEngine}
        onPick={g.pickUnit} swap={g.swap} disabled={g.swap.phase !== "idle"} />
      {g.err ? <div className="notice bad rail-err">{g.err}</div> : null}
      <ResidentStrip status={g.status} unit={g.activeUnit}
        llmRunning={g.running}
        onChanged={g.refreshEngines}
        onOffloaded={() => { g.setChosen(null); g.refreshEngines(); }} />
    </>
  );
}