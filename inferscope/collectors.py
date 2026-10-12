"""Hardware/system telemetry collectors.

Each collector returns Sample rows with per-field provenance. A field the
platform cannot supply stays None (rendered "unavailable"), never 0. GPU
telemetry prefers amdsmi (exact byte counts), enriches temperature from sysfs,
and degrades to sysfs or rocm-smi when amdsmi is absent.

On gfx1100 both rocm-smi and amdsmi report GPU activity as 0 even under load, so
gpu_util_pct is deliberately left None with a reason; VRAM occupancy is the
honest activity signal on this part.
"""
from __future__ import annotations

import glob
import subprocess
from typing import Optional

from .schema import Provenance, Sample, Source, now_mono_ns, now_wall_ns


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:  # noqa: BLE001
        return None


class SystemCollector:
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self._proc = psutil.Process()
            self.available = True
        except Exception:  # noqa: BLE001
            self.psutil = None
            self.available = False

    def poll(self, session_id: str) -> Optional[Sample]:
        if not self.available:
            return None
        ps = self.psutil
        cpu = ps.cpu_percent(interval=None)      # first call primes, returns 0.0
        per_core = ps.cpu_percent(interval=None, percpu=True)
        vm = ps.virtual_memory()
        try:
            la = ps.getloadavg()
        except Exception:  # noqa: BLE001
            la = (None, None, None)
        try:
            rss = self._proc.memory_info().rss
        except Exception:  # noqa: BLE001
            rss = None
        return Sample(
            session_id=session_id, ts_wall_ns=now_wall_ns(), ts_mono_ns=now_mono_ns(),
            source=Source.PSUTIL, provenance=Provenance.MEASURED,
            cpu_util_pct=cpu, load1=la[0], load5=la[1], load15=la[2],
            ram_used_bytes=vm.used, ram_total_bytes=vm.total, proc_rss_bytes=rss,
            per_core=per_core,
        )


class GpuCollector:
    def __init__(self, caps: dict):
        self.backend = self._choose(caps)
        self.amdsmi = None
        self._handle = None
        if self.backend == "amdsmi":
            try:
                import amdsmi
                amdsmi.amdsmi_init()
                hs = amdsmi.amdsmi_get_processor_handles()

                def vt(h):
                    try:
                        return int(amdsmi.amdsmi_get_gpu_memory_total(
                            h, amdsmi.AmdSmiMemoryType.VRAM))
                    except Exception:  # noqa: BLE001
                        return 0

                self.amdsmi = amdsmi
                self._handle = max(hs, key=vt) if hs else None
                if self._handle is None:
                    self.backend = self._fallback(caps)
            except Exception:  # noqa: BLE001
                self.backend = self._fallback(caps)
        self._sysfs = self._find_sysfs()

    @staticmethod
    def _fallback(caps) -> str:
        if caps.get("sysfs_gpu") and caps["sysfs_gpu"].available:
            return "sysfs"
        if caps.get("rocm_smi") and caps["rocm_smi"].available:
            return "rocm-smi"
        return "none"

    def _choose(self, caps) -> str:
        if caps.get("amdsmi") and caps["amdsmi"].available:
            return "amdsmi"
        return self._fallback(caps)

    @staticmethod
    def _find_sysfs() -> Optional[str]:
        best, best_total = None, 0
        for d in glob.glob("/sys/class/drm/card*/device"):
            total = _read_int(d + "/mem_info_vram_total") or 0
            if total > best_total:
                best, best_total = d, total
        return best

    def poll(self, session_id: str) -> Optional[Sample]:
        if self.backend == "amdsmi":
            return self._poll_amdsmi(session_id)
        if self.backend == "sysfs":
            return self._poll_sysfs(session_id)
        if self.backend == "rocm-smi":
            return self._poll_rocm(session_id)
        return None

    def _base(self, session_id: str, source: Source, prov: Provenance) -> Sample:
        return Sample(session_id=session_id, ts_wall_ns=now_wall_ns(),
                      ts_mono_ns=now_mono_ns(), source=source, provenance=prov)

    def _sysfs_temp(self) -> Optional[float]:
        if not self._sysfs:
            return None
        hw = glob.glob(self._sysfs + "/hwmon/*/")
        if not hw:
            return None
        t = _read_int(hw[0] + "temp1_input")
        return t / 1000.0 if t is not None else None

    def _poll_amdsmi(self, session_id: str) -> Optional[Sample]:
        if self._handle is None:
            return None
        a = self.amdsmi
        s = self._base(session_id, Source.AMD_SMI, Provenance.MEASURED)
        # VRAM: exact bytes
        try:
            mem = a.AmdSmiMemoryType.VRAM
            s.vram_used_bytes = int(a.amdsmi_get_gpu_memory_usage(self._handle, mem))
            s.vram_total_bytes = int(a.amdsmi_get_gpu_memory_total(self._handle, mem)) or None
        except Exception:  # noqa: BLE001
            try:
                vu = a.amdsmi_get_gpu_vram_usage(self._handle)
                s.vram_used_bytes = int(vu["vram_used"]) * 2**20
                s.vram_total_bytes = int(vu["vram_total"]) * 2**20 or None
            except Exception:  # noqa: BLE001
                s.vram_used_bytes = s.vram_total_bytes = None
        if s.vram_used_bytes is not None and s.vram_total_bytes:
            s.mem_used_pct = 100.0 * s.vram_used_bytes / s.vram_total_bytes
        # clocks (MHz): SOC as graphics clock, MEM as memory clock
        try:
            soc = a.amdsmi_get_clock_info(self._handle, a.AmdSmiClkType.SOC)
            if soc.get("clk") is not None:
                s.sclk_mhz = float(soc["clk"])
        except Exception:  # noqa: BLE001
            pass
        try:
            mc = a.amdsmi_get_clock_info(self._handle, a.AmdSmiClkType.MEM)
            if mc.get("clk") is not None:
                s.mclk_mhz = float(mc["clk"])
        except Exception:  # noqa: BLE001
            pass
        # temperature: amdsmi, else sysfs
        try:
            tt = a.AmdSmiTemperatureType
            ttype = getattr(tt, "EDGE", None) or next(iter(tt))
            raw = a.amdsmi_get_temp_metric(
                self._handle, ttype, a.AmdSmiTemperatureMetric.CURRENT)
            s.temp_c = raw / 1000.0 if raw and raw > 1000 else (float(raw) if raw else None)
        except Exception:  # noqa: BLE001
            s.temp_c = None
        if s.temp_c is None:
            s.temp_c = self._sysfs_temp()
        # power: µW -> W, guard implausible idle reads
        try:
            pi = a.amdsmi_get_power_info(self._handle)
            for k in ("current_socket_power", "average_socket_power", "socket_power"):
                v = pi.get(k)
                if isinstance(v, (int, float)) and v >= 1e5:
                    s.power_w = float(v) / 1e6
                    break
            lim = pi.get("power_limit")
            if isinstance(lim, (int, float)) and lim:
                s.extra["power_limit_w"] = float(lim) / 1e6
        except Exception:  # noqa: BLE001
            pass
        # GPU / memory-controller utilisation. amdsmi reports this on gfx1100 and
        # it moves with load (verified: gfx_activity 100 -> 0 -> 0 across samples).
        # The earlier "unavailable" claim was a mistake — it conflated rocm-smi's
        # broken `--showuse` (always 0 on RDNA3) with an idle amdsmi sample.
        try:
            act = a.amdsmi_get_gpu_activity(self._handle)
            gfx = act.get("gfx_activity")
            umc = act.get("umc_activity")
            if gfx is not None:
                s.gpu_util_pct = float(gfx)
            if umc is not None:
                s.extra["mem_activity_pct"] = float(umc)   # bandwidth-pressure proxy
        except Exception:  # noqa: BLE001
            s.gpu_util_pct = None
            s.reason = "amdsmi_get_gpu_activity failed on this device"
        return s

    def _poll_sysfs(self, session_id: str) -> Optional[Sample]:
        d = self._sysfs
        if not d:
            return None
        s = self._base(session_id, Source.SYSFS, Provenance.MEASURED)
        s.vram_used_bytes = _read_int(d + "/mem_info_vram_used")
        total = _read_int(d + "/mem_info_vram_total")
        s.vram_total_bytes = total or None
        if s.vram_used_bytes is not None and total:
            s.mem_used_pct = 100.0 * s.vram_used_bytes / total
        s.temp_c = self._sysfs_temp()
        s.gpu_util_pct = None
        s.reason = "gfx1100: GPU util unavailable via sysfs; VRAM occupancy shown."
        return s

    def _poll_rocm(self, session_id: str) -> Optional[Sample]:
        try:
            out = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--showtemperature",
                 "--showpower", "--showclocks", "--csv"],
                capture_output=True, text=True, timeout=8).stdout
        except Exception:  # noqa: BLE001
            return None
        s = self._base(session_id, Source.ROCM_SMI, Provenance.MEASURED)
        for line in out.splitlines():
            cells = [c.strip().strip('"') for c in line.split(",")]
            nums = [float(x) for x in cells if x.replace(".", "", 1).isdigit()]
            if not nums:
                continue
            low = line.lower()
            if "vram" in low and len(nums) >= 2:
                s.vram_used_bytes, s.vram_total_bytes = int(nums[0]), int(nums[1]) or None
            elif "temp" in low and s.temp_c is None:
                s.temp_c = nums[0]
        if s.vram_used_bytes and s.vram_total_bytes:
            s.mem_used_pct = 100.0 * s.vram_used_bytes / s.vram_total_bytes
        s.gpu_util_pct = None
        return s

    def close(self) -> None:
        if self.backend == "amdsmi" and self.amdsmi is not None:
            try:
                self.amdsmi.amdsmi_shut_down()
            except Exception:  # noqa: BLE001
                pass
