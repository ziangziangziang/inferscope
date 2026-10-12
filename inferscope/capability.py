"""Detect what this machine can actually report, and say so precisely.

Nothing here may raise: a missing GPU, a missing CLI, a dead engine endpoint or a
blocked /sys path must each turn into an explicit capability with a reason, never
an exception and never a silently-zeroed metric. This is the module the whole
"unavailable vs unsupported vs not configured vs zero" discipline rests on.
"""
from __future__ import annotations

import glob
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Capability:
    name: str
    available: bool
    detail: str = ""
    notes: list[str] = field(default_factory=list)
    # For GPU metrics: which specific fields this platform exposes.
    supports: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "notes": self.notes,
            "supports": self.supports,
        }


def _run(cmd: list[str], timeout: float = 6.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or r.stderr or "").strip()
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:  # noqa: BLE001 — capability probe must never raise
        return 1, f"{type(e).__name__}: {e}"


def probe_psutil() -> Capability:
    try:
        import psutil  # noqa: F401
        return Capability("system_cpu_ram", True, "psutil present")
    except Exception as e:  # noqa: BLE001
        return Capability("system_cpu_ram", False, f"psutil unavailable: {e}")


def probe_sysfs_gpu() -> Capability:
    devs = []
    for p in sorted(glob.glob("/sys/class/drm/card*/device")):
        if glob.glob1(p, "mem_info_vram_used") or glob.glob1(p, "hwmon"):
            devs.append(p)
    if not devs:
        return Capability("sysfs_gpu", False, "no /sys/class/drm GPU nodes")
    return Capability("sysfs_gpu", True, f"{len(devs)} DRM device node(s)")


def probe_amdsmi() -> Capability:
    """Import amdsmi, init, and probe each getter we rely on individually.

    Field availability differs across ROCm versions and even between identical
    GPUs, so we record support per metric rather than assuming the library
    exposes everything the docs list.
    """
    cap = Capability("amdsmi", False, "not importable")
    try:
        import amdsmi
    except Exception as e:  # noqa: BLE001
        cap.detail = f"import failed: {e}"
        return cap

    cap.available = True
    cap.detail = f"version {getattr(amdsmi, '__version__', '?')}"
    try:
        amdsmi.amdsmi_init()
    except Exception as e:  # noqa: BLE001
        cap.notes.append(f"init failed: {e}")
        return cap
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            cap.notes.append("no GPU handles")
            return cap
        h = handles[0]
        probes = {
            "temp": "amdsmi_get_temp_metric",
            "vram": "amdsmi_get_vram_usage",
            "power": "amdsmi_get_power_info",
            "clock": "amdsmi_get_clock_info",
            "activity": "amdsmi_get_gpu_activity",
        }
        for key, fn in probes.items():
            cap.supports[key] = hasattr(amdsmi, fn)
        # gfx arch string, if exposed.
        try:
            info = amdsmi.amdsmi_get_gpu_device_type(h)
            cap.notes.append(f"device_type={info}")
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        cap.notes.append(f"probe error: {e}")
    finally:
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:  # noqa: BLE001
            pass
    return cap


def probe_rocm_smi() -> Capability:
    path = shutil.which("rocm-smi")
    if not path:
        return Capability("rocm_smi", False, "rocm-smi not on PATH")
    rc, out = _run([path, "--version"], timeout=8)
    first = out.splitlines()[0] if out else ""
    return Capability("rocm_smi", rc == 0, f"{path}: {first}" if rc == 0 else first or f"rc={rc}")


def probe_rocprof() -> Capability:
    path = shutil.which("rocprofv3")
    if not path:
        return Capability("rocprofv3", False, "rocprofv3 not on PATH")
    rc, out = _run([path, "--version"], timeout=8)
    ver = out.splitlines()[0] if out else ""
    cap = Capability("rocprofv3", rc == 0, f"{ver}",
                     notes=["attach to running process requires ptrace (often "
                            "unavailable); captures run only under bounded, "
                            "self-launched processes"])
    # Confirm the tracing knobs we depend on exist, by reading --help.
    rc2, helpout = _run([path, "--help"], timeout=8)
    helpout = helpout.lower()
    cap.supports = {
        "kernel_trace": "--kernel-trace" in helpout,
        "hip_runtime_api": "--hip-runtime-traces" in helpout or "hip-trace" in helpout,
        "perfetto_output": "pp" in helpout or "perfetto" in helpout,
        "output_format": "-o" in helpout or "--output-format" in helpout,
    }
    return cap


def probe_gpu_arch() -> Capability:
    path = shutil.which("rocminfo")
    if path:
        rc, out = _run([path], timeout=10)
        for line in out.splitlines():
            if "gfx" in line and "Name:" in line:
                arch = line.split(":")[-1].strip()
                return Capability("gpu_arch", True, arch)
    return Capability("gpu_arch", False, "arch unknown")


def probe_endpoint(name: str, url: str) -> Capability:
    if not url:
        return Capability(name, False, "not configured",
                          notes=["set INFERSCOPE_%s_METRICS to a /metrics URL" % name.upper()])
    try:
        import httpx
        r = httpx.get(url, timeout=5.0)
        ok = r.status_code == 200 and bool(r.text)
        sample = r.text.splitlines()[0][:80] if ok else f"HTTP {r.status_code}"
        return Capability(name, ok, sample)
    except Exception as e:  # noqa: BLE001
        return Capability(name, False, f"unreachable: {e}")


def probe_all(config) -> dict[str, Capability]:
    """Run every probe, tolerating individual failures. Returns a capability map."""
    caps: dict[str, Capability] = {}
    for name, fn in [
        ("system_cpu_ram", probe_psutil),
        ("sysfs_gpu", probe_sysfs_gpu),
        ("amdsmi", probe_amdsmi),
        ("rocm_smi", probe_rocm_smi),
        ("rocprofv3", probe_rocprof),
        ("gpu_arch", probe_gpu_arch),
    ]:
        try:
            caps[name] = fn()
        except Exception as e:  # noqa: BLE001
            caps[name] = Capability(name, False, f"probe crashed: {e}")

    caps["llamacpp"] = probe_endpoint("llamacpp", getattr(config, "llamacpp_metrics_url", ""))
    caps["vllm"] = probe_endpoint("vllm", getattr(config, "vllm_metrics_url", ""))

    # gfx1100 known gaps, discovered live rather than assumed.
    arch = caps.get("gpu_arch")
    if arch and arch.available and "gfx11" in arch.detail:
        caps.setdefault("notes", Capability("notes", True, "platform caveats"))
        caps["notes"].notes.append(
            "gfx1100: rocm-smi '--showuse' GPU-use% is broken (always 0) — use "
            "amdsmi gpu_activity instead (InferScope's primary path).")
        caps["notes"].notes.append(
            "gfx1100: rocprofv3 DRAM/FETCH_SIZE byte counters read 0, so measured "
            "memory-bandwidth bytes are unavailable; amdsmi umc_activity% is used "
            "as a bandwidth-pressure proxy.")
    return caps
