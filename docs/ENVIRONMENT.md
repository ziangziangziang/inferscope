# Environment findings

Recorded during initial inspection (read-only) of the remote GPU host. Verify
with `inferscope check` — the app re-probes at runtime rather than trusting this
file.

## Host

| | |
|---|---|
| Access | `ssh -p 31699 root@36.150.116.206` (k8s pod, hostname `j-c5yn…`) |
| OS | Ubuntu 24.04 (container) |
| Kernel | `6.8.0-79-generic` (matches `uname -r`) |
| CPU | AMD EPYC 9334, 128 cores |
| RAM | ~503 GiB (≈446 available) |
| Disk | `/` 1.7 TB free; `/workspace` only ~28 GB free → install under `/workspace` but keep large artifacts on `/` |

## GPU

| | |
|---|---|
| Model | AMD Radeon PRO W7900 class, `gfx1100`, 48 GiB VRAM (51,522,830,336 B) |
| CUs | 48 |
| Driver/ROCm | ROCm 7.2.4; amdgpu 6.16.13; ROCM-SMI 4.0.0 |
| rocprofv3 | 1.1.0 — supports `--kernel-trace`, hip-runtime traces, CSV + Perfetto output |

## Kernel-version "discrepancy"

`rocprofv3 --version` reports a different build string than `uname -r`. This is
the ROCm userspace build stamp, not a claim about the running kernel; the running
kernel is `6.8.0-79-generic`. Not relevant to InferScope — we do not act on it and
do not touch the kernel.

## Telemetry tooling reality (probed, not assumed)

- **`amdsmi` Python bindings present (7.0.2)** even though the `amdsmi` CLI is
  absent. Exact VRAM via `amdsmi_get_gpu_memory_usage(VRAM)`; clocks via
  `AmdSmiClkType.SOC/MEM`; power via `amdsmi_get_power_info`.
- **GPU utilisation IS available via amdsmi** — `amdsmi_get_gpu_activity` returns
  a live `gfx_activity`/`umc_activity` that tracks load (verified: 0 → 100 → 98 →
  0 across a generation). The earlier "reads 0 under load" note was wrong: it
  conflated `rocm-smi --showuse` (which genuinely always returns 0 on RDNA3) with
  an idle amdsmi sample. InferScope reads amdsmi activity and reports it
  `measured`; `rocm-smi` is only a fallback where util is then unavailable.
- **rocprofv3 DRAM / `FETCH_SIZE` byte counters read 0 on gfx1100** — measured
  memory-bandwidth *bytes* are unavailable, so bandwidth pressure is modelled
  (`estimated`). amdsmi `umc_activity`% is used as a bandwidth-pressure proxy.
- **rocprofv3 cannot attach to a running process** (no ptrace). Captures run only
  under a bounded, self-launched command.
- **`/sys/class/drm/card*/device`** exposes `mem_info_vram_used/total` and
  `hwmon/temp1_input` — used as a fallback and for temperature.

## Engines

- No llama.cpp or vLLM server was running at inspection time (only defunct
  `vllm`/`EngineCore` zombie entries). The GPU showed heavy VRAM use from a
  workload not matched by our process list — treated as **the user's, hands-off**.
- Consequence: the llama.cpp/vLLM adapters are implemented and verified against
  realistic Prometheus fixtures, and are config-driven (`INFERSCOPE_*_METRICS`),
  but were **not** exercised against a live server (none running; we do not start
  one).

## Egress

- Outbound access is via a filtering proxy that is **intermittent**: PyPI/HF API
  endpoints sometimes return 200, but large file bodies and many hosts time out.
- Therefore InferScope installs **nothing new**: it uses the pre-existing
  `/opt/venv` (fastapi, uvicorn, duckdb, pyarrow, psutil, prometheus-client,
  httpx, pydantic) and builds the frontend on the workstation, shipping only the
  static `dist/`.

## Models on disk (`/workspace/models`)

- `Qwen3.8-27B-INT4` — **a vision-language model** (`Qwen3_5ForConditionalGeneration`,
  64 hybrid LM layers + 27-block vision tower, W4 compressed-tensors, vision tower
  unquantized). Used as the real config-architecture example.
- `Qwen3.5-4B`, `unsloth_Qwen3-VL-8B-Instruct-GGUF`, GGUF/hipfire packs.
