# HOST_W7900 — access, hardware, and network constraints

Everything here was verified by hand against this VM on its rebuild date. Where
something could not be determined it says so rather than carrying a number over
from the previous machine, which was a different machine.

---

## 1. Access

```sh
ssh root@36.150.116.206 -p 32638
```

| what | value |
|---|---|
| host | `36.150.116.206` |
| port | **`32638`** — changed on rebuild; the old 31699 is dead |
| user | `root`, no password, key auth (`/root/.ssh/authorized_keys`) |
| hostname | `j-juuvsv56xfbx6cebrj2xv2axgrj` |
| git remote | `git@github.com:ziangziangziang/inferscope.git` (branch `main`) |

> **Old port 31699 is dead.** Anything scripted against it will hang.

There is no `node`/`npm` and no `docker` on this VM. `git`, `python3` (3.12.3)
and `curl` are present. Install node if you need the Vite build.

---

## 2. Hardware (verified)

| | |
|---|---|
| CPU | 2 × AMD EPYC 9334, 32 cores each — **64 cores / 128 threads** |
| RAM | **503 GiB** total, ~407 GiB available |
| OS | Ubuntu 24.04.4 LTS, kernel `6.8.0-79-generic` |
| ROCm | **7.2.4** at `/opt/rocm-7.2.4` (and `/opt/rocm`) |
| GPU arch | `rocminfo` reports **`gfx1100`** (RDNA3) |
| KFD nodes | **10** exposed under `/sys/class/kfd/kfd/topology/nodes/` |
| amdgpu | kernel module loaded (`amdgpu 20250624`); `/dev/kfd` and `/dev/dri/renderD131` present |

### GPU identity is contradictory — resolve before trusting it

`rocm-smi` and `lspci` disagree, and I could not settle it:

- `rocm-smi --showproductname` → `Card Model: 0x744b`, `Card Series: AMD Radeon Graphics`
- `lspci -nn` → **nine** display-class devices, all `1002:744b`
- `rocminfo` → `gfx1100`, which is the discrete Radeon PRO W7900 / RX 7900 class

`744b` is an APU/integrated part number, not a W7900. So either the name is a
labelling convention, or the 744b devices are on-package graphics sitting
alongside a discrete gfx1100 that `rocm-smi` is not enumerating properly.

**VRAM could not be read.** `rocm-smi --showmemtotalvram` returned nothing and
the KFD `mem_banks/*/size` files came back empty. Do **not** assume the 48 GiB
the previous box had. Check it yourself:

```sh
rocm-smi --showproductname --showmeminfo vram
```

---

## 3. Internet — the important part

**This VM has no direct internet egress at all.** Not slow — nothing resolves
out. `github.com`, `google.com` and `huggingface.co` all fail without a proxy.

Egress is **only** through a cluster proxy:

```
http://notebook-egress-proxy.amd-oneclick-lab.svc.cluster.local:3128
```

Note `lablab`, not `lab`. That is not a typo; the `lab` variant does not resolve.

### It is a domain allowlist, and the list is narrow

| allowed | blocked |
|---|---|
| `hf-mirror.com` | **`huggingface.co`** |
| `pypi.org` | `github.com` |
| `registry.npmjs.org` | `raw.githubusercontent.com` |
| `api.github.com` | `google.com` |

So: `pip install` and `npm install` work through the proxy. Cloning a repo over
SSH or `https://github.com/...` does not.

### DNS for `huggingface.co` is poisoned

On this VM it resolves to `2a03:2880:f10d:183:face:b00c:0:25de` — a **Meta IP
range**. HF does not live there. That is a hijacked or intercepted answer, and
it is why `huggingface.co` fails here instantly even before the proxy is
considered.

### `hf-mirror.com` through the proxy is CORS-clean

Verified with an `Origin:` header:

- The redirect stays **on the same host** (`/api/resolve-cache/...`), not a CDN subdomain.
- Both the `307` and the final `200` return `access-control-allow-origin: <origin>`.

That matters because it means a small server route on this VM can act as a CORS
proxy for model configs and a browser can actually use it. `huggingface.co`,
which is the canonical CORS-enabled source, is blocked here.

### Your own machine, for comparison

| host | curl | browser |
|---|---|---|
| `huggingface.co` | ✅ | ✅ **200, CORS-enabled** |
| `hf-mirror.com` | ✅ | ❌ redirect carries no `Access-Control-Allow-Origin` |

So the browser-side viewer works directly against `huggingface.co` from a normal
network. Only `hf-mirror.com` fails in a browser — and only because of the
missing CORS header on its redirect, not because it is unreachable.

---

## 4. State: this VM was wiped

Everything from the previous deployment is **gone**. Verified absent:

```
/workspace/inferscope      GONE      the project
/opt/venv                  GONE      the python environment
/models                    GONE      model weights
/root/.cache/huggingface   GONE      downloaded models
/workspace/models          GONE      model paths
```

`/workspace` is now an empty 98 GiB loop mount (93 GiB free). Root `/` is a
3.5 TiB overlay with ~1.4 TiB free. No services running, no tunnel process.

**What survives is in git**, not on the box. `run.sh` — which holds the proxy,
the model paths and the ports, and which used to exist only on the machine — was
recovered before the wipe and is committed at `ops/run.sh.host`, with deploy
instructions in `ops/README.md`. Read it first; it is the fastest way to rebuild
the environment.

```sh
scp -P 32638 ops/run.sh.host root@36.150.116.206:/workspace/inferscope/run.sh
```

---

## 5. To rebuild the app on this VM

```sh
# 1. clone over SSH (github.com is blocked over https; SSH git is not listed
#    as blocked, but verify before relying on it)
git clone git@github.com:ziangziangziang/inferscope.git /workspace/inferscope

# 2. environment, with the proxy exported for every install
export HTTPS_PROXY=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
export HTTP_PROXY=$HTTPS_PROXY
python3 -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt

# 3. build the UI (needs node — install it first)
apt-get install -y nodejs npm        # via the proxy's apt mirror if needed
npm --prefix ui install
npm --prefix ui run build            # builds UI *and* copies hfarch into dist

# 4. run.sh back in place
scp -P 32638 ops/run.sh.host root@36.150.116.206:/workspace/inferscope/run.sh

# 5. start
cd /workspace/inferscope && setsid ./run.sh > /tmp/inf.log 2>&1 < /dev/null & disown
```

Large models belong on the root overlay, not `/workspace` (93 GiB).

---

## 6. Traps

**Never restart with `pgrep -f "inferscope serve"`.** That pattern matches the
shell you are running the command in, so you kill your own command and then
conclude the service is down. Use:

```sh
kill -9 $(ps -eo pid,cmd | grep "[i]nferscope serve" | awk '{print $1}')
```

**The static mount is last.** `app.mount("/", StaticFiles(ui/dist))` catches
everything, so a mistyped API path returns the app's `index.html` with a **200,
not a 404**. If an endpoint "works" but returns HTML, check it was registered
above the mount.

**`/workspace/inferscope/run.sh` is not in git as a working file** — it is
committed as `ops/run.sh.host`. Copy it into place; do not edit it on the box and
assume the change is saved.

**Per-turn kernel tracing is unavailable** here: `ptrace_scope` is restricted
and the process lacks `CAP_SYS_PTRACE`. Session-level tracing works. If that has
changed on the rebuild, re-check before assuming it is still blocked.

**Do not trust a bare `except`.** One that swallowed an exception sent several
rounds of debugging after the wrong host; name the exception.

---

## 7. Summary for a new agent

- SSH is `root@36.150.116.206` **port 32638**. Old port is dead.
- The box is **bare**: no project, no venv, no models, no docker, no node.
- **No internet without the cluster proxy**, and the proxy allowlists only
  package registries plus `hf-mirror.com`. `huggingface.co` is blocked *and*
  DNS-poisoned here.
- Clone and rebuild from git; `ops/run.sh.host` has the environment that used to
  live only on the old machine.
- Anything that needs a browser to reach the Hub must go through a server route
  on this VM using `hf-mirror.com`, because `huggingface.co` is unreachable here
  while the mirror through the proxy is CORS-clean.