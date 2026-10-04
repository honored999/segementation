# Linux server download and initial CPU validation

Manual independent Level 3 PASS; original DDP P1 CLOSED. User authorized branch commit/push on 2026-10-04. This branch is not merged. Server execution belongs to the user. No server/CUDA success is claimed.

## Download an isolated checkout

Activate the existing nnU-Net environment (do not install/upgrade dependencies). Use the exact delivered commit from the implementation owner's final message. Clone refuses an existing non-empty target; do not reset an existing checkout. Keep full branch history because compatibility tests read the frozen baseline through git show.

```bash
SERVER_REPO="$HOME/work/segementation-upernet-configurable-stages"
mkdir -p "$HOME/work"
git clone --branch codex/upernet-configurable-stages --single-branch https://github.com/honored999/segementation.git "$SERVER_REPO"
cd "$SERVER_REPO"
git checkout --detach <DELIVERED_COMMIT_SHA>
git rev-parse HEAD
git status --short
# Activate your existing server environment, e.g. conda activate newconda.
export nnUNet_extTrainer="$PWD/nnunet_ext_trainers"
export nnUNet_compile=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
SERVER_EVIDENCE=$(mktemp -d /tmp/upernet-server-check.XXXXXXXX)
export TMPDIR="$SERVER_EVIDENCE"
printf 'Evidence directory: %s\n' "$SERVER_EVIDENCE"
```

Do not execute the Windows-only run_validation.py on Linux. Historical Windows raw-byte manifests can differ from Linux checkout bytes due to Git CRLF/LF conversion; the delivered Git commit and clean diff identify the server source. Do not modify the historical manifests.

## Resource guard before every runtime check/test

Define this shell function once. It uses Python standard-library /proc and installed nvidia-smi, queries all GPUs, fails closed if monitoring is unavailable, and refuses a test at >=80%. No psutil installation is needed. Checks below are serial CPU synthetic work with one thread, not a real model/data workload. Current utilization is not a measured peak guarantee; defer broader or GPU testing unless headroom is established.

```bash
resource_preflight() {
python - <<'PY'
import json, math, subprocess, time
from pathlib import Path

def cpu():
    # Exclude guest/guest_nice, already included in user/nice.
    values = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))[:8]
    return sum(values), values[3] + values[4]
a = cpu()
time.sleep(1)
b = cpu()
assert b[0] > a[0], 'CPU sample unavailable'
cpu_percent = 100 * (1 - (b[1]-a[1])/(b[0]-a[0]))
mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
ram_percent = 100 * (1-mem['MemAvailable']/mem['MemTotal'])
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
assert rows, 'GPU sample unavailable'
gpus = []
for row in rows:
    index, utilization, used, total = map(float, row.split(','))
    assert total > 0 and all(math.isfinite(x) for x in [index, utilization, used, total])
    gpus.append(dict(index=int(index), gpu_percent=utilization, vram_percent=100*used/total))
values = [cpu_percent, ram_percent] + [x for g in gpus for x in [g['gpu_percent'], g['vram_percent']]]
assert all(math.isfinite(x) for x in values)
high = any(x >= 80 for x in values)
print(json.dumps(dict(cpu_percent=cpu_percent, ram_percent=ram_percent, gpus=gpus, decision='DEFERRED_RESOURCE_GUARD' if high else 'NORMAL')))
raise SystemExit(80 if high else 0)
PY
}
```

## Check installed runtime, then run the focused tests

```bash
set -euo pipefail
resource_preflight | tee "$SERVER_EVIDENCE/runtime-preflight.log" &&
python - <<'PY' | tee "$SERVER_EVIDENCE/runtime.log"
import importlib.metadata as md
import sys, torch
print('python', sys.version)
print('nnunetv2', md.version('nnunetv2'))
print('torch', torch.__version__, 'built_cuda', torch.version.cuda)
assert md.version('nnunetv2') == '2.8.1', 'Version mismatch: stop and report; do not upgrade automatically.'
PY
# Continue only if the preceding command exits 0.
resource_preflight | tee "$SERVER_EVIDENCE/focused-preflight.log" &&
python -m pytest nnunet_ext_trainers/tests/test_upernet_selected_stages.py -q -p no:cacheprovider --basetemp "$SERVER_EVIDENCE/focused-tmp" 2>&1 | tee "$SERVER_EVIDENCE/focused.log"
# Continue only if the focused command exits 0. Recheck resources for legacy regression.
resource_preflight | tee "$SERVER_EVIDENCE/legacy-preflight.log" &&
python -m pytest nnunet_ext_trainers/tests/test_official_upernet_trainers.py -q -p no:cacheprovider --basetemp "$SERVER_EVIDENCE/legacy-tmp" 2>&1 | tee "$SERVER_EVIDENCE/legacy.log"
git diff --exit-code
git status --short
```

Expected counts from the reviewed local environment: focused 61 passed, legacy 25 passed. Counts are a reference, not a server PASS until actual exit codes/results are observed. If resource preflight refuses, do not launch the file/suite. Defer it, or independently justify an exact metadata-only node with another fresh preflight. Do not terminate other workloads or reduce production patch/batch/model settings to obtain a PASS.

Return: exact commit, Python/nnunetv2/PyTorch versions, all preflight logs, both pytest summaries/exit codes and any failure traceback. This establishes Linux environment and CPU synthetic compatibility only. CPU DDP fixtures use the disclosed parent initialization shim; they do not establish actual CUDA/SyncBatchNorm/NCCL or multi-rank behavior.

Do not proceed directly to real training/--val/prediction, load real checkpoints, alter raw data/splits, or run full-size eight-stage benchmarks. CUDA/official-initialization/multi-rank preflight is a separate bounded validation step after these logs are reviewed. Plans-copy and real-data operating instructions are in nnunet_ext_trainers/README_upernet_selected_stages.md; preprocessing is reused and each combination has a distinct plans/results identity.

## Windows server: initial CUDA0 synthetic check

User-reported Windows server CPU checks on 2026-10-04: focused 61 passed; legacy 25 passed; both exit 0. Environment: Python 3.10.20, nnunetv2 2.8.1, PyTorch 2.11.0+cu128, built CUDA 12.8. A 279-character synthetic training log path failed; one-character validation labels passed without changing production/tests. Preserve the failed evidence.

A new task-local probe reuses the existing synthetic plans/labels and exercises real official Trainer initialization (no shim), inherited FP32 TopK10 loss, two tiny 17x17 forward/backward iterations, complete encoder execution/used and unused gradients, strict checkpoint reconstruction and pre-write mismatch refusal. Four encoder stages/two channels, FPN128/PPM(1,2,4) retained; selections [0,2] and [0,3]. No optimizer step, real training/data, AMP, torch.compile, DDP or full-size eight-level test. CPU controls passed locally; CUDA remains pending until the user supplies actual server results.

After fetching the delivered probe commit into the existing Windows server worktree (do not reset or overwrite user files), execute in CMD with the existing nnunet5090 environment:

```bat
cd /d C:\lijialin\segementation-upernet-configurable-stages
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set nnUNet_extTrainer=%CD%\nnunet_ext_trainers
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1
python -B .task-notes\upernet-selected-stages\run_validation.py g .task-notes/upernet-selected-stages/test_server_cuda_smoke.py -k cuda0 -s -p no:cacheprovider
```

The guard still samples CPU/RAM and all physical GPUs, labels the GPU load and refuses any CUDA run when a monitored resource is >=80%. It never hides GPU1 utilization. It records the runner and probe hashes alongside production source hashes. CUDA tests fail rather than skip if CUDA/device availability is wrong. Expected: two CUDA_SINGLE_DEVICE_SYNTHETIC_OK records and 2 passed, 2 deselected, exit 0. Peak memory is PyTorch allocated memory only, not whole-device VRAM or a full-size feasibility claim.

Return the entire output/resource preflight and validation/g.json. CUDA single-device PASS does not establish actual DDP/SyncBatchNorm/NCCL, multiple GPUs, AMP/compile, full-size performance or real-data compatibility. Keep production scientific policy unchanged.
