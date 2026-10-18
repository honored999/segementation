# GenSeg short multi-step observation run

Date2026-10-18; USER OVERRIDE: local work is static-only; no local pytest/model/numerical/GPU execution after the instruction. Runtime reproduction/3-step evidence is deferred to data computer. baseline b5b795e506a253c44508f499eba3e87ff69047f4 on codex/genseg-reference initially clean. User asked to increase training after1+1 preview showed near-untrained gray output. Proposed observation budget GD500 then joint100, preceded by3-step preflight per phase. New fresh outputs each, no resume or old76 initialization. Same61/15/19 memberships, target manifest, seed, rates, losses, BN and architecture contract.

## Root-cause boundary

Current run synchronizes then calls gpu_guard before every step, requiring selectedGPU compute/VRAM<80percent, >=12GiB physically free VRAM and fixedUUID. CUDA caching allocator retains unused previous-step reservations. Thus identical repeated steps may fail free-headroom due to reclaimable own cache. Recent own kernels may also remain in nvidia-smi utilization sample. Investigate the code and write inspectable regressions; per latest user instruction do not execute them locally. This is a statically identified risk, not an observed server failure. Do not weaken thresholds, claim peak predictions, kill jobs, or silently override optimizer/data settings. Resource observation failures remain failures. Unknown true CUDA behavior stays pending data-PC3-step run.

## Minimal runtime correction lane

Sole leaf owns experiments/genseg_joint/runtime.py, narrow new tests/test_multistep.py and one no-op CUDA cleanup fixture line in tests/test_pretrain.py. Parent docs/memory/Git only. No nested agents, no commit/push/worktree. All shell calls require_escalated directly, newconda for static AST parsing only.

- [x] Write static-reviewed regression cases (NOT RUN locally): repeated-step cached allocations may cause avoidable guard refusal; cached memory release must follow synchronize and precede telemetry. Preserve live tensors and numerical sequence.
- [x] Implement minimum between-step cleanup (release only unused own allocator cache), with no cooldown or retries for lagged selected-GPU utilization while retaining strict<80percent CPU/RAM/VRAM/headroom/UUID rules. Do not generalize busy/resource errors into indefinite retries. Distinguish permanent failures; no silent skip/update/reseed/resampling. Record cleanup/cooldown policy in protocol/resource logs. Initial guard unchanged.
- [x] Ensure allocator ceiling is not repeatedly shrunk based on own existing reservations; investigate existing set_per_process_memory_fraction calls and keep a consistent safe budget. No increasing initial permitted ceiling; total occupancy remains guarded. If no code change needed outside cache cleanup, keep scope minimal.
- [x] Statically inspect written regression cases for permanent busy/VRAM/lowheadroom/CPU/RAM/UUID/malformed telemetry rejection; execution NOT RUN per user.
- [x] Write or inspect multi-step regression scenarios for eventual server execution: GD/joint3-step numerical path, complete counts/logs/receipts and patientroles. No local execution.
- [x] Local syntax/source inspection and independent final-snapshot STATIC Level3 review only; focused/affected/runtime tests deferred to target PC by user instruction. Parent source/test implementation none.

## Delivery and experiment

- [x] Parent short-run CMD guide:3+3 preflight in fresh dirs; only after success GD500 fresh then joint100 from that exact500 checkpoint; fixed1+1 baseline retained. Must not overwrite generator_5000steps_seed20261014_attempt02 or existing targets/weights/splits.
- [x] Explicit hashes, output suffixes, inspect complete+steps, stop at failures, no guard bypass. Current checkpoints only save at successful run end, no interruption recovery; document honestly, do not add a new recovery subsystem for this short trial.
- [x] Same deterministic inner61 preview sample after final run. This is image-quality exploration, not formal metrics or causal evidence for skull stripping.
- [x] Parent records review/testing/source hashes, branch memory verified status, explicit stage+diffcheck, commit/push then data-PC pull --ff-only.
