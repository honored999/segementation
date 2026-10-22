# Joint repair CPU validation (2026-10-21)

Starting worktree: E:\study\研一\work14-图像分割\segementation\.worktrees\genseg-quality-repair\segementation
Branch codex/genseg-quality-repair; HEAD9a418526df3e39c2b287ac77ac7399b87ec8b666; initial clean. No commit/push, agents, server or real model training. Main agent directly implements because the user explicitly assigned this manual worker and prohibited automatic delegation. Canonical memory untouched.

Final command (newconda, CPU single-thread; guarded before invocation):

```powershell
& experiments/genseg_joint/tests/run_guarded.ps1 experiments/genseg_joint/tests/test_joint_repair.py experiments/genseg_joint/tests/test_core.py experiments/genseg_joint/tests/test_data.py experiments/genseg_joint/tests/test_pretrain.py experiments/genseg_joint/tests/test_preview.py experiments/genseg_joint/tests/test_runtime.py experiments/genseg_joint/tests/test_multistep.py --basetemp "$env:TEMP/genseg-joint-repair-tests"
```

Result:187 passed in82.00s, exit0. Final preflight CPU6%,RAM64.0043%, GPU0 utilization2%,VRAM1357/6144MiB. All executions guarded; observed preflight CPU5-15%,RAM64.0-68.2%,GPU1-2%,VRAM1341-1357/6144MiB. CPU single thread; no CUDA model/tensor training. Temporary fixtures live outside repository in E:\CodexScratch\temp\genseg-joint-repair-tests. The earlier mistaken in-repository pytest temporary root was removed after resolving and checking the exact path boundary; no raw data or old outputs involved.

Focused initial10 pass; expanded16 pass. Failures resolved: synthetic target lacked captured file identity; missing local psutil covered with isolated telemetry dependency fixture (production still fail closed); initial pytest root inside protected repository correctly rejected5 old preview tests; CPU CUDA-availability stub needed stream-capture stub; synthetic label dtype changed to signed integer required by existing preview contract. No production guard weakened.

Evidence:
- three units: actual Adam moments steps1/2/3; BN increments G1/D3/S2; private validation no live model/mode/RNG changes; poison after injected S failure, no retry/save;
- production hyperloss double finite difference tolerance rtol1e-5/atol1e-8; existing direct/indirect proxy finite difference and severed/zero-S-lr negative controls retained;
- pinned author width1/downs5 G/D actual/higher-order path passes; no production width64 checkpoints loaded;
- independent positive sampler preserves real RNG and real-negative availability; all synthetic masks positive; empty positive index stops;
- owned snapshot arrays readonly/non-writeable views, matching raw sample records/arrays; later samples do not call volume loaders or full inventory audits;
- same new engine two-unit raw/snapshot parameter/BN/Adam/alpha equality at predeclared rtol1e-6/atol1e-8; sampling source/z/flip records identical;
- file change during load fails; running stat change fails; changed final content fails; cache estimate over budget stops;
- safe CPU weights_only roundtrip + repeated eval equality; step/nonfinite/source/split corruption rejection; preview restore shape/dtype/finiteness tests in affected group;
- new check/runtime/save/schema/positive-only preview tiny CPU closure with hardware/large transition replaced; actual snapshot/engine/schema/receipt/renderer exercised;
- one selected GPU query per guard; startup CPU interval.5 then None deltas; invalid telemetry refused;
- old legacy repair rejection and old preview/diagnostic entry compatibility pass.

Limitations: production epoch20 gan.pt not read locally; full production author UNet higher-order execution, CUDA RAM/VRAM peaks, fileID behavior on server, psutil availability, real images/quality, throughput NOT_RUN. Tiny timing does not establish speedup. Checkpoint retains pre-serialization timing snapshot; timing.json contains final checkpoint duration. No dedicated CUDA stage event timing; host enqueue stages plus explicit boundary sync/total unit wall reported. No new resume. Standalone CLI joint_cli is the explicit strategy selector; old cli defaults untouched.

Independent Level3 review NOT_RUN per manual-only instructions. This is engineering implementation evidence, not independent acceptance or image-quality pass. Parent must inspect exact final diff/hashes then manually commission Level3 review; only afterward separately decide a real unit budget.


## B1/B2 focused follow-up (2026-10-21)

Starting state: 1 tracked preview.py modification + 11 untracked files, preserved. Branch/HEAD unchanged. Original manifest self SHA256=1f5d4c8b61b7f92409e3445ae854a6f51879a81b33ee870cb3e5df57dfcbf4d9 verified. Historical187 results are not new results; old analytic tests did not establish isolated production routes. STATIC BLOCKING B1/B2 motivated this follow-up; independent review PASS is not claimed.

All tests used newconda, serial CPU, CUDA_VISIBLE_DEVICES=-1, existing resource-guard wrapper, require_escalated. Common command prefix: `& experiments/genseg_joint/tests/run_guarded.ps1`; suffix: `--basetemp "$env:TEMP/genseg-joint-repair-tests"`.

| Arguments between prefix/suffix | Exit/result | Preflight CPU/RAM; GPU0 util; VRAM MiB |
|---|---|---|
| experiments/genseg_joint/tests/test_joint_repair.py -k 'snapshot or allocator' | 0;12 passed,16 deselected,12.60s | 9%/63.9452%;8%;1090/6144 |
| experiments/genseg_joint/tests/test_joint_repair.py -k 'proxy' -s | 1;1 failed,6 passed,21 deselected,3.20s | 8%/63.9892%;6%;1089/6144 |
| experiments/genseg_joint/tests/test_joint_repair.py -k 'proxy or indirect_oracle' -s | 0;8 passed,21 deselected,3.28s | 14%/64.0173%;8%;1090/6144 |
| experiments/genseg_joint/tests/test_joint_repair.py experiments/genseg_joint/tests/test_core.py experiments/genseg_joint/tests/test_data.py experiments/genseg_joint/tests/test_pretrain.py experiments/genseg_joint/tests/test_preview.py experiments/genseg_joint/tests/test_runtime.py experiments/genseg_joint/tests/test_multistep.py | 0;202 passed,84.51s | 9%/64.0888%;9%;1090/6144 |

B1: header-only Blosc2 and NPZ keep-member NPY-header metadata; NPZ/NPY source backends refused explicitly. Float64/int64 regression: final cache fits, transient projection refuses before cohort loader (reads0). Unknown backend, uint64 label and missing shape rejected before load. Loaded dtype/shape/nbytes checked; cache owned/read-only, pre/post hashes and post-load resources retained. Formula and lifecycle in PROTOCOL; separate final_cache_estimated_bytes/max_case_load_transient_bytes/loader_workspace_reserve_bytes/model_reserve_bytes/startup_required_bytes. Workspace reserve is an explicit assumption covering UNKNOWN external codec/pickle/allocator workspace, not a proven bound. Model reserve independent; real production peak UNKNOWN.

B2 actual RepairJointEngine.hyperloss/virtual_chain/virtual_sgd/GAN/segmentation losses: direct grad7.871344123950416e-08 vs FD7.871259199987435e-08; indirect grad1.266600033355551e-10 vs FD1.265654248072678e-10. Double alpha0=.3, eps1e-4; unchanged rtol1e-5/atol1e-8 plus relative-error<1%. Direct freezes V_G(alpha0) ONCE across perturbations; indirect freezes synthetic forward alpha0. Spy asserts actual loss-path calls. Intentional virtual-G detach makes actual indirect oracle raise missing-gradient RuntimeError (RED caught by sensitivity test); normal oracle GREEN. Severed indirect expects None; re-detach across perturbed alpha still has nonzero FD. Initial failed assertion incorrectly expected that FD zero, then corrected (not hidden). Zero-S-lr both connected derivatives exactly zero; sever retains direct. Scalar conversion warning fixed before final202 run. Total-gradient, BN/Adam multiunit, cache/raw parity, runtime/save/preview/legacy tests pass.

Allocator finite-positive guard tested at78.9/79/79.5/80% used VRAM. RUNBOOK now records/skips individual missing-positive patients; fails only when no exportable positives. GPU-util>=80 policy unchanged; previous unit can contribute to sampled utilization, to observe in future separately authorized real short preflight. No sleep/retry/threshold change.

joint_repair.py/core.py unchanged; no optimizer/BN/hyperparameter/loss/sampling/split/model semantics changed. New loader-workspace requirement and model reserve for check are admission-interface changes only. No source/TargetDataset global behavior changed. Complete11-file manifest updated; no self hash. No agents, real data/weights/CUDA/server training, dependency install, staging/commit/push, raw/old-results/project/global-memory edits. Production psutil unavailable/unverified; CPU telemetry fixtures do not establish production readiness. Production peaks/throughput/image quality NOT_RUN. Independent focused re-review NOT_RUN/manual next; parent acceptance then independent review before any server decision.


## 2026-10-22 minimal phase-aware guard repair

Baseline: codex/genseg-quality-repair at 5f2333030590da8bff48cd577b6655e051c1293d,
clean before edits. Direct fixer per explicit user instruction; no agents.
CPU fake telemetry: total32607 MiB, used22.574 GiB (=23115.776 MiB),
free9.269 GiB: startup rejects, between_units passes with unchanged initial
allocator fraction. Global usage80%/81%, changed UUID and GPU utilization80%
remain rejected. Existing startup allocator78.9/79/79.5/80% cases retained.
Runtime spy confirms startup before model construction, synchronization before
between_units, one allocator setter call, and phase on every unit for both old
steps and epochs; existing tiny production engine/BN/update assertions retained.

newconda serial CPU, CUDA_VISIBLE_DEVICES=-1, no dependency changes.
Preflight before RED: CPU7%, RAM67.2268%, GPU0 util6%, VRAM751/6144 MiB;
resident regression: 1 failed in3.59s (old error lacks phase/numeric context).
Preflight before GREEN: CPU12%, RAM67.2514%, GPU0 util8%, VRAM751/6144 MiB.
Command: & experiments/genseg_joint/tests/run_guarded.ps1
experiments/genseg_joint/tests/test_joint_repair.py::test_resource_guard_resident_phase
experiments/genseg_joint/tests/test_joint_repair.py::test_resource_guard_one_query_nonblocking_and_fail_closed
experiments/genseg_joint/tests/test_joint_repair.py::test_allocator_fraction_boundary
experiments/genseg_joint/tests/test_joint_repair.py::test_new_runtime_check_load_save_preview_cpu
--basetemp "$env:TEMP/genseg-phase-tests"
Result:8 passed in13.47s. No202-test rerun, FD, real-model or cache audit rerun.
Only engineering CPU evidence; real CUDA/3-unit retry NOT_RUN; no quality claim.
No train/model/Adam/BN/loss/sampling/80-epoch budget changes. Raw data, targets,
splits, previous outputs/weights and project/global memory untouched. No staging,
commit or push. Independent review NOT_RUN per no-agent scope; parent acceptance
remains next. Manifest entries refreshed; manifest self SHA delivered in HANDOFF
(the manifest cannot contain its own cryptographic hash).
