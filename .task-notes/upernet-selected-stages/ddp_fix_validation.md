# DDP P1 fix evidence — 2026-10-04

Role: user-assigned direct implementation main agent; no agents or independent tasks created.
Worktree: E:\study\研一\work14-图像分割\segementation-upernet-configurable-stages
Branch/base/HEAD: codex/upernet-configurable-stages / 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89. No commit/push/merge.

Original independent Level3 conclusion: BLOCKING P1. Original manifest SHA256 8655a1725ce8ae8da79d3b7d581525a07d70034ff6262c57ca6971e779eb1878 and review_level3_20261003.md are preserved. This fix requires independent re-review; implementer verification is not independent PASS.

## Minimal fix
- New Trainer initialize retains inherited initialization; before first forward, only when is_ddp and deepest selected stage precedes final stage, release initial DDP reducer and rewrap the same compiled/uncompiled module via supported DistributedDataParallel constructor(find_unused_parameters=True). Preserve device_ids/output_device/process_group and official default options. No stage deletion/freezing, dummy gradients, loss or optimizer changes, shared Trainer edits, global monkeypatch or training-loop copy.
- Checkpoint guard validates normalized weights before any inherited state updates, then passes that exact mapping in a shallow copy of the checkpoint to the inherited loader. This prevents wrapper key presence from retaining module. before inner-model loading. Caller checkpoint remains unchanged.
- ddp_fix_delta.patch was reconstructed against original reviewed Trainer/test hashes (handling CRLF correctly). It contains only the new initialize block, checkpoint delegation fix and six DDP regressions. Shared decoder and all old Trainer/test code remain identical to prior snapshot.

## Trust limits of DDP tests
- Real Gloo process group, FileStore, rank 0/world_size 1; tiny four-stage encoder with [0,3] last-stage control and [0,2] unused trailing stage. Two forward/backward iterations; no optimizer step/training epoch.
- Real new Trainer.initialize integration is tested. The CUDA-dependent parent initialize is substituted with CPU compile-then-DDP construction; all actual new conditional DDP code, real reducers, optimizer references and checkpoint load/save/guards execute. This substitution is around environment setup, not the guard/fix.
- Plain and CPU torch.compile(backend=eager) paths pass. Full final-stage forward executes twice; gradients are None for unused trailing parameters and finite for used parameters. Optimizer and parameter object identities are preserved; weakref proves old reducer wrapper is released and final-stage wrapper unchanged.
- Real DDP checkpoint round-trip, module prefix mapping and same-shape mismatch before parameters/optimizer/logger mutation; direct DDP root state load also refuses identity mismatch.
- Actual official CUDA Trainer initialization/SyncBatchNorm execution, multiple ranks or GPUs, NCCL/CUDA compile, real-data training/checkpoints and speed/memory/formal performance remain NOT RUN.

## fix-ddp-affected-final
- Exact argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests", "-p", "no:cacheprovider", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\fix-ddp-affected-final"]`
- Exit code: 0
- Visible result: 111 passed, 8 warnings in 35.28s
- Resource preflight: `{"cpu_percent": 11.764705882352944, "ram_percent": 54.11739290226587, "ram_total_GiB": 15.839164733886719, "gpus": [{"index": 0, "gpu_percent": 6.0, "vram_percent": 8.3984375, "used_MiB": 516.0, "total_MiB": 6144.0}], "decision": "NORMAL", "load": "CPU only, one thread, serial synthetic; no real data or CUDA allocations"}`
- Full stdout/stderr and exact test-source hashes: `validation/fix-ddp-affected-final.json`.

## fix-ddp-final-green
- Exact argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_ddp_two_iterations", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_real_ddp_checkpoint_identity", "-p", "no:cacheprovider", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\fix-ddp-final-green"]`
- Exit code: 0
- Visible result: 6 passed, 4 warnings in 12.70s
- Resource preflight: `{"cpu_percent": 9.329446064139946, "ram_percent": 53.83864521011083, "ram_total_GiB": 15.839164733886719, "gpus": [{"index": 0, "gpu_percent": 7.0, "vram_percent": 8.447265625, "used_MiB": 519.0, "total_MiB": 6144.0}], "decision": "NORMAL", "load": "CPU only, one thread, serial synthetic; no real data or CUDA allocations"}`
- Full stdout/stderr and exact test-source hashes: `validation/fix-ddp-final-green.json`.

## fix-ddp-focused-final
- Exact argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py", "-p", "no:cacheprovider", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\fix-ddp-focused-final"]`
- Exit code: 0
- Visible result: 61 passed, 6 warnings in 33.08s
- Resource preflight: `{"cpu_percent": 7.961165048543695, "ram_percent": 53.31197728786732, "ram_total_GiB": 15.839164733886719, "gpus": [{"index": 0, "gpu_percent": 5.0, "vram_percent": 8.430989583333334, "used_MiB": 518.0, "total_MiB": 6144.0}], "decision": "NORMAL", "load": "CPU only, one thread, serial synthetic; no real data or CUDA allocations"}`
- Full stdout/stderr and exact test-source hashes: `validation/fix-ddp-focused-final.json`.

## fix-ddp-green
- Exact argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_ddp_two_iterations", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_real_ddp_checkpoint_identity", "-p", "no:cacheprovider", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\fix-ddp-green"]`
- Exit code: 1
- Visible result: 3 failed, 3 passed, 4 warnings in 11.49s
- Resource preflight: `{"cpu_percent": 29.98046875, "ram_percent": 59.43862228218592, "ram_total_GiB": 15.839164733886719, "gpus": [{"index": 0, "gpu_percent": 24.0, "vram_percent": 22.102864583333332, "used_MiB": 1358.0, "total_MiB": 6144.0}], "decision": "NORMAL", "load": "CPU only, one thread, serial synthetic; no real data or CUDA allocations"}`
- Full stdout/stderr and exact test-source hashes: `validation/fix-ddp-green.json`.

## fix-ddp-prefix-red
- Exact argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_real_ddp_checkpoint_identity[plain]", "-p", "no:cacheprovider", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\fix-ddp-prefix-red"]`
- Exit code: 1
- Visible result: 1 failed, 4 warnings in 7.82s
- Resource preflight: `{"cpu_percent": 11.579980372914623, "ram_percent": 58.803745151297804, "ram_total_GiB": 15.839164733886719, "gpus": [{"index": 0, "gpu_percent": 0.0, "vram_percent": 20.100911458333332, "used_MiB": 1235.0, "total_MiB": 6144.0}], "decision": "NORMAL", "load": "CPU only, one thread, serial synthetic; no real data or CUDA allocations"}`
- Full stdout/stderr and exact test-source hashes: `validation/fix-ddp-prefix-red.json`.

## fix-ddp-red
- Exact argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_selected_trainer_ddp_two_iterations", "-p", "no:cacheprovider", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\fix-ddp-red"]`
- Exit code: 1
- Visible result: 2 failed, 2 passed, 4 warnings in 10.60s
- Resource preflight: `{"cpu_percent": 14.091350826044701, "ram_percent": 61.22249672578634, "ram_total_GiB": 15.839164733886719, "gpus": [{"index": 0, "gpu_percent": 24.0, "vram_percent": 21.516927083333332, "used_MiB": 1322.0, "total_MiB": 6144.0}], "decision": "NORMAL", "load": "CPU only, one thread, serial synthetic; no real data or CUDA allocations"}`
- Full stdout/stderr and exact test-source hashes: `validation/fix-ddp-red.json`.

## Failure history
- fix-ddp-red: 2 control cases passed; plain/eager [0,2] failed on second forward with Expected to have finished reduction (exit 1), reproducing original P1.
- fix-ddp-green: second-forward bug was gone, but id() reuse caused two erroneous wrapper identity test assertions; legitimate module.-prefixed plain DDP checkpoint exposed a prefix delegation defect (3 failed/3 passed, exit 1). Replaced id comparison with weakref; no production weakening.
- fix-ddp-prefix-red: separately reproduced normalized identity accepted but inherited loader retained wrapper prefix (1 failed, exit 1). Minimal new Trainer-only shallow-copy/delegation fix followed.
- fix-ddp-final-green: 6 passed; fix-ddp-focused-final: 61 passed; fix-ddp-affected-final: 111 passed, all exit 0.
- A task-local delta reconstruction initially compared normalized LF text against CRLF byte hashes and failed; corrected byte comparison without changing source. Original hashes were subsequently matched.

## Status / next action
- Fix implementation COMPLETED; original reviewer BLOCKING verdict has not been superseded by independent evidence. RE_REVIEW_PENDING.
- Run resource preflight before each later test; all fix samples were below 80%, CPU-only single-thread synthetic. Current readings are not peak/projected measurements.
- Protected root/baseline/NoStage7 remain read-only and clean. Branch memory only updated to original BLOCKING + fix-ready/re-review-PENDING; master canonical untouched.
- User manually dispatches re_review_ddp_prompt.md with the new snapshot_manifest_ddp_fix.json and separately supplied manifest hash. No reviewer automatically dispatched.
