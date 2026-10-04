# Next

## Immediate server continuation (2026-10-04)

1. User downloads task-local CUDA0 probe delivery; retains server-generated validation JSONs and failed long-path evidence.
2. With fresh safe resource preflight, run test_server_cuda_smoke.py -k cuda0 on physical GPU0, short label g; return full output and validation/g.json.
3. Evaluate actual CUDA result before separately authorizing DDP/AMP/compile/full-size/real-data steps; no real training or formal evaluation in this task.

## Branch-local next actions (2026-10-04, after independent PASS)

1. Preserve the accepted source snapshot and review evidence; P1 is closed, no implementation/review blocker remains.
2. User authorized branch commit/push; verify remote delivered commit, then user downloads an isolated server checkout using server_validation.md. Merge/integration remains separately authorized.
3. Before separately authorized server testing, define resource preflight and CUDA/SyncBatchNorm/NCCL/multi-rank checks with isolated plans/results; preserve preprocessing, fixed patient splits and scientific policies. No performance or formal-evaluation claim follows from CPU synthetic evidence.

## Retained baseline next actions (historical; outside this task)

## Current focus

- Decide whether to run a separately authorized, isolated Lite-UPerNet
  engineering preflight or experiment. Implementation itself is complete.

## Next actions

1. Keep both baseline models and their checkpoints unchanged.
2. If experimentation is requested, define isolated output names and preserve
   the fixed patient-level five-fold split and matched single-output contract.
3. Treat any synthetic smoke/preflight evidence separately from formal OOF
   results.

## Blockers

- None for the Lite-UPerNet implementation.
- Whole-repository `python -m pytest -q` still encounters six pre-existing
  `non_teacher_student_files` import-collection errors outside this task diff.
