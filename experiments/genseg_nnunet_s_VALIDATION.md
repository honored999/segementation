# GenSeg nnU-Net S — CPU validation record (2026-10-23)

Status: IMPLEMENTED / LOCAL_ENGINEERING_PASS / INDEPENDENT_LEVEL3_PENDING.
This chat directly implements the explicitly assigned manual implementation brief; no agents were created.
Actual runtime: Codex current chat. No verifiable LunaMax/DSH worker profile is available, so no such profile is claimed.

Worktree: `C:\Users\Windows\.codex\worktrees\genseg-nnunet2d-s-trial\segementation`.
Branch `codex/genseg-nnunet2d-s-trial`; start=end HEAD `890ee77d34c08c528d82ebdf2b0fead76386782c`.
App-native worktree creation chose the managed C: path rather than the suggested repository-local path.
The checkout started clean; changes remain unstaged/uncommitted/unpushed. Master is not integrated.

## Final exact command / evidence

PowerShell, shell require_escalated; guard uses `D:/Anaconda/envs/newconda/python.exe`, Python3.14.4;
nnunetv2 2.8.1, dynamic-network-architectures0.4.4, torch2.11.0+cu126.

```powershell
& experiments/genseg_joint/tests/run_guarded.ps1 experiments/genseg_joint/tests/test_nnunet_s.py experiments/genseg_joint/tests/test_joint_repair.py::test_fresh_adam_multistep_bn_and_private_validation experiments/genseg_joint/tests/test_joint_repair.py::test_joint_checkpoint_roundtrip_and_malformed_state experiments/genseg_joint/tests/test_joint_repair.py::test_new_cli_explicit_budget_and_legacy_rejection experiments/genseg_joint/tests/test_joint_repair.py::test_new_runtime_check_load_save_preview_cpu experiments/genseg_joint/tests/test_segment.py::test_full_joint_schema_safe_load_and_malformed_moments experiments/genseg_joint/tests/test_segment.py::test_actual_prediction_completion_and_scoring_gate experiments/genseg_joint/tests/test_segment.py::test_prediction_failure_removes_complete_preserves_original experiments/genseg_joint/tests/test_segment.py::test_audit_rejects_partial_wrong_hash_split_source_and_failure experiments/genseg_joint/tests/test_segment.py::test_exact19_rejects_missing_extra_duplicate experiments/genseg_joint/tests/test_segment.py::test_crop_pad_transpose_and_real_resampler experiments/genseg_joint/tests/test_segment.py::test_real_metadata_binding_before_any_patient_array
```

Final preflight: CPU11%, RAM67.9535285642928%, GPU0 utilization9%, VRAM967/6144MiB (~15.74%).
Decision: LIGHTWEIGHT by task scope, serial tiny CPU with CUDA_VISIBLE_DEVICES=-1,
OMP/MKL/OPENBLAS threads1, no pytest-xdist/cache, no real patient arrays or real checkpoints.
Observed result: `27 passed, 4 warnings in 39.40s`, exit0. Third-party batchgenerators/SciPy deprecation warnings only.
Final new file alone previously: `9 passed, 4 warnings in 13.65s`.
Directly affected legacy subset previously: `16 passed, 4 warnings in 27.75s`.
No production model resource peak is inferred from these results.

## Evidence by contract (8 new test functions / 9 parameterized cases)

1. Real official get_network_from_plans -> actual PlainConvUNet (three tiny stages [2,3,4], 540 unique parameters),
   1logit output and shared decoder.encoder identity. Norm/no-buffer contract and changed IN eps rejected.
   Actual production pad_unit padding arithmetic/content checked (zero image, ignore label); no 256/512 production-capacity model forward.
2. Production RepairJointEngine three actual Adam units; private hyperloss leaves S values intact;
   G/D BN stays audited, S BN0/no buffers; unused official segmentation heads have no moments.
3. Production alpha direct frozen-anchor finite difference: grad=-2.9838912786353716e-8,
   FD=-2.983879809903556e-8 (double, eps1e-3). Indirect: grad=1.1536630189307386e-10,
   FD=1.1546319456101628e-10. Both require <1% relative error plus assert_close.
   Fixed spatial context in fixture G avoids artificial affine-scale invariance of IN.
4. Actual indirect graph-sever control returns no alpha gradient. No analytic replacement for production proxy.
5. Actual tiny S state weights_only roundtrip gives bit-exact fixed-input eval output;
   aliases, finite/shape/dtype, norm, wrong plans/kind reject. Geometry-independent primitive verifies support suppression.
6. Full schema3 save/load validates G/D BN, alpha, moments/steps and deduplicated S ownership;
   schema2/schema3 cross-reading rejected; three-unit run rejected by final80 inference gate.
   Synthetic final80 metadata is explicitly fabricated solely to exercise the gate/reader, NOT an 80-epoch training claim.
7. Tiny real S passes production predict -> image-only preprocessing/support-v2 -> asymmetric raw-space export ->
   every-and-only-fixed19 verification -> GT-late scoring admission. Synthetic images only.
   HD-BET, GPU transport and checkpoint loader are replaced at external boundaries; actual S forward,
   geometry, cache/support hashes, output paths and admission execute. No real HD-BET/CUDA claim.
8. Explicit new runtime check -> private InputSnapshot -> production engine3 units -> real schema3 save -> preview admission/renderer.
   Hardware transport, real-GD20 transition, G/D size and preview generator replaced by named fixtures.
   Data/S CPU tensors8x8; full-plane padding transport replaced here (separately tested above).
   SOURCE metadata and tiny plans are fixture-bound, never substituted for server plans in production code.

Legacy affected evidence covers live BN/Adam, schema2 save/safe load/malformed moments,
explicit CLI budgets/defaults, runtime saved receipt/preview/resource and receipt failures,
old support-v2 prediction/scoring/exception propagation, exact19/crop/resampling and no-patient-read metadata binding.

## Failure history / fixes

- Initial expected RED: new module missing during collection (no tests collected).
- Shared module alias reproduction: functional_call polluted live S by ~8.94e-7; direct/indirect FD drifted.
  Explicit aliases alone still failed. Restoring original Parameter objects in finally fixed live ownership/state;
  final tests retain value checks, multistep Adam and relative FD sensitivity.
- Old affine1x1 fixture G made direct gradients near1e-12 due IN scale invariance;
  add fixed spatial context to fixture only, use eps1e-3 to resolve the indirect double-precision FD signal.
  No production rate/norm/optimizer was changed to accommodate the test.
- First synthetic19 fixture omitted the CPU model.to transport override, causing 'No CUDA GPUs are available'
  under CUDA_VISIBLE_DEVICES=-1. Explicit CPU-only transport override fixed the fixture; no CUDA test passed or ran.

## NOT_RUN / pending

Real server SOURCE plans/dataset/split/version binding; real GD20 weights read;
complete production-capacity512 network allocation/forward; actual CUDA higher-order path;
real three-unit preflight; actual RAM/VRAM peaks; real80-epoch run; real19 prediction/metrics;
G image-quality review, synthetic-data suitability and benefit. All NOT_RUN, not implementation blockers.
No dataset/split/GT/old weights/output/author reference/.project-memory/global memory edits.
Unrelated two historical HANDOFF files remain only in the protected baseline checkout.
No installations, AMP/compile/multicard, new augmentation, checkpoint selection or early stopping.

Independent manual Level3 review remains required before integration/server execution.
Review exact SHA snapshot in HANDOFF; this implementation self-inspection is not independent review.
