# Status

Updated: 2026-09-21

## Current state

- The active feature branch contains standalone 2D PlainConvUNet and H2Former
  model families with explicit model/supervision checkpoint identities.
- Existing H2Former uses single-output supervision; PlainConvUNet supports its
  default deep supervision and a matched single-output mode.
- Independent `h2former_lite_upernet` and `plain_conv_unet_lite_upernet`
  single-output variants are implemented at `53472ae`. Both use the shared
  lightweight PPM/FPN decoder while preserving the baseline model identities.

## Verified capabilities/results

- Fresh main-agent validation: standalone suite `424 passed in 47.67s`; root
  `tests/` suite `26 passed in 6.67s`.
- Independent Level 3 review of `bbe65b5..53472ae` returned PASS with no
  blocking findings.
- Synthetic complexity evidence shows both new decoders have fewer parameters
  than their corresponding baseline decoders. This is engineering evidence,
  not medical-performance evidence.

## Verified constraints

- Dataset501 remains the established DWI-only baseline.
- Preserve the existing patient-level five-fold split and original full-volume
  formal-evaluation semantics.
- Synthetic validation is engineering evidence only.

## Active work

- Lite-UPerNet implementation is accepted. No real-data training, preflight,
  fold evaluation, or formal five-fold OOF evaluation has been run.

## Branch-local GenSeg30 preparation (2026-10-14)

- codex/genseg-reference contains a synthetic-validated offline GenSeg30 slice-pool workflow and separate plain EarlyStopping Trainer. Adds ceil(30% of training-only 2D slices), retaining patient-level fold0 grouping and real validation geometry.
- Local engineering evidence: 74 combined synthetic and EarlyStopping tests passed; final independent 15-file snapshot PASS. Server compatibility, generated-image QA, training and quantitative benefit remain unverified. Original Blosc2 preprocessed backend required.

## Branch-local GD pretraining handoff (2026-10-18)

- User confirmed Dataset501 still95 (94 was a typo); fixed76/19 and inner61/15 remain unchanged.
- Added inner61-only G/D pretraining with frozen architecture/BN, completed gan.pt receipt and strict same-protocol handoff to fresh author U-Net/alphaAdam. Cold-start remains supported; interrupted-run resume is absent.
- Worker guarded CPU tests73 joint +15 target +63 offline passed separately; parent43 key tests and three CLI helps passed. Independent Level3 review PASS on the exact five-file snapshot in reference/GENSEG-JOINT-PRETRAIN-REVIEW.json.
- Target-PC native512x512 one-step cold-start pilot success was user-reported; user subsequently reported completed1+1 pretrain-to-joint CUDA receipts with finite alpha change; those remote files were not read locally. No performance/OOF claim.
- This is feature-branch state only; master canonical memory not modified.

## Branch-local joint preview (2026-10-18)

- Added completed-joint-checkpoint preview: only G inference, saved learned alpha/scaling, deterministic first4 inner61 cases, native four-panel PNG and provenance manifest. No inner15/external19 image reads or training/data-export changes.
- Guarded worker32 focused/105 joint affected PASS; parent32 focused and CLI PASS. Parent synthetic visual inspection and final exact-three-file independent Level3 PASS; receipt reference/GENSEG-JOINT-PREVIEW-REVIEW.json.
- Target-PC real-patient/CUDA preview not run locally. User can view PNGs after pulling branch and executing PREVIEW-RUNBOOK. PNG output is qualitative only, not a generated training dataset or formal metric.

## Branch-local multi-step preparation (2026-10-18; static only)

- Added synchronize/unused-cache release before unchanged per-step guard; fixed allocator ceiling to minimum initial/pre-model budget. No resource thresholds or numerical/data settings changed; no cooldown/retry.
- Latest user instruction: local static review only. Three files AST/diff checks and independent Level3 PASS(STATIC); pytest/model/CUDA/numerical execution NOT RUN. Pre-override test start interrupted without result.
- Proposed observation budget: target-PC3+3 first, then new GD500 and joint100 plus same-inner61 previews. Server multi-step success and image-quality benefit unverified; no resume/periodic recovery added.

## Branch-local selected-utilization cooldown (2026-10-18; static only)

- Server user evidence: GD3 completed; joint3 stopped beforestep2 at81percent selectedGPUutil whileVRAM5.4percent. No completejoint3/500+100success; utilization attribution unresolved.
- Typed util-only finite retry only between completed steps (6attempts/5x1sec); all non-util guards checked first, no initial/first-step wait. Events/failed diagnostics recorded; thresholds/numerical protocol unchanged.
- Parent/workerAST3files+diffcheck and independent Level3 PASS STATIC; localruntime NOT RUN per user. ReusecompletedGD3, freshjointtrial02, retainfailedtrial01; no resume.
## Branch-local GD500 preview comparison (2026-10-18; static)

- User reported GD500 and joint100 completed; copied joint preview receipt confirms same initialization/protocol and selected slices. Reviewed PNGs remain nearly constant gray; no image-quality or segmentation-benefit claim. BN/optimizer causality remains unverified.
- Added explicit preview phase gan_pretrain for completed gan.pt, reusing G-only restoration, saved scale and inner61 selection; default joint retained. Training, raw/splits/targets and old weights unchanged.
- Leaf/parent/reviewer final three-file AST/diff checks PASS; independent Level3 PASS STATIC. No local tests/imports/model/numerical/CUDA/resource probes per user. User subsequently supplied completed GD500 preview; receipt/hash/same slices/scale verified locally, inspected images already gray before joint.
- Branch-local state only; canonical master memory absent/not modified. Leaf implementation, no nested agents.

## Branch-local fixed-slice diagnostic (2026-10-18; static)

- Added diagnose-single-slice: exact completed GD500 G/D initialization, fixed inner61 first lesion plane, unchanged PretrainEngine/BN/rates/L1/alpha and saved scale. Before/after scalar probes and PNGs; no saved weights or formal learning claim.
- Final three-file AST/diff checks and independent Level3 PASS STATIC after moving identity audit before resource sampling. No local tests/imports/models/CUDA/numerical/resource probes. Local copied diagnostic receipt subsequently verified complete=true/completed_steps=100/weights_saved=false; no efficacy claim.
- Source/test implementation and fix leaf-owned, no nested agents; parent branch records/docs/Git only. Raw/splits/old weights and master untouched.

## Branch-local HD-BET nnU-Net fold0 preparation (2026-10-19; static)

- User approved real-DWI fixed95/fold0 76train+19development comparison with ordinary deep supervision, existing Dice+TopK10 and early stopping. GenSeg61/15 roles do not apply to this separate segmentation run.
- New isolated builder check/prepare/preprocess-check/ready/guard/trainer-check: audited76 native masks reused; remaining19 use frozen HD-BET2.0.1/weights/defaultTTA, native-plane3mm and empty-plane retain. Native geometry/kept pixels and byte-exactGT preserved; fresh derived raw/preprocessed/results below HD-BET_outputs.
- Original plans/splits retained; baseline fingerprint copied as explicitly inherited compatibility metadata, not newly measured derived statistics. No replanning. Existingconfig/fold paths refused; selectedGPU guards and success-chainedCMD prevent continuing after failures.
- Leaf implementation; parent coordination/docs/Git only, no nested workers. Parent7PythonAST/diff PASS and exact10file independentLevel3 STATIC PASS, receipt reference/HDBET-NNUNET-FOLD0-REVIEW.json. All pytest/import/model/patient/CUDA/runtime/training/evaluation deferred by local-static-only instruction.
- Server runbook reference/HDBET-NNUNET-FOLD0-RUNBOOK.md; manual training76 QA before long run, fixed19 original-volume best-checkpoint evaluation, matched originalDWI control needed if prior Trainer differs. No effectiveness claim; canonical master memory absent/unmodified.
## Branch-local ISLES initialization steering (2026-10-19; documentation only)

- User changed the next run from cold-start to original ISLES508 source checkpoint initialization, full-parameter ordinary nnU-Net TopK10+ES/normalDS on stripped Dataset501 fold0. Historical pretrained_ES and CKPD_ES roots are private fine-tuning results, not arbitrary source choices.
- Added HDBET-NNUNET-FINETUNE-FOLD0-RUNBOOK with historical source path, guarded SERVER CPU identity/architecture audit, exclusive source-hash receipt and pre-train hash recheck; -pretrained_weights with fresh optimizer/stopping, no --c/CKPD/LR change. Separate hdbet3mm_isles508_finetune output; prior builder/source untouched.
- Parent embedded3snippetAST/diff and independent scientific Level3 STATIC PASS. No local checkpoint/model/import/runtime tests or server execution. New source identity/current files require server verification; old full-tuning source hash is not verified, so causal A/B claims require matched control. Branch memory only, canonical master absent/unchanged.

## Branch-local GenSeg quality repair phase B (2026-10-19)

- Isolated codex/genseg-quality-repair implements explicit v2 Adam/trainBN/epoch candidate, both G/D gradients from pre-update weights, per-layer BN G1/D3, persisted Adam/BN/RNG/sampler. Legacy remains default; no interrupted resume. Live failures poison the engine, no atomic rollback claim.
- Fixed-slice cold and locked-GD500 recovery diagnostics share eval-pure probes and constant-L1/lesion gates; scalar success leaves manual structure/semantics pending. New pretrained state is explicitly refused by legacy joint/check pending phase C.
- New user instruction allows local CPU pytest for this task: 186 related tests passed, final focused3 passed; test-function count unchanged. Parent AST11/diff and independent12-file exact STATIC Level3 PASS, receipt reference/GENSEG-QUALITY-REPAIR-REVIEW.json. No real-patient/CUDA/quality/long-run evidence.
- Source/test edits leaf-owned, no nested agents. Raw/fixed splits/old weights/primary fine-tuning checkout unchanged. Branch-local accepted engineering state only; canonical master memory absent and unchanged.

## GenSeg fixed-slice probe follow-up (2026-10-19; branch-local)

- Server stopped before its first update on repeated-eval exact comparison. Fixed confirmed context mismatch (grad enabled vs no_grad); both forwards now match, exact/state/RNG checks retained, failure reports numerical/context details. CUDA cause/effect still awaits server evidence.
- Existing regression RED2 -> GREEN2 and affected31 CPU passed, no new test functions; independent exact2-file STATIC PASS. No training/optimizer/BN protocol, raw/split/weight change; leaf implementation only.

## GenSeg numerical repeatability follow-up (2026-10-19; branch-local)
- Server matched-context float32 CUDA output differed by7.45058e-9 before update1. Revised only diagnostic output repeatability to predeclared atol1e-7/rtol0 with shape/device/dtype/finite/context guards and truthful exact/within metrics. All live-state purity and learning-quality thresholds unchanged; old failed results retained.
- Existing regression RED2/GREEN2, affected31 CPU passed; independent exact2-file STATIC PASS. Zero new test functions, leaf implementation; no CUDA/quality pass claimed. Canonical master/current fine-tuning checkout untouched.

## GenSeg probe-local cuDNN follow-up (2026-10-20; branch-local)
- User supplied failed.json confirms100 updates completed, after-probe repeat maxdiff1.49e-7. No weights were saved; preserve this failed run, do not fabricate completion or imply loss of all training execution.
- Probe-only cuDNN deterministic=True/benchmark=False now scoped/restored by official context, compatible optional API inspection; caller enabled/TF32 and subsequent actual training unchanged. No tolerance increase (atol1e-7/rtol0), quality/strict state checks unchanged.
- Existing CPU regression scope/restore and old-signature RED/GREEN, final affected31 passed; exact2-file independent STATIC PASS. Zero new test functions; real CUDA/old-version execution and generated quality pending. Primary fine-tuning/master memory untouched.

## GenSeg same-weights BN diagnosis (2026-10-20; branch-local)
- Locally read deterministic cold100 receipt: complete100, state/repeated-output checks passed; learning gate failed. Valid L1 0.6040 exceeds constant 0.1392; no weights saved. Single-slice engineering evidence only.
- Added opt-in --compare-bn-modes: same private weights/input, only G BN batchstats, private buffers and isolated RNG; original eval PNG/statistics/gates remain authoritative. Existing focused3/affected31 CPU pytest passed; zero new test functions, exact3-file independent STATIC PASS. Actual BN CUDA comparison pending; no causal or quality claim. Source/test leaf-owned; primary checkout/master memory untouched.

## GenSeg private BN calibration (2026-10-20; branch-local)
- Received complete100 BN-mode diagnostic: after eval/batchstats valid L1=0.5817/0.2315 vs constant0.1392; quality remains failed. Mode affects output, not a unique-cause claim.
- Added default-off --compare-bn-calibration: after-only same private parameters, one BN-only momentum1 no_grad pass into persistent private buffers, then matched-context eval; original eval/gates/live state unchanged. Existing focused3/affected31 CPU passed; zero new test functions; exact3-file independent STATIC PASS. Real CUDA calibration pending; no formal inference/calibration policy accepted.

## GenSeg G-objective ablation (2026-10-20; branch-local)
- Received private BN calibration diagnostic: complete100, original/batchstats/calibrated validL1=.5696/.2282/.2556 vs constant.1392; quality remains failed. Calibration explains only part of error.
- DSH implemented explicit cold+repair l1_only diagnostic: G keeps100*valid_L1; D objective/update/G1-D3 and original gates unchanged; cannot hand off as pretraining. User-supplied96 CPU pytest plus parent6 key cases passed; exact7-file independent STATIC PASS; zero new test functions. CUDA quality pending. Parent source/test unchanged; only docs/receipt/branch memory/Git integration. Master memory untouched.

## GenSeg diagnostic evidence (2026-10-20; branch-local)
- L1-only complete100 did not improve matched-budget quality. Accepted manual DSH ready.json and default-off same-forward mode PNG patch; no core/data/target/seed/optimizer/gate change. Immutable CPU array/readiness hashes support failure evidence; extra images reuse existing private BN comparisons.
- Independent manual DSH STATIC PASS exact5-file diff; implementer reported198 CPU passed; parent32 targeted passed10.70s, zero new test functions. Parent only corrected dynamic-shape/hash docs and integrated; actual CUDA mode-preview run pending. Primary/master memory unchanged.

## GenSeg bounded convergence engineering (2026-10-20; branch-local)
- Mode-preview cold100 received: partial structure visible, quality still failed. Added default-off fixed500 cold GAN+L1 convergence observation plan100/250/500, private same-mode plots/trajectory and one diagnostic-only G/D/alpha snapshot; no resume/pretrain/joint handoff or training-rule changes.
- Initial B1/B2 blockers fixed: strict tiny restored-eval/BN/alpha evidence and partial write/close/readback/hash/stat failure receipt handling. Independent manual focused STATIC PASS exact5 SHA; fixer42 CPU passed, parent2 passed5.36s. Parent production/test unchanged; historical pre-run status superseded by the local500 acceptance below. Primary/canonical master memory untouched.


## GenSeg500 local acceptance (2026-10-20; branch-local)
- Approved500 run completed at830642c; local receipt/log hashes and all10 PNG hashes verified, steps1..500/G1D3 and10 implementation hashes match. Diagnostic state snapshot absent locally; server bytes/hash/readback remain receipt-only.
- Original eval validL1 .0319455 vs constant .1391926; foreground .0613226 at250 -> .0638616 at500. All10 images inspected: major structure recovered, residual circular halo/smoothing; complete quality not accepted. Original manual-pending/learning_quality_passed=false untouched. Single-slice engineering evidence only.
- Original preview accepts gan/joint only; new independent preview-diagnostic supports this diagnostic snapshot with strict audit, frozen G-only restore, fixed inner61 coverage <=13 and historical anchor statistics. Implementer reported107 CPU passed; independent four-file STATIC Level3 PASS. Parent verified exact hashes/diff; real snapshot/CUDA/inventory/peak memory remain unverified. Master canonical memory absent; only branch-local sync.


## Joint Adam positive strategy accepted (2026-10-21; branch-local)
- New explicit joint_adam_trainbn_positive_sgdproxy_v1: epoch20 G/D weights-only transition, fresh S/Adam state, actual trainBN G1/D3/S2; virtual single-SGD/eval-BN approximation and staged alpha Adam. Real negatives retained; independent inner61 positive synthetic masks, inner15 alpha only, external19 excluded.
- RAM snapshot and lightweight unit audits reduce repeated loading/hash work. Final cache, transient, external-loader workspace and model reserves separated. B1 pre-load refusal and B2 production direct/indirect FD/negative controls independently STATIC re-reviewed PASS; final manifest d403e688d02b668614fde4e277bc2bd4727bee8232ea3da93f513f6f39dcdf3e. Fixer reported202 CPU passed; parent exact11 hashes/diff checked, no rerun.
- Production checkpoint/CUDA/psutil/peaks/throughput/quality NOT_RUN. GPU-util boundary self-load-window risk remains disclosed; no resource threshold relaxation. Historical pre-epoch authorization state superseded below; no automatic training. Parent changed coordination/memory/Git only; implementation/fix delegated manually. Canonical master unchanged.


## Joint epoch mode accepted (2026-10-21; branch-local)
- User approved80 complete real-inner61 epochs and requested minimal validation. Added mutually exclusive --epochs/--steps, reusing EpochSampler on RAM snapshot, independent shuffle RNG, progress output and protocol/log epoch metadata; old steps random path retained. Expected970 planes ->77600 units; actual N measured.
- Worker reported3 focused CPU tests passed12.88s; parent checked5 hashes/full manifest/diff, no large rerun or repeated independent core review. Joint engine/Adam/BN/resources unchanged. Epoch validation RNG sequence differs from steps mode because real draws are separated; disclosed.
- User server telemetry confirms nnunet5090 psutil works, RAM127.73GiB/free100.52GiB, physicalGPU1 RTX5090 32607MiB free/idle. Real production load/CUDA preflight/throughput/quality NOT_RUN. Only final checkpoint/no resume;80epoch budget is approved, resource verification remains before launch. Parent coordination/Git only; canonical master unchanged.


## Joint resident VRAM guard accepted (2026-10-22; branch-local)
- Server preflight3 trial01 completed1 unit then stopped before unit2: freeVRAM12GiB gate, not OOM; allocated/reserved peaks21.347/22.574GiB, unit wall1.928s. Trial01 failed/non-resumable, not3-unit pass.
- Startup retains12GiB free and80% checks; synchronized between_units retains80% global VRAM/CPU/RAM/GPU/identity/RAM checks without startup12GiB rule. Initial allocator fixed once; no cache flush, wait/retry or model/optimizer/sampling change. Worker8 focused CPU passed13.47s; parent diff/final11 hashes checked; no large rerun. Manifest f4efb5e85a4c6e6e6339f8397c33662045f6af80e2ee5eb19524d29f570368c8.
- New-directory real3unit retry/CUDA performance still pending; own previous-unit GPU-util sampling risk unchanged.80epoch approval retained; no automatic launch.

## Joint runtime GPU-util policy and failure diagnostics accepted (2026-10-22; supersedes prior runtime-util rule)
- Startup GPU utilization remains <80%; synchronized between_units validates finite 0..100 and records only. Whole-card samples cannot distinguish our own work from other processes. CPU/RAM/global VRAM <80%, projected RAM/headroom, startup12GiB, UUID/index and fixed initial allocator retained; no model/optimizer/BN/sampling/budget change.
- Failure exceptions and failed.json retain resource_snapshot and violated_rules; invalid numeric telemetry is null with reason. Receipt-write failure cannot mask original error; add_note is capability-checked with protected fallback for older Python.
- Worker reported18 focused CPU passes, then2 actual-guard injection passes and2 compatibility passes (7.78s). Parent inspected final diff and all11 manifest hashes; manifest b869ac28fddcc5ac96c4b64e10095340b630efd859142d8ea3711358e1ff70ce. No parent source/test edit, broad rerun, new independent review or real CUDA validation.
- Subsequent server 80% boundary refusal did not record CPU/RAM/GPU-util individually; historical trigger UNKNOWN, not a proven GPU-util failure. Preserve all failed outputs. Next fresh3-unit preflight must finish checkpoint and complete receipt before retained80epoch budget; no resume. Canonical master unchanged.

## Joint epoch80 S prediction accepted (2026-10-23; branch-local)
- User reports3-unit preflight and80epochs completed on server; actual epoch80 complete/checkpoint/metadata have not yet been independently verified. No new training/resume authorized or needed for this evaluation step.
- New genseg_segment audit/predict/accept reuses full joint schema and S-only strict restore; fixed outer19 image-only DefaultPreprocessor(seg=None), bound plans and HD-BET2.0.1 recipe/weights,3mm keep, baseline fill and saved inner61 scale. Supervised-mask replacement and possible masked-normalization differences are explicit, not claimed pixel-equivalent. No GT input, tuning, S TTA or postprocessing.
- Original-space binary export, exact19 manifests/hash/geometry/state checks and independent scoring admission precede unchanged seven-metric script (--f2-mode paper). B1 canonical CACHE ownership/disjoint METRICS and B2 preservation of original resource errors despite subprocess cleanup failures independently STATIC re-reviewed PASS; no blockers.
- Final source/test hashes22d4e6515950608171ab2bd982b69c62ff9cae60c58f00f8c5a13696ab65cf64 / f10fae2aaa8ffb0c81cf3d8cc8f052843519609b05a9c4f92b35f8e6b11e3922. Implementer reported20 different CPU continuation cases, then11 focused B1/B2 cases across runs; parent hashes/diff/CRLF-aware whitespace inspected, no repeat pytest and no parent source/test implementation.
- Actual server metadata/production checkpoint/HD-BET/CUDA/19-case prediction and metrics NOT_RUN by local agents; no efficacy claim. LAS-copy/additional HD-BET-receipt hash strengthening remains nonblocking deferred. Next publish reviewed source/test/RUNBOOK plus branch memory after scope authorization, then server audit→predict→accept→original metrics. Canonical master unchanged.

## Joint S output support correction accepted (2026-10-23; supersedes v1 export)
- User server reports completed19-case v1 prediction/accept/scoring: macro Dice4.2887%, AVD7546.0607%, HD95 139.5312mm. User-supplied TP/FP/FN totals406367/16348422/351537; raw-image nonzero-support diagnostic places16284256 FP (99.6075%) outside that support. Raw support is a diagnostic clue, not identical to the exact resampled image-side mask. Original trial01 results preserved; not a training-effect conclusion.
- v2 stores bool valid_support from actual DefaultPreprocessor(seg=None) mask>=0 with shape/source/protocol/hash. Same .5 threshold, zero outside support before inverse geometry and outside the same inverse support before writing; writer/accept independently verify support/plans/properties/hash and original geometry. Valid-domain threshold decisions remain unchanged; no GT/HD-BET/zero-intensity substitute for support.
- Fixer reported9 focused CPU passes10.28s; independent new-support STATIC review PASS, no blockers, exact source12f3f7ab4b712f7ce935a4ab66f9b937aa8ce2c6cbcdbe84ad3a202da09feb6d / test6a5f6d4e9794ba16094f58db64af52e7ec1fedefd3df18f93d241825980cd95f. Parent verified snapshot/diff without source/test edits or repeated tests. B1/B2 protection retained.
- Use same final epoch80 checkpoint (preview hash824a3912cdabd6136dd970d042b12a29ffce5680c3de2ee53e53afe13fae9a9a; verify complete/file); no validation-best checkpoint exists, no retraining/selection. Fresh CACHE/PRED/METRICS trial02; v2 actual clearing counts/19-case metrics NOT_RUN. Label new result post-error-observation inference correction/development evidence, not blind test or training gain. Canonical master unchanged.
- Parent inspected all4 epoch80 G preview images and matched same-input/window epoch20 examples:3 show new extracerebral ring textures and some dark holes, one lacks that ring. This G quality issue is separate from S export-support correction; no augmentation quality/effectiveness pass.

## Accepted nnU-Net S epoch recovery (2026-10-24; branch-local)
- The nnU-Net S architecture trial is published separately at 6a6be887. User-reported server SOURCE check and three-unit CUDA preflight produced a completed checkpoint; this validates the prior final-only entry, not periodic-save/resume behavior. The current 80-epoch run status is unknown; never update an active server checkout.
- Epoch-only recovery is accepted on archived implementation e9fc06d plus the reviewed Python3.10-compatible working patch. Complete-epoch checkpoints retain two generations; new attempts restore model/alpha/Adam/BN/modes/RNG/samplers and continue the original total budget without refitting scale or recopying GD20 weights. Final schema3 completion remains separate.
- Independent full static Level3 PASS, no blockers. Parent verified HANDOFF SHA256 5a806ddbde4de3b16dee05e01789d21b143402647f336104790845cad9f92dc4 and all eight delivered hashes. Initial implementer17 CPU passes are historical; fixer15 passes13.28s and missing-file_digest RED are reported evidence, not parent/reviewer reruns.
- Actual Python3.10, CUDA periodic-save/restore, CPU/global-VRAM peaks and throughput remain NOT_RUN. Pointer pre-read symlink/manifest hardening is a nonblocking suggestion, deferred. No scientific-quality or reproducibility-on-CUDA claim.
- User approved scoped publication to the new epoch-resume branch and a fresh resumable total80-epoch server run. Only commands are provided here; server execution is not performed by the parent. This is branch-local accepted engineering state; master and global memory are unchanged.
