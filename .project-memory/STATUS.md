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
- Production checkpoint/CUDA/psutil/peaks/throughput/quality NOT_RUN. GPU-util boundary self-load-window risk remains disclosed; no resource threshold relaxation. No joint unit budget approved, no automatic training. Parent changed coordination/memory/Git only; implementation/fix delegated manually. Canonical master unchanged.
