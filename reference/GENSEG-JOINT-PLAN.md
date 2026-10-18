# GenSeg joint author-U-Net engineering pilot

Approved scope: user selected the author U-Net first to exercise the paper three-level optimization chain. Baseline 89e1bf2; isolated codex/genseg-joint-unet. No auto-push or raw/split changes. Outputs on target PC remain C:/lijialin/HD-BET_outputs/genseg_joint_trial01.

## Protocol
- Paper DOI10.1038/s41467-025-61754-6 Eqs4-11 is primary; pinned author source6480e178 is secondary. This is an engineering adaptation, not formal reproduction or proof of benefit.
- Existing76/19 outer fold0 remains intact. Deterministic seeded inner61/15 split from76 only. No outer19 image/label reads for fitting, scaling, previews, architecture feedback or checkpoint selection. Inner split receipt written in fresh external run, never canonical splits_final.json.
- Reuse completed 3mm target manifest with exact supplied SHA256. Do not load weights from old76-trained generator. Use fresh seeded generator, discriminator and exact author UNet(1,1). Scaling fitted only inner61. Fixed inherited target-construction fill/minimum and nnUNet preprocessing provenance are documented, not claimed refitted inner-only preprocessing.
- Pretraining optional explicit budget on inner61 only. Defaults and optimizer/deviation decisions must be recorded. No hidden warmstart or early-stop selection on19.
- Virtual G step with GAN+L1 using create_graph; virtual S step with real+synthetic segmentation loss using create_graph; internal15 real-image architecture loss through virtual S and G. Clone functional buffers to avoid mutation; preserve graph through generator outputs and normalization. No uint8, NumPy, detach or no_grad along meta path. One-step approximation and optimizer choice explicitly recorded. Active upstream upconv architecture parameters learned; unused conv tensor handling explicit. After alpha step perform actual GAN and S updates with graph boundaries deliberate.
- Native preprocessing grid retained; bottom/right padding only; signed -1 pixels ignored by losses, fake-mask augmentation optional minimal documented seeded flip. Eval/frozen BatchNorm in virtual path explicitly recorded if needed for stable functional higher derivatives. All models/checkpoint optimizers, arch tensors, RNG, protocol and completion state serialized; no resume unless genuinely supported.

## Deliverables and ownership
- Leaf implementer owns experiments/genseg_joint/** (source/tests), optionally surgical existing helper compatibility fixes only if required and approved to parent. Reuse genseg30 and genseg3mm helpers, no upstream edits or Trainer changes.
- Parent owns reference/GENSEG-JOINT-PLAN.md (this), protocol/runbook/review snapshot. Independent read-only Level3 review after exact snapshot stable.
- Minimal CLI check-only then short real-data pilot, no automatic long training or outer19 evaluation. Fresh outputs, resource80percent guards before models/steps, fixed UUID GPU, immutable manifest/hash/inventory checks. Fail nonfinite/absent meta gradients; report finite nonzero active-alpha gradient norm and changes, plus negative-control severed path in synthetic test.

## Validation plan
1. RED/GREEN tiny analytical test verifies hypergradient through virtualG and virtualS against finite differences; changing internalval target changesalpha; detaching generated image removes the complete architecture path. For indirect-path isolation, hold explicit synthesis-alpha constant and preserve A dependence only through virtual G weights; cutting virtual-G dependence then removes this isolated indirect hypergradient. No fake callback substituted for actual optimizer chain.
2. CPU tiny actual pinned generator and authorUNet run same production jointstep, verify at least activealpha finitegradient and update, params persist, bufferisolation, ignoredpixel invariance, allfourweights snapshot unchanged onfailure where promised.
3. Cohort leakage guards tested before reads, disjoint inner61/15/external19, source arrays immutable, exactmanifesthash and source tampering rejected, freshprotectedoutputs, checkpoint roundtrip and CLI arguments. All CPU tests singlethread and guarded resource preflight. No actual patient training available locally.
4. Independent Level3 checks exactfinal snapshot mathematical path, isolation, provenance and source boundaries. Resolveblockers and retestfocusedpaths.
5. Parent verifies status/diffcheck, publishes concise WindowsCMD runbook and deliverable. Integration/push not implied by this trial; coordinate with user server-first Git workflow.

Clarified real-input contract: all G/D real training targets, S real training inputs and inner15 architecture-validation inputs use the same completed3mm-masked preprocessed DWI and the same inner61-fitted scaling. No unmasked real segmentation branch. Enforce inner roles BEFORE TargetDataset.load_train. External19 evaluation is not implemented; future evaluation must use the same prediction-only masking pipeline.

