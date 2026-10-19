# Acceptance checklist: HD-BET nnU-Net fold0

Scope: user-approved real-DWI preprocessing comparison, ordinary deep supervision, TopK10+existing early stopping, fixed76/19. Not GenSeg synthesis, not formal5fold evidence.

- [ ] Metadata-only check has no patient image/label arrays, model load, CUDA context or output writes; audits pinned target receipt and canonical five-fold membership.
- [ ] Reuse only76 native HD-BET masks from exacthash auditedreceipt; process19 withsame fixed HD-BET version/weights/TTA, never usingGTtofixmask.
- [ ] Reorient mask to native rawgrid before physical inplane3mm; no z dilation; emptyplane keep and emptyvolume fail. NIfTIheader mm/orthogonality and shape/affine validated.
- [ ] Original per-voxel kept image values preserved, outsidezero, nativefullvolume geometry preserved; labels byteexact copied. QA/provenance cannotchangeinputsemantics.
- [ ] Fresh resolved independentHD-BET_outputs root excludes source/repo/targets/weights; protect referencedcache files and writeboundaries, nooverwrite/resume/destructiveactions.
- [ ] Before eachcase/HD-BET model subprocess, guardselectedGPU physicalindex+UUID withCPU/RAM/VRAM/util<80 and freeheadroom. Guard samples after expensiveaudits and is not presented as a runtime utilizationcap.
- [ ] Complete receipt onlyafter all95 readback/hashaudit; partialoutputs remainincomplete. No95patientlist or traininglog in projectmemory.
- [ ] Isolated raw/pp/results allthree envroots enforced; original plans/splits copied unchanged; required baseline fingerprint compatibility file explicitly inherited and hash-checked; no datasetID collision outside controlledroots. Preprocess-check refuses existingconfigdir because officialpreprocess deletesit.
- [ ] ready checks exact95 expectedb2nd/seg/pkl and fullreceipt/split/plans/env; separate server-only trainer-check verifies actual externalTrainer class/source withoutconstructingmodels. No freshtraining overwritesexistingfold0.
- [ ] ComboTrainer uses existingTopK10loss andESlifecycle bycooperativeMRO, samePlainConvUNet/deepsupervision; no sourceclass edits.
- [ ] Serverrunbook has CMDnnunet5090 realpaths, check->prepare->trainonlyQA->preprocess-check->preprocess->ready->guard->freshtrainbest->fixed19originalspaceeval. No inventedcommandflags or genotypeclaims.
- [ ] Local syntax/source/diff checks only; runtime/serverchecks explicitly deferred. Independent exactsnapshotLevel3 STATIC PASS, blockersresolved throughleafworker.
- [ ] Explicitfiles staged/reviewed/committed/pushed; originalraw/splits/weights and otherworktrees unchanged. Branch-localmemory synchronized; canonicalmaster memoryabsent/untouched. Globalmemorynotmodified.