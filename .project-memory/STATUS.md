# Status

Updated: 2026-10-27

## Integration state
- Canonical integration branch: master; starting commit for this conversion integration: 8592d18.
- Master contains the original H2Former implementation and the existing nnU-Net early-stopping/evaluation work recorded in its history.
- Most experiment code remains on separate feature branches. This conversion integration does not merge or establish acceptance of those other workstreams.
- Branch pointers were checked locally on 2026-10-21; remote freshness and current full-suite behavior were not verified.

- CI-1 conversion update from task commit `0df8d43` is integrated in this master working tree: explicit high/low/numeric b selection, DWIADC eligibility, strict geometry/output guards and separate rejection provenance. Multiple valid DWI series are resolved by a unique label-header geometry match, otherwise the unique minimum numeric SeriesNumber; selection reasons are recorded and final geometry refusal remains. Timepoint recognition accepts both name-Dx-DWI and nameDx-DWI; a local filename-only audit finds 131 candidate patient/timepoint keys versus 98 before the fix, not a successful-conversion count. Label selection prefers pure DWI filenames per patient/timepoint, falling back to combined DWIADC only when pure DWI is absent. Within each label type, normal .nii/.nii.gz suffixes outrank extra-dot ..nii/..nii.gz suffixes, as explicitly selected by the user. Byte-identical duplicates are deduplicated; differing contents within the selected priority remain an ambiguity error. Independent Level 3 review: PASS STATIC; final runtime tests and real conversion were not run per user instruction. Other feature-branch history was not merged.

## Active project workstreams
- `codex/genseg-reference` at 5472832: HD-BET3mm preparation/runbooks, GenSeg reference/adaptation and detailed branch-local project memory.
- Prior server-returned evidence reports HD-BET3mm95-case preparation and fixed-fold0 ISLES-initialized/scratch evaluation completed. These historical reports were not rechecked against server files in this update; they remain single-fold development evidence. Consult the branch memory for provenance and metric limitations.
- GenSeg GD500/joint previews were reported gray; completed100-step fixed-slice diagnostics do not establish useful augmentation. BN/optimizer causes remain unresolved hypotheses.
- `codex/genseg-quality-repair` at 9a41852: separate diagnostic work exists; its current acceptance/runtime status was not audited here.
- `codex/nnunet-ckpd-finetune` at a0455a0: isolated CKPD adaptation exists; do not label the Conv2d adaptation a faithful paper reproduction without the source audit and explicit deviations.
- `codex/isles2022-dwi-conversion` at 414cdac: ISLES conversion/evaluation work exists; `codex/nnunet-exclude15-ab` at ccbf1b1 and `codex/nnunet-result-report` at abedcd8 contain separate cohort/reporting work. Cohort selection effects must be distinguished from training gains.
- Standalone nnU-Net, H2Former transfer/source-space evaluation, UPerNet variants, multichannel/asymmetry, optical models and Stage2/ROI work remain in separate feature branches. A branch pointer is inventory evidence, not proof of tests, integration or experimental benefit.

## Scientific invariants
- Dataset501 remains the established 95-entry DWI baseline. The supplied conversion manifest contains 95 examinations under 25 distinct patient fields; do not call these 95 independent patients. Preserve existing five-fold splits and the existing fold0 76/19 and GenSeg inner61/15 entry partitions; patient-group separation has NOT yet been audited. No split regeneration is authorized.
- Real images/labels and old checkpoints are protected sources; derived outputs remain separate. GT must not influence inference-time input construction; prediction-guided Stage2 training requires OOF predictions.
- Formal metrics use original full-volume space. Synthetic/static/preflight and fold0 development results are distinct from formal five-fold OOF evidence.
- Compare identical cohorts, source checkpoints, Trainer protocols and metric definitions before causal conclusions. Skull stripping and registration are distinct operations.

## Operational constraints
- Prefer newconda for Python and request direct command approval per user instruction. No nested subagents.
- Latest acceptance is static-only conversion tooling, not dataset/clinical validation. No real conversion, training, raw-data changes, split changes or remote push occurred in this task.
