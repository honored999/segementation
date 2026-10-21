# Status

Updated: 2026-10-21

## Integration state
- Canonical integration branch: master; starting commit for this documentation update: 10079c4.
- Master contains the original H2Former implementation and the existing nnU-Net early-stopping/evaluation work recorded in its history.
- Most recent experiments live on separate feature branches. This project-wide overview does not merge their production code or establish their acceptance into master.
- Branch pointers were checked locally on 2026-10-21; remote freshness and current full-suite behavior were not verified.

## Active project workstreams
- `codex/genseg-reference` at 5472832: HD-BET3mm preparation/runbooks, GenSeg reference/adaptation and detailed branch-local project memory.
- Prior server-returned evidence reports HD-BET3mm95-case preparation and fixed-fold0 ISLES-initialized/scratch evaluation completed. These historical reports were not rechecked against server files in this update; they remain single-fold development evidence. Consult the branch memory for provenance and metric limitations.
- GenSeg GD500/joint previews were reported gray; completed100-step fixed-slice diagnostics do not establish useful augmentation. BN/optimizer causes remain unresolved hypotheses.
- `codex/genseg-quality-repair` at 9a41852: separate diagnostic work exists; its current acceptance/runtime status was not audited here.
- `codex/nnunet-ckpd-finetune` at a0455a0: isolated CKPD adaptation exists; do not label the Conv2d adaptation a faithful paper reproduction without the source audit and explicit deviations.
- `codex/isles2022-dwi-conversion` at 414cdac: ISLES conversion/evaluation work exists; `codex/nnunet-exclude15-ab` at ccbf1b1 and `codex/nnunet-result-report` at abedcd8 contain separate cohort/reporting work. Cohort selection effects must be distinguished from training gains.
- Standalone nnU-Net, H2Former transfer/source-space evaluation, UPerNet variants, multichannel/asymmetry, optical models and Stage2/ROI work remain in separate feature branches. A branch pointer is inventory evidence, not proof of tests, integration or experimental benefit.

## Scientific invariants
- Dataset501 is the DWI-only baseline with95 cases; preserve patient-level five-fold splits. Ordinary fold0 is76/19; GenSeg inner61/15 is a distinct training-only partition.
- Real images/labels and old checkpoints are protected sources; derived outputs remain separate. GT must not influence inference-time input construction; prediction-guided Stage2 training requires OOF predictions.
- Formal metrics use original full-volume space. Synthetic/static/preflight and fold0 development results are distinct from formal five-fold OOF evidence.
- Compare identical cohorts, source checkpoints, Trainer protocols and metric definitions before causal conclusions. Skull stripping and registration are distinct operations.

## Operational constraints
- Prefer newconda for Python and request direct command approval per user instruction. No nested subagents.
- This update changes documentation only; no runtime tests, server data access or fresh experimental validation occurred.
