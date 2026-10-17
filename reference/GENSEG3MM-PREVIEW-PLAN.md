# GenSeg3mm preview implementation plan

Status: user authorized generator-image exploration; no segmentation training or automatic pool approval.
Baseline code: 909df3ec131d9ac323576428e5b206a05f4fd8f6.
Implementation checkout: codex/genseg3mm-preview, independent managed worktree.

## Accepted scope

- Preserve Dataset501 canonical five-fold split; fold0 generator uses exactly 76 training patients, no validation images/labels/QA for fitting or choosing outputs.
- Existing baseline generator is complete at 5000 updates, seed20261014, width64, downs8, discriminator width64, batch1, learning rate2e-5, L1weight100, epochs ceiling200.
- Derive HD-BET masks from all original Dataset501 training DWI, not from the five-case development pilot. The former pilot includes case005, a fold0 validation patient; it is excluded from generator fitting.
- Fixed 3mm in-plane mask margin; no through-plane expansion; retain whole original native slice when its brain mask is empty. This retention rule includes extracranial signal and is not anatomical brain truth. Reject a wholly empty predicted volume.
- Generate a small training-only image comparison, not a 30percent synthetic pool or segmentation result.

## Minimal architecture and deliberate adaptation

New experiments/genseg3mm package only; do not change Source, existing GenSeg30 pipeline, upstream networks, Trainer or original/preprocessed data.

1. Prepare train-only retention masks in native space with HD-BET2.0.1, record weight/raw hashes and commands. Reorient arrays with geometry, never edit headers to pretend alignment.
2. Map retention masks through original nnU-Net reader, transpose, original crop bbox and segmentation resampling to the exact stored preprocessed data grid. Do not independently crop or normalize masks; no preprocessing writes against source.
3. Generator-only adapter copies preprocessed targets and fills outside retention mask with the previous generator's training-global minimum. This is PREPROCESSED-TARGET MASKING, not raw-zero extraction followed by newly fitted normalization. Segmentation arrays, sentinel semantics, plane shapes and original source remain unchanged. Training uses the existing min/max scaling algorithm on transformed targets, recorded afresh.
4. Bind masks/recipe/baseline/source identity into immutable target manifest. Adapter identity and inventory must validate it during old-pipeline train/load. No global Source.load_train changes.
5. Retrain from the same seeded initialization using old generator settings; verify 5000 actual updates. Preserve old generator.
6. Preview old/new generators using same training lesion-mask conditions; output original target, transformed target, old/new fake and label overlay. Restore each model's recorded scaling and architecture. No heldout or test data is read for preview selection.

Brain extraction does not add anatomical information to an empty lesion mask; a quality improvement is unverified.

## Work ownership

- Leaf implementation worker: experiments/genseg3mm/** only, including focused synthetic tests.
- Main coordinator: this plan, reference/GENSEG3MM-RUNBOOK.md, independent review, delivery bundle and final report.
- Independent Level3 reviewer: final exact working-tree files/hashes; read-only geometry, leakage, provenance, source safety and runtime/checkpoint contracts.

## Checks and acceptance

- [ ] Native physical-distance / anisotropic spacing and orientation checks; no hidden z expansion.
- [ ] Original crop/transpose/resampling mapped correctly to stored source grid.
- [ ] Foreign validation patients and incomplete/tampered targets refused before inference/fitting.
- [ ] Original source images/labels remain unchanged; adapter is generator-only.
- [ ] All source/mask/model/protocol identities bound; output directories fresh and disjoint.
- [ ] Actual old settings carried over; no accidental architecture tensor reset.
- [ ] Resource preflight before focused CPU synthetic tests; no default-size GPU/model training locally.
- [ ] Separate Level3 PASS and clear unverified target-computer/real-data boundary.
- [ ] Portable new-file-only bundle and exact Windows CMD commands, use GPU1 if selected and available.

No commits or pushes requested. Canonical project memory is not changed by this branch-local exploratory preparation.
