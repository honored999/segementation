# Repeat-Oracle Inference Parity Design

## Goal

Replace the non-reproducible single-run exact-mask inference gate with a
repeat-oracle stability gate. The gate must distinguish stable official output
from voxels that the installed `nnunetv2` predictor itself changes across
identical CUDA runs.

The report remains evidence for an external alignment decision. It always
writes `run_state="official_alignment_pending"` and never promotes a training
or inference run to `official_aligned` by itself.

## Evidence and problem boundary

For the same case, checkpoint, fold, device, seed, plans, and input:

- official oracle run 1 and run 2 differ at two mask voxels;
- standalone differs from each official run at one of those voxels;
- official and standalone input arrays are exactly equal;
- converted weights are exactly equal;
- official and standalone FP16 logits are exactly equal when evaluated under
  the same initialized CUDA execution context.

Therefore a single official mask is not a reproducible exact oracle. The new
gate changes only inference-mask comparison. Transform parity, checkpoint
mapping, preprocessing, metrics, and training labels are unchanged.

## Inputs

The repeated gate accepts:

- at least three independently generated official inference artifact roots;
- exactly one standalone inference artifact root;
- artifacts for the same case, fold, plans hash, capture mode, array schema,
  and source-space NIfTI metadata.

Each official artifact must come from a separate invocation of the installed
server `nnunetv2` predictor. Reusing or copying one artifact does not satisfy
the repeat requirement.

The existing single-oracle comparison remains available and unchanged for
transform parity and diagnostic exact comparisons.

## Mask stability policy

Stack the official masks in repeat order and classify every voxel:

- **Stable voxel:** every official repeat has the same integer label.
- **Unstable voxel:** at least two official repeats have different labels.

The standalone mask passes only when both conditions hold:

1. Every stable voxel equals the unanimous official label exactly.
2. At every unstable voxel, the standalone label is one of the labels observed
   at that voxel in the official repeats.

There is no image tolerance, Dice threshold, morphological cleanup, retry-until-
match behavior, or silent substitution. Integer comparison remains exact.
Official instability is disclosed rather than ignored.

## Non-mask components

The gate compares every official repeat and the standalone artifact for:

- readable and compatible manifests;
- exact image array shape, dtype, and values;
- exact label array shape, dtype, and values;
- matching case ID, plans hash, capture mode, and NIfTI spatial metadata.

Any disagreement in these components fails the report. Repeated-oracle
handling applies only to the inference mask.

## Report contract

The JSON report adds:

- `parity_policy: "repeat_oracle_stability_v1"`;
- `oracle_roots` and `oracle_repeat_count`;
- `oracle_pairwise_mask_difference_counts`;
- `oracle_unstable_voxel_count` and complete unstable voxel coordinates;
- `stable_mask_mismatch_count` and coordinates;
- `unobserved_standalone_label_count` and coordinates;
- component-level pass/fail diagnostics;
- `status`, while retaining `run_state="official_alignment_pending"`.

The report passes only when the repeat count is at least three, all non-mask
components pass, and both standalone mask error counts are zero.

## CLI compatibility

`standalone_nnunet2d.tools.parity_report` keeps the existing command compatible.
`--oracle-root` becomes repeatable:

- one occurrence uses the existing exact comparison;
- three or more occurrences use `repeat_oracle_stability_v1`;
- two occurrences are rejected because they demonstrate variability but do
  not satisfy the accepted repeat gate.

`--standalone-root`, `--output`, and `--image-atol` retain their existing
meaning. The repeated inference gate requires `--image-atol 0`.

## Error handling

The command fails explicitly for fewer than three repeated roots, duplicate
resolved oracle roots, non-inference artifacts, inconsistent manifests,
missing arrays, shape or dtype differences, non-integer masks, and negative or
nonzero image tolerance in repeated mode.

Failures never rewrite source artifacts and never emit `official_aligned`.

## Testing

Tests follow strict red-green TDD and cover:

- three identical oracle masks and an exact standalone mask;
- official variability with a standalone mask assembled only from observed
  official labels;
- a mismatch on an official-stable voxel;
- an unobserved standalone label at an unstable voxel;
- fewer than three or duplicate oracle roots;
- inconsistent image, label, manifest, and spatial metadata;
- unchanged single-oracle transform behavior;
- CLI JSON output and persistent pending run state;
- absence of automatic `official_aligned` promotion.

## Acceptance boundary

A passed repeated inference report, together with the already passed transform
report, establishes that the standalone implementation is within the observed
behavior of the installed official predictor. It does not retroactively relabel
older checkpoints or training runs. Any formal training result produced before
the aligned augmentation and inference path remains
`official_alignment_pending` unless separately rerun and validated.
