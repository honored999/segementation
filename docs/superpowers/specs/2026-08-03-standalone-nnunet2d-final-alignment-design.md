# Standalone nnU-Net 2D Final Alignment Design

## Goal

Deliver a pure-PyTorch standalone 2D nnU-Net workflow that can train, infer,
validate all five supplied folds, and produce a 95-case out-of-fold report.
It must not claim official alignment until server-side differential checks
against the installed `nnunetv2` implementation have passed.

## Scope and non-goals

The project remains independent of `nnunetv2` at runtime. The server's
installed `nnunetv2` is used only as an oracle that exports reproducible
reference artifacts. No official data, splits, results, or checkpoints are
overwritten. A run before all parity checks pass is labelled
`official_alignment_pending`; a passing report may be labelled
`official_aligned` and must include the oracle version and report path.

## Architecture

The workflow has five explicit layers:

1. **Reference capture.** A server-only oracle command records the official
   preprocessing result, sampled batch geometry, transformed image/label,
   deep-supervision labels, selected logits, and inference masks for fixed
   case IDs and seeds. It writes portable arrays plus JSON metadata without
   changing official training artifacts.
2. **Standalone training data.** A plan-derived preprocessing and batch-patch
   sampler reproduces the captured geometry. The sampler creates the same
   foreground/background composition per batch as the oracle, uses the
   official initial patch size, and applies paired spatial and intensity
   transforms before generating deep-supervision targets.
3. **Trainer.** The trainer retains the confirmed SGD, PolyLR, deep
   supervision, 250/50 iteration schedule, and checkpoints, but saves all
   reproducibility state required to continue a run consistently.
4. **Inference and fold evaluation.** A formal prediction command loads a
   checkpoint in full-resolution mode, applies the captured inference policy
   (including mirroring and sliding-window aggregation when required), restores
   the raw-image geometry, writes NIfTI masks, and checks their metadata. Fold
   validation runs this prediction path for every held-out case.
5. **OOF reporting and parity gate.** Fold reports are aggregated only after
   every held-out case has one prediction. The report records case-level
   foreground Dice/IoU, empty-mask policy, argmax rule, aggregation, cohort,
   and run state. A separate parity report marks each oracle comparison pass or
   fail; no metric report may elevate the run state itself.

## Transform and sampler contract

The standalone transform implementation is structured as independently tested
operators rather than one opaque augmentation function. It must reproduce the
oracle's operation order, probabilities, interpolation orders, label padding
and removal, crop centre, initial patch dimensions, rotation, scaling,
Gaussian noise, blur, brightness, contrast, low-resolution simulation, both
gamma operations, mirroring, and target downsampling. The batch sampler's
foreground decision is deterministic for a supplied batch index and seed;
the exact batch-position policy is captured from the installed official source
rather than inferred from the scalar 0.33 alone.

## Inference and metric contract

Inference always evaluates full volumes, never random validation patches. It
uses class argmax, reassembles `(z, y, x)`, and exports a binary `uint8` NIfTI
in the source image's spacing, origin, and direction. The final Dice path is:
model output -> configured inference transforms -> argmax -> full-volume mask
-> TP/FP/FN -> one case Dice -> macro mean across the unique 95 OOF cases.
Both-empty and one-empty behaviour are written into `summary.json`; they are
not inferred from the training loss or online validation value.

## Acceptance checks

The final implementation is accepted only when all of the following are
recorded in files under `standalone_nnunet2d/outputs/`:

- Unit tests cover every new sampling, transform, inference, restore, metric,
  and report edge case.
- The fixed-seed parity report compares standalone and oracle preprocessing,
  transformed labels exactly, transformed image values within declared
  tolerance, deep-supervision target shapes/values, and sampling decisions.
- A fixed checkpoint parity report compares standalone and oracle inference
  masks and validates written NIfTI spatial metadata.
- Fold 0 produces all 19 held-out predictions and a case-level report.
- Five-fold OOF aggregation contains exactly 95 unique cases and explicitly
  records its evaluation convention.

Failure of any parity check leaves the run state as
`official_alignment_pending` and identifies the failing component; it does not
silently substitute an approximate result.

## Error handling and reproducibility

Commands reject malformed plans, invalid fold membership, missing oracle
artifacts, inconsistent image/label geometry, unknown checkpoint state, and
duplicate/missing OOF case IDs. Checkpoints persist model, optimizer,
scheduler, epoch, global step, and random-generator states. Every formal run
writes a resolved configuration containing the plan fingerprint, source paths,
seed, device, transform/inference policy, and run-state label.
