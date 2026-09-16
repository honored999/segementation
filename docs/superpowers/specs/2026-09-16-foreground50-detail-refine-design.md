# Foreground50 DetailRefine Design

## Scope

Add one isolated official nnU-Net v2 Trainer experiment,
`nnUNetTrainerForeground50DetailRefine`. It preserves the existing
`nnUNetTrainerForeground50` training protocol and changes only the highest-resolution
segmentation path. This is an engineering implementation and synthetic validation
task; it does not train or evaluate real Dataset501 data.

## Architecture

Build the baseline network through the installed nnU-Net 2.8.1 constructor. After
the official constructor and its network-wide initialization have completed, replace
only `decoder.seg_layers[-1]` with a serializable head that owns the original
segmentation layer and a residual feature refiner:

`C -> 16 (1x1) -> 16 (3x3) -> norm -> activation -> 16 (3x3) -> norm -> activation -> C (1x1)`.

The head computes `segmentation_head(F + detail_refiner(F))`. It reads `C`, the
convolution dimensionality, convolution bias convention, normalization type and
activation type from the constructed decoder/encoder. The final projection is zero
initialized after module construction, including its bias when present. Because the
replacement happens after official initialization, no later initializer overwrites
the zero projection. Lower-resolution deep-supervision heads remain untouched.

The implementation rejects unsupported network layouts with a clear error instead
of guessing. It does not use hooks, monkey patches, site-package edits, adjacent
slices, 2.5D input, TopK, Tversky, ADC, ROI, or a copied training framework.

## Runtime contracts

- The Trainer uses the current five-argument `build_network_architecture` signature.
- Training, resume, validation and predictor reconstruction all resolve the same
  Trainer class and therefore rebuild the same architecture.
- Official strict `state_dict` loading remains in force. A DetailRefine checkpoint
  round-trips strictly; a baseline architecture checkpoint is rejected rather than
  silently accepted as a resume checkpoint.
- Deep supervision keeps its original list order (highest resolution first); only
  output index zero is refined. Disabling deep supervision returns the same single
  highest-resolution path.
- Sliding-window and mirrored TTA inference call the rebuilt network normally and
  require no special inference implementation.

## Validation

Use synthetic 2D networks and inputs to verify baseline non-mutation, output shapes
and order, zero-init parity, initialization order, staged gradient flow, paired
backbone initialization, strict checkpoint behavior, resolver discovery, deep-
supervision toggling, and predictor sliding-window/TTA calls. Measure parameter
counts from actual constructors. Probe GPU availability, but do not alter the
Dataset501 batch or patch size; without the real server plans/checkpoint, matched
resource measurements remain explicitly pending.

## Experiment handoff

Document A/B commands for baseline Foreground50 versus Foreground50+DetailRefine,
using the same Dataset501 plans, fixed patient-level folds, budget, seed and backbone
initialization rule. Start with fold 0 and compare full reconstructed-volume
case-macro Dice, precision, recall and HD95 plus paired per-case differences. The
pre-fixed engineering screen is Dice improvement at least 0.5 percentage points,
recall non-decrease, and no unacceptable HD95 or false-positive degradation; it is
not a significance claim.
