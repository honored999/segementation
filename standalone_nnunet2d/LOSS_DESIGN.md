# Loss and Deep-Supervision Design

## Scope

Implement loss computation only: batch Dice, categorical cross-entropy, their
sum, and aggregation over the model's ordered deep-supervision logits. No
optimizer, backward pass, data augmentation, trainer, checkpoint, or formal
training run is included.

## Interfaces

`SoftDiceLoss` accepts logits of shape `(B, C, H, W)` and integer targets of
shape `(B, H, W)`. It applies softmax internally, calculates class Dice over
the complete batch (`batch_dice=true`), and excludes background by default for
the binary lesion objective. It raises clear errors for invalid shapes, class
labels, or non-finite logits.

`DiceCrossEntropyLoss` returns the sum of cross-entropy and Soft Dice. Targets
are resized with nearest-neighbor interpolation for auxiliary logits only;
class labels remain integers. The single main output needs no target resize.

`DeepSupervisionLoss` receives the existing model order `(512, 256, 128, 64,
32, 16, 8)` and aggregates each level's compound loss with caller-provided,
positive weights normalized to sum to one. It therefore does not silently
invent official nnU-Net deep-supervision weights. `deep_supervision=False`
uses the same compound loss directly on a single tensor.

## Tests and safety

Tests will use small synthetic logits/labels to verify near-zero Dice loss for
correct confident predictions, finite compound loss, nearest-neighbor target
resizing, normalized weighted aggregation, invalid-input errors, and gradient
existence. They will not access external NIfTI data and will not run an
optimizer or training loop.

## Confirmed and unconfirmed behavior

The plan directly confirms `batch_dice=true`; the output ordering is documented
by the current model. The exact official Dice smoothing constants, background
inclusion, CE weighting, and deep-supervision scale weights remain unverified
because the compatible official nnU-Net source environment is unavailable.
Those values will be explicit constructor/configuration choices and recorded in
the reproduction notes rather than presented as exact official defaults.
