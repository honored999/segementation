# Train and Validation Engine Design

## Goal

Provide explicit single-epoch training and validation functions that compose
the existing model, loss, metric, and checkpoint foundations. The functions
must be testable with synthetic tensors and must not make formal training start
implicitly.

## Scope

`engine/trainer.py` will retain the existing `train_step` and add
`run_train_epoch`. It will consume a caller-supplied iterable of
`(image, target)` tensor batches, call `train_step` once for each batch, and
return an aggregate containing batch count, mean loss, and output shapes from
the final batch. Empty iterables are invalid because they cannot establish an
epoch loss.

`engine/validator.py` will add `run_validation_epoch`. It will consume the same
batch form under `torch.no_grad()`, calculate the supplied loss function, turn
full-resolution logits into binary foreground masks with `argmax`, and aggregate
the existing binary Dice/IoU metrics across batches. It will restore the model's
previous training/evaluation mode before returning, so validation has no hidden
mode side effect.

## Boundaries

The epoch functions accept only already-created iterables; they do not discover
data, construct DataLoaders, choose folds, schedule epochs, save checkpoints,
or write results. A future CLI may explicitly compose these functions after
server preflight checks, but this phase does not change `train.py` from its
formal-training-disabled state.

The validator supports binary segmentation only, matching the project's
existing binary metric helpers and Dataset501 target. It rejects logits that do
not have exactly two channels or targets that do not match the full-resolution
spatial shape, instead of silently calculating an invalid metric.

## Error Handling

Both functions reject empty batch iterables with `ValueError`. Training inherits
the existing finite-loss guard from `train_step`. Validation rejects non-finite
losses and incompatible output/target shapes with clear `ValueError` messages.
Neither function catches device, model, or loss exceptions: those retain their
original stack traces for server diagnostics.

## Tests

New unit tests will use a tiny `Conv2d` model, deterministic synthetic batches,
the current Dice-plus-cross-entropy loss, and CPU execution. They will prove
that the train epoch updates model parameters and returns the expected aggregate;
that validation computes the known perfect Dice/IoU result without gradients;
and that both reject empty iterables. Existing tests remain the regression suite.

## Alternatives Considered

1. Add a full CLI and epoch scheduler now. This would make the server command
   appear complete, but would combine unverified data paths, policy choices, and
   long-running execution with the core aggregation logic.
2. Add only a server preflight command. This improves diagnostics but leaves no
   reusable train/validation orchestration.
3. Add explicit single-epoch functions first. This is the selected design: it
   isolates computation, establishes deterministic tests, and leaves future CLI
   policy explicit.
