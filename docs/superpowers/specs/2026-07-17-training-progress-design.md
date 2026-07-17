# Training Progress Display Design

## Goal

Make long-running server training observable from a terminal without changing
the model, data loading, loss, optimizer, scheduler, checkpoint selection, or
CSV training log format.

## Terminal Behaviour

Each training epoch uses a `tqdm` progress bar over its batches. The dynamic
postfix reports the current batch loss, mean loss so far, current CUDA memory
usage in MiB when CUDA is active, and the estimate for the remainder of that
epoch. The progress bar leaves a compact completed line for every epoch so an
SSH log remains readable.

At the end of each epoch, after validation completes, the script emits one
summary line containing the epoch number, epoch duration, global validation
Dice, mean patient Dice, and estimated remaining time for all unfinished
epochs. Total ETA is calculated from the mean duration of completed epochs;
the first epoch reports an unavailable total ETA rather than a misleading
guess.

## Component Boundaries

A small training-progress helper converts seconds to a stable human-readable
duration and builds the progress-bar postfix. `train.py` owns the per-epoch
timer and passes only measured values to that helper. Existing CSV logging
continues to receive the same fields and is not used as a live console output
mechanism.

## Error Handling

The display is non-essential. It never suppresses training errors. On CPU,
the memory field is shown as zero MiB. ETA is displayed as unavailable until
there is enough completed-epoch timing information.

## Tests

Tests are written first and cover duration formatting, CPU/CUDA memory display
selection, ETA unavailable on the first epoch, and the summary payload's
validation Dice fields. Existing training CLI and package tests remain the
regression suite.
