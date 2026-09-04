# Single-Step Dry-Run Engine Design

## Scope

Implement one explicit training step for synthetic or caller-supplied batches:
forward pass, selected main/deep-supervision loss, backward pass, optimizer
step, and scalar result reporting. No epoch loop, dataloader traversal,
checkpoint write, external-data read, or formal training command is included.

## Interface

`train_step(model, batch, loss_fn, optimizer, device)` accepts image/label
tensors, moves them to the selected device, calls the model, computes loss,
checks that it is finite, applies backward/optimizer step, and returns a Python
float plus output shape metadata. The optimizer is supplied by the caller; no
optimizer, learning-rate, scheduler, or epoch default is claimed to reproduce
official nnU-Net.

## Safety and tests

Tests use a tiny synthetic convolution model and one small batch to verify that
parameters change, loss is finite, deep-supervision output accepts the supplied
loss function, and non-finite loss raises before the optimizer step. The public
`train.py` remains a hard exit with no override in this phase, so invoking it
cannot begin training.
