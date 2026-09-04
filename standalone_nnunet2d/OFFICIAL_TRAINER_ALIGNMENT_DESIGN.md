# Official Trainer Alignment Design

## Goal

Add a separate, pure-PyTorch 2D Trainer configuration layer aligned to the
inspected nnUNetTrainer source, without importing nnunetv2 or starting formal
cross-validation.

## Confirmed Contract

The formal layer uses SGD(lr=0.01, momentum=0.99, nesterov=True,
weight_decay=3e-5), PolyLR over 1000 epochs, 250 training and 50 validation
iterations per epoch, foreground oversampling 0.33, full 2D rotation and
mirror axes (0,1), and seven deep-supervision scales. The lowest-resolution
loss output has zero weight; remaining exponential weights normalize to one.

## Boundaries

This layer is distinct from `smoke_run_only`. It exposes configuration,
scheduler, loss-weight, sampler, and augmentation interfaces with synthetic
tests first. It neither starts a 5-fold run nor claims that every official
transform has been independently reproduced until each is implemented and
validated.
