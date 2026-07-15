# Scratch 6 GB Smoke Training Design

## Goal

Run a fair, no-ImageNet-pretraining smoke check on the local RTX 3060 Laptop
GPU with 6 GB VRAM before committing to full training.

## Configurations

Create independent electronic and Hybrid YAML files. Both set
`encoder_weights: null`, `batch_size: 2`, `num_workers: 0` and `epochs: 1`.
They retain seed 2026, fold 0, native 512 by 512 images, loss, optimizer,
threshold and all other common settings. Existing pretraining configs remain
unchanged.

## Execution and Success

Run electronic then Hybrid from the project root on local CUDA. Success means
each run completes one epoch without CUDA OOM, writes last/best checkpoint and
training log, and reports peak GPU memory. This is an engineering smoke test,
not an accuracy comparison; full scratch training remains a later explicit run.
