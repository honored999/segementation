# Standalone nnU-Net 2D Reproduction

This project is a pure-PyTorch foundation for reproducing the 2D `PlainConvUNet`
configuration of `Dataset501_StrokeLesion`. It does not import `nnunetv2` or
`dynamic_network_architectures` at runtime.

Current progress: reference validation, environment inspection, JSON-driven
network construction, and CPU model-shape tests are implemented. Formal
training, preprocessing, augmentation, loss, inference, and evaluation are not
implemented and must not be started in this phase.

Run the checks from the worktree root (using the required environment):

```powershell
conda run -n newconda python standalone_nnunet2d/tools/inspect_environment.py
conda run -n newconda python standalone_nnunet2d/tools/inspect_reference.py
conda run -n newconda python standalone_nnunet2d/tools/inspect_dataset.py --raw-root C:\path\to\Dataset501_StrokeLesion
conda run -n newconda python -m pytest standalone_nnunet2d/tests -v
```

The three external Dataset501 directories are strictly read-only. Do not copy
the NIfTI dataset, NNUNet outputs, large NPZ files, or model weights into this
repository; any future generated artifacts belong under `outputs/`.

`inspect_dataset.py` only checks paths unless a single `--case-id` is supplied.
It never scans all images. Data loading currently supports only the read-only,
on-demand slice pipeline; no training, validation, or prediction command is
enabled.
