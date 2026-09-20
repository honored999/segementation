# Lite UPerNet Decoder Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Do not create subagents, reviewers, fixers, validators, or new tasks; this worker is a leaf because the user manually dispatches every delegated role.

**Goal:** Add isolated `h2former_lite_upernet` and `plain_conv_unet_lite_upernet` single-output model variants while preserving both baseline models and all non-architecture experiment policies.

**Architecture:** A shared four-level `LiteUPerDecoder` applies a normalization-free `(1,2,4)` PPM to the deepest feature, a 64-channel FPN with exact-size bilinear interpolation, multi-level concatenation, and a two-class logits head. Thin H2Former and PlainConvUNet adapters expose the approved encoder feature levels and keep model/checkpoint identities distinct.

**Tech Stack:** Python 3, PyTorch, pytest, standalone nnU-Net 2D, PowerShell, `newconda`.

---

## Execution boundaries

- Repository: `E:\study\研一\work14-图像分割\segementation`
- Starting branch: `feat/h2former-stroke`
- Baseline design commit: `1e6d796`; execute from the later commit that contains
  this implementation plan.
- Authoritative design: `docs/superpowers/specs/2026-09-20-lite-upernet-decoders-design.md`
- Read and obey root `AGENTS.md` and closest subdirectory instructions.
- Keep `third_party/H2Former`, real datasets, raw/preprocessed/results roots, and
  user data read-only. Do not access real medical data.
- Do not change splits, data handling, augmentation, sampling, loss, scheduler,
  checkpoint selection, early stopping, batch size, or inference postprocessing.
- Do not edit `.project-memory/`; the main agent owns canonical memory sync.
- Do not push. Commit only the intended implementation and test files.

Before every test, validation, model-loading check, or FLOP command, run this
PowerShell preflight immediately beforehand and record the values:

```powershell
$cpu = ((Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 3).CounterSamples | Measure-Object CookedValue -Average).Average
$os = Get-CimInstance Win32_OperatingSystem
$ram = 100 * (1 - ($os.FreePhysicalMemory / $os.TotalVisibleMemorySize))
Write-Output ("CPU_AVG_PERCENT={0:N1}" -f $cpu)
Write-Output ("RAM_USED_PERCENT={0:N1}" -f $ram)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
} else {
    Write-Output 'GPU_UNAVAILABLE'
}
```

If current or credibly projected CPU, RAM, GPU, or VRAM is at least 80%, do not
run the planned command. Run the narrowest semantically faithful substitute or
record `DEFERRED_RESOURCE_GUARD` / `NOT_RUN_RESOURCE_GUARD`. Use visible test
output via `conda run -n newconda --no-capture-output`.

## File map

Create:

- `standalone_nnunet2d/models/lite_upernet.py` — shared decoder and validation.
- `standalone_nnunet2d/models/h2former_lite_upernet.py` — H2Former adapter.
- `standalone_nnunet2d/models/plain_conv_unet_lite_upernet.py` — PlainConvUNet adapter.
- `standalone_nnunet2d/tools/profile_lite_upernet.py` — reproducible parameter/FLOP engineering report.
- `standalone_nnunet2d/tests/test_lite_upernet.py` — decoder and adapter contracts.

Modify:

- `standalone_nnunet2d/models/h2former.py` — expose encoder features without changing baseline output.
- `standalone_nnunet2d/models/plain_conv_unet.py` — expose encoder features without changing baseline output.
- `standalone_nnunet2d/models/__init__.py` — export the two variants.
- `standalone_nnunet2d/models/factory.py` — names, contracts, and construction.
- `standalone_nnunet2d/training/official_config.py` — permit the H2Former variant to use the existing H2-only AdamW preset.
- `standalone_nnunet2d/tests/test_model_factory.py` — CLI/config/loss/factory coverage.
- `standalone_nnunet2d/tests/test_official_trainer_config.py` — optimizer-family coverage.
- `standalone_nnunet2d/tests/test_formal_checkpoint.py` — cross-identity rejection.
- `standalone_nnunet2d/tests/test_predict_command.py` — metadata-driven reconstruction.

Do not modify baseline checkpoint formats or add dependencies.

### Task 1: Shared LiteUPerDecoder

**Files:**
- Create: `standalone_nnunet2d/models/lite_upernet.py`
- Create: `standalone_nnunet2d/tests/test_lite_upernet.py`

- [ ] **Step 1: Write failing decoder contract tests**

Add tests with these exact behavioral assertions:

```python
from __future__ import annotations

import pytest
import torch
from torch import nn

from standalone_nnunet2d.models.lite_upernet import LiteUPerDecoder


def _batch_norm(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels)


def test_lite_uper_decoder_returns_exact_output_size_and_finite_logits() -> None:
    decoder = LiteUPerDecoder(
        in_channels=(8, 16, 32, 64),
        num_classes=2,
        fpn_channels=8,
        pool_scales=(1, 2, 4),
        norm_factory=_batch_norm,
    ).eval()
    features = (
        torch.randn(2, 8, 64, 64),
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
    )
    with torch.inference_mode():
        logits = decoder(features, output_size=(128, 128))
    assert logits.shape == (2, 2, 128, 128)
    assert torch.isfinite(logits).all()
    assert not any(isinstance(module, (nn.Softmax, nn.LogSoftmax, nn.Sigmoid)) for module in decoder.modules())


def test_ppm_branches_are_normalization_free_and_support_one_by_one_pooling() -> None:
    decoder = LiteUPerDecoder(
        in_channels=(8, 16, 32, 64), num_classes=2, fpn_channels=8,
        pool_scales=(1, 2, 4), norm_factory=lambda channels: nn.InstanceNorm2d(channels, affine=True),
    ).train()
    features = (
        torch.randn(2, 8, 32, 32), torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4),
    )
    logits = decoder(features, output_size=(64, 64))
    logits.square().mean().backward()
    assert logits.shape == (2, 2, 64, 64)
    assert all(not any(isinstance(module, nn.InstanceNorm2d) for module in branch.modules()) for branch in decoder.ppm_branches)


@pytest.mark.parametrize(
    ("features", "output_size", "message"),
    [
        ((torch.zeros(1, 8, 16, 16),), (32, 32), "four"),
        ((torch.zeros(1, 8, 16),) * 4, (32, 32), "BCHW"),
        ((torch.zeros(1, 7, 16, 16), torch.zeros(1, 16, 8, 8), torch.zeros(1, 32, 4, 4), torch.zeros(1, 64, 2, 2)), (32, 32), "channels"),
        ((torch.zeros(1, 8, 8, 8), torch.zeros(1, 16, 16, 16), torch.zeros(1, 32, 4, 4), torch.zeros(1, 64, 2, 2)), (32, 32), "highest to lowest"),
        ((torch.zeros(1, 8, 16, 16), torch.zeros(2, 16, 8, 8), torch.zeros(1, 32, 4, 4), torch.zeros(1, 64, 2, 2)), (32, 32), "batch"),
        ((torch.zeros(1, 8, 16, 16), torch.zeros(1, 16, 8, 8), torch.zeros(1, 32, 4, 4), torch.zeros(1, 64, 2, 2)), (0, 32), "output_size"),
    ],
)
def test_lite_uper_decoder_rejects_invalid_contracts(features, output_size, message: str) -> None:
    decoder = LiteUPerDecoder((8, 16, 32, 64), 2, 8, (1, 2, 4), _batch_norm)
    with pytest.raises((TypeError, ValueError), match=message):
        decoder(features, output_size=output_size)
```

- [ ] **Step 2: Run the focused tests and observe RED**

After resource preflight, run:

```powershell
conda run -n newconda --no-capture-output python -m pytest standalone_nnunet2d/tests/test_lite_upernet.py -q
```

Expected: collection fails because `standalone_nnunet2d.models.lite_upernet` does not exist.

- [ ] **Step 3: Implement the minimal shared decoder**

Implement the shared module with this complete structure (minor formatting is
allowed, behavior is not):

```python
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NormFactory = Callable[[int], nn.Module]


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, norm_factory: NormFactory) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            norm_factory(out_channels),
            nn.ReLU(inplace=True),
        )


class LiteUPerDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        fpn_channels: int = 64,
        pool_scales: Sequence[int] = (1, 2, 4),
        norm_factory: NormFactory = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        channels = tuple(in_channels)
        scales = tuple(pool_scales)
        if len(channels) != 4 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in channels):
            raise ValueError("in_channels must contain four positive integers")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if isinstance(fpn_channels, bool) or not isinstance(fpn_channels, int) or fpn_channels <= 0:
            raise ValueError("fpn_channels must be a positive integer")
        if not scales or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in scales):
            raise ValueError("pool_scales must contain positive integers")
        if not callable(norm_factory):
            raise TypeError("norm_factory must be callable")

        self.in_channels = channels
        self.pool_scales = scales
        deepest_channels = channels[-1]
        self.ppm_branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((scale, scale)),
                nn.Conv2d(deepest_channels, fpn_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for scale in scales
        )
        self.ppm_bottleneck = ConvNormAct(
            deepest_channels + len(scales) * fpn_channels,
            fpn_channels,
            kernel_size=3,
            norm_factory=norm_factory,
        )
        self.lateral_projections = nn.ModuleList(
            ConvNormAct(value, fpn_channels, kernel_size=1, norm_factory=norm_factory)
            for value in channels[:-1]
        )
        self.refinements = nn.ModuleList(
            ConvNormAct(fpn_channels, fpn_channels, kernel_size=3, norm_factory=norm_factory)
            for _ in channels[:-1]
        )
        self.fusion = ConvNormAct(
            4 * fpn_channels,
            fpn_channels,
            kernel_size=3,
            norm_factory=norm_factory,
        )
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1, bias=True)

    def _validate_features(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = tuple(features)
        if len(values) != 4:
            raise ValueError("LiteUPerDecoder requires exactly four feature maps")
        batch_size = None
        previous_spatial = None
        for index, (feature, expected_channels) in enumerate(zip(values, self.in_channels, strict=True)):
            if not isinstance(feature, Tensor) or feature.ndim != 4:
                raise ValueError(f"feature {index} must be a BCHW tensor")
            if feature.shape[1] != expected_channels:
                raise ValueError(f"feature {index} channels must be {expected_channels}, got {feature.shape[1]}")
            if batch_size is None:
                batch_size = feature.shape[0]
            elif feature.shape[0] != batch_size:
                raise ValueError("all features must have the same batch size")
            spatial = tuple(feature.shape[-2:])
            if previous_spatial is not None and not (
                spatial[0] < previous_spatial[0] and spatial[1] < previous_spatial[1]
            ):
                raise ValueError("features must be ordered from highest to lowest spatial resolution")
            previous_spatial = spatial
        return values  # type: ignore[return-value]

    def forward(
        self,
        features: Sequence[Tensor],
        *,
        output_size: tuple[int, int],
    ) -> Tensor:
        values = self._validate_features(features)
        if (
            not isinstance(output_size, tuple)
            or len(output_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in output_size)
        ):
            raise ValueError("output_size must contain two positive integers")

        deepest = values[-1]
        ppm_values = [deepest]
        for branch in self.ppm_branches:
            pooled = branch(deepest)
            ppm_values.append(
                F.interpolate(pooled, size=deepest.shape[-2:], mode="bilinear", align_corners=False)
            )
        pyramid: list[Tensor] = [deepest, deepest, deepest, self.ppm_bottleneck(torch.cat(ppm_values, dim=1))]
        for index in reversed(range(3)):
            lateral = self.lateral_projections[index](values[index])
            top_down = F.interpolate(
                pyramid[index + 1], size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid[index] = self.refinements[index](lateral + top_down)

        highest_size = pyramid[0].shape[-2:]
        fused_inputs = [pyramid[0]] + [
            F.interpolate(value, size=highest_size, mode="bilinear", align_corners=False)
            for value in pyramid[1:]
        ]
        logits = self.classifier(self.fusion(torch.cat(fused_inputs, dim=1)))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


__all__ = ["LiteUPerDecoder"]
```

Implementation requirements:

- Convert channel and scale sequences to immutable positive-integer tuples.
- Require exactly four declared input channels and non-empty pool scales.
- Build every PPM branch as `AdaptiveAvgPool2d(scale) -> Conv2d(deepest, 64, 1, bias=False) -> ReLU`; do not call `norm_factory` inside PPM branches.
- Concatenate the untouched deepest feature and all upsampled PPM outputs, then
  use the defined 3x3 `ConvNormAct` block to produce the deepest FPN feature.
- Build three lateral `1x1 ConvNormAct` blocks and three `3x3 ConvNormAct` refinement blocks.
- Traverse from deepest to highest feature using exact `size=lateral.shape[-2:]`, bilinear interpolation, and `align_corners=False`.
- Resize all four pyramid outputs to the highest selected resolution, concatenate them, fuse with `3x3 ConvNormAct(4*fpn_channels, fpn_channels)`, apply a bias-enabled `1x1` logits head, and resize logits to `output_size`.
- Validate the feature count, BCHW rank, common batch size, declared channels, strictly decreasing height and width, and two positive integer output dimensions before computation.
- Export `LiteUPerDecoder` in `__all__`.

- [ ] **Step 4: Run focused decoder tests and observe GREEN**

After a new resource preflight, rerun the Task 1 pytest command. Expected:
all tests in `test_lite_upernet.py` currently pass with a visible summary.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- standalone_nnunet2d/models/lite_upernet.py standalone_nnunet2d/tests/test_lite_upernet.py
git diff --cached --check
git commit -m "feat: add shared lightweight UPerNet decoder"
```

### Task 2: H2Former Lite-UPerNet adapter

**Files:**
- Modify: `standalone_nnunet2d/models/h2former.py`
- Create: `standalone_nnunet2d/models/h2former_lite_upernet.py`
- Modify: `standalone_nnunet2d/models/__init__.py`
- Test: `standalone_nnunet2d/tests/test_lite_upernet.py`
- Test: `standalone_nnunet2d/tests/test_h2former.py`

- [ ] **Step 1: Add failing H2 variant tests**

Append tests that build `H2FormerLiteUPerNet(in_channels=1, num_classes=2, image_size=512)`, assert a finite `[1,2,512,512]` tensor, assert selected feature shapes `[(1,64,256,256),(1,128,128,128),(1,256,64,64),(1,512,32,32)]`, and backpropagate an input containing one non-zero central pixel. Require finite non-zero gradients for:

```python
(
    "conv1.weight",
    "swin_layers.0.blocks.0.attn.qkv.weight",
    "swin_layers.3.blocks.0.attn.qkv.weight",
    "decoder.classifier.weight",
)
```

Also extend `test_h2former.py` with a baseline regression asserting that
`H2Former.forward_features(image)` returns the approved four feature shapes and
that ordinary `H2Former(image)` still returns the same full-resolution tensor
contract.

- [ ] **Step 2: Run the H2-focused tests and observe RED**

After resource preflight:

```powershell
conda run -n newconda --no-capture-output python -m pytest standalone_nnunet2d/tests/test_lite_upernet.py standalone_nnunet2d/tests/test_h2former.py -q
```

Expected: failure for missing `H2FormerLiteUPerNet` and `forward_features`.

- [ ] **Step 3: Refactor H2Former encoder extraction without changing baseline behavior**

In `H2Former`, extract input validation plus the existing encoder body into:

```python
def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return encoder features ordered from highest to lowest resolution."""
```

Return the four tensors currently accumulated in `encoder`. Keep baseline
`forward` as:

```python
def forward(self, x: Tensor) -> Tensor:
    encoder = self.forward_features(x)
    decoded = self.decode4(encoder[3], encoder[2])
    decoded = self.decode3(decoded, encoder[1])
    decoded = self.decode2(decoded, encoder[0])
    return self.decode0(decoded)
```

Do not rename baseline state-dict keys or alter its decoder modules.

- [ ] **Step 4: Implement the H2 adapter**

Create a subclass that invokes the baseline constructor, deletes only the
inherited baseline decoder modules from the new instance, installs:

```python
self.decoder = LiteUPerDecoder(
    in_channels=(64, 128, 256, 512),
    num_classes=num_classes,
    fpn_channels=64,
    pool_scales=(1, 2, 4),
    norm_factory=nn.BatchNorm2d,
)
```

Its `forward` calls `forward_features(x)` and
`self.decoder(features, output_size=tuple(x.shape[-2:]))`. Export the class from
`models/__init__.py`. Do not retain unused inherited decoder parameters.

- [ ] **Step 5: Run H2-focused tests and observe GREEN**

After a new resource preflight, rerun the Task 2 pytest command. If the full
512 backward test is resource-guarded, run the exact forward tests and decoder
gradient tests, record the H2 adapter backward node as deferred, and do not claim
full H2 gradient validation.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- standalone_nnunet2d/models/h2former.py standalone_nnunet2d/models/h2former_lite_upernet.py standalone_nnunet2d/models/__init__.py standalone_nnunet2d/tests/test_h2former.py standalone_nnunet2d/tests/test_lite_upernet.py
git diff --cached --check
git commit -m "feat: add H2Former lightweight UPerNet variant"
```

### Task 3: PlainConvUNet Lite-UPerNet adapter

**Files:**
- Modify: `standalone_nnunet2d/models/plain_conv_unet.py`
- Create: `standalone_nnunet2d/models/plain_conv_unet_lite_upernet.py`
- Modify: `standalone_nnunet2d/models/__init__.py`
- Test: `standalone_nnunet2d/tests/test_lite_upernet.py`
- Test: `standalone_nnunet2d/tests/test_model_shapes.py`

- [ ] **Step 1: Add failing Plain variant tests**

Add tests that construct the new class from `load_model_config()`, run a 512x512
input, and assert:

```python
assert logits.shape == (1, 2, 512, 512)
assert model.last_encoder_shapes == (
    (1, 32, 512, 512), (1, 64, 256, 256), (1, 128, 128, 128),
    (1, 256, 64, 64), (1, 512, 32, 32), (1, 512, 16, 16),
    (1, 512, 8, 8), (1, 512, 4, 4),
)
assert model.last_selected_feature_shapes == (
    (1, 64, 256, 256), (1, 256, 64, 64),
    (1, 512, 16, 16), (1, 512, 4, 4),
)
```

Backpropagate a finite DiceCrossEntropyLoss and require gradients on
`encoder_stages.0`, `encoder_stages.2`, `encoder_stages.4`,
`encoder_stages.6`, `encoder_stages.7`, and `decoder.classifier`. This proves
unselected intermediate stages remain on the gradient path.

- [ ] **Step 2: Run Plain-focused tests and observe RED**

After resource preflight:

```powershell
conda run -n newconda --no-capture-output python -m pytest standalone_nnunet2d/tests/test_lite_upernet.py standalone_nnunet2d/tests/test_model_shapes.py -q
```

Expected: missing Plain adapter and missing encoder helper.

- [ ] **Step 3: Extract the baseline encoder helper**

Add this method to `PlainConvUNet2D`:

```python
def forward_features(
    self, image: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return all encoder outputs in increasing downsampling order."""
```

Move only the existing encoder loop into it and keep assigning
`last_encoder_shapes`. Make baseline `forward` call `forward_features(image)`
before its unchanged seven-stage decoder loop. Preserve all baseline module
names and output ordering.

- [ ] **Step 4: Implement the Plain adapter**

Subclass `PlainConvUNet2D`, call `super().__init__(config, deep_supervision=False)`,
remove inherited transposed convolutions, decoder stages, and segmentation heads
from the new instance, then construct:

```python
norm_factory = lambda channels: nn.InstanceNorm2d(
    channels,
    eps=config.norm_eps,
    affine=config.norm_affine,
)
self.decoder = LiteUPerDecoder(
    in_channels=(
        config.features_per_stage[1], config.features_per_stage[3],
        config.features_per_stage[5], config.features_per_stage[7],
    ),
    num_classes=config.output_channels,
    fpn_channels=64,
    pool_scales=(1, 2, 4),
    norm_factory=norm_factory,
)
```

Validate that the plan has at least eight stages before indexing. In `forward`,
call `forward_features`, select indices `(1,3,5,7)`, store
`last_selected_feature_shapes`, and decode to `image.shape[-2:]`. Export the
class from `models/__init__.py`.

- [ ] **Step 5: Run Plain-focused tests and observe GREEN**

After a new resource preflight, rerun the Task 3 pytest command and retain the
visible summary.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- standalone_nnunet2d/models/plain_conv_unet.py standalone_nnunet2d/models/plain_conv_unet_lite_upernet.py standalone_nnunet2d/models/__init__.py standalone_nnunet2d/tests/test_model_shapes.py standalone_nnunet2d/tests/test_lite_upernet.py
git diff --cached --check
git commit -m "feat: add PlainConvUNet lightweight UPerNet variant"
```

### Task 4: Factory, supervision, optimizer, and resolved-config contracts

**Files:**
- Modify: `standalone_nnunet2d/models/factory.py`
- Modify: `standalone_nnunet2d/training/official_config.py`
- Modify: `standalone_nnunet2d/tests/test_model_factory.py`
- Modify: `standalone_nnunet2d/tests/test_official_trainer_config.py`

- [ ] **Step 1: Add failing contract tests**

Define and import exact names:

```python
H2FORMER_LITE_UPERNET = "h2former_lite_upernet"
PLAIN_CONV_UNET_LITE_UPERNET = "plain_conv_unet_lite_upernet"
```

Test that both appear in `MODEL_NAMES`, the parser accepts them, and both
contracts resolve only to `single_output`, `deep_supervision=False`, and
`DiceCrossEntropyLoss`. Require `deep_supervision` to raise `ValueError` for
both variants. Test that `build_model` with each new model name and both values
of `inference` returns the correct adapter class and one tensor in both modes.

Extend resolved-config coverage to require five distinct plan hashes for:
baseline Plain deep supervision, matched Plain single output, H2Former, and each
new model identity, while all non-model/non-plan-hash fields remain equal.

Update optimizer tests so AdamW is accepted for `h2former` and
`h2former_lite_upernet`, and rejected for both Plain model names.

- [ ] **Step 2: Run focused contract tests and observe RED**

After resource preflight:

```powershell
conda run -n newconda --no-capture-output python -m pytest standalone_nnunet2d/tests/test_model_factory.py standalone_nnunet2d/tests/test_official_trainer_config.py -q
```

- [ ] **Step 3: Register model contracts and builders**

In `factory.py`, add both constants to `MODEL_NAMES` and `_CONTRACTS`. The H2
variant uses `image_size=512`; the Plain variant uses `image_size=None`; both use
`SINGLE_OUTPUT`. Replace model-specific supervision conditionals with explicit
sets so the two variants reject deep supervision without changing baseline
Plain's two supported modes.

Construct the adapters in `build_model`. `inference=True` must not alter either
new contract. Export both constants.

In `official_config.py`, replace the single H2 string equality with an immutable
H2-family tuple/set containing exactly `h2former` and
`h2former_lite_upernet`. Preserve every AdamW hyperparameter and the rejection
message's H2Former meaning.

- [ ] **Step 4: Run focused contract tests and observe GREEN**

After a new resource preflight, rerun the Task 4 pytest command.

- [ ] **Step 5: Commit Task 4**

```powershell
git add -- standalone_nnunet2d/models/factory.py standalone_nnunet2d/training/official_config.py standalone_nnunet2d/tests/test_model_factory.py standalone_nnunet2d/tests/test_official_trainer_config.py
git diff --cached --check
git commit -m "feat: register lightweight UPerNet model contracts"
```

### Task 5: Checkpoint and prediction identity isolation

**Files:**
- Modify: `standalone_nnunet2d/tests/test_formal_checkpoint.py`
- Modify: `standalone_nnunet2d/tests/test_predict_command.py`

- [ ] **Step 1: Extend checkpoint identity fixtures and failing tests**

Update `_model_identity` so `image_size` is 512 only for the H2 family and None
for the Plain family. Add parameterized cross-model pairs covering:

```python
(
    ("h2former", "h2former_lite_upernet"),
    ("plain_conv_unet", "plain_conv_unet_lite_upernet"),
    ("h2former_lite_upernet", "plain_conv_unet_lite_upernet"),
)
```

All use `single_output` for the Lite variants. Preserve the existing
`fail_if_loaded` sentinel and assert it is not called.

Add prediction-loader parameterization asserting metadata reconstructs each new
model via `build_model(name, supervision_mode="single_output", inference=True)`.
Add conflicting nested/top-level metadata cases and preserve the existing
sentinel proving failure occurs before model construction.

- [ ] **Step 2: Run identity tests**

After resource preflight:

```powershell
conda run -n newconda --no-capture-output python -m pytest standalone_nnunet2d/tests/test_formal_checkpoint.py standalone_nnunet2d/tests/test_predict_command.py -q
```

Expected after Task 4: GREEN without production checkpoint-format changes. If a
test fails, fix only incorrect fixtures or the centralized factory identity
logic; do not weaken pre-load rejection.

- [ ] **Step 3: Commit Task 5**

```powershell
git add -- standalone_nnunet2d/tests/test_formal_checkpoint.py standalone_nnunet2d/tests/test_predict_command.py
git diff --cached --check
git commit -m "test: isolate lightweight UPerNet checkpoint identities"
```

### Task 6: Complexity evidence and affected validation

**Files:**
- Create: `standalone_nnunet2d/tools/profile_lite_upernet.py`
- Modify: `standalone_nnunet2d/tests/test_lite_upernet.py`

- [ ] **Step 1: Add decoder parameter-count assertions**

In `test_lite_upernet.py`, instantiate each baseline and variant and compare only
decoder parameter groups. Assert each Lite-UPerNet decoder has fewer parameters
than its corresponding baseline decoder. Compute baseline groups explicitly:

```python
def _parameters(modules) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())

h2_baseline_decoder = _parameters((baseline.decode4, baseline.decode3, baseline.decode2, baseline.decode0))
plain_baseline_decoder = _parameters((baseline.transposed_convolutions, baseline.decoder_stages, baseline.segmentation_heads))
assert sum(p.numel() for p in h2_variant.decoder.parameters()) < h2_baseline_decoder
assert sum(p.numel() for p in plain_variant.decoder.parameters()) < plain_baseline_decoder
```

Print all four exact counts for evidence.

- [ ] **Step 2: Run the complete focused architecture group**

After resource preflight:

```powershell
conda run -n newconda --no-capture-output python -m pytest standalone_nnunet2d/tests/test_lite_upernet.py standalone_nnunet2d/tests/test_h2former.py standalone_nnunet2d/tests/test_model_shapes.py standalone_nnunet2d/tests/test_model_factory.py standalone_nnunet2d/tests/test_official_trainer_config.py standalone_nnunet2d/tests/test_formal_checkpoint.py standalone_nnunet2d/tests/test_predict_command.py -q
```

Expected: visible pass summary with no unexplained failures.

- [ ] **Step 3: Add a reproducible parameter/FLOP reporting tool**

Create `profile_lite_upernet.py` with this complete behavior:

```python
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from standalone_nnunet2d.models.factory import (
    H2FORMER,
    H2FORMER_LITE_UPERNET,
    PLAIN_CONV_UNET,
    PLAIN_CONV_UNET_LITE_UPERNET,
    SINGLE_OUTPUT,
    build_model,
)


PROFILE_MODELS = (
    H2FORMER,
    H2FORMER_LITE_UPERNET,
    PLAIN_CONV_UNET,
    PLAIN_CONV_UNET_LITE_UPERNET,
)


def _parameter_count(modules: Sequence[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _decoder_parameter_count(model_name: str, model: nn.Module) -> int:
    if model_name == H2FORMER:
        return _parameter_count((model.decode4, model.decode3, model.decode2, model.decode0))
    if model_name == PLAIN_CONV_UNET:
        return _parameter_count((model.transposed_convolutions, model.decoder_stages, model.segmentation_heads))
    if model_name in {H2FORMER_LITE_UPERNET, PLAIN_CONV_UNET_LITE_UPERNET}:
        return _parameter_count((model.decoder,))
    raise ValueError(f"unsupported profile model {model_name!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile one standalone 2D decoder variant")
    parser.add_argument("--model", required=True, choices=PROFILE_MODELS)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    device = torch.device(arguments.device)
    supervision_mode = SINGLE_OUTPUT if arguments.model == PLAIN_CONV_UNET else None
    model = build_model(arguments.model, supervision_mode=supervision_mode, inference=True).to(device).eval()
    image = torch.zeros((1, 1, 512, 512), device=device)
    with torch.inference_mode():
        model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch.inference_mode(), profile(activities=activities, with_flops=True) as profiler:
        output = model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "model": arguments.model,
        "supervision_mode": SINGLE_OUTPUT,
        "device": str(device),
        "input_shape": list(image.shape),
        "output_shape": list(output.shape),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decoder_parameters": _decoder_parameter_count(arguments.model, model),
        "profiler_flops": sum(int(event.flops or 0) for event in profiler.key_averages()),
        "flop_scope": "torch.profiler supported operators only",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Measure all four fixed-input contracts**

Perform a fresh resource preflight before each command. Use the same safe device
for all four commands:

```powershell
conda run -n newconda --no-capture-output python -m standalone_nnunet2d.tools.profile_lite_upernet --model h2former --device cpu
conda run -n newconda --no-capture-output python -m standalone_nnunet2d.tools.profile_lite_upernet --model h2former_lite_upernet --device cpu
conda run -n newconda --no-capture-output python -m standalone_nnunet2d.tools.profile_lite_upernet --model plain_conv_unet --device cpu
conda run -n newconda --no-capture-output python -m standalone_nnunet2d.tools.profile_lite_upernet --model plain_conv_unet_lite_upernet --device cpu
```

Confirm each output shape is `[1,2,512,512]`, both Lite decoder parameter counts
are lower than their baselines, and report FLOPs explicitly as estimates for
PyTorch-profiler-supported operators. Do not interpret these numbers as
segmentation performance.

- [ ] **Step 5: Run the affected standalone suite**

After a new resource preflight:

```powershell
$env:TMP='D:\codex-pytest-temp\stroke-lesion-segmentation'
$env:TEMP=$env:TMP
conda run -n newconda --no-capture-output python -m pytest standalone_nnunet2d/tests -q --basetemp 'D:\codex-pytest-temp\stroke-lesion-segmentation\pytest'
```

If `D:` is unavailable, use a verified disposable temp root outside the repo and
report it. Treat this suite as heavy: if safe headroom cannot be established,
record it as deferred and retain the complete focused group as the strongest
executed evidence.

- [ ] **Step 6: Inspect final scope and whitespace**

```powershell
git status --short
git diff --check 1e6d796..HEAD
git diff --stat 1e6d796..HEAD
git diff 1e6d796..HEAD -- standalone_nnunet2d
```

Confirm no data, generated checkpoints, profiler traces, temp files,
`.project-memory`, unrelated files, or third-party sources changed.

- [ ] **Step 7: Commit complexity tooling and assertions**

Commit the exact Task 6 files:

```powershell
git add -- standalone_nnunet2d/tools/profile_lite_upernet.py standalone_nnunet2d/tests/test_lite_upernet.py
git diff --cached --check
git commit -m "test: report lightweight decoder complexity"
```

## Completion HANDOFF

End with exactly one terminal HANDOFF and no important content afterward:

```text
HANDOFF

- status: COMPLETED / BLOCKED / FAILED
- role: implementation worker
- requested model/profile: LunaMax
- observed model/profile/reasoning, when verifiable:
- repository/worktree:
- branch:
- starting commit: report the exact commit checked out when the worker starts
- ending commit, if any:
- changed files:
- implemented behavior:
- important design decisions:
- tests/validation:
- resource preflight/deferred validation, when relevant:
- failing tests or blockers:
- complexity evidence:
- git status:
- raw/user data status: not accessed / details
- project-memory changes: must be none
- scope deviations:
- unresolved issues:
- next recommended action:
```
