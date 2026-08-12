from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch import nn

from standalone_nnunet2d.data.nifti_io import NiftiVolume
import standalone_nnunet2d.engine.predictor as predictor
from standalone_nnunet2d.engine.predictor import predict_logits_2d, predict_volume


class _BinaryModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.stack((-image[:, 0], image[:, 0]), dim=1)


class _OrientationModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        signal = image[:, :1]
        return torch.cat((torch.zeros_like(signal), signal), dim=1)


class _CountingOrientationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batches: list[int] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_batches.append(int(image.shape[0]))
        signal = image[:, :1]
        return torch.cat((torch.zeros_like(signal), signal), dim=1)


class _FixedSpatialLogitModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        difference = torch.tensor(
            [[1.0, 1.0], [1.0, -5.0]], dtype=image.dtype, device=image.device
        ).view(1, 1, 2, 2)
        difference = difference.expand(image.shape[0], -1, -1, -1)
        return torch.cat((torch.zeros_like(difference), difference), dim=1)


class _TileMeanLogitModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        foreground = image.mean(dim=(1, 2, 3), keepdim=True).expand(
            image.shape[0], 1, image.shape[2], image.shape[3]
        )
        return torch.cat((torch.zeros_like(foreground), foreground), dim=1)


class _RecordingOrientationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.inputs.append(image.detach().clone())
        signal = image[:, :1]
        return torch.cat((torch.zeros_like(signal), signal), dim=1)


class _RecordingCoordinateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.inputs.append(image.detach().clone())
        coordinates = torch.arange(
            image.shape[2] * image.shape[3], dtype=image.dtype, device=image.device
        ).reshape(1, 1, image.shape[2], image.shape[3])
        coordinates = coordinates.expand(image.shape[0], -1, -1, -1)
        return torch.cat((torch.zeros_like(coordinates), coordinates), dim=1)


class _ConstantFloat32LogitModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        foreground = torch.full(
            (image.shape[0], 1, image.shape[2], image.shape[3]),
            -2.9975,
            dtype=torch.float32,
            device=image.device,
        )
        return torch.cat((torch.zeros_like(foreground), foreground), dim=1)


class _DropsBatchModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        foreground = image[:1, :1]
        return torch.cat((torch.zeros_like(foreground), foreground), dim=1)


def test_predict_logits_2d_unflips_each_mirror_before_averaging() -> None:
    image = torch.tensor([[[[2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]]])

    logits = predict_logits_2d(
        _OrientationModel(), image, torch.device("cpu"), mirror_axes=(0, 1)
    )

    expected = torch.cat((torch.zeros_like(image), image), dim=1)
    torch.testing.assert_close(logits, expected.half())


def test_predict_volume_preserves_zyx_shape_and_binary_labels() -> None:
    image = NiftiVolume(np.array([[[-1.0, 1.0]], [[2.0, -2.0]]], dtype=np.float32), (1, 1, 1), (0, 0, 0))
    prediction = predict_volume(_BinaryModel(), image, torch.device("cpu"))
    assert prediction.shape == image.array.shape
    assert prediction.dtype == np.uint8
    assert set(np.unique(prediction)) <= {0, 1}


def test_predict_volume_slice_batch_matches_serial_orientation_sensitive_masks_and_reduces_forwards() -> None:
    image = NiftiVolume(
        np.arange(30, dtype=np.float32).reshape(5, 2, 3),
        (1, 1, 1),
        (0, 0, 0),
    )
    serial_model = _CountingOrientationModel()
    batched_model = _CountingOrientationModel()

    serial = predict_volume(
        serial_model,
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 3),
    )
    batched = predict_volume(
        batched_model,
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 3),
        slice_batch_size=2,
    )

    np.testing.assert_array_equal(batched, serial)
    assert batched.dtype == np.uint8
    assert len(batched_model.forward_batches) < len(serial_model.forward_batches)
    assert batched_model.forward_batches == [2] * 8 + [1] * 4


def test_predict_volume_slice_batch_argmaxes_after_all_tta_logits_are_averaged() -> None:
    image = NiftiVolume(np.zeros((3, 2, 2), dtype=np.float32), (1, 1, 1), (0, 0, 0))

    serial = predict_volume(
        _FixedSpatialLogitModel(),
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 2),
    )
    batched = predict_volume(
        _FixedSpatialLogitModel(),
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 2),
        slice_batch_size=2,
    )

    np.testing.assert_array_equal(batched, serial)
    np.testing.assert_array_equal(batched, np.zeros((3, 2, 2), dtype=np.uint8))


@pytest.mark.parametrize("slice_batch_size", [0, -1])
def test_predict_volume_rejects_nonpositive_slice_batch_size(slice_batch_size: int) -> None:
    image = NiftiVolume(np.zeros((1, 2, 2), dtype=np.float32), (1, 1, 1), (0, 0, 0))

    with pytest.raises(ValueError, match="slice_batch_size"):
        predict_volume(_BinaryModel(), image, torch.device("cpu"), slice_batch_size=slice_batch_size)


def test_compute_gaussian_matches_official_fp16_nonzero_deterministic_map() -> None:
    first = predictor.compute_gaussian(
        (8, 10), value_scaling_factor=10.0, device=torch.device("cpu")
    )
    second = predictor.compute_gaussian(
        (8, 10), value_scaling_factor=10.0, device=torch.device("cpu")
    )

    expected = np.zeros((8, 10), dtype=np.float64)
    expected[(4, 5)] = 1.0
    from scipy.ndimage import gaussian_filter

    expected = gaussian_filter(expected, (1.0, 1.25), order=0, mode="constant", cval=0)
    expected /= np.max(expected) / 10.0

    assert first.dtype == torch.float16
    assert torch.equal(first, second)
    assert torch.all(first != 0)
    np.testing.assert_array_equal(first.cpu().numpy(), expected.astype(np.float16))
    assert float(first[4, 5]) == 10.0


def test_predict_logits_2d_uses_official_gaussian_and_fp16_tile_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    image = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    gaussian = predictor.compute_gaussian(
        (2, 2), value_scaling_factor=10.0, device=torch.device("cpu")
    )
    expected_logits = torch.zeros((1, 2, 2, 3), dtype=torch.half)
    expected_counts = torch.zeros((2, 3), dtype=torch.half)
    for x_start in (0, 1):
        tile = image[:, :, :, x_start : x_start + 2]
        foreground = tile.mean()
        prediction = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        prediction[:, 1] = foreground
        prediction *= gaussian
        expected_logits[:, :, :, x_start : x_start + 2] += prediction
        expected_counts[:, x_start : x_start + 2] += gaussian
    expected = expected_logits / expected_counts

    observed_dtypes: list[torch.dtype | None] = []
    real_zeros = torch.zeros

    def record_zeros(*args: object, **kwargs: object) -> torch.Tensor:
        observed_dtypes.append(kwargs.get("dtype"))
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(predictor.torch, "zeros", record_zeros)
    actual = predict_logits_2d(
        _TileMeanLogitModel(),
        image,
        torch.device("cpu"),
        mirror_axes=(),
        patch_size=(2, 2),
        tile_step_size=0.5,
    )

    assert actual.dtype == torch.half
    assert observed_dtypes.count(torch.half) >= 2
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_predict_logits_2d_applies_mirror_tta_in_official_order_inside_one_tile() -> None:
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    model = _RecordingOrientationModel()

    logits = predict_logits_2d(
        model,
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 2),
        tile_step_size=1.0,
    )

    expected_inputs = [
        image,
        torch.flip(image, dims=(2,)),
        torch.flip(image, dims=(3,)),
        torch.flip(image, dims=(2, 3)),
    ]
    assert len(model.inputs) == 4
    for actual, expected_input in zip(model.inputs, expected_inputs):
        torch.testing.assert_close(actual, expected_input)
    torch.testing.assert_close(logits, torch.cat((torch.zeros_like(image), image), dim=1).half())


def test_predict_volume_forwards_mirror_axes_once_per_slice_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    image = NiftiVolume(np.arange(12, dtype=np.float32).reshape(3, 2, 2), (1, 1, 1), (0, 0, 0))
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_predict_logits_2d(
        model: nn.Module,
        tensor: torch.Tensor,
        device: torch.device,
        *,
        mirror_axes: tuple[int, ...],
        patch_size: tuple[int, int],
        tile_step_size: float,
    ) -> torch.Tensor:
        del model, device, patch_size, tile_step_size
        calls.append((mirror_axes, tuple(tensor.shape)))
        return torch.zeros((tensor.shape[0], 2, tensor.shape[2], tensor.shape[3]), dtype=torch.half)

    monkeypatch.setattr(predictor, "predict_logits_2d", fake_predict_logits_2d)
    predict_volume(
        _BinaryModel(),
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        slice_batch_size=2,
    )

    assert calls == [((0, 1), (2, 1, 2, 2)), ((0, 1), (1, 1, 2, 2))]


def test_autocast_context_is_enabled_only_for_cuda_without_requiring_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_autocast(device_type: str, *, enabled: bool) -> object:
        calls.append((device_type, enabled))
        return nullcontext()

    monkeypatch.setattr(predictor.torch, "autocast", fake_autocast)
    with predictor._autocast_context(torch.device("cpu")):
        pass
    assert calls == []

    with predictor._autocast_context(torch.device("cuda")):
        pass
    assert calls == [("cuda", True)]


def test_configure_inference_backend_enables_cudnn_benchmark_for_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor.torch.backends.cudnn, "benchmark", False)

    predictor._configure_inference_backend(torch.device("cuda"))

    assert torch.backends.cudnn.benchmark is True


@pytest.mark.parametrize("initial_benchmark", [False, True])
def test_configure_inference_backend_preserves_cudnn_benchmark_for_cpu(
    monkeypatch: pytest.MonkeyPatch,
    initial_benchmark: bool,
) -> None:
    monkeypatch.setattr(
        predictor.torch.backends.cudnn,
        "benchmark",
        initial_benchmark,
    )

    predictor._configure_inference_backend(torch.device("cpu"))

    assert torch.backends.cudnn.benchmark is initial_benchmark


def test_cuda_backend_configuration_preserves_all_other_global_backend_state() -> None:
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic
    original_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    original_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_matmul_precision = torch.get_float32_matmul_precision()
    original_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    original_deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    before = {
        "deterministic": original_deterministic,
        "cudnn_allow_tf32": original_cudnn_allow_tf32,
        "matmul_allow_tf32": original_matmul_allow_tf32,
        "matmul_precision": original_matmul_precision,
        "deterministic_algorithms": original_deterministic_algorithms,
    }

    try:
        torch.backends.cudnn.benchmark = False
        predictor._configure_inference_backend(torch.device("cuda"))
        after = {
            "deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }

        assert torch.backends.cudnn.benchmark is True
        assert after == before
    finally:
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.allow_tf32 = original_cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_allow_tf32
        torch.set_float32_matmul_precision(original_matmul_precision)
        torch.use_deterministic_algorithms(
            original_deterministic_algorithms,
            warn_only=original_deterministic_warn_only,
        )


def test_predict_logits_2d_configures_inference_backend_before_model_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def record_backend_configuration(device: torch.device) -> None:
        del device
        events.append("backend")

    class _RecordingModel(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            events.append("forward")
            return torch.cat((torch.zeros_like(image), image), dim=1)

    monkeypatch.setattr(
        predictor,
        "_configure_inference_backend",
        record_backend_configuration,
        raising=False,
    )
    predict_logits_2d(
        _RecordingModel(),
        torch.zeros((1, 1, 2, 2)),
        torch.device("cpu"),
        mirror_axes=(),
        patch_size=(2, 2),
        tile_step_size=1.0,
    )

    assert events == ["backend", "forward"]


def test_predict_logits_2d_cuda_configures_backend_before_transfer_or_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BackendConfigured(RuntimeError):
        pass

    class _ForwardMustNotRun(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_called = False

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            self.forward_called = True
            return torch.cat((torch.zeros_like(image), image), dim=1)

    def stop_after_backend_configuration(device: torch.device) -> None:
        assert device == torch.device("cuda:0")
        raise _BackendConfigured("backend configured before CUDA transfer")

    model = _ForwardMustNotRun()
    monkeypatch.setattr(
        predictor,
        "_configure_inference_backend",
        stop_after_backend_configuration,
    )

    with pytest.raises(_BackendConfigured, match="before CUDA transfer"):
        predict_logits_2d(
            model,
            torch.zeros((1, 1, 2, 2)),
            "cuda:0",
            mirror_axes=(),
            patch_size=(2, 2),
            tile_step_size=1.0,
        )

    assert model.forward_called is False


def test_predict_logits_2d_centers_constant_padding_and_restores_exact_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    model = _RecordingCoordinateModel()

    def unit_gaussian(tile_size: tuple[int, ...], **kwargs: object) -> torch.Tensor:
        return torch.ones(
            tile_size,
            dtype=kwargs["dtype"],
            device=kwargs["device"],
        )

    monkeypatch.setattr(predictor, "compute_gaussian", unit_gaussian)
    logits = predict_logits_2d(
        model,
        image,
        torch.device("cpu"),
        mirror_axes=(),
        patch_size=(5, 8),
        tile_step_size=0.5,
    )

    expected_padded = torch.zeros((1, 1, 5, 8))
    expected_padded[:, :, 1:3, 2:5] = image
    torch.testing.assert_close(model.inputs[0], expected_padded)
    expected_foreground = torch.arange(40, dtype=torch.half).reshape(5, 8)[1:3, 2:5]
    torch.testing.assert_close(logits[0, 1], expected_foreground)
    assert tuple(logits.shape) == (1, 2, 2, 3)


@pytest.mark.parametrize(
    ("size", "patch", "step", "expected"),
    [
        (1000, 512, 0.5, (0, 244, 488)),
        (500, 512, 0.5, (0,)),
    ],
)
def test_tile_starts_matches_official_sliding_window_steps(
    size: int,
    patch: int,
    step: float,
    expected: tuple[int, ...],
) -> None:
    assert predictor._tile_starts(size, patch, step) == expected


def test_compute_gaussian_default_is_unit_scaled_cached_and_cpu_safe() -> None:
    predictor.compute_gaussian.cache_clear()

    first = predictor.compute_gaussian((8, 10))
    second = predictor.compute_gaussian((8, 10))

    assert first is second
    assert first.device.type == "cpu"
    assert float(first.max()) == 1.0
    cache_info = predictor.compute_gaussian.cache_info()
    assert cache_info.maxsize == 2
    assert cache_info.hits == 1


def test_cpu_multiplies_float32_prediction_by_half_gaussian_before_half_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gaussian_value = torch.tensor(0.1, dtype=torch.half)

    def constant_gaussian(tile_size: tuple[int, ...], **kwargs: object) -> torch.Tensor:
        return torch.full(
            tile_size,
            float(gaussian_value),
            dtype=kwargs["dtype"],
            device=kwargs["device"],
        )

    monkeypatch.setattr(predictor, "compute_gaussian", constant_gaussian)
    logits = predict_logits_2d(
        _ConstantFloat32LogitModel(),
        torch.zeros((1, 1, 1, 1)),
        torch.device("cpu"),
        mirror_axes=(),
        patch_size=(1, 1),
        tile_step_size=1.0,
    )

    prediction = torch.tensor(-2.9975, dtype=torch.float32)
    expected = (prediction * gaussian_value).half() / gaussian_value
    premature_half = prediction.half() * gaussian_value / gaussian_value
    assert expected != premature_half
    assert logits[0, 1, 0, 0] == expected


def test_predict_logits_2d_rejects_model_batch_size_changes() -> None:
    with pytest.raises(ValueError, match="batch size"):
        predict_logits_2d(
            _DropsBatchModel(),
            torch.zeros((2, 1, 2, 2)),
            torch.device("cpu"),
            mirror_axes=(),
            patch_size=(2, 2),
            tile_step_size=1.0,
        )
