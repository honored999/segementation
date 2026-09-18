from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from standalone_nnunet2d.data.data_source import PreparedCase
from standalone_nnunet2d.engine import formal_validation
from standalone_nnunet2d import formal_train
from standalone_nnunet2d.training.formal_checkpoint import FormalTrainerState


class _RecordingSource:
    case_ids = ("case-b", "case-a")

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def prepare_case(self, case_id: str) -> PreparedCase:
        label = np.zeros((2, 2, 2), dtype=np.uint8)
        if case_id == "case-b":
            label[0, 0, 0] = 1
        else:
            label[:, :, :] = 1
        image = np.full(label.shape, 0.0 if case_id == "case-b" else 1.0, dtype=np.float32)

        class _Case(PreparedCase):
            def image_slice(inner_self, z_index: int) -> np.ndarray:
                self.calls.append((case_id, z_index))
                return super(_Case, inner_self).image_slice(z_index)

        return _Case(label=label, shape=label.shape, _image_volume=image)


def test_selection_is_deterministic_case_macro_and_covers_each_slice_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _RecordingSource()
    monkeypatch.setattr(formal_validation, "make_formal_case_source", lambda *args, **kwargs: source)
    predictions = {
        "case-b": np.zeros((2, 2, 2), dtype=np.uint8),
        "case-a": np.ones((2, 2, 2), dtype=np.uint8),
    }
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def fake_predict(model, image, device, **kwargs):
        calls.append((str(image.array[0, 0, 0]), image.array.shape, kwargs["mirror_axes"]))
        return predictions["case-b" if image.array[0, 0, 0] == 0 else "case-a"]

    monkeypatch.setattr(formal_validation, "predict_volume", fake_predict)
    result = formal_validation.select_fold(
        object(),
        source_root=None,
        data_source="nnunet_preprocessed_b2nd",
        fold=0,
        device=torch.device("cpu"),
    )
    assert result["case_ids"] == ("case-b", "case-a")
    assert result["selection_dice"] == pytest.approx((0.0 + 1.0) / 2.0)
    assert source.calls == [("case-b", 0), ("case-b", 1), ("case-a", 0), ("case-a", 1)]
    assert calls == [("0.0", (2, 2, 2), ()), ("1.0", (2, 2, 2), ())]


def test_selection_repeated_runs_have_identical_cases_metrics_and_slice_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_sources = [_RecordingSource(), _RecordingSource()]
    remaining_sources = list(all_sources)
    monkeypatch.setattr(
        formal_validation,
        "make_formal_case_source",
        lambda *args, **kwargs: remaining_sources.pop(0),
    )
    predictions = {
        "case-b": np.zeros((2, 2, 2), dtype=np.uint8),
        "case-a": np.ones((2, 2, 2), dtype=np.uint8),
    }
    prediction_calls: list[tuple[tuple[int, ...], bool]] = []

    def fake_predict(model, image, device, **kwargs):
        del model, device
        prediction_calls.append((kwargs["mirror_axes"], kwargs["normalise_inputs"]))
        return predictions["case-b" if image.array[0, 0, 0] == 0 else "case-a"]

    monkeypatch.setattr(formal_validation, "predict_volume", fake_predict)
    results = [
        formal_validation.select_fold(
            object(),
            source_root=None,
            data_source="nnunet_preprocessed_b2nd",
            fold=0,
            device=torch.device("cpu"),
        )
        for _ in range(2)
    ]

    assert results[0]["case_ids"] == results[1]["case_ids"] == ("case-b", "case-a")
    assert results[0]["metric_per_case"] == results[1]["metric_per_case"]
    assert results[0]["selection_dice"] == results[1]["selection_dice"] == pytest.approx(0.5)
    expected_slice_access = [("case-b", 0), ("case-b", 1), ("case-a", 0), ("case-a", 1)]
    assert [source.calls for source in all_sources] == [expected_slice_access, expected_slice_access]
    assert prediction_calls == [((), False), ((), False), ((), False), ((), False)]


def test_selection_fails_without_scoring_a_partial_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _RecordingSource()
    monkeypatch.setattr(formal_validation, "make_formal_case_source", lambda *args, **kwargs: source)

    def fail_on_second_case(model, image, device, **kwargs):
        if image.array[0, 0, 0] != 0:
            raise RuntimeError("case failure")
        return np.zeros_like(image.array, dtype=np.uint8)

    monkeypatch.setattr(formal_validation, "predict_volume", fail_on_second_case)
    with pytest.raises(RuntimeError, match="case-a"):
        formal_validation.select_fold(
            object(), source_root=None, data_source="raw_nifti_online", fold=0, device=torch.device("cpu")
        )


@pytest.mark.parametrize("initial_benchmark", [False, True])
def test_selection_restores_cudnn_benchmark_after_success(
    monkeypatch: pytest.MonkeyPatch,
    initial_benchmark: bool,
) -> None:
    source = _RecordingSource()
    monkeypatch.setattr(formal_validation, "make_formal_case_source", lambda *args, **kwargs: source)

    def fake_predict(model, image, device, **kwargs):
        del model, image, device, kwargs
        torch.backends.cudnn.benchmark = True
        return np.zeros((2, 2, 2), dtype=np.uint8)

    monkeypatch.setattr(formal_validation, "predict_volume", fake_predict)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", initial_benchmark)

    formal_validation.select_fold(
        object(), source_root=None, data_source="raw_nifti_online", fold=0, device=torch.device("cpu")
    )

    assert torch.backends.cudnn.benchmark is initial_benchmark


def test_selection_restores_cudnn_benchmark_without_swallowing_case_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _RecordingSource()
    monkeypatch.setattr(formal_validation, "make_formal_case_source", lambda *args, **kwargs: source)
    case_failure = RuntimeError("case failure")

    def fail_on_second_case(model, image, device, **kwargs):
        del model, device, kwargs
        torch.backends.cudnn.benchmark = True
        if image.array[0, 0, 0] != 0:
            raise case_failure
        return np.zeros_like(image.array, dtype=np.uint8)

    monkeypatch.setattr(formal_validation, "predict_volume", fail_on_second_case)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)

    with pytest.raises(RuntimeError, match="case-a") as raised:
        formal_validation.select_fold(
            object(), source_root=None, data_source="raw_nifti_online", fold=0, device=torch.device("cpu")
        )

    assert raised.value.__cause__ is case_failure
    assert torch.backends.cudnn.benchmark is False


def test_random_slice_dice_cannot_update_best_but_strict_selection_can() -> None:
    state = FormalTrainerState(9, 9, .99, 0)
    state, save_best, stop = formal_train.update_selection_state(state, .2, completed_epoch=10)
    assert state.best_validation_dice == pytest.approx(.99)
    assert save_best is True
    assert stop is False

    state = replace(state, best_selection_dice=.2, best_selection_epoch=10)
    state, save_best, _ = formal_train.update_selection_state(state, .3, completed_epoch=20)
    assert state.best_selection_dice == pytest.approx(.3)
    assert state.best_selection_epoch == 20
    assert save_best is True


def test_selection_min_delta_only_controls_patience_and_start_gate() -> None:
    state = FormalTrainerState(99, 0, -1.0, 0, -1.0, 0, .5, 0)
    state, _, stop = formal_train.update_selection_state(state, .5005, completed_epoch=100)
    assert state.early_stop_reference_dice == pytest.approx(.5)
    assert state.checks_without_improvement == 1
    assert stop is False

    state = replace(state, checks_without_improvement=9)
    state, _, stop = formal_train.update_selection_state(state, .5005, completed_epoch=110)
    assert state.checks_without_improvement == 10
    assert stop is True

    state = replace(state, checks_without_improvement=10)
    state, _, stop = formal_train.update_selection_state(state, .501, completed_epoch=120)
    assert state.early_stop_reference_dice == pytest.approx(.501)
    assert state.checks_without_improvement == 0
    assert stop is False

    pre_start = FormalTrainerState(0, 0, -1.0, 0, -1.0, 0, .5, 0)
    pre_start, _, stop = formal_train.update_selection_state(pre_start, .1, completed_epoch=90)
    assert pre_start.checks_without_improvement == 0
    assert stop is False
