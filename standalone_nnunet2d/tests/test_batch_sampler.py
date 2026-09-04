from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from standalone_nnunet2d.training.batch_sampler import (
    FormalBatchSampler,
    PatchRequest,
    choose_patch_location,
)


def test_patch_request_is_immutable() -> None:
    request = PatchRequest(case_id="a", force_foreground=True, z_index=2, center_yx=(3, 4))

    with pytest.raises(FrozenInstanceError):
        request.z_index = 0  # type: ignore[misc]


def test_batch_request_has_oracle_comparable_foreground_slots() -> None:
    requests = FormalBatchSampler(
        case_ids=("a", "b", "c", "d"),
        batch_size=4,
        foreground_slots=(2, 3),
        seed=7,
    ).batch(0)

    assert [request.force_foreground for request in requests] == [False, False, True, True]
    assert [request.case_id for request in requests] == ["a", "b", "c", "d"]


def test_batch_requests_are_reproducible_without_stateful_rng() -> None:
    sampler = FormalBatchSampler(case_ids=("a", "b"), batch_size=3, foreground_slots=(1,), seed=17)

    first = sampler.batch(4)
    sampler.batch(0)
    second = sampler.batch(4)

    assert first == second


def test_no_foreground_location_sampling_falls_back_to_valid_random_location() -> None:
    labels = np.zeros((3, 4, 5), dtype=np.int16)

    z_index, center_yx = choose_patch_location(labels, np.random.default_rng(2), force_foreground=True)

    assert 0 <= z_index < 3
    assert 0 <= center_yx[0] < 4
    assert 0 <= center_yx[1] < 5
