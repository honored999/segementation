from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.data.symmetry_alignment import (
    AlignmentEstimate,
    AlignmentResult,
    QuasiSymmetricAlignmentError,
    align_case as real_align_case,
    align_case_result as real_align_case_result,
    build_alignment_qc as real_build_alignment_qc,
)
from standalone_nnunet2d.tools import quasi_symmetric_qc as qc


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
TARGET_SPACING = (0.5, 0.625)


def _image_volume(
    *,
    z: int = 8,
    direction: tuple[float, ...] = IDENTITY_DIRECTION,
    circular: bool = False,
    empty: bool = False,
) -> NiftiVolume:
    yy, xx = np.indices((33, 33), dtype=np.float64)
    if empty:
        plane = np.zeros((33, 33), dtype=np.float32)
    elif circular:
        plane = (((xx - 16.0) ** 2 + (yy - 16.0) ** 2) <= 8.0**2).astype(np.float32)
    else:
        plane = (
            (((xx - 19.0) / 10.0) ** 2 + ((yy - 14.0) / 5.0) ** 2) <= 1.0
        ).astype(np.float32)
    array = np.repeat(plane[None, ...], z, axis=0)
    return NiftiVolume(
        array=array,
        spacing_xyz=(1.0, 1.25, 4.0),
        origin_xyz=(10.0, 20.0, 30.0),
        direction=direction,
    )


def _label_volume(*, z: int = 8, marker: tuple[int, int] = (10, 24)) -> NiftiVolume:
    array = np.zeros((z, 33, 33), dtype=np.int16)
    array[:, marker[0], marker[1]] = 1
    image = _image_volume(z=z)
    return NiftiVolume(array, image.spacing_xyz, image.origin_xyz, image.direction)


def _estimate(*, foreground_voxel_count: int = 100) -> AlignmentEstimate:
    return AlignmentEstimate(
        center_xyz=(26.0, 38.0, 44.0),
        reference_center_xyz=(26.0, 40.0, 44.0),
        rotation_angle_radians=0.125,
        output_to_input_matrix=IDENTITY_DIRECTION,
        output_to_input_translation_xyz=(1.0, 2.0, 3.0),
        foreground_voxel_count=foreground_voxel_count,
    )


def _result(image: NiftiVolume, label: NiftiVolume | None = None) -> AlignmentResult:
    return AlignmentResult(image=image, label=label, estimate=_estimate())


def _args(image: Path, output_dir: Path, label: Path | None = None) -> list[str]:
    arguments = ["--image", str(image), "--output-dir", str(output_dir)]
    if label is not None:
        arguments.extend(("--label", str(label)))
    arguments.extend(("--target-spacing-xy", str(TARGET_SPACING[0]), str(TARGET_SPACING[1])))
    return arguments


def _summary(output_dir: Path) -> dict[str, object]:
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_output_directory_is_rejected_before_input_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "image.nii.gz"
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()
    reads: list[Path] = []

    monkeypatch.setattr(qc, "read_nifti", lambda path: reads.append(path) or _image_volume())

    with pytest.raises(FileExistsError, match="output directory already exists"):
        qc.main(_args(image_path, output_dir))

    assert reads == []
    assert list(output_dir.iterdir()) == []


def test_cli_orders_reads_geometry_check_resampling_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    output_dir = tmp_path / "qc"
    image = _image_volume()
    label = _label_volume()
    events: list[str] = []

    def fake_read(path: Path) -> NiftiVolume:
        events.append("read:image" if path == image_path else "read:label")
        return image if path == image_path else label

    def fake_resample(volume: NiftiVolume, target: tuple[float, float], *, is_segmentation: bool) -> NiftiVolume:
        events.append("resample:label" if is_segmentation else "resample:image")
        assert target == TARGET_SPACING
        return volume

    def fake_align(image_arg: NiftiVolume, label_arg: NiftiVolume) -> tuple[NiftiVolume, NiftiVolume, AlignmentEstimate]:
        events.append("align")
        assert image_arg is image
        assert label_arg is label
        return image, label, _estimate()

    monkeypatch.setattr(qc, "read_nifti", fake_read)
    monkeypatch.setattr(qc, "resample_inplane", fake_resample)
    monkeypatch.setattr(qc, "align_case", fake_align)

    assert qc.main(_args(image_path, output_dir, label_path)) == 0
    assert events == [
        "read:image",
        "read:label",
        "resample:image",
        "resample:label",
        "align",
    ]


def test_label_cannot_change_estimate_transform_or_slice_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.nii.gz"
    label_a_path = tmp_path / "label-a.nii.gz"
    label_b_path = tmp_path / "label-b.nii.gz"
    image = _image_volume()
    label_a = _label_volume(marker=(8, 8))
    label_b = _label_volume(marker=(27, 27))
    captured: list[tuple[np.ndarray, AlignmentEstimate]] = []

    def fake_read(path: Path) -> NiftiVolume:
        if path == image_path:
            return image
        return label_a if path == label_a_path else label_b

    def wrapped_align(image_arg: NiftiVolume, label_arg: NiftiVolume) -> tuple[NiftiVolume, NiftiVolume, AlignmentEstimate]:
        result = real_align_case(image_arg, label_arg)
        captured.append((result[0].array.copy(), result[2]))
        return result

    monkeypatch.setattr(qc, "read_nifti", fake_read)
    monkeypatch.setattr(qc, "align_case", wrapped_align)

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    assert qc.main(_args(image_path, first_output, label_a_path)) == 0
    assert qc.main(_args(image_path, second_output, label_b_path)) == 0

    first = _summary(first_output)
    second = _summary(second_output)
    for field in (
        "center_xyz",
        "reference_center_xyz",
        "center_to_reference_xyz",
        "output_to_input_translation_xyz",
        "rotation_angle_radians",
        "foreground_voxel_count",
        "chosen_slice_indices",
    ):
        assert first[field] == second[field]
    np.testing.assert_array_equal(captured[0][0], captured[1][0])
    assert captured[0][1] == captured[1][1]


def test_success_writes_summary_three_panels_and_preserves_input_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _image_volume()
    label = _label_volume()
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    write_nifti(image_path, image)
    write_nifti(label_path, label)
    before_image = _sha256(image_path)
    before_label = _sha256(label_path)
    output_dir = tmp_path / "qc"
    original_arguments: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

    def record_qc(original: NiftiVolume, *, result: AlignmentResult, slice_index: int | None = None):
        original_arguments.append((tuple(original.array.shape), original.spacing_xyz))
        return real_build_alignment_qc(original, result=result, slice_index=slice_index)

    monkeypatch.setattr(qc, "build_alignment_qc", record_qc)

    assert qc.main(_args(image_path, output_dir, label_path)) == 0

    summary = _summary(output_dir)
    assert summary["status"] == "aligned"
    assert sorted(path.name for path in output_dir.glob("*.png")) == [
        "slice_00_z002.png",
        "slice_01_z004.png",
        "slice_02_z005.png",
    ]
    assert len(original_arguments) == 3
    assert all(shape == (8, 66, 66) for shape, _ in original_arguments)
    assert all(spacing == TARGET_SPACING + (4.0,) for _, spacing in original_arguments)

    for geometry_name in ("original_geometry", "resampled_geometry", "aligned_geometry"):
        assert set(summary[geometry_name]) == {"shape_zyx", "spacing_xyz", "origin_xyz", "direction"}
    assert summary["original_geometry"]["shape_zyx"] == [8, 33, 33]
    assert summary["original_geometry"]["spacing_xyz"] == [1.0, 1.25, 4.0]
    assert summary["resampled_geometry"]["shape_zyx"] == [8, 66, 66]
    assert summary["resampled_geometry"]["spacing_xyz"] == [0.5, 0.625, 4.0]
    assert summary["aligned_geometry"] == summary["resampled_geometry"]
    assert summary["label_geometry"]["original"]["shape_zyx"] == [8, 33, 33]
    assert summary["label_geometry"]["resampled"] == summary["resampled_geometry"]
    assert summary["label_geometry"]["aligned"] == summary["aligned_geometry"]
    assert isinstance(summary["orientation_handling"], str)
    for field in (
        "center_xyz",
        "reference_center_xyz",
        "center_to_reference_xyz",
        "output_to_input_translation_xyz",
        "rotation_angle_radians",
        "foreground_voxel_count",
        "chosen_slice_indices",
    ):
        assert field in summary
    assert summary["chosen_slice_indices"] == [2, 4, 5]
    assert summary["foreground_voxel_count"] > 0

    assert _sha256(image_path) == before_image
    assert _sha256(label_path) == before_label


@pytest.mark.parametrize("z", [2, 4])
def test_cli_fails_without_three_distinct_deterministic_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, z: int
) -> None:
    image_path = tmp_path / "image.nii.gz"
    output_dir = tmp_path / f"qc-{z}"
    image = _image_volume(z=z)
    monkeypatch.setattr(qc, "read_nifti", lambda path: image)
    monkeypatch.setattr(qc, "align_case_result", lambda image_arg: _result(image_arg))

    assert qc.main(_args(image_path, output_dir)) != 0
    summary = _summary(output_dir)
    assert summary["status"] == "failed"
    assert "three distinct" in summary["error"]
    assert list(output_dir.glob("*.png")) == []


def test_slice_selection_is_deterministic_valid_and_unique() -> None:
    assert qc.choose_slice_indices(8) == (2, 4, 5)
    assert len(set(qc.choose_slice_indices(8))) == 3
    assert all(0 <= index < 8 for index in qc.choose_slice_indices(8))


@pytest.mark.parametrize(
    ("image", "error_match"),
    [
        (_image_volume(direction=(1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), "direction"),
        (_image_volume(empty=True), "foreground"),
        (_image_volume(circular=True), "principal axis"),
    ],
)
def test_alignment_failures_write_failed_summary_without_pngs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: NiftiVolume,
    error_match: str,
) -> None:
    image_path = tmp_path / "image.nii.gz"
    output_dir = tmp_path / f"qc-{error_match.replace(' ', '-') }"
    monkeypatch.setattr(qc, "read_nifti", lambda path: image)
    if error_match == "principal axis":
        monkeypatch.setattr(
            qc,
            "align_case_result",
            lambda image_arg: (_ for _ in ()).throw(
                QuasiSymmetricAlignmentError(
                    "degenerate principal axis for quasi-symmetric alignment"
                )
            ),
        )

    assert qc.main(_args(image_path, output_dir)) != 0
    summary = _summary(output_dir)
    assert summary["status"] == "failed"
    assert error_match in summary["error"]
    assert list(output_dir.glob("*.png")) == []
