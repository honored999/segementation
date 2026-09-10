"""Build an offline Dataset505 DWI/ADC/Fusion raw dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from standalone_nnunet2d.data.fusion import build_dwi_adc_fusion_channel
from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti


SOURCE_DATASET_NAME = "Dataset502_StrokeLesion_DWI_ADC"
EXPECTED_CHANNELS = {"0": "DWI", "1": "ADC"}
EXPECTED_NUM_TRAINING = 95


def _load_source_metadata(source_root: Path) -> tuple[dict[str, object], int]:
    dataset_path = source_root / "dataset.json"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"source dataset.json does not exist: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or metadata.get("channel_names") != EXPECTED_CHANNELS:
        raise ValueError(
            "source dataset.json channel_names must be exactly "
            f"{EXPECTED_CHANNELS}"
        )
    num_training = metadata.get("numTraining")
    if isinstance(num_training, bool) or not isinstance(num_training, int) or num_training < 1:
        raise ValueError("source dataset.json numTraining must be a positive integer")
    if num_training != EXPECTED_NUM_TRAINING:
        raise ValueError(
            f"source dataset.json numTraining must be {EXPECTED_NUM_TRAINING}, "
            f"got {num_training}"
        )
    return metadata, num_training


def _source_cases(source_root: Path, num_training: int) -> tuple[str, ...]:
    images = source_root / "imagesTr"
    labels = source_root / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError("source root must contain imagesTr and labelsTr directories")
    dwi_suffix = "_0000.nii.gz"
    adc_suffix = "_0001.nii.gz"
    nifti_suffix = ".nii.gz"
    dwi_ids = {path.name[: -len(dwi_suffix)] for path in images.glob(f"*{dwi_suffix}")}
    adc_ids = {path.name[: -len(adc_suffix)] for path in images.glob(f"*{adc_suffix}")}
    label_ids = {path.name[: -len(nifti_suffix)] for path in labels.glob(f"*{nifti_suffix}")}
    if dwi_ids != adc_ids or dwi_ids != label_ids:
        raise ValueError(
            "source DWI, ADC, and label case ID sets must match exactly"
        )
    case_ids = tuple(sorted(dwi_ids))
    if len(case_ids) != num_training:
        raise ValueError(
            f"source case count {len(case_ids)} does not match numTraining={num_training}"
        )
    for case_id in case_ids:
        required = (
            images / f"{case_id}_0000.nii.gz",
            images / f"{case_id}_0001.nii.gz",
            labels / f"{case_id}.nii.gz",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"case {case_id} is missing required source files: {missing}"
            )
    return case_ids


def _geometry_mismatch(dwi: NiftiVolume, adc: NiftiVolume) -> str | None:
    if dwi.array.shape != adc.array.shape:
        return f"shape {dwi.array.shape} != {adc.array.shape}"
    if dwi.spacing_xyz != adc.spacing_xyz:
        return f"spacing {dwi.spacing_xyz} != {adc.spacing_xyz}"
    if dwi.origin_xyz != adc.origin_xyz:
        return f"origin {dwi.origin_xyz} != {adc.origin_xyz}"
    if dwi.direction != adc.direction:
        return "direction differs"
    return None


def _dataset_json(num_training: int) -> dict[str, object]:
    return {
        "channel_names": {"0": "DWI", "1": "ADC", "2": "noNorm"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def _provenance() -> dict[str, object]:
    return {
        "source_dataset": SOURCE_DATASET_NAME,
        "channels": {
            "0": {"semantic_name": "DWI"},
            "1": {"semantic_name": "ADC"},
            "2": {
                "semantic_name": "DWI_ADC_FUSION",
                "nnunet_channel_name": "noNorm",
                "recipe": "DWI_01 + (1 - ADC_01)",
                "valid_region": "finite and nonzero intersection",
                "output_range": [0, 2],
                "normalization_during_nnunet_preprocessing": "none",
            },
        },
    }


def build_dataset(source_root: Path, output_root: Path) -> None:
    """Validate Dataset502 and atomically publish a separate Dataset505."""
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"output root must not already exist: {output}")
    if source == output or source in output.parents:
        raise ValueError("output root must not be inside the read-only source dataset")
    _, num_training = _load_source_metadata(source)
    case_ids = _source_cases(source, num_training)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}_building_", dir=output.parent)
    )
    try:
        images = temporary / "imagesTr"
        labels = temporary / "labelsTr"
        images.mkdir()
        labels.mkdir()
        for case_id in case_ids:
            source_dwi = source / "imagesTr" / f"{case_id}_0000.nii.gz"
            source_adc = source / "imagesTr" / f"{case_id}_0001.nii.gz"
            source_label = source / "labelsTr" / f"{case_id}.nii.gz"
            dwi = read_nifti(source_dwi)
            adc = read_nifti(source_adc)
            mismatch = _geometry_mismatch(dwi, adc)
            if mismatch is not None:
                raise ValueError(f"case {case_id} DWI/ADC geometry mismatch: {mismatch}")

            shutil.copy2(source_dwi, images / f"{case_id}_0000.nii.gz")
            shutil.copy2(source_adc, images / f"{case_id}_0001.nii.gz")
            shutil.copy2(source_label, labels / f"{case_id}.nii.gz")
            fusion = build_dwi_adc_fusion_channel(dwi.array, adc.array)
            write_nifti(
                images / f"{case_id}_0002.nii.gz",
                NiftiVolume(
                    np.asarray(fusion, dtype=np.float32),
                    dwi.spacing_xyz,
                    dwi.origin_xyz,
                    dwi.direction,
                ),
            )

        (temporary / "dataset.json").write_text(
            json.dumps(_dataset_json(num_training), indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "provenance.json").write_text(
            json.dumps(_provenance(), indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    build_dataset(arguments.source_root, arguments.output_root)
    print(json.dumps({"output_root": str(arguments.output_root.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
