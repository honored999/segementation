import json

import numpy as np
import pytest

from experiments.genseg3mm.targets import (
    TargetDataset, baseline_training_kwargs, expand_keep_mask_3mm, map_keep_mask_to_preprocessed,
)
from experiments.genseg30.contract import sha256


def test_3mm_expansion_respects_anisotropic_spacing_without_z_growth():
    brain = np.zeros((3, 13, 13), dtype=bool)
    brain[0, 0, 0] = True
    brain[1, 6, 6] = True
    brain[2, 12, 12] = True
    keep = expand_keep_mask_3mm(brain, spacing_xyz=(2.0, 1.0, 9.0))
    assert keep[1, 9, 6]  # y+3 voxels at 1 mm
    assert not keep[1, 10, 6]
    assert keep[1, 6, 7]  # x+1 voxel at 2 mm
    assert not keep[1, 6, 8]
    assert not keep[0, 6, 6] and not keep[2, 6, 6]


def test_empty_native_plane_retains_everything_and_whole_empty_volume_fails():
    brain = np.zeros((2, 5, 7), dtype=bool)
    brain[0, 2, 2] = True
    keep = expand_keep_mask_3mm(brain, spacing_xyz=(1.0, 1.0, 5.0))
    assert keep[1].all() and keep[0].any() and not keep[0].all()
    with pytest.raises(ValueError, match="empty"):
        expand_keep_mask_3mm(np.zeros((2, 5, 7), dtype=bool), spacing_xyz=(1, 1, 5))


def test_map_mask_uses_posttranspose_crop_and_real_segmentation_resampler():
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

    class Plans:
        transpose_forward = [0, 2, 1]

    class Configuration:
        spacing = [1.0, 1.0]

        @staticmethod
        def resampling_fn_seg(data, new_shape, current_spacing, new_spacing):
            return resample_data_or_seg_to_shape(
                data, new_shape, current_spacing, new_spacing, is_seg=True, order=1, order_z=0
            )

    raw = np.zeros((4, 6, 8), dtype=bool)
    raw[1:3, 1:5, 2:7] = True
    properties = {
        "spacing": [3.0, 1.0, 1.0],
        "shape_before_cropping": [4, 8, 6],
        "bbox_used_for_cropping": [[0, 4], [1, 7], [1, 5]],
        "shape_after_cropping_and_before_resampling": [4, 6, 4],
    }
    mask = map_keep_mask_to_preprocessed(raw, Plans(), Configuration(), properties,
                                         expected_shape=(1, 4, 6, 4))
    assert mask.shape == (4, 6, 4) and mask.dtype == np.bool_
    assert not mask[0].any() and mask[1:3].any() and not mask[3].any()
    assert not mask[:, 0].any() and mask[:, -1].any()


def _fake_source(root):
    class Source:
        train, val = ["trainA", "trainB"], ["heldoutA"]

        def __init__(self):
            self.base = root / "source" / "Dataset501_StrokeLesion"
            self.folder = self.base / "nnUNetPlans_2d"
            self.folder.mkdir(parents=True)
            self.raw_root = root / "raw"
            raw_folder = self.raw_root / self.base.name / "imagesTr"
            raw_folder.mkdir(parents=True)
            self.dataset_path, self.plans_path = self.base / "dataset.json", self.base / "nnUNetPlans.json"
            self.splits = self.base / "splits_final.json"
            for path in (self.dataset_path, self.plans_path, self.splits):
                path.write_text("fixture", encoding="utf-8")
            self.protected, self.arrays, self.source_files, self.raw_files = [], {}, {}, {}
            for patient in self.train:
                source_file = self.folder / (patient + ".fixture")
                source_file.write_bytes(patient.encode())
                raw_file = raw_folder / (patient + "_0000.nii.gz")
                raw_file.write_bytes((patient + "-raw").encode())
                self.source_files[patient], self.raw_files[patient] = source_file, raw_file
                self.arrays[patient] = (
                    np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4),
                    np.zeros((1, 2, 3, 4), dtype=np.int16), {"patient": patient},
                )
                self.arrays[patient][1][0, 0, 0, 0] = -1
            self._inventory = {patient: self.inventory(patient) for patient in self.train}
            self.identity = {
                "source_folder": str(self.folder), "dataset_base": str(self.base),
                "dataset_name": self.base.name, "train_ids": self.train, "val_ids": self.val,
            }

        def verify_identity(self, identity):
            if identity != self.identity:
                raise ValueError("source identity mismatch")

        def inventory(self, identifier):
            return {self.source_files[identifier].name: sha256(self.source_files[identifier])}

        def verify_inventory(self, inventory):
            if inventory != self._inventory:
                raise ValueError("source inventory mismatch")

        def case_files(self, identifier):
            return [self.source_files[identifier]]

        def training_dataset(self):
            return type("Dataset", (), {"identifiers": self.train})()

        def load_train(self, dataset, identifier):
            image, seg, props = self.arrays[identifier]
            return image.copy(), seg.copy(), props

        def require_no_unpack_writes(self):
            return None

    return Source()


def _write_targets(root, source, monkeypatch):
    from experiments.genseg3mm import targets

    root.mkdir()
    (root / "masks").mkdir()
    cases, hd_cli = {}, r"C:\fixture\hd-bet.exe"
    for patient in source.train:
        keep = np.ones((2, 3, 4), dtype=bool)
        keep[:, 0, 0] = False
        mask = root / "masks" / (patient + ".npz")
        np.savez_compressed(mask, keep=keep)
        case_dir = root / "cases" / patient
        case_dir.mkdir(parents=True)
        native, las, log = case_dir / "native.nii.gz", case_dir / "input_las.nii.gz", case_dir / "hd-bet.log"
        native.write_bytes(b"native-mask")
        las.write_bytes(b"las-image")
        log.write_text("HD-BET complete", encoding="utf-8")
        command = [hd_cli, "-i", str(las), "-o", str(case_dir / "hd_bet"), "-device", "cuda",
                   "--save_bet_mask", "--no_bet_image"]
        started = case_dir / "command_started.json"
        started.write_text(json.dumps({
            "patient": patient, "status": "running", "command": command,
            "raw_sha256": sha256(source.raw_files[patient]),
        }), encoding="utf-8")
        result = case_dir / "command_result.json"
        result.write_text(json.dumps({
            "patient": patient, "status": "command_completed", "return_code": 0,
            "mask_sha256": sha256(native), "log_sha256": sha256(log),
        }), encoding="utf-8")
        cases[patient] = {
            "source_inventory": source.inventory(patient), "source_shape": [1, 2, 3, 4],
            "raw_file": source.raw_files[patient].name, "raw_sha256": sha256(source.raw_files[patient]),
            "input_las_file": str(las.relative_to(root)), "input_las_sha256": sha256(las),
            "native_mask_file": str(native.relative_to(root)), "native_mask_sha256": sha256(native),
            "log_file": str(log.relative_to(root)), "log_sha256": sha256(log),
            "command_started_file": str(started.relative_to(root)),
            "command_started_sha256": sha256(started),
            "command_result_file": str(result.relative_to(root)),
            "command_result_sha256": sha256(result),
            "command": command,
            "return_code": 0, "status": "complete",
            "mask_file": str(mask.relative_to(root)), "mask_sha256": sha256(mask), "mask_shape": [2, 3, 4],
        }
    settings = {
        "epochs": 200, "max_steps": 5000, "width": 64, "downs": 8,
        "discriminator_width": 64, "seed": 20261014, "lr": 2e-5,
        "l1_weight": 100.0, "batch_size": 1, "betas": [0.5, 0.999],
        "norm": "batch", "dropout": False, "gan_loss": "vanilla", "scheduler": "constant_lr",
    }
    metadata = {
        "complete": True, "source": source.identity, "inventory": source._inventory,
        "cases": {patient: {"shape": [1, 2, 3, 4]} for patient in source.train},
        "actual_steps": 5000, "settings": settings,
        "scaling": {"method": "global_train_minmax", "min": -7.25, "max": 8.5},
    }
    baseline = root.parent / "baseline"
    baseline.mkdir()
    checkpoint = baseline / "generator.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    record = {"metadata": metadata, "checkpoint_sha256": sha256(checkpoint)}
    monkeypatch.setattr(targets, "load_run", lambda *args: (record, checkpoint))
    manifest = {
        "format": 1, "classification": "exploratory_dataset501_fold0_genseg3mm_preview",
        "complete": True, "source": source.identity, "source_inventory": source._inventory,
        "baseline": {"run": str(baseline.resolve()), "metadata": metadata,
                     "checkpoint_sha256": record["checkpoint_sha256"]},
        "background_fill": -7.25, "recipe": targets.RECIPE,
        "hd_bet": {"version": "2.0.1", "default_tta": True,
                   "weights_sha256": "c" * 64, "cli": hd_cli},
        "cases": cases,
    }
    (root / "targets.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sha256(root / "targets.json")


def test_adapter_masks_only_copied_image_and_preserves_labels_and_sentinels(tmp_path, monkeypatch):
    source = _fake_source(tmp_path)
    root = tmp_path / "targets"
    digest = _write_targets(root, source, monkeypatch)
    adapter = TargetDataset(source, root, tmp_path / "baseline", expected_manifest_sha256=digest)
    original_image, original_seg, _ = source.arrays["trainA"]
    image, seg, _ = adapter.load_train(source.training_dataset(), "trainA")
    assert image is not original_image and seg is not original_seg
    assert image[0, :, 0, 0].tolist() == [-7.25, -7.25]
    assert np.array_equal(seg, original_seg)
    assert np.array_equal(source.arrays["trainA"][0], original_image)
    assert np.array_equal(source.arrays["trainA"][1], original_seg)
    assert adapter.identity["genseg3mm"]["target_manifest_sha256"] == digest


@pytest.mark.parametrize("mutation", ["validation", "missing", "tampered_mask", "baseline_source"])
def test_receipt_rejects_bad_inputs_before_source_load(tmp_path, monkeypatch, mutation):
    source = _fake_source(tmp_path)
    root = tmp_path / "targets"
    digest = _write_targets(root, source, monkeypatch)
    manifest_path = root / "targets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "validation":
        manifest["cases"]["heldoutA"] = manifest["cases"].pop("trainB")
    elif mutation == "missing":
        del manifest["cases"]["trainB"]
    elif mutation == "tampered_mask":
        (root / manifest["cases"]["trainA"]["mask_file"]).write_bytes(b"tampered")
    else:
        manifest["baseline"]["metadata"]["source"] = {"wrong": "source"}
    if mutation != "tampered_mask":
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        digest = sha256(manifest_path)
    monkeypatch.setattr(source, "load_train", lambda *args: (_ for _ in ()).throw(
        AssertionError("source array read before receipt rejection")))
    with pytest.raises((ValueError, KeyError)):
        TargetDataset(source, root, tmp_path / "baseline", expected_manifest_sha256=digest)


def test_baseline_training_settings_and_update_budget_are_reused():
    metadata = {
        "settings": {
            "epochs": 200, "max_steps": 5000, "width": 64, "downs": 8,
            "discriminator_width": 64, "seed": 20261014, "lr": 2e-5,
            "l1_weight": 100.0, "batch_size": 1, "betas": [0.5, 0.999],
            "norm": "batch", "dropout": False, "gan_loss": "vanilla", "scheduler": "constant_lr",
        },
        "actual_steps": 5000,
    }
    kwargs = baseline_training_kwargs(metadata, device="cuda:0")
    assert kwargs == {
        "epochs": 200, "max_steps": 5000, "width": 64, "downs": 8,
        "discriminator_width": 64, "seed": 20261014, "lr": 2e-5,
        "l1_weight": 100.0, "device": "cuda:0",
    }
    metadata["settings"]["max_steps"] = 4000
    with pytest.raises(ValueError):
        baseline_training_kwargs(metadata, device="cuda:0")


def test_oblique_sitk_mask_roundtrip_preserves_the_original_physical_grid():
    import SimpleITK as sitk
    from experiments.genseg3mm.targets import mask_image_to_raw_grid

    angle = np.deg2rad(17.0)
    direction = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    voxels = np.zeros((4, 6, 8), dtype=np.uint8)
    voxels[1:3, 2:5, 3:7] = 1
    raw = sitk.GetImageFromArray(voxels)
    raw.SetSpacing((2.0, 1.0, 5.0))
    raw.SetOrigin((13.0, -21.0, 8.0))
    raw.SetDirection(tuple(direction.reshape(-1)))
    las_mask = sitk.DICOMOrient(raw, "LAS")
    aligned, keep = mask_image_to_raw_grid(las_mask, raw)
    assert np.array_equal(keep, voxels > 0)
    assert aligned.GetSize() == raw.GetSize()
    assert np.allclose(aligned.GetOrigin(), raw.GetOrigin())
    assert np.allclose(aligned.GetDirection(), raw.GetDirection())


def test_target_cache_tamper_and_protected_output_are_rejected(tmp_path, monkeypatch):
    from experiments.genseg30.contract import check_output

    source = _fake_source(tmp_path)
    root = tmp_path / "targets"
    digest = _write_targets(root, source, monkeypatch)
    adapter = TargetDataset(source, root, tmp_path / "baseline", expected_manifest_sha256=digest)
    with pytest.raises(ValueError, match="Output must be outside"):
        check_output(root / "nested-output", adapter.protected)
    mask = root / adapter.manifest["cases"]["trainA"]["mask_file"]
    mask.write_bytes(b"changed after adapter construction")
    monkeypatch.setattr(source, "load_train", lambda *args: (_ for _ in ()).throw(
        AssertionError("source read occurred after target cache changed")))
    with pytest.raises(ValueError):
        adapter.load_train(source.training_dataset(), "trainA")


def test_prepare_hd_bet_invocation_uses_official_single_file_gzip_contract(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from experiments.genseg3mm import targets

    case_dir = tmp_path / "trainA"
    case_dir.mkdir()
    input_file = case_dir / "input_las.nii.gz"
    output_file = case_dir / "hd_bet.nii.gz"
    input_file.write_bytes(b"synthetic input")
    log_path = case_dir / "hd_bet.log"
    observed = {}

    def fake_official_hdbet(command, *, cwd, env, stdout, stderr, text, check):
        output_arg = command[command.index("-o") + 1]
        # HD-BET 2.0.1 hdbet_predict requires a single-file output ending in .nii.gz.
        assert output_arg.lower().endswith(".nii.gz")
        assert "--save_bet_mask" in command and "--no_bet_image" in command
        assert "--disable_tta" not in command
        observed["command"] = command
        observed["cwd"] = Path(cwd)
        mask = Path(output_arg).with_name(Path(output_arg).name[:-7] + "_bet.nii.gz")
        mask.write_bytes(b"synthetic HD-BET mask")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(targets.subprocess, "run", fake_official_hdbet)
    command = targets._build_hd_bet_command("hdbet", input_file, output_file)
    result, mask_path = targets._run_hd_bet(
        command, case_dir=case_dir, env={"CUDA_VISIBLE_DEVICES": "GPU-test"}, log_path=log_path
    )

    assert result.returncode == 0
    assert observed["cwd"] == case_dir
    assert observed["command"][observed["command"].index("-o") + 1] == str(output_file)
    assert mask_path == case_dir / "hd_bet_bet.nii.gz"
    assert mask_path.read_bytes() == b"synthetic HD-BET mask"
    assert log_path.is_file()

    with pytest.raises(ValueError, match=r"end in \.nii\.gz"):
        targets._build_hd_bet_command("hdbet", input_file, case_dir / "hd_bet")


def test_permuted_anisotropic_las_mask_roundtrips_to_exact_native_grid():
    import SimpleITK as sitk

    from experiments.genseg3mm.targets import mask_image_to_raw_grid

    native = np.zeros((4, 5, 6), dtype=np.uint8)
    native[1:4, 1:3, 2:5] = 1
    raw = sitk.GetImageFromArray(native)
    raw.SetSpacing((0.5, 0.8, 7.0))
    raw.SetOrigin((15.0, -22.0, 9.0))
    # Raw x and z point along different patient axes; LAS reorientation swaps them.
    direction = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    raw.SetDirection(tuple(direction.reshape(-1)))
    las_mask = sitk.DICOMOrient(raw, "LAS")
    assert not np.allclose(las_mask.GetSpacing(), raw.GetSpacing())

    aligned, keep = mask_image_to_raw_grid(las_mask, raw)

    assert np.array_equal(keep, native > 0)
    assert aligned.GetSize() == raw.GetSize()
    assert np.allclose(aligned.GetSpacing(), raw.GetSpacing(), atol=1e-5, rtol=0)
    assert np.allclose(aligned.GetOrigin(), raw.GetOrigin(), atol=1e-4, rtol=0)
    assert np.allclose(aligned.GetDirection(), raw.GetDirection(), atol=1e-5, rtol=0)
