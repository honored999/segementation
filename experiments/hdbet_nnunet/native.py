"""Runtime-only native processing, imported after the selected-GPU resource guard."""
from pathlib import Path
import os
import shutil

from .cli import NAME, PROTOCOL, FINGERPRINT_STATUS, require, sha, write, verify_files


def mask_values(data, keep, np):
    require(data.shape == keep.shape and data.ndim == 3 and keep.dtype == np.bool_, "Native mask/grid mismatch")
    require(data.dtype.kind in "iuf" and np.isfinite(data).all(), "Finite numeric native DWI required")
    return np.where(keep, data, 0).astype(data.dtype, copy=False)


def qa(directory, image, brain, keep, np):
    """Every native train plane, including empty brain planes; no label inputs."""
    from PIL import Image
    lo, hi = np.percentile(image, [1, 99])
    scale = max(float(hi - lo), 1e-12)
    qa_dir = directory / "qa"
    qa_dir.mkdir()
    for z in range(image.shape[0]):
        plane = np.clip((image[z] - lo) / scale * 255, 0, 255).astype(np.uint8)
        raw = np.repeat(plane[..., None], 3, axis=-1)
        overlay = raw.copy()
        overlay[brain[z]] = (overlay[brain[z]].astype(float) * .5 + np.array([0, 127, 0])).astype(np.uint8)
        overlay[~keep[z]] = (overlay[~keep[z]].astype(float) * .5 + np.array([127, 0, 0])).astype(np.uint8)
        masked = np.repeat(np.where(keep[z], plane, 0)[..., None], 3, axis=-1)
        Image.fromarray(np.concatenate([raw, overlay, masked], axis=1)).save(qa_dir / f"z{z:04d}.png")


def save_native_dwi(source, masked, path, nib, np):
    """Keep both original transform encodings; never re-encode qform via SVD."""
    header = source.header.copy()
    header.set_slope_inter(1, 0)
    saved = source.__class__(masked, source.affine, header)
    nib.save(saved, path)
    readback = nib.load(path)
    actual = np.asanyarray(readback.dataobj)
    require(readback.shape == source.shape and readback.get_data_dtype() == source.get_data_dtype() and
            (readback.dataobj.slope, readback.dataobj.inter) == (1.0, 0.0) and
            np.array_equal(readback.affine, source.affine) and np.array_equal(actual, masked), "DWI readback failed")
    for field in ("qform_code", "sform_code", "pixdim", "xyzt_units", "quatern_b", "quatern_c", "quatern_d",
                  "qoffset_x", "qoffset_y", "qoffset_z", "srow_x", "srow_y", "srow_z"):
        require(np.array_equal(readback.header[field], source.header[field]), "NIfTI geometry header changed: " + field)
    for get in ("get_qform", "get_sform"):
        require(np.array_equal(getattr(readback, get)(), getattr(source, get)()), "NIfTI transform changed")

def process_case(args, info, patient, uid):
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk
    from .cli import guard
    from experiments.genseg3mm.targets import (
        validate_nifti_mm_orthogonal, mask_image_to_raw_grid, expand_keep_mask_3mm,
        _build_hd_bet_command, _run_hd_bet,
    )
    case = info["cases"][patient]
    # Hash reads finish BEFORE guard. Cached76 have the same CPU/RAM/GPU guard.
    require(sha(case["image"]) == case["raw_sha256"] and sha(case["label"]) == case["label_sha256"], "Source changed before case")
    guard(args.gpu, uid)
    source = validate_nifti_mm_orthogonal(case["image"])
    require(len(source.shape) == 3, "Only native3D DWI supported")
    dtype = source.get_data_dtype()
    require(dtype.kind in "iuf" and (source.dataobj.slope, source.dataobj.inter) == (1.0, 0.0),
            "Nonidentity NIfTI scaling unsupported; refusing silent requantization")
    label_header = nib.load(case["label"])
    require(label_header.shape == source.shape and np.allclose(label_header.affine, source.affine, atol=1e-5, rtol=0),
            "Label header grid differs from native DWI")
    raw_image = sitk.ReadImage(case["image"])
    output = Path(info["output"])
    case_dir = output / "cases" / patient
    case_dir.mkdir(parents=True, exist_ok=False)
    command = None
    if "cached_mask" in case:
        mask_path = Path(case["cached_mask"])
        require(sha(mask_path) == info["files"][str(mask_path)], "Cached native mask changed")
    else:
        las = case_dir / "input_las.nii.gz"
        sitk.WriteImage(sitk.DICOMOrient(raw_image, "LAS"), str(las), True)
        command = _build_hd_bet_command(info["executable"], las, case_dir / "hd_bet.nii.gz")
        write(case_dir / "command_started.json", {"command": command, "gpu_uuid": uid, "default_tta": True,
              "raw_sha256": case["raw_sha256"], "weights_sha256": info["weights_sha256"], "input_las_sha256": sha(las)})
        require(sha(info["weights"]) == info["weights_sha256"], "Fixed weights changed before inference")
        guard(args.gpu, uid)  # Adjacent to actual19 inference, after disk audits.
        process, mask_path = _run_hd_bet(command, case_dir=case_dir, env=os.environ.copy(), log_path=case_dir / "hd-bet.log")
        write(case_dir / "command_result.json", {"return_code": process.returncode,
              "mask_sha256": sha(mask_path) if mask_path.is_file() else None, "log_sha256": sha(case_dir / "hd-bet.log")})
        require(process.returncode == 0 and mask_path.is_file(), "HD-BET failed; preserve incomplete output")
        guard(args.gpu, uid)  # Inference may have changed resource availability.
    _, brain = mask_image_to_raw_grid(sitk.ReadImage(str(mask_path)), raw_image)
    keep = expand_keep_mask_3mm(brain, spacing_xyz=raw_image.GetSpacing())
    keep_xyz = keep.transpose(2, 1, 0)  # SITK ZYX -> native NIfTI XYZ; no interpolation.
    data = np.asanyarray(source.dataobj)
    masked = mask_values(data, keep_xyz, np)
    image_out = output / "nnUNet_raw" / NAME / "imagesTr" / Path(case["image"]).name
    save_native_dwi(source, masked, image_out, nib, np)
    label_out = output / "nnUNet_raw" / NAME / "labelsTr" / Path(case["label"]).name
    shutil.copyfile(case["label"], label_out)
    require(sha(label_out) == case["label_sha256"], "Label byte identity failed")
    for name, values in (("brain_native.nii.gz", brain), ("keep_native3mm.nii.gz", keep)):
        native = sitk.GetImageFromArray(values.astype(np.uint8))
        native.CopyInformation(raw_image)
        sitk.WriteImage(native, str(case_dir / name), True)
        _, check_mask = mask_image_to_raw_grid(sitk.ReadImage(str(case_dir / name)), raw_image)
        require(np.array_equal(check_mask, values), "Saved mask readback failed: " + name)
    is_train = patient in info["train"]
    if is_train:
        qa(case_dir, data.transpose(2, 1, 0), brain, keep, np)
    result = {"role": "train" if is_train else "development_validation",
              "mask_source": "cached76" if "cached_mask" in case else "new19", "command": command,
              "mask_sha256": sha(mask_path), "keep_voxels": int(keep.sum()),
              "empty_native_planes": np.flatnonzero(~brain.any(axis=(1, 2))).tolist(),
              "qa_planes": int(data.shape[2]) if is_train else 0,
              "raw_sha256": case["raw_sha256"], "label_sha256": case["label_sha256"],
              "image_sha256": sha(image_out), "readback_verified": True, "coverage_audit": "not_requested"}
    if args.audit_label_coverage and is_train:
        labels = np.asanyarray(label_header.dataobj)
        require(np.isin(labels, [0, 1]).all(), "Coverage audit requires binary GT")
        result["coverage_audit"] = {"lesion_voxels": int((labels == 1).sum()),
                                    "lesion_outside_keep": int(((labels == 1) & ~keep_xyz).sum()), "report_only": True}
    write(case_dir / "case.json", result)
    paths = [image_out, label_out, *[p for p in case_dir.rglob("*") if p.is_file()]]
    artifacts = {}
    for path in paths:
        require(output in path.resolve(strict=True).parents, "Generated artifact escaped output")
        artifacts[str(path.relative_to(output))] = sha(path)
    return result, artifacts


def build(args, info, snap):
    from .cli import guard
    from experiments.genseg3mm.targets import _hd_bet_runtime
    version, executable, weights, digest = _hd_bet_runtime()
    require(str(weights) == info["weights"] and digest == info["weights_sha256"], "HD-BET runtime changed")
    info["executable"] = executable
    output = Path(info["output"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "started.json", {"complete": False, "protocol": PROTOCOL, "inputs": info, "gpu_uuid": snap["gpu_uuid"]})
    raw, pre = output / "nnUNet_raw" / NAME, output / "nnUNet_preprocessed" / NAME
    for directory in (raw / "imagesTr", raw / "labelsTr", pre, output / "nnUNet_results", output / "provenance"):
        directory.mkdir(parents=True, exist_ok=False)
    artifacts = {"started.json": sha(output / "started.json")}
    def copy(source, target):
        shutil.copyfile(source, target)
        require(sha(target) == info["files"][str(Path(source).resolve())], "Metadata copy failed")
        artifacts[str(target.relative_to(output))] = sha(target)
    # nnUNet2.8.1 on_train_start copies this inherited fingerprint unconditionally.
    # It is compatibility metadata; no extraction, fitting, or replanning occurs.
    for name in ("nnUNetPlans.json", "splits_final.json", "dataset.json", "dataset_fingerprint.json"):
        copy(Path(info["pre"]) / name, pre / name)
    copy(Path(info["raw"]) / "dataset.json", raw / "dataset.json")
    copy(Path(info["pre"]) / "dataset_fingerprint.json", output / "provenance/inherited_fixed_baseline_dataset_fingerprint.json")
    results = {}
    for patient in info["train"] + info["val"]:
        result, case_files = process_case(args, info, patient, snap["gpu_uuid"])
        results[patient] = result
        artifacts.update(case_files)
        print("PREPARE", len(results), "/95", patient, flush=True)
    verify_files(info["files"])
    verify_files({str(output / path): digest for path, digest in artifacts.items()})
    require(len(results) == 95 and all(r["readback_verified"] for r in results.values()), "Incomplete native readback")
    write(output / "complete.json", {"complete": True, "protocol": PROTOCOL, "inputs": info, "cases": results,
          "artifacts": artifacts, "gpu_uuid": snap["gpu_uuid"], "hd_bet_version": version,
          "fingerprint_status": FINGERPRINT_STATUS,
          "preprocessing": "not_started", "qa_status": "all_native_train76_planes_generated_manual_review_required_no_GT",
          "normalization": "inherited_singleDWI_per_case_ZScore_no_cohort_fitting",
          "cropping": "nnUNet preprocessing may change nonzero bbox; plans unchanged; native labels byte_exact"})
    print("PREPARE COMPLETE: " + str(output / "complete.json") + " SHA256=" + sha(output / "complete.json"), flush=True)
