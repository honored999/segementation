import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generate_nnunet_result_report import (
    _add_window, _architecture, _diagnostic_feature, _inverse_feature, _channel_ids, _native_slice_index, _confusion_masks, _overlay_rgba, _confusion_legend_handles,
    _normalize_display, check_geometry, main,
    protected_output, read_metrics, require_simpleitk_reader, resolve_external_trainer,
    select_cases, select_slices, verify_full_geometry,
)


def test_rank_ties_and_disjoint():
    rows = {name: {"dice": str(value)} for name, value in
            (("b", .9), ("a", .9), ("c", .8), ("d", .2), ("f", .1), ("e", .1), ("g", .1))}
    high, low = select_cases(rows)
    assert high == ["a", "b", "c"]
    assert low == ["e", "f", "g"]
    assert set(high).isdisjoint(low)
    with pytest.raises(ValueError, match="at least six"):
        select_cases({"a": {"dice": "1"}})


def test_metric_coverage_and_nan(tmp_path):
    headers = "case_id,dice,iou,f2,avd_percent,lcd,recall,hd95_mm\n"
    rows = "a,0.5,0.4,0.3,NaN,1,0.6,NaN\n"
    (tmp_path / "case_metrics.csv").write_text(headers + rows, encoding="utf-8")
    summary = {"n_cases": 1, "metrics": {key: {"valid_cases": 0 if key in
               ("avd_percent", "hd95_mm") else 1} for key in
               ("dice", "iou", "f2", "avd_percent", "lcd", "recall", "hd95_mm")}}
    (tmp_path / "summary_metrics.json").write_text(json.dumps(summary), encoding="utf-8")
    assert read_metrics(tmp_path, {"a"})[1]["n_cases"] == 1
    with pytest.raises(ValueError, match="coverage mismatch"):
        read_metrics(tmp_path, {"b"})
    (tmp_path / "case_metrics.csv").write_text(headers + rows.replace("0.5", "NaN"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Dice"):
        read_metrics(tmp_path, {"a"})
    (tmp_path / "case_metrics.csv").write_text(headers + rows + rows, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_metrics(tmp_path, {"a"})


def test_slices_and_constant():
    mask = np.zeros((5, 3, 4), dtype=np.uint8)
    mask[3, :2] = 1
    mask[1, :2] = 1
    mask[4, 0, 0] = 1
    assert select_slices(mask) == [1, 3, 4]
    assert select_slices(np.zeros_like(mask)) == []
    assert np.all(_normalize_display(np.ones((3, 4))) == 0)


def test_window_overlap_inverse_crop_transpose():
    total = np.zeros((3, 4), dtype=np.float32)
    weight = np.zeros_like(total)
    _add_window(total, weight, np.ones((2, 3)), 2, (slice(0, 2), slice(0, 3)))
    _add_window(total, weight, np.full((2, 3), 3), 1, (slice(1, 3), slice(1, 4)))
    assert total[1, 1] / weight[1, 1] == pytest.approx(5/3)
    with pytest.raises(ValueError, match="mismatch"):
        _add_window(total, weight, np.ones((1, 1)), 1, (slice(0, 2), slice(0, 3)))

    class Config:
        spacing = (1, 1)
        def resampling_fn_probabilities(self, array, shape, *unused):
            assert tuple(shape) == (2, 3, 4)
            return array
    props = {"shape_after_cropping_and_before_resampling": (2, 3, 4),
             "shape_before_cropping": (3, 4, 5),
             "bbox_used_for_cropping": ((1, 3), (1, 4), (1, 5)),
             "spacing": (2, 1, 1)}
    plans = SimpleNamespace(transpose_forward=(2, 0, 1), transpose_backward=(1, 2, 0))
    original = _inverse_feature(np.ones((2, 3, 4)), props, plans, Config())
    assert original.shape == (4, 5, 3)
    assert original.sum() == 24


def test_path_boundary(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    assert protected_output(tmp_path / "report", [source]) == (tmp_path / "report").resolve()
    with pytest.raises(ValueError, match="overlaps"):
        protected_output(source / "report", [source])
    with pytest.raises(ValueError, match="already exists"):
        protected_output(source, [source])


def test_physical_geometry():
    sitk = pytest.importorskip("SimpleITK")
    a = sitk.GetImageFromArray(np.zeros((2, 3, 4), dtype=np.uint8))
    b = sitk.GetImageFromArray(np.zeros((2, 3, 4), dtype=np.uint8))
    check_geometry(a, b, "x")
    b.SetDirection((0, 1, 0, 1, 0, 0, 0, 0, -1))
    with pytest.raises(ValueError, match="Direction"):
        check_geometry(a, b, "x")


def test_tiny_hook_two_windows_and_cleanup():
    torch = pytest.importorskip("torch")
    class Pre:
        def __init__(self, verbose=False):
            pass
        def run_case(self, images, seg, plans, config, dataset):
            assert seg is None
            data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=np.float32)
            props = {"shape_after_cropping_and_before_resampling": (1, 2, 4),
                     "shape_before_cropping": (1, 2, 4),
                     "bbox_used_for_cropping": ((0, 1), (0, 2), (0, 4)),
                     "spacing": (1, 1, 1)}
            return data, None, props
    class Config:
        patch_size = (2, 2)
        spacing = (1, 1)
        preprocessor_class = Pre
        def resampling_fn_probabilities(self, array, shape, *unused):
            assert tuple(shape) == (1, 2, 4)
            return array
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.stages = torch.nn.ModuleList([torch.nn.Identity()])
            self.fail = False
        def forward(self, x):
            value = self.encoder.stages[0](x)
            if self.fail:
                raise RuntimeError("synthetic forward failure")
            return value
    network = Net()
    predictor = SimpleNamespace(configuration_manager=Config(),
        plans_manager=SimpleNamespace(transpose_forward=(0, 1, 2),
                                      transpose_backward=(0, 1, 2)),
        dataset_json={}, network=network, device=torch.device("cpu"),
        _internal_get_sliding_window_slicers=lambda shape: [
            (slice(None), 0, slice(0, 2), slice(0, 2)),
            (slice(None), 0, slice(0, 2), slice(2, 4))])
    mapped, provenance, native = _diagnostic_feature(Path("synthetic.nii.gz"), predictor, 0, 0)
    assert mapped.shape == (1, 2, 4)
    assert np.allclose(mapped[0], [[1, 2, 3, 4], [5, 6, 7, 8]])
    assert provenance["module"] == "encoder.stages.0"
    assert provenance["windows"] == 2
    assert provenance["covered_preprocessed_voxels"] == 8
    assert native["channel_ids"] == [0]
    assert native["channels"].shape == (1, 2, 2)
    assert native["window_index"] == 1
    assert native["window_yx_preprocessed"] == [[0, 2], [2, 4]]
    assert not network.encoder.stages[0]._forward_hooks
    network.fail = True
    with pytest.raises(RuntimeError, match="synthetic"):
        _diagnostic_feature(Path("synthetic.nii.gz"), predictor, 0)
    assert not network.encoder.stages[0]._forward_hooks


def test_architecture_png_smoke(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    decoder = SimpleNamespace(pool_scales=(1, 2, 4), fpn_channels=128,
                              classifier=SimpleNamespace(out_channels=2))
    net = SimpleNamespace(encoder=SimpleNamespace(stages=[1, 2, 3, 4],
                         output_channels=(32, 64, 128, 256)),
                         selected_feature_indices=(0, 1, 2, 3), decoder=decoder)
    fig, ax = plt.subplots(figsize=(14, 2))
    _architecture(ax, net, 3)
    target = tmp_path / "architecture.png"
    fig.savefig(target)
    plt.close(fig)
    assert target.stat().st_size > 1000


def test_cli_check_metadata_only(tmp_path, capsys):
    model = tmp_path / "Dataset501_StrokeLesion" / "nnUNetTrainerUPerNetTopK10EarlyStopping__nnUNetPlans__2d"
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    prediction = model / "fold_0" / "validation"
    metrics = model / "fold_0" / "multi_metrics"
    for directory in (images, labels, prediction, metrics):
        directory.mkdir(parents=True)
    (model / "dataset.json").write_text(json.dumps({"channel_names": {"0": "DWI"}}))
    (model / "plans.json").write_text(json.dumps({"image_reader_writer": "SimpleITKIO", "configurations": {"2d": {}}}))
    checkpoint = model / "fold_0" / "checkpoint_best.pth"
    checkpoint.touch()
    names = [f"case_{i}" for i in range(6)]
    for i, name in enumerate(names):
        (images / f"{name}_0000.nii.gz").touch()
        (labels / f"{name}.nii.gz").touch()
        (prediction / f"{name}.nii.gz").touch()
    cols = "case_id,dice,iou,f2,avd_percent,lcd,recall,hd95_mm\n"
    (metrics / "case_metrics.csv").write_text(cols + "".join(
        f"{name},{i/10},0,0,0,0,0,0\n" for i, name in enumerate(names)))
    (metrics / "summary_metrics.json").write_text(json.dumps({"n_cases": 6,
        "metrics": {key: {"valid_cases": 6} for key in
        ("dice", "iou", "f2", "avd_percent", "lcd", "recall", "hd95_mm")}}))
    assert main(["--model-dir", str(model), "--fold", "0", "--images-dir", str(images),
                 "--labels-dir", str(labels), "--checkpoint", str(checkpoint),
                 "--output-dir", str(tmp_path / "new-report"), "--check"]) == 0
    assert "METADATA ONLY" in capsys.readouterr().out
    assert not (tmp_path / "new-report").exists()


def test_reader_axis_contract():
    require_simpleitk_reader({"image_reader_writer": "SimpleITKIO"})
    for reader in ("SimpleITKIOWithReorient", "NibabelIO", None):
        with pytest.raises(ValueError, match="image_reader_writer"):
            require_simpleitk_reader({"image_reader_writer": reader})


def test_external_trainer_lookup_and_namesake(monkeypatch, tmp_path):
    import importlib.util
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    extension = Path(__file__).resolve().parents[1] / "nnunet_ext_trainers"
    monkeypatch.setenv("nnUNet_extTrainer", str(extension))
    trainer = resolve_external_trainer(recursive_find_trainer_class_by_name)
    assert trainer.__module__ == "nnUNetTrainerUPerNetTopK10EarlyStopping"
    assert Path(trainer.build_network_architecture.__code__.co_filename).resolve() == (
        extension / "nnUNetTrainerMixins.py").resolve()
    impostor = tmp_path / "nnUNetTrainerUPerNetTopK10EarlyStopping.py"
    impostor.write_text("class nnUNetTrainerUPerNetTopK10EarlyStopping: pass\n")
    spec = importlib.util.spec_from_file_location("nnUNetTrainerUPerNetTopK10EarlyStopping", impostor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_external_trainer(lambda name: module.nnUNetTrainerUPerNetTopK10EarlyStopping)


def test_full_geometry_rejects_seventh_case(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    images, labels, predictions = {}, {}, {}
    for i in range(7):
        cid = f"case_{i}"
        paths = [tmp_path / f"{cid}_{kind}.nii.gz" for kind in ("image", "gt", "pred")]
        for kind, path in enumerate(paths):
            image = sitk.GetImageFromArray(np.zeros((2, 3, 4), dtype=np.uint8))
            if i == 6 and kind == 2:
                image.SetOrigin((1.0, 0.0, 0.0))
            sitk.WriteImage(image, str(path))
        images[cid], labels[cid], predictions[cid] = paths
    info = {"rows": {cid: {} for cid in images}, "images": images,
            "labels": labels, "predictions": predictions}
    with pytest.raises(ValueError, match="case_6.*Origin"):
        verify_full_geometry(info)


@pytest.mark.parametrize("missing_channel", [False, True])
def test_atomic_delivery_removes_partial_png(tmp_path, monkeypatch, capsys, missing_channel):
    import generate_nnunet_result_report as report
    model = tmp_path / "Dataset501_StrokeLesion" / (
        "nnUNetTrainerUPerNetTopK10EarlyStopping__nnUNetPlans__2d")
    images, labels = tmp_path / "images", tmp_path / "labels"
    prediction, metrics = model / "fold_0" / "validation", model / "fold_0" / "multi_metrics"
    for folder in (images, labels, prediction, metrics):
        folder.mkdir(parents=True)
    (model / "dataset.json").write_text(json.dumps({"channel_names": {"0": "DWI"}}))
    (model / "plans.json").write_text(json.dumps(
        {"image_reader_writer": "SimpleITKIO", "configurations": {"2d": {}}}))
    checkpoint = model / "fold_0" / "checkpoint_best.pth"
    checkpoint.touch()
    names = [f"case_{i}" for i in range(6)]
    for cid in names:
        for path in (images / f"{cid}_0000.nii.gz", labels / f"{cid}.nii.gz",
                     prediction / f"{cid}.nii.gz"):
            path.touch()
    cols = "case_id,dice,iou,f2,avd_percent,lcd,recall,hd95_mm\n"
    (metrics / "case_metrics.csv").write_text(cols + "".join(
        f"{cid},0.5,0,0,0,0,0,0\n" for cid in names))
    (metrics / "summary_metrics.json").write_text(json.dumps({"n_cases": 6,
        "metrics": {key: {"valid_cases": 6} for key in
        ("dice", "iou", "f2", "avd_percent", "lcd", "recall", "hd95_mm")}}))
    monkeypatch.setattr(report, "verify_full_geometry", lambda info: 6)
    monkeypatch.setattr(report, "_predictor", lambda info, args: object())
    def fail_after_png(info, predictor, declaration):
        (info["output"] / "summary.png").write_bytes(b"partial")
        raise RuntimeError("injected write failure")
    def omit_channel_png(info, predictor, declaration):
        (info["output"] / "summary.png").write_bytes(b"synthetic")
        (info["output"] / "report.txt").write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(report, "create_report",
                        omit_channel_png if missing_channel else fail_after_png)
    target = tmp_path / "new-report"
    with pytest.raises(SystemExit if missing_channel else RuntimeError,
                       match=None if missing_channel else "injected write failure"):
        main(["--model-dir", str(model), "--fold", "0", "--images-dir", str(images),
              "--labels-dir", str(labels), "--checkpoint", str(checkpoint),
              "--output-dir", str(target),
              "--prediction-checkpoint-declaration", "synthetic declaration"])
    if missing_channel:
        assert "report output incomplete" in capsys.readouterr().err
    assert not target.exists()
    assert not list(tmp_path.glob(".new-report-*"))


def test_fixed_channel_ids_and_confusion_overlay():
    assert _channel_ids(512) == [0, 73, 146, 219, 292, 365, 438, 511]
    gt = np.array([[1, 0], [1, 0]], dtype=bool)
    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    tp, fp, fn = _confusion_masks(gt, pred)
    assert np.array_equal(tp, [[1, 0], [0, 0]])
    assert np.array_equal(fp, [[0, 1], [0, 0]])
    assert np.array_equal(fn, [[0, 0], [1, 0]])
    assert [handle.get_label() for handle in _confusion_legend_handles()] == ["TP", "FP", "FN"]
    rgba = _overlay_rgba(gt, pred)
    assert np.allclose(rgba[0, 0, :3], (.12, .82, .30))
    assert np.allclose(rgba[0, 1, :3], (1., .18, .18))
    assert np.allclose(rgba[1, 0, :3], (.16, .42, 1.))


def test_native_slice_axis_contract_rejects_transpose():
    props = {"bbox_used_for_cropping": ((1, 3), (0, 2), (0, 3)),
             "shape_after_cropping_and_before_resampling": (2, 2, 3)}
    identity = SimpleNamespace(transpose_forward=(0, 1, 2),
                               transpose_backward=(0, 1, 2))
    assert _native_slice_index(2, props, identity, (2, 2, 3)) == 1
    swapped = SimpleNamespace(transpose_forward=(2, 0, 1),
                              transpose_backward=(1, 2, 0))
    with pytest.raises(ValueError, match="identity axis transpose"):
        _native_slice_index(2, props, swapped, (2, 2, 3))
