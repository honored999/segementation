"""Server-deferred metadata contracts. No tests were executed on the local host."""
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.hdbet_nnunet import cli


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = tmp_path / "original_raw" / cli.NAME
    pre = tmp_path / "original_preprocessed" / cli.NAME
    targets = tmp_path / "HD-BET_outputs" / "old_targets"
    home = tmp_path / "user"
    weights = home / "hd-bet_params/release_2.0.0/fold_all/checkpoint_final.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic pinned weights")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli, "versions", lambda: None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "synthetic-hd-bet")
    for p in (raw / "imagesTr", raw / "labelsTr", pre, targets):
        p.mkdir(parents=True, exist_ok=True)
    reference = cli.REPO / "standalone_nnunet2d/reference"
    for name in ("dataset.json", "nnUNetPlans.json", "splits_final.json"):
        (pre / name).write_bytes((reference / name).read_bytes())
    (raw / "dataset.json").write_bytes((reference / "dataset.json").read_bytes())
    cli.write(pre / "dataset_fingerprint.json", {"spacings": [[8, 2, 2]] * 95, "shapes_after_crop": [[3, 4, 5]] * 95})
    train, val = cli.splits(pre / "splits_final.json")
    identifier = cli.read(pre / "nnUNetPlans.json")["configurations"]["2d"]["data_identifier"]
    (pre / identifier).mkdir()
    inventory, cases = {}, {}
    for patient in train + val:
        image = raw / "imagesTr" / (patient + "_0000.nii.gz")
        image.write_bytes(b"metadata fixture image")
        (raw / "labelsTr" / (patient + ".nii.gz")).write_bytes(b"metadata fixture label")
        inv = {}
        for ext in (".b2nd", "_seg.b2nd", ".pkl"):
            p = pre / identifier / (patient + ext)
            p.write_bytes(b"metadata fixture, never unpickle/decompress")
            inv[p.name] = cli.sha(p)
        if patient not in train:
            continue
        inventory[patient] = inv
        directory = targets / patient
        directory.mkdir()
        rec = {"status": "complete", "return_code": 0, "raw_file": image.name,
               "raw_sha256": cli.sha(image), "source_inventory": inv}
        for field in ("mask", "native_mask", "input_las", "log"):
            p = directory / field
            p.write_bytes(b"receipt-only fixture")
            rec[field + "_file"], rec[field + "_sha256"] = str(p.relative_to(targets)), cli.sha(p)
        command = ["synthetic-hd-bet", "-i", "input", "-o", "output.nii.gz", "-device", "cuda", "--save_bet_mask", "--no_bet_image"]
        rec["command"] = command
        for field, value in (
            ("command_started", {"patient": patient, "command": command, "default_tta": True,
              "raw_sha256": cli.sha(image), "input_las_file": "input", "output_file": "output.nii.gz"}),
            ("command_result", {"patient": patient, "status": "command_completed", "return_code": 0,
              "mask_sha256": rec["native_mask_sha256"], "log_sha256": rec["log_sha256"]}),
        ):
            p = directory / (field + ".json")
            cli.write(p, value)
            rec[field + "_file"], rec[field + "_sha256"] = str(p.relative_to(targets)), cli.sha(p)
        cases[patient] = rec
    identity = {"dataset_name": cli.NAME, "configuration": "2d", "fold": 0, "backend": "nnUNetDatasetBlosc2",
                "train_ids": train, "val_ids": val, "reference_sha256": cli.SPLIT_SHA}
    for field, name in (("plans_sha256", "nnUNetPlans.json"), ("dataset_sha256", "dataset.json"), ("splits_sha256", "splits_final.json")):
        identity[field] = cli.sha(pre / name)
    cli.write(targets / "targets.json", {"format": 1, "complete": True, "recipe": cli.RECIPE,
              "classification": "exploratory_dataset501_fold0_genseg3mm_preview", "source": identity,
              "source_inventory": inventory, "cases": cases, "hd_bet": {"version": "2.0.1", "default_tta": True,
              "weights_file": str(weights), "weights_sha256": cli.sha(weights), "cli": "synthetic-hd-bet"}})
    digest = cli.sha(targets / "targets.json")
    monkeypatch.setattr(cli, "TARGET_SHA", digest)
    return SimpleNamespace(raw_dataset=str(raw), preprocessed_dataset=str(pre), targets=str(targets),
                           targets_sha256=digest, output=str(tmp_path / "HD-BET_outputs/trial"), gpu=1)


def test_check_metadata_only_does_not_import_compute_or_create_output(source, monkeypatch):
    original = builtins.__import__
    def no_compute(name, *a, **kw):
        assert name.split(".")[0] not in {"numpy", "torch", "nibabel", "SimpleITK", "HD_BET", "scipy"}
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", no_compute)
    monkeypatch.setattr(cli, "guard", lambda *a, **kw: pytest.fail("metadata audit invoked telemetry"))
    info = cli.check(source)
    assert len(info["cases"]) == 95
    for relative in ("experiments/hdbet_nnunet/cli.py", "experiments/hdbet_nnunet/native.py",
                     "nnunet_ext_trainers/nnUNetTrainerTopK10EarlyStopping.py",
                     "experiments/genseg3mm/targets.py", "experiments/genseg_joint/runtime.py"):
        path = (cli.REPO / relative).resolve()
        assert info["files"][str(path)] == cli.sha(path)
    assert sum("cached_mask" in c for c in info["cases"].values()) == 76
    assert not Path(source.output).exists()


@pytest.mark.parametrize("kind", ["raw", "mask", "plan", "split", "weight", "foreign"])
def test_check_fails_closed_on_changed_sources(source, kind):
    pre, raw, target = Path(source.preprocessed_dataset), Path(source.raw_dataset), Path(source.targets)
    if kind == "raw":
        next((raw / "imagesTr").iterdir()).write_bytes(b"tampered")
    elif kind == "mask":
        next(target.glob("*/native_mask")).write_bytes(b"tampered")
    elif kind == "plan":
        p = pre / "nnUNetPlans.json"
        p.write_text(p.read_text() + " ")
    elif kind == "split":
        p = pre / "splits_final.json"
        value = cli.read(p)
        value[1]["train"][0] = "foreign"
        p.write_text(__import__("json").dumps(value))
    elif kind == "weight":
        Path(cli.read(target / "targets.json")["hd_bet"]["weights_file"]).write_bytes(b"tampered")
    else:
        (raw / "imagesTr/foreign_0000.nii.gz").write_bytes(b"foreign")
    with pytest.raises(ValueError):
        cli.check(source)
    assert not Path(source.output).exists()


def test_split_all_folds_crlf_safe_and_mutation_rejected(tmp_path):
    source = cli.REPO / "standalone_nnunet2d/reference/splits_final.json"
    p = tmp_path / "split.json"
    p.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    assert tuple(map(len, cli.splits(p))) == (76, 19)
    value = cli.read(p)
    value[4]["val"][0] = value[4]["train"][0]
    p.write_text(__import__("json").dumps(value))
    with pytest.raises(ValueError):
        cli.splits(p)


def test_output_overlap_and_confined_artifact_escape(tmp_path):
    protected = tmp_path / "HD-BET_outputs/source"
    protected.mkdir(parents=True)
    p = tmp_path / "outside"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        cli.separate(protected / "child", [protected])
    with pytest.raises(ValueError):
        cli.confined(protected, "../../outside")


def prepared(source, monkeypatch):
    info = cli.check(source)
    output = Path(source.output)
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        (output / name).mkdir(parents=True)
        monkeypatch.setenv(name, str(output / name))
    raw, pre = output / "nnUNet_raw" / cli.NAME, output / "nnUNet_preprocessed" / cli.NAME
    cli.shutil.copytree(info["raw"], raw)
    pre.mkdir()
    for name in ("dataset.json", "nnUNetPlans.json", "splits_final.json", "dataset_fingerprint.json"):
        cli.shutil.copyfile(Path(info["pre"]) / name, pre / name)
    (output / "provenance").mkdir()
    cli.shutil.copyfile(Path(info["pre"]) / "dataset_fingerprint.json",
                        output / "provenance/inherited_fixed_baseline_dataset_fingerprint.json")
    cases = {p: {"readback_verified": True, "mask_source": "cached76" if p in info["train"] else "new19",
                 "raw_sha256": info["cases"][p]["raw_sha256"], "label_sha256": info["cases"][p]["label_sha256"],
                 "image_sha256": info["cases"][p]["raw_sha256"]}
             for p in info["train"] + info["val"]}
    artifacts = {str(p.relative_to(output)): cli.sha(p) for p in output.rglob("*") if p.is_file()}
    cli.write(output / "complete.json", {"complete": True, "protocol": cli.PROTOCOL, "inputs": info, "cases": cases, "artifacts": artifacts,
              "fingerprint_status": cli.FINGERPRINT_STATUS})
    monkeypatch.setattr(cli, "check_trainer_metadata", lambda: {})
    return info, output, pre


def test_preprocess_refuses_existing_directory_even_empty(source, monkeypatch):
    info, output, pre = prepared(source, monkeypatch)
    cli.ready(source, before_preprocess=True)
    (pre / info["identifier"]).mkdir()
    with pytest.raises(ValueError, match="deletes"):
        cli.ready(source, before_preprocess=True)


def test_ready_triplets_and_fresh_training_boundary(source, monkeypatch):
    info, output, pre = prepared(source, monkeypatch)
    folder = pre / info["identifier"]
    folder.mkdir()
    for patient in info["train"] + info["val"]:
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            (folder / (patient + suffix)).write_bytes(b"metadata presence only")
    cli.ready(source)
    victim = next(folder.glob("*_seg.b2nd"))
    victim.unlink()
    with pytest.raises(ValueError, match="triplets"):
        cli.ready(source)
    victim.write_bytes(b"restored fixture")
    fold = output / "nnUNet_results" / cli.NAME / (cli.TRAINER + "__nnUNetPlans__2d") / "fold_0"
    fold.mkdir(parents=True)
    with pytest.raises(ValueError, match="fold_0"):
        cli.ready(source)


def test_prepare_guard_after_check_before_runtime_import(source, monkeypatch):
    events = []
    def audit(args):
        events.append("audit")
        return {}
    def refused(*a):
        events.append("guard")
        raise ValueError("busy")
    monkeypatch.setattr(cli, "check", audit)
    monkeypatch.setattr(cli, "guard", refused)
    with pytest.raises(ValueError, match="busy"):
        cli.prepare(source)
    assert events == ["audit", "guard"]
    assert not Path(source.output).exists()

@pytest.mark.parametrize("which", ["compatibility", "provenance", "status"])
def test_preprocess_check_requires_inherited_fingerprint_copies(source, monkeypatch, which):
    info, output, pre = prepared(source, monkeypatch)
    source_digest = info["files"][str(Path(info["pre"]) / "dataset_fingerprint.json")]
    assert cli.sha(pre / "dataset_fingerprint.json") == source_digest
    assert cli.sha(output / "provenance/inherited_fixed_baseline_dataset_fingerprint.json") == source_digest
    cli.ready(source, before_preprocess=True)
    if which == "status":
        path = output / "complete.json"
        receipt = cli.read(path)
        receipt["fingerprint_status"] = "newly_measured"
        path.write_text(__import__("json").dumps(receipt))
    else:
        path = pre / "dataset_fingerprint.json" if which == "compatibility" else output / "provenance/inherited_fixed_baseline_dataset_fingerprint.json"
        path.write_bytes(b"tampered fingerprint")
    with pytest.raises(ValueError):
        cli.ready(source, before_preprocess=True)


def test_results_nested_redirect_is_refused(source, monkeypatch, tmp_path):
    info, output, pre = prepared(source, monkeypatch)
    outside = tmp_path / "external_results"
    outside.mkdir()
    redirect = output / "nnUNet_results" / cli.NAME
    try:
        redirect.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Filesystem does not permit directory symlinks; Windows junction check remains server-deferred")
    with pytest.raises(ValueError, match="Results path redirected"):
        cli.ready(source, before_preprocess=True)
    assert not list(outside.iterdir())
