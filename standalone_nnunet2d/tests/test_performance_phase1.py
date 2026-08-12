from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from standalone_nnunet2d import formal_train
from standalone_nnunet2d.alignment_evidence import build_alignment_evidence
from standalone_nnunet2d.engine import trainer
from standalone_nnunet2d.engine.validator import ValidationEpochResult
from standalone_nnunet2d.engine.trainer import TrainEpochResult, TrainStepResult
from standalone_nnunet2d.training.formal_trainer import run_formal_epoch
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler


class _TinyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        return torch.zeros(1, 4, 4), torch.zeros(4, 4, dtype=torch.long)


def _components() -> dict[str, dict[str, object]]:
    return {
        name: {"status": "passed", "diagnostics": []}
        for name in ("image", "label", "manifest", "mask")
    }


def _write_alignment_reports(tmp_path: Path, *, suffix: str = "") -> tuple[Path, Path]:
    transform = {
        "status": "passed",
        "run_state": "official_alignment_pending",
        "oracle_root": f"/oracle/transform/{suffix}",
        "standalone_root": f"/standalone/transform/{suffix}",
        "image_atol": 0.0,
        "components": _components(),
        "diagnostics": [],
    }
    inference = {
        "parity_policy": "repeat_oracle_stability_v1",
        "oracle_roots": [
            f"/oracle/inference/{suffix}/0",
            f"/oracle/inference/{suffix}/1",
            f"/oracle/inference/{suffix}/2",
        ],
        "oracle_repeat_count": 3,
        "stable_mask_mismatch_count": 0,
        "stable_mask_mismatch_coordinates": [],
        "unobserved_standalone_label_count": 0,
        "unobserved_standalone_label_coordinates": [],
        "status": "passed",
        "run_state": "official_alignment_pending",
        "standalone_root": f"/standalone/inference/{suffix}",
        "image_atol": 0.0,
        "components": _components(),
        "diagnostics": [],
    }
    transform_path = tmp_path / f"transform{suffix}.json"
    inference_path = tmp_path / f"inference{suffix}.json"
    transform_path.write_text(json.dumps(transform), encoding="utf-8")
    inference_path.write_text(json.dumps(inference), encoding="utf-8")
    return transform_path, inference_path


def _write_plans(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"configurations": {"2d": {"patch_size": [64, 80], "use_mask_for_norm": [False]}}}
        ),
        encoding="utf-8",
    )


def _required(module: object, name: str):
    value = getattr(module, name, None)
    assert value is not None, f"expected {module!r} to expose {name}"
    return value


def test_alignment_profile_preserves_conservative_loader_and_runtime_defaults() -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    config = resolve("alignment", device="cuda:0")

    assert config.profile == "alignment"
    assert config.num_workers == 0
    assert config.pin_memory is False
    assert config.persistent_workers is False
    assert config.prefetch_factor is None
    assert config.non_blocking is False
    assert config.amp is False
    assert config.tf32 is False
    assert config.compile is False


def test_throughput_profile_resolves_safe_cuda_loader_defaults() -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    config = resolve("throughput", device="cuda:0")

    assert (config.num_workers, config.pin_memory) == (2, True)
    assert config.persistent_workers is True
    assert config.prefetch_factor == 2
    assert config.non_blocking is True
    assert config.amp is False
    assert config.tf32 is False
    assert config.compile is False


def test_explicit_cpu_pin_memory_does_not_enable_non_blocking_transfer() -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    config = resolve("alignment", device="cpu", pin_memory="on")

    assert config.pin_memory is True
    assert config.non_blocking is False


def test_loader_factory_applies_resolved_settings_to_train_and_validation() -> None:
    resolve = _required(formal_train, "resolve_performance_config")
    build_loaders = _required(formal_train, "build_formal_loaders")
    performance = resolve("throughput", device="cuda:0")

    train_loader, val_loader = build_loaders(
        _TinyDataset(),
        _TinyDataset(),
        performance=performance,
        batch_size=2,
    )

    for loader in (train_loader, val_loader):
        assert loader.num_workers == 2
        assert loader.pin_memory is True
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 2


def test_formal_epoch_helper_reuses_the_same_loaders_for_every_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    run_epochs = _required(formal_train, "run_formal_epochs")
    train_loader = object()
    val_loader = object()
    seen_train_loaders: list[object] = []
    seen_val_loaders: list[object] = []

    def fake_train_epoch(*args, **kwargs):
        seen_train_loaders.append(args[1])
        return TrainEpochResult(1, 0.0, ()) , 0.01

    def fake_validation(*args, **kwargs):
        seen_val_loaders.append(args[1])
        return ValidationEpochResult(1, 0.0, 0.0, 0.0)

    monkeypatch.setattr(formal_train, "run_formal_epoch", fake_train_epoch)
    monkeypatch.setattr(formal_train, "run_formal_validation", fake_validation)

    rows = list(
        run_epochs(
            model=object(),
            train_loader=train_loader,
            val_loader=val_loader,
            loss=object(),
            validation_loss=object(),
            optimizer=object(),
            scheduler=object(),
            device=torch.device("cpu"),
            start_epoch=0,
            end_epoch=3,
            schedule=object(),
            non_blocking=False,
        )
    )

    assert len(rows) == 3
    assert seen_train_loaders == [train_loader] * 3
    assert seen_val_loaders == [val_loader] * 3


def test_run_train_epoch_forwards_non_blocking_to_each_train_step(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_train_step(*args, non_blocking: bool = False, **kwargs):
        seen.append(non_blocking)
        return TrainStepResult(0.0, ())

    monkeypatch.setattr(trainer, "train_step", fake_train_step)

    result = trainer.run_train_epoch(
        model=nn.Identity(),
        batches=[(torch.zeros(1), torch.zeros(1))] * 2,
        loss_fn=nn.Identity(),
        optimizer=object(),
        device=torch.device("cpu"),
        non_blocking=True,
    )

    assert result.batch_count == 2
    assert seen == [True, True]


def test_formal_epoch_forwards_non_blocking_to_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_run_train_epoch(*args, non_blocking: bool = False, **kwargs):
        seen.append(non_blocking)
        return TrainEpochResult(1, 0.0, ())

    monkeypatch.setattr("standalone_nnunet2d.training.formal_trainer.run_train_epoch", fake_run_train_epoch)

    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = PolyLRScheduler(optimizer, 0.01, 1000)
    schedule = OfficialTrainerSchedule(num_iterations_per_epoch=1)

    run_formal_epoch(
        model,
        [(torch.zeros(1), torch.zeros(1))],
        nn.Identity(),
        optimizer,
        scheduler,
        torch.device("cuda:0"),
        0,
        schedule,
        non_blocking=True,
    )

    assert seen == [True]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"persistent_workers": True}, "persistent_workers.*num_workers"),
        ({"prefetch_factor": 2}, "prefetch_factor.*num_workers"),
        ({"num_workers": -1}, "num_workers"),
        ({"prefetch_factor": 0, "num_workers": 1}, "prefetch_factor"),
        ({"pin_memory": "invalid"}, "pin_memory"),
    ],
)
def test_performance_options_reject_invalid_combinations(kwargs: dict[str, object], message: str) -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    with pytest.raises(ValueError, match=message):
        resolve("alignment", device="cpu", **kwargs)


def test_parser_exposes_explicit_performance_profile_and_loader_options() -> None:
    build_parser = _required(formal_train, "build_parser")
    parser = build_parser()

    args = parser.parse_args(
        [
            "--raw-root",
            "raw",
                "--output-root",
                "out",
                "--plans",
                "plans.json",
                "--performance-profile",
            "throughput",
            "--num-workers",
            "3",
            "--pin-memory",
            "on",
            "--persistent-workers",
            "--prefetch-factor",
            "4",
        ]
    )

    assert args.performance_profile == "throughput"
    assert args.num_workers == 3
    assert args.pin_memory == "on"
    assert args.persistent_workers is True
    assert args.prefetch_factor == 4


def test_parser_exposes_alignment_report_paths_as_paths() -> None:
    parser = formal_train.build_parser()

    args = parser.parse_args(
        [
            "--raw-root",
            "raw",
            "--output-root",
            "out",
            "--plans",
            "plans.json",
            "--transform-parity-report",
            "transform.json",
            "--inference-parity-report",
            "inference.json",
        ]
    )

    assert args.transform_parity_report == Path("transform.json")
    assert args.inference_parity_report == Path("inference.json")


def test_build_formal_config_defaults_to_pending_without_alignment_evidence() -> None:
    config = formal_train.build_formal_config(
        fold=0,
        epochs=2,
        schedule=OfficialTrainerSchedule(),
    )

    assert config["run_type"] == "official_alignment_pending"
    assert config["run_state"] == "official_alignment_pending"
    assert "alignment_evidence" not in config


def test_build_formal_config_embeds_deep_copied_evidence_and_hashes_it(tmp_path: Path) -> None:
    transform_path, inference_path = _write_alignment_reports(tmp_path, suffix="_one")
    evidence = build_alignment_evidence(transform_path, inference_path)

    config = formal_train.build_formal_config(
        fold=0,
        epochs=2,
        schedule=OfficialTrainerSchedule(),
        alignment_evidence=evidence,
    )

    assert config["run_type"] == "official_aligned"
    assert config["run_state"] == "official_aligned"
    assert config["alignment_evidence"] == evidence
    assert config["alignment_evidence"] is not evidence
    evidence["sources"]["transform"]["snapshot"]["status"] = "tampered"
    assert config["alignment_evidence"]["sources"]["transform"]["snapshot"]["status"] == "passed"

    second_transform, second_inference = _write_alignment_reports(tmp_path, suffix="_two")
    second_evidence = build_alignment_evidence(second_transform, second_inference)
    second_config = formal_train.build_formal_config(
        fold=0,
        epochs=2,
        schedule=OfficialTrainerSchedule(),
        alignment_evidence=second_evidence,
    )
    assert config["plan_hash"] != second_config["plan_hash"]


def test_main_rejects_one_alignment_report_before_output_side_effect(tmp_path: Path) -> None:
    plans_path = tmp_path / "plans.json"
    _write_plans(plans_path)
    output_root = tmp_path / "output"

    with pytest.raises(SystemExit) as error:
        formal_train.main(
            [
                "--raw-root",
                str(tmp_path / "raw"),
                "--output-root",
                str(output_root),
                "--plans",
                str(plans_path),
                "--transform-parity-report",
                str(tmp_path / "transform.json"),
            ]
        )

    assert error.value.code == 2
    assert not output_root.exists()


def test_main_rejects_invalid_alignment_pair_before_output_side_effect(tmp_path: Path) -> None:
    plans_path = tmp_path / "plans.json"
    _write_plans(plans_path)
    transform_path, inference_path = _write_alignment_reports(tmp_path)
    inference_path.write_text("{invalid", encoding="utf-8")
    output_root = tmp_path / "output"

    with pytest.raises(SystemExit) as error:
        formal_train.main(
            [
                "--raw-root",
                str(tmp_path / "raw"),
                "--output-root",
                str(output_root),
                "--plans",
                str(plans_path),
                "--transform-parity-report",
                str(transform_path),
                "--inference-parity-report",
                str(inference_path),
            ]
        )

    assert error.value.code == 2
    assert not output_root.exists()


def test_main_valid_alignment_dry_run_reports_aligned_without_output_side_effect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plans_path = tmp_path / "plans.json"
    _write_plans(plans_path)
    transform_path, inference_path = _write_alignment_reports(tmp_path)
    output_root = tmp_path / "output"

    assert formal_train.main(
        [
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(output_root),
            "--plans",
            str(plans_path),
            "--transform-parity-report",
            str(transform_path),
            "--inference-parity-report",
            str(inference_path),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["run_type"] == "official_aligned"
    assert payload["config"]["run_state"] == "official_aligned"
    assert output_root.exists() is False


def test_parser_requires_plans_path() -> None:
    parser = formal_train.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--raw-root", "raw", "--output-root", "out"])


def test_load_2d_plan_config_reads_patch_and_mask_fields_only(tmp_path: Path) -> None:
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [64, 80], "use_mask_for_norm": [True]},
                    "3d_fullres": {"patch_size": [12, 384, 384], "use_mask_for_norm": [False]},
                }
            }
        ),
        encoding="utf-8",
    )

    assert formal_train.load_2d_plan_config(plans_path) == ((64, 80), (True,))


def test_load_2d_plan_config_fails_normally_when_required_field_is_missing(tmp_path: Path) -> None:
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps({"configurations": {"2d": {"patch_size": [64, 80]}}}),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        formal_train.load_2d_plan_config(plans_path)


def test_formal_dataset_builder_passes_plan_config_to_train_and_validation() -> None:
    seen: list[dict[str, object]] = []

    class RecordingDataset:
        def __init__(self, raw_root: Path, **kwargs: object) -> None:
            seen.append({"raw_root": raw_root, **kwargs})

    original = formal_train.FormalPatchDataset
    formal_train.FormalPatchDataset = RecordingDataset  # type: ignore[assignment]
    try:
        train, validation = formal_train.build_formal_datasets(
            Path("raw"), fold=2, patch_size=(64, 80), use_mask_for_norm=(True,)
        )
    finally:
        formal_train.FormalPatchDataset = original

    assert train is not None and validation is not None
    assert seen == [
        {
            "raw_root": Path("raw"),
            "fold": 2,
            "split": "train",
            "patch_size": (64, 80),
            "use_mask_for_norm": (True,),
            "augment": True,
        },
        {
            "raw_root": Path("raw"),
            "fold": 2,
            "split": "val",
            "patch_size": (64, 80),
            "use_mask_for_norm": (True,),
            "augment": False,
            "oversample_foreground_percent": 0.0,
        },
    ]


def test_resolved_config_records_profile_loader_settings_and_disabled_optimizations() -> None:
    resolve = _required(formal_train, "resolve_performance_config")
    performance = resolve("alignment", device="cuda:0")

    config = formal_train.build_formal_config(
        fold=0,
        epochs=2,
        schedule=OfficialTrainerSchedule(),
        performance=performance,
    )

    assert config["performance_profile"] == "alignment"
    assert config["performance"]["profile"] == "alignment"
    assert config["performance"]["loader"] == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "non_blocking": False,
    }
    assert config["performance"]["optimizations"] == {
        "amp": False,
        "tf32": False,
        "compile": False,
    }


def test_public_transfer_signatures_expose_non_blocking_keyword() -> None:
    assert "non_blocking" in inspect.signature(trainer.train_step).parameters
    assert "non_blocking" in inspect.signature(trainer.run_train_epoch).parameters
