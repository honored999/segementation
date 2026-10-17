import numpy as np
import pytest

from experiments.genseg3mm.cli import parser, select_preview_slices


def test_cli_exposes_only_prepare_train_preview_and_requires_physical_gpu():
    root = parser()
    subparsers = next(action for action in root._actions if action.dest == "command")
    assert set(subparsers.choices) == {"prepare", "train", "preview"}
    common = ["--source", "source", "--raw-root", "raw", "--baseline-run", "baseline", "--gpu", "1"]
    args = root.parse_args(["prepare", *common, "--output", "targets"])
    assert args.gpu == 1 and args.output.name == "targets"
    args = root.parse_args(["train", *common, "--targets", "targets", "--targets-sha256", "a" * 64,
                            "--output", "run"])
    assert args.output.name == "run"
    args = root.parse_args(["preview", *common, "--targets", "targets", "--targets-sha256", "a" * 64,
                            "--run", "run", "--output", "preview"])
    assert args.run.name == "run"
    with pytest.raises(SystemExit):
        root.parse_args(["train", "--source", "source", "--raw-root", "raw",
                         "--baseline-run", "baseline", "--targets", "targets",
                         "--targets-sha256", "a" * 64, "--output", "run"])


def test_preview_selects_two_distinct_background_planes_and_a_lesion_plane():
    train = ["p001", "p002", "p003", "p004", "p005"]
    labels = {}
    for patient in train[:4]:
        seg = np.zeros((1, 5, 4, 4), dtype=np.int16)
        seg[0, 3, 1, 1] = 1
        seg[0, 4, 0, 0] = -1
        labels[patient] = seg
    selected = select_preview_slices(train, labels)
    assert [(row["patient"], row["z"]) for row in selected] == [
        (patient, z) for patient in train[:4] for z in (0, 1, 3)
    ]
    assert [row["has_lesion"] for row in selected[:3]] == [False, False, True]
    assert all(row["patient"] not in train[4:] for row in selected)
