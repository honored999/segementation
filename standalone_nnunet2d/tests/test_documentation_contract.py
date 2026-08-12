from __future__ import annotations

import re
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    "standalone_nnunet2d/README.md",
    "standalone_nnunet2d/REPRODUCTION_NOTES.md",
)


def _states_smoke_and_online_validation_are_not_official_reproduction(
    text: str,
) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower())
    return bool(
        re.search(
            r"(?:smoke[^.]{0,120}online validation|"
            r"online validation[^.]{0,120}smoke)"
            r"[^.]{0,120}(?:not|never|exclude|excluded|does not|is not|are not)"
            r"[^.]{0,120}official reproduction",
            normalized,
        )
    )


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_declares_pending_server_parity_and_non_official_validation(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    text = document.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text.lower())

    required_phrases = (
        "official_alignment_pending",
        "parity report",
        "server oracle capture",
    )
    missing_phrases = [
        phrase for phrase in required_phrases if phrase not in normalized
    ]
    assert not missing_phrases, f"{relative_path} is missing: {missing_phrases}"
    assert "smoke" in normalized
    assert "online validation" in normalized
    assert _states_smoke_and_online_validation_are_not_official_reproduction(text)


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_records_repeat_oracle_inference_gate(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    normalized = re.sub(r"\s+", " ", document.read_text(encoding="utf-8").lower())

    required_phrases = (
        "repeat_oracle_stability_v1",
        "at least three independent oracle runs",
        "the stable voxel exact rule",
        "every stable voxel must match the unanimous official label exactly",
        "an unstable voxel only accepts labels observed across oracle repeats",
        "report all unstable coordinates and pairwise differences",
        "the report remains `official_alignment_pending`",
        "does not automatically relabel historical runs `official_aligned`",
    )
    missing_phrases = [
        phrase for phrase in required_phrases if phrase not in normalized
    ]
    assert not missing_phrases, f"{relative_path} is missing: {missing_phrases}"


def _parity_command_blocks(text: str) -> tuple[str, ...]:
    """Return each documented parity command, including its continuation lines."""
    return tuple(
        match.group(0).strip()
        for match in re.finditer(
            r"(?ms)^conda run[^\n]*standalone_nnunet2d\.tools\.parity_report.*?"
            r"(?=^conda run|\n```|\Z)",
            text,
        )
    )


def _only_command(commands: tuple[str, ...], output_name: str) -> str:
    matches = [command for command in commands if output_name in command]
    assert len(matches) == 1, f"expected one {output_name} command, got {matches}"
    return matches[0]


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"(?ms)^## {re.escape(heading)}\s*$.*?(?=^## |\Z)",
        text,
    )
    assert match is not None, f"missing section: {heading}"
    return match.group(0)


def _powershell_command_blocks(text: str) -> tuple[str, ...]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("conda run"):
            if current:
                blocks.append("\n".join(current))
            current = [stripped]
        elif current and (not stripped or stripped.startswith("--")):
            current.append(stripped)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return tuple(blocks)


def test_reproduction_notes_marks_legacy_pending_sections_and_requires_plans(
    ) -> None:
    document = REPOSITORY_ROOT / "standalone_nnunet2d/REPRODUCTION_NOTES.md"
    text = document.read_text(encoding="utf-8")

    section_headings = (
        "Fold-0 formal train, prediction, and validation",
        "Five-fold training and OOF sequence",
    )
    for heading in section_headings:
        section = _section(text, heading).lower()
        assert "legacy pending" in section
        assert "diagnostic only" in section
        assert "must not be used as the final `official_aligned` workflow" in section
        assert "official_alignment_pending" in section

        train_commands = tuple(
            block
            for block in _powershell_command_blocks(section)
            if "formal_train.py" in block
        )
        assert train_commands, f"{heading} has no pending training command"
        assert all("--plans" in command for command in train_commands)


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documented_parity_examples_use_strict_zero_image_atol(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    commands = _parity_command_blocks(document.read_text(encoding="utf-8"))

    assert commands, f"{relative_path} has no documented parity command"
    assert all("--image-atol 1e-6" not in command for command in commands)
    assert all("--image-atol 0" in command for command in commands)

    transform = _only_command(commands, "transform_parity_case005_v3.json")
    assert transform.count("--oracle-root") == 1

    single_root_inference = _only_command(commands, "inference_parity_report.json")
    assert single_root_inference.count("--oracle-root") == 1

    repeated_inference = _only_command(
        commands,
        "inference_repeat_parity_report.json",
    )
    assert repeated_inference.count("--oracle-root") >= 3


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_requires_repeat_gate_for_alignment_promotion(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    normalized = re.sub(r"\s+", " ", document.read_text(encoding="utf-8").lower())

    assert "passed inference parity report" not in normalized
    assert "passed inference parity reports" not in normalized
    required_phrases = (
        "passed `repeat_oracle_stability_v1` report",
        "single-root inference",
        "diagnostic",
        "inference_context",
        "must be recaptured",
    )
    missing_phrases = [
        phrase for phrase in required_phrases if phrase not in normalized
    ]
    assert not missing_phrases, f"{relative_path} is missing: {missing_phrases}"


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_requires_explicit_training_evidence_promotion(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    normalized = re.sub(r"\s+", " ", document.read_text(encoding="utf-8").lower())

    required_phrases = (
        "only when both `--transform-parity-report` and `--inference-parity-report`",
        "omitting both reports leaves the run `official_alignment_pending`",
        "epoch 999/fold 0 is not retroactively upgraded",
        "synchronize the current final source",
        "do not copy `outputs/`",
        "use a fresh output root",
        "parity reports themselves remain `official_alignment_pending`",
        "only newly generated artifacts may be labeled `official_aligned`",
    )
    missing_phrases = [
        phrase for phrase in required_phrases if phrase not in normalized
    ]
    assert not missing_phrases, f"{relative_path} is missing: {missing_phrases}"


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_records_final_windows_paths(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    normalized = re.sub(r"\s+", " ", document.read_text(encoding="utf-8").lower())

    required_paths = (
        r"c:\lijialin\models3d\nnunet\nnunet_raw\dataset501_strokelesion",
        r"c:\lijialin\models3d\nnunet\nnunet_preprocessed\dataset501_strokelesion\nnunetplans.json",
        r"c:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json",
        r"c:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json",
        r"c:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold",
    )
    missing_paths = [path for path in required_paths if path not in normalized]
    assert not missing_paths, f"{relative_path} is missing: {missing_paths}"


def _formal_train_command_lines(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in text.splitlines()
        if "python standalone_nnunet2d\\formal_train.py" in line
        and line.strip().startswith("conda run")
        and "official_aligned_5fold" in line
    )


def _fold_validation_command_lines(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in text.splitlines()
        if "validate_cv.py fold" in line
        and line.strip().startswith("conda run")
        and "official_aligned_5fold" in line
    )


def _commands_for_fold(commands: tuple[str, ...], fold: int) -> tuple[str, ...]:
    return tuple(
        command
        for command in commands
        if re.search(rf"--fold\s+{fold}(?=\s|$)", command.lower())
    )


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_has_aligned_commands_for_all_folds(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    text = document.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text.lower())
    train_commands = _formal_train_command_lines(text)
    validation_commands = _fold_validation_command_lines(text)

    assert len(train_commands) == 5
    assert len(validation_commands) == 5
    for fold in range(5):
        fold_train_commands = _commands_for_fold(train_commands, fold)
        fold_validation_commands = _commands_for_fold(validation_commands, fold)
        assert len(fold_train_commands) == 1
        assert len(fold_validation_commands) == 1

        train_command = fold_train_commands[0].lower()
        validation_command = fold_validation_commands[0].lower()
        assert "--output-root" in train_command
        assert f"official_aligned_5fold\\formal\\fold_{fold}" in train_command
        assert "--checkpoint" in validation_command
        assert (
            f"official_aligned_5fold\\formal\\fold_{fold}\\checkpoint"
            in validation_command
        )
    for command in train_commands:
        lowered = command.lower()
        assert "--plans" in lowered
        assert "--transform-parity-report" in lowered
        assert "--inference-parity-report" in lowered
        assert "--performance-profile throughput" in lowered
        assert "--epochs 1000" in lowered
        assert "--confirm-run" in lowered
        assert "--device cuda:0" in lowered
        assert "official_aligned_5fold" in lowered
    for command in validation_commands:
        lowered = command.lower()
        assert "official_aligned_5fold\\crossval" in lowered
        assert "--allow-pending" not in lowered

    assert "batch size is fixed at 12" in normalized
    assert "throughput profile does not enable amp, tf32, or compile" in normalized
    assert "does not require a separate `predict.py` run" in normalized
    assert "full-volume prediction" in normalized
    assert "prediction manifest" in normalized
    assert "fold reports share one `crossval` directory" in normalized


@pytest.mark.parametrize("relative_path", DOCUMENTS)
def test_documentation_contract_requires_strict_aligned_oof_promotion_and_output_check(
    relative_path: str,
) -> None:
    document = REPOSITORY_ROOT / relative_path
    normalized = re.sub(r"\s+", " ", document.read_text(encoding="utf-8").lower())

    required_phrases = (
        "aggregate requires fold_0_report.json through fold_4_report.json",
        "95 unique ids",
        "zero failed cases",
        "identical evidence",
        "writes an `official_aligned` `oof_summary.json`",
        "type oof_summary.json",
        "smoke and online validation are never official",
    )
    missing_phrases = [
        phrase for phrase in required_phrases if phrase not in normalized
    ]
    assert not missing_phrases, f"{relative_path} is missing: {missing_phrases}"
