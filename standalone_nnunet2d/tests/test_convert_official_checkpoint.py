from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch


RULES = {
    "encoder_stages.2.blocks.3.0.param": "encoder.stages.2.0.convs.3.conv.param",
    "encoder_stages.2.blocks.3.1.param": "encoder.stages.2.0.convs.3.norm.param",
    "decoder_stages.4.blocks.1.0.param": "decoder.stages.4.convs.1.conv.param",
    "decoder_stages.4.blocks.1.1.param": "decoder.stages.4.convs.1.norm.param",
    "transposed_convolutions.5.param": "decoder.transpconvs.5.param",
    "segmentation_heads.6.param": "decoder.seg_layers.6.param",
}


def _converter():
    try:
        return importlib.import_module("standalone_nnunet2d.tools.convert_official_checkpoint")
    except ModuleNotFoundError as error:
        pytest.fail(f"converter module is missing: {error}")


def _target_contract() -> dict[str, torch.Tensor]:
    return {target_key: torch.full((1,), index, dtype=torch.float32) for index, target_key in enumerate(RULES, 1)}


def _official_weights() -> dict[str, torch.Tensor]:
    return {source_key: torch.full((1,), index, dtype=torch.float32) for index, source_key in enumerate(RULES.values(), 1)}


@pytest.fixture(scope="module")
def real_contract() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    converter = _converter()
    target = dict(converter.get_target_state_dict())
    official = {
        converter.target_to_source_key(target_key): value.detach().clone()
        for target_key, value in target.items()
    }
    official["encoder.stages.0.0.convs.0.all_modules.0.weight"] = next(iter(target.values())).detach().clone()
    return target, official


def test_semantic_mapper_uses_target_to_source_rules() -> None:
    converter = _converter()

    mapped = {target_key: converter.target_to_source_key(target_key) for target_key in RULES}

    assert mapped == RULES


def test_semantic_mapper_rejects_unsupported_target_name() -> None:
    with pytest.raises(ValueError, match="unsupported target"):
        _converter().target_to_source_key("optimizer.param")


def test_mapping_copies_synthetic_contract_from_official_tensors() -> None:
    target = _target_contract()

    mapped = _converter().map_official_weights(_official_weights(), target)

    assert set(mapped) == set(target)
    for target_key, value in target.items():
        assert torch.equal(mapped[target_key], value)


def test_mapping_rejects_missing_source_and_shape_mismatch() -> None:
    converter = _converter()
    target = _target_contract()
    official = _official_weights()
    missing_source = next(iter(official))
    official.pop(missing_source)

    with pytest.raises(KeyError, match="missing source"):
        converter.map_official_weights(official, target)

    official = _official_weights()
    official[next(iter(official))] = torch.zeros(2)
    with pytest.raises(ValueError, match="shape mismatch"):
        converter.map_official_weights(official, target)


def test_mapping_rejects_duplicate_semantic_source_key(monkeypatch: pytest.MonkeyPatch) -> None:
    converter = _converter()
    target = {
        "first.weight": torch.zeros(1),
        "second.weight": torch.zeros(1),
    }
    official = {"duplicate.source": torch.zeros(1)}
    monkeypatch.setattr(converter, "target_to_source_key", lambda _target_key: "duplicate.source")

    with pytest.raises(ValueError, match="unique"):
        converter.map_official_weights(official, target)


def test_real_target_contract_has_148_tensors_and_expected_groups(
    real_contract: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
) -> None:
    target, _ = real_contract

    assert len(target) == 148
    assert Counter(key.split(".", 1)[0] for key in target) == Counter(
        {
            "encoder_stages": 64,
            "decoder_stages": 56,
            "transposed_convolutions": 14,
            "segmentation_heads": 14,
        }
    )


def test_real_semantic_mapping_strict_loads_plain_conv_unet(
    real_contract: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
) -> None:
    converter = _converter()
    target, official = real_contract

    mapped = converter.map_official_weights(official, target)
    converter.validate_mapped_state_dict(mapped)

    assert len({converter.target_to_source_key(key) for key in target}) == len(target)


def test_converted_checkpoint_is_readable_by_predict_checkpoint_reader(
    tmp_path: Path,
    real_contract: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    converter = _converter()
    target, official = real_contract
    official_path = tmp_path / "official-real.pth"
    output_path = tmp_path / "standalone-real.pth"
    torch.save({"network_weights": official}, official_path)

    converter.convert_checkpoint(official_path, output_path, fold=0)

    predict = importlib.import_module("standalone_nnunet2d.predict")
    state_dict, metadata = predict._read_checkpoint(output_path)
    assert set(state_dict) == set(target)
    assert metadata["run_state"] == "official_alignment_pending"


def test_read_network_weights_uses_weights_only_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converter = _converter()
    official_path = tmp_path / "official.pth"
    torch.save({"network_weights": _official_weights()}, official_path)
    original_load = converter.torch.load
    observed: dict[str, object] = {}

    def load(*args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(converter.torch, "load", load)

    converter._read_network_weights(official_path)

    assert observed["weights_only"] is True


def test_read_network_weights_loads_numpy_scalar_metadata_in_weights_only_mode(tmp_path: Path) -> None:
    converter = _converter()
    official_path = tmp_path / "official-with-numpy-metadata.pth"
    torch.save(
        {
            "network_weights": _official_weights(),
            "metadata": {
                "score": np.float64(0.5),
                "float32_score": np.float32(0.5),
            },
        },
        official_path,
    )

    loaded = converter._read_network_weights(official_path)

    assert set(loaded) == set(_official_weights())


def test_conversion_rejects_target_contract_with_unexpected_tensor_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = _converter()
    target = _target_contract()
    official_path = tmp_path / "official-fold-3.pth"
    output_path = tmp_path / "standalone-fold-3.pth"
    torch.save({"network_weights": _official_weights()}, official_path)
    monkeypatch.setattr(converter, "get_target_state_dict", lambda: target)

    with pytest.raises(ValueError, match="148"):
        converter.convert_checkpoint(official_path, output_path, fold=3)

    assert not output_path.exists()


def test_conversion_writes_pending_minimal_checkpoint_with_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converter = _converter()
    target = _target_contract()
    official_path = tmp_path / "official-fold-3.pth"
    output_path = tmp_path / "standalone-fold-3.pth"
    torch.save(
        {
            "network_weights": _official_weights(),
            "optimizer": {"must_not": "be copied"},
        },
        official_path,
    )
    monkeypatch.setattr(converter, "EXPECTED_TENSOR_COUNT", len(target))
    monkeypatch.setattr(converter, "get_target_state_dict", lambda: target)
    validated_calls: list[set[str]] = []

    def validate_mapped_state_dict(mapped: dict[str, torch.Tensor]) -> None:
        validated_calls.append(set(mapped))

    monkeypatch.setattr(converter, "validate_mapped_state_dict", validate_mapped_state_dict)

    converter.convert_checkpoint(official_path, output_path, fold=3)

    assert validated_calls == [set(target)]
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1
    assert set(payload["model_state_dict"]) == set(target)
    for target_key, value in target.items():
        assert torch.equal(payload["model_state_dict"][target_key], value)
    assert payload["metadata"]["run_state"] == "official_alignment_pending"
    assert payload["metadata"]["fold"] == 3
    assert payload["metadata"]["source_sha256"] == hashlib.sha256(official_path.read_bytes()).hexdigest()
    assert payload["metadata"]["mapping_policy"] == "semantic_name_v1"
    assert "optimizer" not in payload


def test_conversion_does_not_write_when_mapped_state_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = _converter()
    target = _target_contract()
    official_path = tmp_path / "official-fold-3.pth"
    output_path = tmp_path / "standalone-fold-3.pth"
    torch.save({"network_weights": _official_weights()}, official_path)
    monkeypatch.setattr(converter, "EXPECTED_TENSOR_COUNT", len(target))
    monkeypatch.setattr(converter, "get_target_state_dict", lambda: target)

    def validate_mapped_state_dict(mapped: dict[str, torch.Tensor]) -> None:
        raise RuntimeError("mapped state_dict validation failed")

    monkeypatch.setattr(converter, "validate_mapped_state_dict", validate_mapped_state_dict)

    with pytest.raises(RuntimeError, match="validation failed"):
        converter.convert_checkpoint(official_path, output_path, fold=3)

    assert not output_path.exists()


def test_conversion_rejects_same_resolved_input_and_output_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = _converter()
    target = _target_contract()
    official_path = tmp_path / "official-fold-3.pth"
    torch.save({"network_weights": _official_weights()}, official_path)
    original_bytes = official_path.read_bytes()
    monkeypatch.setattr(converter, "EXPECTED_TENSOR_COUNT", len(target))
    monkeypatch.setattr(converter, "get_target_state_dict", lambda: target)

    with pytest.raises(ValueError, match="same.*path"):
        converter.convert_checkpoint(official_path, official_path, fold=3)

    assert official_path.read_bytes() == original_bytes


def test_cli_prints_json_and_writes_pending_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    converter = _converter()
    target = _target_contract()
    official_path = tmp_path / "official-fold-3.pth"
    output_path = tmp_path / "nested" / "standalone-fold-3.pth"
    torch.save(
        {
            "network_weights": _official_weights(),
            "optimizer": {"must_not": "be copied"},
        },
        official_path,
    )
    monkeypatch.setattr(converter, "EXPECTED_TENSOR_COUNT", len(target))
    monkeypatch.setattr(converter, "get_target_state_dict", lambda: target)
    monkeypatch.setattr(converter, "validate_mapped_state_dict", lambda _mapped: None)

    assert converter.main(
        [
            "--official-checkpoint",
            str(official_path),
            "--output",
            str(output_path),
            "--fold",
            "3",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "mapped_count": len(target),
        "output": str(output_path.resolve()),
        "run_state": "official_alignment_pending",
    }
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert payload["metadata"] == {
        "fold": 3,
        "mapping_policy": "semantic_name_v1",
        "run_state": "official_alignment_pending",
        "run_type": "official_alignment_pending",
        "source_checkpoint": str(official_path.resolve()),
        "source_format": "nnunetv2_network_weights",
        "source_sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
    }
    assert "optimizer" not in payload
