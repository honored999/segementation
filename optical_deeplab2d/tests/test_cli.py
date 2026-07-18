from pathlib import Path
import subprocess
import sys

import yaml

from optical_deeplab2d.models.electronic_deepseg_decoder import ElectronicDeepSegDecoder
from optical_deeplab2d.models.electronic_densenet_deepseg_decoder import (
    ElectronicDenseNetDeepSegDecoder,
)

ROOT = Path(__file__).parents[2]

def test_direct_train_script_exposes_help() -> None:
    result = subprocess.run([sys.executable, "optical_deeplab2d/train.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_electronic_deepseg_decoder_smoke_config() -> None:
    config_path = ROOT / "optical_deeplab2d/configs/electronic_deepseg_decoder_6gb_smoke.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["model"]["type"] == "electronic_deepseg_decoder"
    assert config["model"]["encoder_name"] == "mobilenet_v2"
    assert config["model"]["encoder_weights"] is None


def test_train_model_types_selects_electronic_deepseg_decoder() -> None:
    from optical_deeplab2d import train

    assert train.MODEL_TYPES["electronic_deepseg_decoder"] is ElectronicDeepSegDecoder


def test_densenet121_deepseg_no_aspp_smoke_config() -> None:
    config_path = (
        ROOT
        / "optical_deeplab2d/configs/"
        "electronic_densenet121_deepseg_no_aspp_6gb_smoke.yaml"
    )
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["model"] == {
        "type": "electronic_densenet121_deepseg_no_aspp",
        "encoder_name": "densenet121",
        "encoder_weights": "imagenet",
    }


def test_train_model_types_selects_densenet_deepseg_without_aspp() -> None:
    from optical_deeplab2d import train

    assert (
        train.MODEL_TYPES["electronic_densenet121_deepseg_no_aspp"]
        is ElectronicDenseNetDeepSegDecoder
    )


def test_train_script_declares_live_progress_contract() -> None:
    import pathlib

    train_source = (pathlib.Path(__file__).resolve().parents[1] / "train.py").read_text(
        encoding="utf-8"
    )

    assert "from tqdm.auto import tqdm" in train_source
    assert "build_batch_postfix" in train_source
    assert "format_epoch_summary" in train_source
    assert "leave=False" in train_source
    assert "complete_epoch_timing" in train_source
    assert "logged_epoch_seconds=time.time()-began" in train_source
    assert "'epoch_time':logged_epoch_seconds" in train_source
    assert (
        "progress.set_postfix(build_batch_postfix(value.item(),running_loss_total/batch_index,gpu_mib,batch_eta_seconds),refresh=False)"
        in train_source
    )
    assert (
        "print(format_epoch_summary(epoch+1,cfg['training']['epochs'],logged_epoch_seconds"
        in train_source
    )
    assert train_source.index("logged_epoch_seconds=time.time()-began") > train_source.index(
        "else:\n   stale+=1"
    )
    assert train_source.index("completed_epoch_seconds,total_eta_seconds=complete_epoch_timing") > train_source.index(
        "else:\n   stale+=1"
    )
