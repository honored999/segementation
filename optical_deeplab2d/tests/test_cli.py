from pathlib import Path
import subprocess
import sys

import yaml

from optical_deeplab2d.models.electronic_deepseg_decoder import ElectronicDeepSegDecoder

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
