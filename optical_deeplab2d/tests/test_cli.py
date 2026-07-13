from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[2]

def test_direct_train_script_exposes_help() -> None:
    result = subprocess.run([sys.executable, "optical_deeplab2d/train.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
