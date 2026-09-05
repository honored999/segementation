"""Regression tests for TorchScript NVAUTO weight inspection."""

from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "inspect_nvauto_weights.py"
SPEC = importlib.util.spec_from_file_location("inspect_nvauto_weights", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestTorchScriptInspection(unittest.TestCase):
    def test_finds_torchscript_files_in_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_dir = Path(temp_dir) / "weights" / "NVAUTO"
            nested_path = weights_dir / "ts" / "model0.ts"
            nested_path.parent.mkdir(parents=True)
            nested_path.touch()

            paths = MODULE.find_torchscript_files(weights_dir)

        self.assertEqual(paths, [nested_path])

    def test_inspects_script_module_state_and_forward_shapes(self) -> None:
        model = torch.nn.Conv3d(2, 1, kernel_size=1).eval()
        example = torch.zeros(1, 2, 16, 128, 128)
        scripted = torch.jit.trace(model, example)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nvauto.ts"
            torch.jit.save(scripted, path)
            output = io.StringIO()
            with redirect_stdout(output):
                state = MODULE.inspect_torchscript(path)

        self.assertIn("ScriptModule", output.getvalue())
        self.assertIn("state_dict keys: 2", output.getvalue())
        self.assertIn("forward [1, 2, 16, 128, 128]: output", output.getvalue())
        self.assertEqual(set(state), {"weight", "bias"})


if __name__ == "__main__":
    unittest.main()
