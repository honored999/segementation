"""Tiny CPU regressions using nnU-Net's actual external-import cleanup; no models/data."""
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trainer_source_run", ROOT / "run_experiment.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class TrainerSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "trainers"
        self.folder.mkdir()
        self.name = "SyntheticTrainer_" + uuid.uuid4().hex
        self.source = self.folder / (self.name + ".py")
        self.source.write_text(
            "class Parent:\n    def __init__(self): pass\n"
            f"class {self.name}(Parent):\n    \"\"\"Composition with no own methods.\"\"\"\n",
            encoding="utf-8",
        )
        self.addCleanup(sys.modules.pop, self.name, None)
        root_patch = patch.object(run, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def find(self):
        return recursive_find_python_class(
            str(self.folder), self.name, None, cleanup_imports_from_base_folder=True,
        )

    def preimport(self):
        with patch.object(sys, "path", [str(self.folder)] + sys.path):
            return importlib.import_module(self.name)

    def test_legacy_getfile_fails_after_actual_cleanup(self):
        cls = self.find()
        self.assertNotIn(self.name, sys.modules)
        self.assertNotIn("__init__", vars(cls))
        with self.assertRaises(TypeError):
            inspect.getfile(cls)

    def test_preimport_survives_cleanup_and_resolves_exact_source(self):
        retained = self.preimport()
        cls = self.find()
        self.assertIs(sys.modules[self.name], retained)
        self.assertIs(cls, getattr(retained, self.name))
        self.assertNotIn("__init__", vars(cls))
        self.assertEqual(
            run.verify_trainer_source(cls(), retained, self.name), self.source.resolve(),
        )

    def test_wrong_module_path_rejected(self):
        retained = self.preimport()
        cls = self.find()
        with patch.object(retained, "__file__", str(self.root / "other" / self.source.name)):
            with self.assertRaisesRegex(ValueError, "Unexpected Trainer source"):
                run.verify_trainer_source(cls(), retained, self.name)

    def test_same_name_different_class_identity_rejected(self):
        retained = self.preimport()
        expected = self.find()
        other = type(expected.__name__, (), {"__module__": expected.__module__})
        with self.assertRaisesRegex(ValueError, "Unexpected Trainer class"):
            run.verify_trainer_source(other(), retained, self.name)

    def test_subclass_identity_rejected(self):
        retained = self.preimport()
        subclass = type("OtherTrainer", (self.find(),), {})
        with self.assertRaisesRegex(ValueError, "Unexpected Trainer class"):
            run.verify_trainer_source(subclass(), retained, self.name)


if __name__ == "__main__":
    unittest.main()
