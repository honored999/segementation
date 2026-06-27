import csv
import tempfile
import unittest
from pathlib import Path

import torch

from audit_ci1_dwi_overlays import build_overlay_audit


class CI1DwiOverlayAuditTest(unittest.TestCase):
    def test_build_overlay_audit_writes_positive_sample_pngs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sample_path = root / "sample.pt"
            manifest_path = root / "cache_manifest.csv"
            output_dir = root / "audit"

            image = torch.zeros((1, 8, 8), dtype=torch.float32)
            image[:, 2:6, 2:6] = 0.5
            mask = torch.zeros((1, 8, 8), dtype=torch.float32)
            mask[:, 3:5, 3:5] = 1.0
            torch.save({"image": image, "mask": mask}, sample_path)

            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "patient",
                        "timepoint",
                        "slice_index",
                        "has_mask",
                        "tensor_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patient": "p1",
                        "timepoint": "DWI",
                        "slice_index": "3",
                        "has_mask": "1",
                        "tensor_path": str(sample_path),
                    }
                )

            outputs = build_overlay_audit(
                manifest_path=manifest_path,
                output_dir=output_dir,
                max_samples=1,
                seed=1,
            )

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].exists())
            self.assertGreater(outputs[0].stat().st_size, 0)
            self.assertTrue((output_dir / "overlay_grid.png").exists())


if __name__ == "__main__":
    unittest.main()
