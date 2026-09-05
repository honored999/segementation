import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from filter_ci1_mask_ratio_dataset import filter_manifest_by_mask_ratio


class FilterCI1MaskRatioDatasetTest(unittest.TestCase):
    def test_filter_manifest_keeps_rows_at_or_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            manifest_path = root / "manifest.csv"
            output_manifest = root / "filtered" / "manifest.csv"

            fieldnames = ["patient", "image_path", "mask_path", "mask_area", "has_mask"]
            rows = []
            for index, mask_area in enumerate([9, 10, 25], start=1):
                image_path = image_dir / f"image_{index}.png"
                mask_path = mask_dir / f"mask_{index}.png"
                Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(image_path)
                Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(mask_path)
                rows.append(
                    {
                        "patient": f"p{index}",
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "mask_area": str(mask_area),
                        "has_mask": "1" if mask_area else "0",
                    }
                )

            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            kept_rows = filter_manifest_by_mask_ratio(
                input_manifest=manifest_path,
                output_manifest=output_manifest,
                min_mask_ratio=0.10,
                copy_files=False,
            )

            self.assertEqual([row["patient"] for row in kept_rows], ["p2", "p3"])
            self.assertTrue(output_manifest.exists())

            with output_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                output_rows = list(csv.DictReader(handle))

            self.assertEqual(len(output_rows), 2)
            self.assertEqual(output_rows[0]["mask_ratio"], "0.1")
            self.assertEqual(output_rows[1]["mask_ratio"], "0.25")


if __name__ == "__main__":
    unittest.main()
