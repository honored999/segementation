import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from train_ci1_dwi_student_noskip_32ch import (
    CI1DwiSliceDataset,
    CI1DwiTensorCacheDataset,
    split_manifest_rows_by_patient,
)


class CI1DwiTrainingDataTest(unittest.TestCase):
    def test_split_manifest_rows_by_patient_has_no_patient_overlap(self):
        rows = [
            {"patient": "p1", "timepoint": "D1"},
            {"patient": "p1", "timepoint": "D2"},
            {"patient": "p2", "timepoint": "D1"},
            {"patient": "p3", "timepoint": "D1"},
            {"patient": "p4", "timepoint": "D1"},
        ]

        train_rows, val_rows = split_manifest_rows_by_patient(
            rows,
            train_split=0.5,
            seed=7,
        )

        train_patients = {row["patient"] for row in train_rows}
        val_patients = {row["patient"] for row in val_rows}
        self.assertTrue(train_patients)
        self.assertTrue(val_patients)
        self.assertFalse(train_patients & val_patients)

    def test_dataset_returns_single_channel_image_and_binary_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            manifest_path = root / "manifest.csv"

            Image.fromarray(np.full((4, 5), 128, dtype=np.uint8)).save(image_path)
            Image.fromarray(
                np.array(
                    [
                        [0, 0, 255, 0, 0],
                        [0, 255, 255, 0, 0],
                        [0, 0, 0, 0, 0],
                        [255, 0, 0, 0, 0],
                    ],
                    dtype=np.uint8,
                )
            ).save(mask_path)

            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "patient",
                        "timepoint",
                        "image_path",
                        "mask_path",
                        "has_mask",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patient": "p1",
                        "timepoint": "D1",
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "has_mask": "1",
                    }
                )

            dataset = CI1DwiSliceDataset(
                manifest_path=manifest_path,
                rows=None,
                image_height=8,
                image_width=10,
            )

            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (1, 8, 10))
            self.assertEqual(tuple(mask.shape), (1, 8, 10))
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
            self.assertTrue(set(mask.unique().tolist()).issubset({0.0, 1.0}))

    def test_tensor_cache_dataset_returns_cached_tensors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_path = root / "sample.pt"
            manifest_path = root / "cache_manifest.csv"

            import torch

            torch.save(
                {
                    "image": torch.full((1, 6, 7), 0.5, dtype=torch.float32),
                    "mask": torch.zeros((1, 6, 7), dtype=torch.float32),
                },
                cache_path,
            )

            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["patient", "timepoint", "tensor_path", "has_mask"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patient": "p1",
                        "timepoint": "D1",
                        "tensor_path": str(cache_path),
                        "has_mask": "0",
                    }
                )

            dataset = CI1DwiTensorCacheDataset(manifest_path=manifest_path, rows=None)
            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (1, 6, 7))
            self.assertEqual(tuple(mask.shape), (1, 6, 7))
            self.assertEqual(float(image.mean()), 0.5)
            self.assertEqual(float(mask.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
