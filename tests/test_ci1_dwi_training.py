import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from train_ci1_dwi_student_noskip_32ch import (
    BCEDiceLoss,
    CI1DwiSliceDataset,
    CI1DwiTensorCacheDataset,
    OpticalElectronicCI1DwiNoSkip32,
    calculate_binary_metrics,
    configure_torch_threads,
    split_manifest_rows_by_patient,
    visualize_predictions,
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

    def test_configure_torch_threads_limits_cpu_thread_pools(self):
        script = (
            "import torch;"
            "from train_ci1_dwi_student_noskip_32ch import configure_torch_threads;"
            "configure_torch_threads(torch_threads=1, torch_interop_threads=1);"
            "print(torch.get_num_threads(), torch.get_num_interop_threads())"
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "1 1")

    def test_bce_dice_loss_penalizes_missing_foreground(self):
        import torch
        import torch.nn as nn

        logits = torch.full((1, 1, 4, 4), -4.0)
        mask = torch.zeros((1, 1, 4, 4))
        mask[:, :, 1:3, 1:3] = 1.0

        bce = nn.BCEWithLogitsLoss()(logits, mask)
        combined = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)(logits, mask)

        self.assertGreater(float(combined), float(bce))

    def test_calculate_binary_metrics_reports_positive_only_scores(self):
        import torch

        logits = torch.full((2, 1, 2, 2), -10.0)
        masks = torch.zeros((2, 1, 2, 2))
        masks[0, 0, 0, 0] = 1.0

        metrics = calculate_binary_metrics(logits, masks)

        self.assertEqual(metrics.positive_mask_slices, 1)
        self.assertEqual(metrics.empty_mask_slices, 1)
        self.assertEqual(metrics.predicted_positive_slices, 0)
        self.assertAlmostEqual(metrics.positive_dice, 0.0, places=6)
        self.assertGreater(metrics.dice, metrics.positive_dice)

    def test_visualize_predictions_can_include_probability_panel(self):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class TinyModel(nn.Module):
            def forward(self, images):
                logits = torch.full_like(images, -2.0)
                logits[:, :, 1:3, 1:3] = 2.0
                return logits

        images = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        masks = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        masks[:, :, 1:3, 1:3] = 1.0
        loader = DataLoader(TensorDataset(images, masks), batch_size=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "predictions.png"
            visualize_predictions(
                TinyModel(),
                loader,
                torch.device("cpu"),
                output_path,
                num_samples=1,
                show_probability=True,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_student_decoder_uses_two_convs_per_upsampling_stage(self):
        import torch
        import torch.nn as nn

        model = OpticalElectronicCI1DwiNoSkip32(
            image_height=64,
            image_width=80,
            in_channels=1,
            num_kernels=8,
            out_channels=1,
        )
        images = torch.randn(2, 1, 64, 80)

        logits = model(images)

        self.assertEqual(tuple(logits.shape), (2, 1, 64, 80))
        self.assertIsInstance(model.up1_refine, nn.Sequential)
        self.assertIsInstance(model.up2_refine, nn.Sequential)
        self.assertGreaterEqual(
            sum(isinstance(layer, nn.Conv2d) for layer in model.up1_refine),
            2,
        )
        self.assertGreaterEqual(
            sum(isinstance(layer, nn.Conv2d) for layer in model.up2_refine),
            2,
        )


if __name__ == "__main__":
    unittest.main()
