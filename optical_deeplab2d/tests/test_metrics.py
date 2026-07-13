import numpy as np

from optical_deeplab2d.evaluation.metrics import dice_score, binary_metrics, summarize_by_patient


def test_empty_mask_dice_rules() -> None:
    empty = np.zeros(4, dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0
    assert dice_score(empty, np.array([1, 0, 0, 0])) == 0.0


def test_patient_metrics_merge_slices_before_scoring() -> None:
    rows = [
        {"patient": "p1", "target": np.array([1, 0]), "prediction": np.array([1, 0])},
        {"patient": "p1", "target": np.array([1, 0]), "prediction": np.array([0, 0])},
    ]
    result = summarize_by_patient(rows)
    assert result[0]["patient"] == "p1"
    assert result[0]["dice"] == 2 / 3


def test_binary_metrics_contains_required_counts() -> None:
    result = binary_metrics(np.array([1, 0, 0, 0]), np.array([1, 1, 0, 0]))
    assert result["false_positive_pixels"] == 1
    assert result["predicted_lesion_area"] == 2
