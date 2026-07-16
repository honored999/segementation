from pathlib import Path
import numpy as np

from optical_deeplab2d.evaluation.visualization import select_representative_rows, save_validation_grid


def test_selection_is_unique_and_grid_is_written(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        target = np.zeros((8, 8), dtype=np.uint8)
        if index != 3: target[: index + 1, : index + 1] = 1
        prediction = np.zeros_like(target)
        prediction[0:index, 0:index] = 1
        rows.append({"sample_id": str(index), "patient": "p1", "timepoint": "D1", "slice_index": index, "image": np.zeros((8, 8)), "target": target, "prediction": prediction})
    selected = select_representative_rows(rows, seed=2026, limit=6)
    output = tmp_path / "validation_predictions_best.png"
    save_validation_grid(selected, output)
    assert len(selected) <= 6
    assert len({row["sample_id"] for row in selected}) == len(selected)
    assert output.exists() and output.stat().st_size > 0


def test_random_selection_is_deterministic_and_unique() -> None:
    from optical_deeplab2d.evaluation.visualization import select_random_rows
    rows = [{"sample_id": str(index)} for index in range(10)]
    assert [row["sample_id"] for row in select_random_rows(rows, 6)] == [row["sample_id"] for row in select_random_rows(rows, 6)]
    assert len({row["sample_id"] for row in select_random_rows(rows, 6)}) == 6
