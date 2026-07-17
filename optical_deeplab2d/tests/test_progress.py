from optical_deeplab2d.training.progress import (
    build_batch_postfix,
    format_duration,
    format_epoch_summary,
)


def test_format_duration_formats_expected_values():
    assert format_duration(None) == "N/A"
    assert format_duration(0) == "00:00"
    assert format_duration(65) == "01:05"
    assert format_duration(3661) == "01:01:01"


def test_format_duration_rounds_up_fractional_seconds_at_boundaries():
    assert format_duration(59.001) == "01:00"
    assert format_duration(3599.001) == "01:00:00"


def test_format_duration_rejects_invalid_values():
    assert format_duration(-1) == "N/A"
    assert format_duration(float("inf")) == "N/A"
    assert format_duration(float("nan")) == "N/A"


def test_build_batch_postfix_uses_fixed_progress_fields():
    assert build_batch_postfix(0.25, 0.5, 1024.4, 65) == {
        "loss": "0.2500",
        "avg_loss": "0.5000",
        "gpu_mib": "1024",
        "epoch_eta": "01:05",
    }


def test_format_epoch_summary_includes_metrics_and_total_eta():
    summary = format_epoch_summary(
        epoch=2,
        total_epochs=5,
        epoch_seconds=60,
        global_dice=0.4,
        patient_dice=0.3,
        total_eta_seconds=180,
    )

    assert "Epoch 2/5" in summary
    assert "epoch_time=01:00" in summary
    assert "val_dice=0.4000" in summary
    assert "patient_dice=0.3000" in summary
    assert "total_eta=03:00" in summary
