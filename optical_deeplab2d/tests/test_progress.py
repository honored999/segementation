import optical_deeplab2d.training.progress as progress

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


def test_update_running_loss_tracks_a_sequence_in_constant_space():
    total, average = progress.update_running_loss(0.0, 0, 0.25)
    assert total == 0.25
    assert average == 0.25

    total, average = progress.update_running_loss(total, 1, 0.75)
    assert total == 1.0
    assert average == 0.5

    total, average = progress.update_running_loss(total, 2, 2.0)
    assert total == 3.0
    assert average == 1.0


def test_complete_epoch_timing_uses_the_complete_current_epoch_duration():
    completed, total_eta = progress.complete_epoch_timing([], 40.0, 4)
    assert completed == [40.0]
    assert total_eta is None

    completed, total_eta = progress.complete_epoch_timing(completed, 60.0, 3)
    assert completed == [40.0, 60.0]
    assert total_eta == 150.0


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
