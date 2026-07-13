import torch

from optical_deeplab2d.training.losses import CombinedBCEDiceLoss


def test_combined_loss_is_finite_for_empty_and_positive_masks() -> None:
    loss = CombinedBCEDiceLoss(pos_weight=2.0)
    for target in (torch.zeros(2, 1, 3, 3), torch.ones(2, 1, 3, 3)):
        assert torch.isfinite(loss(torch.zeros_like(target), target))

