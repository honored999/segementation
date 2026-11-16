"""Source-separated TopK CE and per-plane Dice; no mutable network input."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SourceLoss(nn.Module):
    def __init__(self, baseline, ds_weights=None, synthetic_coefficient=1., ignore_label=None):
        super().__init__()
        self.baseline = baseline
        self.ds_weights = ds_weights
        self.synthetic_coefficient = float(synthetic_coefficient)
        if not math.isfinite(self.synthetic_coefficient) or self.synthetic_coefficient < 0:
            raise ValueError('INVALID_SYNTHETIC_COEFFICIENT')
        self.ignore_label = ignore_label
        self.metadata = None
        self.training_context = False

    def forward(self, output, target):
        if not self.training_context:
            # Validation uses the untouched original loss/checkpoint metric.
            return self.baseline(output, target)
        if self.metadata is None:
            raise ValueError('Q_SOURCE_MISSING')
        source, quality = self.metadata
        outputs = output if isinstance(output, (list, tuple)) else [output]
        targets = target if isinstance(target, (list, tuple)) else [target]
        scales = self.ds_weights if self.ds_weights is not None else [1.]
        if len(outputs) != len(targets) or len(outputs) != len(scales):
            raise ValueError('DEEP_SUPERVISION_MISMATCH')
        return sum(w * self.level(x, y, source, quality) for x, y, w in zip(outputs, targets, scales) if w > 0)

    def level(self, logits, target, source, quality):
        if logits.ndim != 4 or logits.shape[1] != 2 or target.shape != (logits.shape[0], 1, *logits.shape[2:]):
            raise ValueError('BINARY_2D_REQUIRED')
        source = torch.as_tensor(source, device=logits.device)
        quality = torch.as_tensor(quality, dtype=torch.float32, device=logits.device)
        if source.shape != (len(logits),) or quality.shape != source.shape:
            raise ValueError('SOURCE_LENGTH_MISMATCH')
        if not ((source == 0) | (source == 1)).all() or not torch.isfinite(quality).all() or not ((quality >= 0) & (quality <= 1)).all():
            raise ValueError('INVALID_SOURCE_OR_QUALITY')
        if not (quality[source == 0] == 1).all():
            raise ValueError('REAL_WEIGHT_MUST_BE_ONE')
        y = target[:, 0].long()
        valid = torch.ones_like(y, dtype=torch.bool) if self.ignore_label is None else y != self.ignore_label
        if not ((y[valid] == 0) | (y[valid] == 1)).all():
            raise ValueError('INVALID_TARGET')
        safe_y = y.masked_fill(~valid, 0)
        ce = F.cross_entropy(logits.float(), safe_y, reduction='none')
        prob = logits.float().softmax(1)[:, 1]
        mask = valid.float()
        fg = safe_y.float()
        axes = (1, 2)
        dice = 1 - (2 * (prob * fg * mask).sum(axes) + 1e-5) / (
            (prob * mask).sum(axes) + (fg * mask).sum(axes) + 1e-5)
        total = logits.sum() * 0.
        for kind, coefficient in ((0, 1.), (1, self.synthetic_coefficient)):
            group = source == kind
            pixels = valid & group[:, None, None]
            n = int(pixels.sum())
            if n == 0:
                continue
            values = ce[pixels]
            k = max(1, math.floor(n * .10))
            # Rank unweighted hardness *inside each source*. Fixed k denominator.
            indices = torch.topk(values, k, sorted=False).indices
            expanded_weights = quality[:, None, None].expand_as(ce)[pixels]
            ce_term = (values[indices] * expanded_weights[indices]).sum() / k
            planes = group & valid.flatten(1).any(1)
            dice_term = (dice[planes] * quality[planes]).sum() / planes.sum()
            total = total + coefficient * (ce_term + dice_term)
        return total
