from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.engine.checkpoint import load_checkpoint
from standalone_nnunet2d.models import PlainConvUNet2D


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only smoke checkpoint restore verification")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = PlainConvUNet2D(load_model_config()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    metadata = load_checkpoint(model, optimizer, args.checkpoint, {"smoke_run_only": True})
    with torch.no_grad():
        shape = tuple(model(torch.zeros(1, 1, 512, 512, device=device)).shape)
    print(json.dumps({"metadata": metadata, "output_shape": shape, "smoke_run_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
