"""Train a MONAI SegResNet baseline on the CI-1 DWI+ADC datalist."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def require_monai():
    try:
        from monai.data import CacheDataset, DataLoader, decollate_batch
        from monai.inferers import sliding_window_inference
        from monai.losses import DiceCELoss
        from monai.metrics import DiceMetric
        from monai.networks.nets import SegResNet
        from monai.transforms import (
            Activations,
            AsDiscrete,
            Compose,
            EnsureChannelFirstd,
            LoadImaged,
            RandCropByPosNegLabeld,
            ScaleIntensityd,
            SpatialPadd,
            ToTensord,
        )
    except ImportError as exc:
        raise ImportError("MONAI is required. Install it with: pip install monai") from exc
    return locals()


def resolve_items(items: list[dict[str, object]], root: Path) -> list[dict[str, object]]:
    resolved = []
    for item in items:
        resolved.append(
            {
                "image": [str(root / path) for path in item["image"]],
                "label": str(root / item["label"]),
            }
        )
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MONAI SegResNet on CI-1 DWI+ADC.")
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--roi-size", type=int, nargs=3, default=[96, 96, 32])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained-path", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device selection. 'auto' uses CUDA only when this PyTorch build supports the GPU architecture.",
    )
    return parser.parse_args()


def load_pretrained(model: torch.nn.Module, path: Path, strict: bool) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    matched = {key: value for key, value in state.items() if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)}
    model_state.update(matched)
    model.load_state_dict(model_state, strict=strict)
    print(f"Loaded matched pretrained tensors: {len(matched)}")


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        required_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if required_arch in supported_arches:
            return torch.device("cuda")
        message = (
            f"CUDA device requires {required_arch}, but this PyTorch build supports "
            f"{sorted(supported_arches)}. Install a newer PyTorch build for this GPU."
        )
        if requested == "cuda":
            raise RuntimeError(message)
        print(f"[WARN] {message} Falling back to CPU for this run.")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    monai = require_monai()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.datalist.open("r", encoding="utf-8") as handle:
        datalist = json.load(handle)
    train_files = resolve_items(datalist.get("training", []), args.dataset_root)
    val_files = resolve_items(datalist.get("validation", []), args.dataset_root)

    transforms = monai["Compose"](
        [
            monai["LoadImaged"](keys=["image", "label"]),
            monai["EnsureChannelFirstd"](keys=["image", "label"]),
            monai["ScaleIntensityd"](keys=["image"]),
            monai["SpatialPadd"](keys=["image", "label"], spatial_size=tuple(args.roi_size)),
            monai["RandCropByPosNegLabeld"](
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(args.roi_size),
                pos=3,
                neg=1,
                num_samples=2,
                image_key="image",
            ),
            monai["ToTensord"](keys=["image", "label"]),
        ]
    )
    val_transforms = monai["Compose"](
        [
            monai["LoadImaged"](keys=["image", "label"]),
            monai["EnsureChannelFirstd"](keys=["image", "label"]),
            monai["ScaleIntensityd"](keys=["image"]),
            monai["SpatialPadd"](keys=["image", "label"], spatial_size=tuple(args.roi_size)),
            monai["ToTensord"](keys=["image", "label"]),
        ]
    )
    train_ds = monai["CacheDataset"](train_files, transform=transforms, cache_rate=0.2)
    val_ds = monai["CacheDataset"](val_files, transform=val_transforms, cache_rate=0.2)
    train_loader = monai["DataLoader"](train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = monai["DataLoader"](val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = select_device(args.device)
    print(f"Using device: {device}")
    model = monai["SegResNet"](spatial_dims=3, in_channels=2, out_channels=1, init_filters=16).to(device)
    if args.pretrained_path:
        load_pretrained(model, args.pretrained_path, strict=args.strict)
    loss_fn = monai["DiceCELoss"](sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    post_pred = monai["Compose"]([monai["Activations"](sigmoid=True), monai["AsDiscrete"](threshold=0.5)])
    post_label = monai["AsDiscrete"](threshold=0.5)
    dice_metric = monai["DiceMetric"](include_background=False, reduction="mean")

    metrics_path = args.output_dir / "validation_metrics.csv"
    best_dice = -1.0
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_dice"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            steps = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item())
                steps += 1
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    outputs = monai["sliding_window_inference"](images, tuple(args.roi_size), 1, model)
                    preds = [post_pred(item) for item in monai["decollate_batch"](outputs)]
                    labs = [post_label(item) for item in monai["decollate_batch"](labels)]
                    dice_metric(y_pred=preds, y=labs)
                val_dice = float(dice_metric.aggregate().item())
                dice_metric.reset()
            train_loss = total_loss / max(steps, 1)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice})
            handle.flush()
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_dice={val_dice:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, args.output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
