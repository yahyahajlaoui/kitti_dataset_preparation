# train.py
import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import create_dataloaders
from model import UNetSmall
from utils import gt_valid_mask, masked_mae, masked_rmse, ensure_dir


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["x"].to(device, non_blocking=True)  # [B,5,H,W]
        y = batch["y"].to(device, non_blocking=True)  # [B,1,H,W]

        pred = model(x)

        # Compute loss only where GT is valid
        m_gt = gt_valid_mask(y)
        loss = (torch.abs(pred - y) * m_gt).sum() / m_gt.sum().clamp_min(1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(model, loader, device) -> Dict[str, float]:
    model.eval()
    mae_sum = 0.0
    rmse_sum = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        pred = model(x)
        m_gt = gt_valid_mask(y)

        mae = masked_mae(pred, y, m_gt).item()
        rmse = masked_rmse(pred, y, m_gt).item()

        mae_sum += mae
        rmse_sum += rmse
        n_batches += 1

    return {
        "mae": mae_sum / max(1, n_batches),
        "rmse": rmse_sum / max(1, n_batches),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/kitti_supervised_3k")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--width", type=int, default=1216)
    ap.add_argument("--out_dir", type=str, default="runs/unet_kitti")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ensure_dir(args.out_dir)
    ckpt_path = os.path.join(args.out_dir, "best.pt")

    train_loader, val_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        height=args.height,
        width=args.width,
        rgb_normalize=True,
        include_intrinsics=False,
        pin_memory=True,
    )

    model = UNetSmall(in_channels=5, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = validate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={loss:.4f} | val_MAE={metrics['mae']:.3f} | val_RMSE={metrics['rmse']:.3f}")

        # Save best checkpoint by RMSE
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_rmse": best_rmse,
                    "args": vars(args),
                },
                ckpt_path
            )
            print(f"  ✅ Saved best checkpoint: {ckpt_path} (best_rmse={best_rmse:.3f})")

    print("Training finished.")
    print("Best checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
