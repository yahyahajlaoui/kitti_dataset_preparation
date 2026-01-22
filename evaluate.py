# evaluate.py
import argparse
import os

import torch
from tqdm import tqdm

from dataloader import create_dataloaders
from model import UNetSmall
from utils import gt_valid_mask, masked_mae, masked_rmse, save_prediction_grid, ensure_dir


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/kitti_supervised_3k")
    ap.add_argument("--ckpt", type=str, default="runs/unet_kitti/best.pt")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--width", type=int, default=1216)
    ap.add_argument("--out_dir", type=str, default="runs/unet_kitti/vis")
    ap.add_argument("--max_save", type=int, default=30)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _, val_loader = create_dataloaders(
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

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Loaded checkpoint:", args.ckpt)

    ensure_dir(args.out_dir)

    mae_sum = 0.0
    rmse_sum = 0.0
    n_batches = 0

    saved = 0

    for batch in tqdm(val_loader, desc="Evaluate"):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        pred = model(x)
        m_gt = gt_valid_mask(y)

        mae_sum += masked_mae(pred, y, m_gt).item()
        rmse_sum += masked_rmse(pred, y, m_gt).item()
        n_batches += 1

        # Save a few visualizations
        for i in range(x.shape[0]):
            if saved >= args.max_save:
                continue

            rgb = x[i, 0:3].cpu()
            sparse = x[i, 3:4].cpu()
            pred_i = pred[i].cpu()
            gt_i = y[i].cpu()

            out_path = os.path.join(args.out_dir, f"val_{saved:03d}.png")
            save_prediction_grid(out_path, rgb, sparse, pred_i, gt_i, vmax=80.0)
            saved += 1

    mae = mae_sum / max(1, n_batches)
    rmse = rmse_sum / max(1, n_batches)

    print(f"\nFinal VAL metrics:")
    print(f"  MAE : {mae:.3f} m")
    print(f"  RMSE: {rmse:.3f} m")
    print(f"Saved visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()
