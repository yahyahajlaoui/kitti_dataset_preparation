# utils.py
import os
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def masked_mae(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: [B, 1, H, W]
    mask:     [B, 1, H, W] float in {0,1}
    """
    diff = torch.abs(pred - gt) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


@torch.no_grad()
def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = ((pred - gt) ** 2) * mask
    denom = mask.sum().clamp_min(1.0)
    return torch.sqrt(diff2.sum() / denom)


def gt_valid_mask(gt: torch.Tensor) -> torch.Tensor:
    """
    KITTI GT has 0 where invalid.
    gt: [B,1,H,W] or [1,H,W]
    """
    return (gt > 0.0).float()


def tensor_to_rgb_vis(rgb_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert RGB tensor (possibly ImageNet-normalized) into a displayable uint8 image.
    rgb_tensor: [3,H,W]
    """
    x = rgb_tensor.detach().cpu().float()

    # Try to detect if it's ImageNet normalized: values around [-2, +2]
    if x.min() < -0.5 or x.max() > 1.5:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = x * std + mean

    x = x.clamp(0, 1)
    x = (x * 255.0).byte()
    img = x.permute(1, 2, 0).numpy()
    return img


def depth_to_vis(depth: torch.Tensor, vmin: float = 0.0, vmax: float = 80.0) -> np.ndarray:
    """
    depth: [H,W] or [1,H,W] torch tensor in meters
    returns: uint8 RGB image via matplotlib colormap
    """
    d = depth.detach().cpu().float().squeeze(0).numpy()
    d = np.clip(d, vmin, vmax)
    d = (d - vmin) / (vmax - vmin + 1e-6)

    cmap = plt.get_cmap("plasma")
    colored = cmap(d)[:, :, :3]  # drop alpha
    colored = (colored * 255.0).astype(np.uint8)
    return colored


def save_prediction_grid(
    out_path: str,
    rgb: torch.Tensor,
    sparse: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    vmax: float = 80.0
) -> None:
    """
    Save a 2x3 grid:
      [RGB | Sparse | Pred]
      [GT  | Error  | Mask]
    """
    import PIL.Image as Image

    rgb_img = tensor_to_rgb_vis(rgb)

    sparse_vis = depth_to_vis(sparse, vmax=vmax)
    pred_vis   = depth_to_vis(pred, vmax=vmax)
    gt_vis     = depth_to_vis(gt, vmax=vmax)

    err = torch.abs(pred - gt)
    err_vis = depth_to_vis(err, vmin=0.0, vmax=10.0)  # error range in meters

    m = (sparse > 0).float()
    mask_vis = (m.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
    mask_vis = np.stack([mask_vis]*3, axis=-1)

    top = np.concatenate([rgb_img, sparse_vis, pred_vis], axis=1)
    bot = np.concatenate([gt_vis, err_vis, mask_vis], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(grid).save(out_path)
