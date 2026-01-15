"""
dataset_kitti_depth_completion.py
=================================

Master IMA4201 — Supervised Depth Completion (KITTI subset)

This file gives you a READY-TO-USE PyTorch Dataset + DataLoader.

It reads the manifest files created by our dataset-preparation script:

  data/kitti_supervised_3k/manifests/train.txt
  data/kitti_supervised_3k/manifests/val.txt

Each line in the manifest is:
  <rgb_path> <sparse_depth_path> <gt_depth_path> <intrinsics.npy>

What this Dataset returns (per sample):
  x : torch.FloatTensor of shape [5, H, W]
      channels = [R, G, B, sparse_depth, validity_mask]

  y : torch.FloatTensor of shape [1, H, W]
      dense ground truth depth

We also return a small 'meta' dictionary with useful debug info.

----------------------------------------------------------------------
IMPORTANT NOTE ABOUT KITTI DEPTH PNG FORMAT
----------------------------------------------------------------------
KITTI depth maps are stored as 16-bit PNG where:
  depth_meters = png_value / 256.0
and 0 means "invalid / missing depth".

This file handles that conversion for you.

----------------------------------------------------------------------
HOW TO USE
----------------------------------------------------------------------
from dataset_kitti_depth_completion import create_dataloaders

train_loader, val_loader = create_dataloaders(
    root_dir="data/kitti_supervised_3k",
    batch_size=4,
    num_workers=4,
    height=352,
    width=1216
)

for batch in train_loader:
    x = batch["x"]   # [B, 5, H, W]
    y = batch["y"]   # [B, 1, H, W]
    ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1) Small configuration container (optional but clean)
# =============================================================================

@dataclass
class DataConfig:
    """
    Configuration for how we load and prepare the data.

    root_dir:
        Path to the dataset folder produced by the preparation script.
        Example: "data/kitti_supervised_3k"

    height, width:
        The model needs all images in the same shape.
        We resize everything to (height, width).

    rgb_normalize:
        If True, we normalize RGB images to typical ImageNet stats
        (this often helps training stability).
        If False, we simply scale RGB to [0, 1].

    include_intrinsics:
        If True, we also load the 3x3 camera intrinsics matrix and return it in meta.
        For a first baseline, intrinsics are not strictly required (we can ignore them).
    """
    root_dir: str
    height: int = 352
    width: int = 1216
    rgb_normalize: bool = True
    include_intrinsics: bool = False


# =============================================================================
# 2) Utility functions to load and convert data
# =============================================================================

def _read_manifest(manifest_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Read a manifest file and return a list of file paths.

    Each line is:
      rgb_path sparse_depth_path gt_depth_path intrinsics_path
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items: List[Tuple[str, str, str, str]] = []

    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Bad manifest line (expected 4 paths):\n{line}\n"
                    f"Got {len(parts)} parts."
                )

            rgb_path, sparse_path, gt_path, K_path = parts
            items.append((rgb_path, sparse_path, gt_path, K_path))

    if len(items) == 0:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    return items


def _load_rgb(path: str, size_hw: Tuple[int, int], normalize_imagenet: bool) -> torch.Tensor:
    """
    Load an RGB image and return a torch tensor of shape [3, H, W].

    Steps:
      - read with PIL
      - convert to RGB
      - resize to target size
      - convert to float32
      - scale to [0, 1]
      - optionally normalize by mean/std (ImageNet)
    """
    H, W = size_hw

    img = Image.open(path).convert("RGB")
    img = img.resize((W, H), resample=Image.BILINEAR)  # PIL uses (width, height)

    arr = np.asarray(img).astype(np.float32) / 255.0   # [H, W, 3] in [0,1]
    arr = np.transpose(arr, (2, 0, 1))                 # [3, H, W]

    x = torch.from_numpy(arr)

    if normalize_imagenet:
        # ImageNet normalization (common for CNNs)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std

    return x  # [3, H, W]


def _load_kitti_depth_png(path: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Load a KITTI depth PNG (16-bit) and convert to meters.

    KITTI encoding:
      depth_meters = png_value / 256.0
      png_value == 0 means invalid depth

    Returns:
      depth: torch.FloatTensor of shape [1, H, W] in meters
    """
    H, W = size_hw

    # Open with PIL, keep as 16-bit
    depth = Image.open(path)

    # Convert to numpy; dtype is usually uint16
    depth = np.array(depth).astype(np.float32)  # [H_orig, W_orig]

    # Convert encoding to meters
    depth = depth / 256.0

    # Resize depth using NEAREST (important!)
    # Because depth is not a "color image": bilinear would create fake values.
    depth_img = Image.fromarray(depth)
    depth_img = depth_img.resize((W, H), resample=Image.NEAREST)
    depth = np.array(depth_img).astype(np.float32)

    # Add channel dimension: [1, H, W]
    depth = depth[None, :, :]

    return torch.from_numpy(depth)


def _make_validity_mask(sparse_depth: torch.Tensor) -> torch.Tensor:
    """
    Build a validity mask for sparse depth.

    sparse_depth: [1, H, W] in meters
    mask:         [1, H, W] where 1 means depth exists, 0 means missing
    """
    return (sparse_depth > 0.0).float()


# =============================================================================
# 3) The PyTorch Dataset
# =============================================================================

class KittiDepthCompletionDataset(Dataset):
    """
    A Dataset that returns everything needed for supervised depth completion.

    Output dictionary per item:
      {
        "x":   [5, H, W] float32,
        "y":   [1, H, W] float32,
        "meta": { ... useful debug info ... }
      }
    """

    def __init__(self, manifest_path: str, cfg: DataConfig):
        """
        manifest_path:
            Path to train.txt or val.txt.

        cfg:
            DataConfig describing resize size and normalization.
        """
        self.cfg = cfg
        self.items = _read_manifest(manifest_path)
        self.size_hw = (cfg.height, cfg.width)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rgb_path, sparse_path, gt_path, K_path = self.items[idx]

        # ---- Load RGB as [3, H, W]
        rgb = _load_rgb(rgb_path, self.size_hw, normalize_imagenet=self.cfg.rgb_normalize)

        # ---- Load sparse depth as [1, H, W] (meters)
        sparse = _load_kitti_depth_png(sparse_path, self.size_hw)

        # ---- Validity mask as [1, H, W] (0/1)
        mask = _make_validity_mask(sparse)

        # ---- Load GT dense depth as [1, H, W] (meters)
        gt = _load_kitti_depth_png(gt_path, self.size_hw)

        # ---- Build input x by concatenation
        # Channels: RGB(3) + sparse(1) + mask(1) = 5 channels total
        x = torch.cat([rgb, sparse, mask], dim=0).float()   # [5, H, W]
        y = gt.float()                                      # [1, H, W]

        meta = {
            "rgb_path": rgb_path,
            "sparse_path": sparse_path,
            "gt_path": gt_path,
        }

        # Intrinsics are optional; for a baseline we usually do not need them.
        if self.cfg.include_intrinsics:
            K = np.load(K_path).astype(np.float32)          # [3,3]
            meta["K"] = torch.from_numpy(K)
            meta["K_path"] = K_path

        return {"x": x, "y": y, "meta": meta}


# =============================================================================
# 4) Create DataLoaders (train + val)
# =============================================================================

def create_dataloaders(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    height: int = 352,
    width: int = 1216,
    rgb_normalize: bool = True,
    include_intrinsics: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from the dataset root.

    root_dir:
        Example: "data/kitti_supervised_3k"

    batch_size:
        How many samples per batch.

    num_workers:
        How many background processes to load data faster.
        On a server: 4-8 is common. On a laptop: 0-2 might be safer.

    pin_memory:
        If you use a GPU, pin_memory=True speeds up CPU→GPU transfer.
    """
    train_manifest = os.path.join(root_dir, "manifests", "train.txt")
    val_manifest   = os.path.join(root_dir, "manifests", "val.txt")

    cfg = DataConfig(
        root_dir=root_dir,
        height=height,
        width=width,
        rgb_normalize=rgb_normalize,
        include_intrinsics=include_intrinsics,
    )

    train_ds = KittiDepthCompletionDataset(train_manifest, cfg)
    val_ds   = KittiDepthCompletionDataset(val_manifest, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,         # shuffle only in train
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,       # stable batch shapes
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


# =============================================================================
# 5) Small test (so you can quickly verify everything)
# =============================================================================
# =============================================================================
# 5) VERY CLEAR DEBUG TEST (for students)
# =============================================================================

if __name__ == "__main__":

    print("\n================ DATA LOADER DEBUG MODE ================\n")

    train_loader, val_loader = create_dataloaders(
        root_dir="data/kitti_supervised_3k",
        batch_size=2,
        num_workers=2,
        height=352,
        width=1216,
        rgb_normalize=True,
        include_intrinsics=False,
    )

    # ---- Take ONE batch only
    batch = next(iter(train_loader))

    x = batch["x"]   # [B, 5, H, W]
    y = batch["y"]   # [B, 1, H, W]
    meta = batch["meta"]

    print("✅ Batch successfully loaded\n")

    # -------------------------------------------------------
    # SHAPES
    # -------------------------------------------------------
    print("---- Tensor Shapes ----")
    print(f"Input  x shape : {x.shape}  -> [B, 5, H, W]")
    print(f"Target y shape : {y.shape}  -> [B, 1, H, W]\n")

    # -------------------------------------------------------
    # CHANNEL SEMANTICS
    # -------------------------------------------------------
    print("---- Input Channel Breakdown ----")
    print("Channel 0-2 : RGB image")
    print("Channel 3   : Sparse depth (meters)")
    print("Channel 4   : Validity mask (0 or 1)\n")

    # -------------------------------------------------------
    # VALUE RANGES (very important)
    # -------------------------------------------------------
    print("---- Value Ranges (sanity check) ----")

    rgb = x[:, 0:3, :, :]
    sparse = x[:, 3:4, :, :]
    mask = x[:, 4:5, :, :]

    print(f"RGB    min / max : {rgb.min():.3f} / {rgb.max():.3f}")
    print(f"Sparse min / max : {sparse.min():.3f} / {sparse.max():.3f}")
    print(f"Mask   unique    : {torch.unique(mask)}")
    print(f"GT     min / max : {y.min():.3f} / {y.max():.3f}\n")

    # -------------------------------------------------------
    # VALIDITY CHECK
    # -------------------------------------------------------
    print("---- Validity Mask Check ----")

    valid_sparse_pixels = (sparse > 0).sum().item()
    valid_mask_pixels = (mask > 0).sum().item()

    print(f"Valid sparse depth pixels : {valid_sparse_pixels}")
    print(f"Valid mask pixels         : {valid_mask_pixels}")

    if valid_sparse_pixels == valid_mask_pixels:
        print("✅ Mask correctly matches sparse depth\n")
    else:
        print("❌ Mask mismatch (this should NOT happen)\n")

    # -------------------------------------------------------
    # PATH CHECK
    # -------------------------------------------------------
    print("---- Example File Paths ----")
    print("RGB path   :", meta["rgb_path"][0])
    print("Sparse path:", meta["sparse_path"][0])
    print("GT path    :", meta["gt_path"][0])
    print()

    # -------------------------------------------------------
    # NUMERICAL SAFETY
    # -------------------------------------------------------
    print("---- Numerical Safety ----")

    if torch.isnan(x).any() or torch.isinf(x).any():
        print("❌ NaN or Inf detected in input!")
    else:
        print("✅ No NaN / Inf in input")

    if torch.isnan(y).any() or torch.isinf(y).any():
        print("❌ NaN or Inf detected in target!")
    else:
        print("✅ No NaN / Inf in target")

    print("\n================ DEBUG COMPLETE ================\n")
