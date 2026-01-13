#!/usr/bin/env python3
"""
====================================================================
Master IMA4201 — Deep Learning for Dense Depth Completion (KITTI)
====================================================================

Hello 👋

This script is prepared for YOU as part of the Master IMA4201 project.

Your goal in this project is supervised depth completion:
  Input  : RGB image + sparse depth (LiDAR projected)
  Output : dense depth (ground truth)

To help you focus on deep learning (Dataset/Dataloader/Model/Training/Evaluation),
this script automatically prepares a SMALL supervised KITTI subset (~3K samples).

--------------------------------------------------------------------
How to run
--------------------------------------------------------------------


Prepare the project dataset (~3000 train samples):
  python3 kitti.py

You will get:
  data/kitti_supervised_3k/
    train/{rgb,sparse_depth,gt_depth,intrinsics}/...
    val/{rgb,sparse_depth,gt_depth,intrinsics}/...
    manifests/train.txt
    manifests/val.txt

Each line in the manifest:
  <rgb_path> <sparse_depth_path> <gt_depth_path> <intrinsics.npy>

--------------------------------------------------------------------
Important note (not an error)
if you want more examples , change the list DEFAULT_DRIVES by adding more drivers. this is the full list : 
2011_09_26_drive_0001
2011_09_26_drive_0002
2011_09_26_drive_0005
2011_09_26_drive_0009
2011_09_26_drive_0011
2011_09_26_drive_0013
2011_09_26_drive_0014
2011_09_26_drive_0015
2011_09_26_drive_0017
2011_09_26_drive_0018
2011_09_26_drive_0019
2011_09_26_drive_0020
2011_09_26_drive_0022
2011_09_26_drive_0023
2011_09_26_drive_0027
2011_09_26_drive_0028
2011_09_26_drive_0029
2011_09_26_drive_0032
2011_09_26_drive_0035
2011_09_26_drive_0036
2011_09_26_drive_0039
2011_09_26_drive_0046
2011_09_26_drive_0048
2011_09_26_drive_0051
2011_09_26_drive_0052
2011_09_26_drive_0056
2011_09_26_drive_0057
2011_09_26_drive_0059
2011_09_26_drive_0060
2011_09_26_drive_0061
2011_09_26_drive_0064
2011_09_26_drive_0070
2011_09_26_drive_0079
2011_09_26_drive_0084
2011_09_26_drive_0086
2011_09_26_drive_0087
2011_09_26_drive_0091
2011_09_26_drive_0093
2011_09_26_drive_0095
2011_09_26_drive_0096
2011_09_26_drive_0101
2011_09_26_drive_0104
2011_09_26_drive_0106
2011_09_26_drive_0113
2011_09_26_drive_0117
2011_09_26_drive_0119
2011_09_28_drive_0001
2011_09_28_drive_0002
2011_09_28_drive_0016
2011_09_28_drive_0021
2011_09_28_drive_0034
2011_09_28_drive_0035
2011_09_28_drive_0037
2011_09_28_drive_0038
2011_09_28_drive_0039
2011_09_28_drive_0043
2011_09_28_drive_0045
2011_09_28_drive_0047
2011_09_28_drive_0053
2011_09_28_drive_0054
2011_09_28_drive_0057
2011_09_28_drive_0065
2011_09_28_drive_0066
2011_09_28_drive_0068
2011_09_28_drive_0070
2011_09_28_drive_0071
2011_09_28_drive_0075
2011_09_28_drive_0077
2011_09_28_drive_0078
2011_09_28_drive_0080
2011_09_28_drive_0082
2011_09_28_drive_0086
2011_09_28_drive_0087
2011_09_28_drive_0089
2011_09_28_drive_0090
2011_09_28_drive_0094
2011_09_28_drive_0095
2011_09_28_drive_0096
2011_09_28_drive_0098
2011_09_28_drive_0100
2011_09_28_drive_0102
2011_09_28_drive_0103
2011_09_28_drive_0104
2011_09_28_drive_0106
2011_09_28_drive_0108
2011_09_28_drive_0110
2011_09_28_drive_0113
2011_09_28_drive_0117
2011_09_28_drive_0119
2011_09_28_drive_0121
2011_09_28_drive_0122
2011_09_28_drive_0125
2011_09_28_drive_0126
2011_09_28_drive_0128
2011_09_28_drive_0132
2011_09_28_drive_0134
2011_09_28_drive_0135
2011_09_28_drive_0136
2011_09_28_drive_0138
2011_09_28_drive_0141
2011_09_28_drive_0143
2011_09_28_drive_0145
2011_09_28_drive_0146
2011_09_28_drive_0149
2011_09_28_drive_0153
2011_09_28_drive_0154
2011_09_28_drive_0155
2011_09_28_drive_0156
2011_09_28_drive_0160
2011_09_28_drive_0161
2011_09_28_drive_0162
2011_09_28_drive_0165
2011_09_28_drive_0166
2011_09_28_drive_0167
2011_09_28_drive_0168
2011_09_28_drive_0171
2011_09_28_drive_0174
2011_09_28_drive_0177
2011_09_28_drive_0179
2011_09_28_drive_0183
2011_09_28_drive_0184
2011_09_28_drive_0185
2011_09_28_drive_0186
2011_09_28_drive_0187
2011_09_28_drive_0191
2011_09_28_drive_0192
2011_09_28_drive_0195
2011_09_28_drive_0198
2011_09_28_drive_0199
2011_09_28_drive_0201
2011_09_28_drive_0204
2011_09_28_drive_0205
2011_09_28_drive_0208
2011_09_28_drive_0209
2011_09_28_drive_0214
2011_09_28_drive_0216
2011_09_28_drive_0220
2011_09_28_drive_0222
2011_09_28_drive_0225
2011_09_29_drive_0004
2011_09_29_drive_0026
2011_09_29_drive_0071
2011_09_29_drive_0108
2011_09_30_drive_0016
2011_09_30_drive_0018
2011_09_30_drive_0020
2011_09_30_drive_0027
2011_09_30_drive_0028
2011_09_30_drive_0033
2011_09_30_drive_0034
2011_09_30_drive_0072
2011_10_03_drive_0027
2011_10_03_drive_0034
2011_10_03_drive_0042
2011_10_03_drive_0047
2011_10_03_drive_0058
)
--------------------------------------------------------------------
If you downloaded only a few KITTI drives, you may get fewer usable samples
than requested (because RGB must exist for each frame). That is normal.
"""

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np

# ==============================================================
# Online sources (public KITTI S3 mirror)
# ==============================================================
KITTI_S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
RAW_BASE = KITTI_S3 + "raw_data/"

DEPTH_ZIPS = {
    "data_depth_velodyne.zip":  KITTI_S3 + "data_depth_velodyne.zip",
    "data_depth_annotated.zip": KITTI_S3 + "data_depth_annotated.zip",
    "data_depth_selection.zip": KITTI_S3 + "data_depth_selection.zip",
}

# ==============================================================
# A SMALL list of KITTI drives (RGB sequences)
# - This is intentionally small to keep downloads reasonable.
# - If you ever need more samples, add a few more drives here.
# ==============================================================
DEFAULT_DRIVES = [
    "2011_09_26_drive_0001",
    "2011_09_26_drive_0002",
    "2011_09_26_drive_0005",
    "2011_09_26_drive_0009",
    "2011_09_26_drive_0011",
    "2011_09_26_drive_0013",
    "2011_09_26_drive_0014",
    "2011_09_26_drive_0015",
    "2011_09_26_drive_0017",
    "2011_09_26_drive_0018",
]

# ==============================================================
# Local folders
# ==============================================================
RAW_DIR = Path("data/kitti_raw_data")
DC_DIR  = Path("data/kitti_depth_completion")

SPARSE_ROOT = DC_DIR / "train_val_split" / "sparse_depth"
GT_ROOT     = DC_DIR / "train_val_split" / "ground_truth"

# ==============================================================
# Small helper functions
# ==============================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download(url: str, dst: Path) -> None:
    """Download a file only if it is not already present."""
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[OK] Already downloaded: {dst}")
        return
    print(f"[DOWNLOAD] {url}")
    print(f"           → {dst}")
    urlretrieve(url, dst)
    print("[OK] Download complete.")

def unzip(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip file into out_dir."""
    ensure_dir(out_dir)
    print(f"[EXTRACT] {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print("[OK] Extraction done.")

def copy_or_link(src: Path, dst: Path, do_copy: bool) -> None:
    """
    By default we create symlinks (saves disk).
    If --copy is used, files are physically copied.
    """
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        try:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        except Exception:
            dst.symlink_to(src)

# ==============================================================
# KITTI raw download helpers
# ==============================================================
def required_dates_from_drives(drives: List[str]) -> List[str]:
    """Extract unique dates (YYYY_MM_DD) from drive names."""
    return sorted({d[:10] for d in drives})

def raw_zip_url(name: str) -> str:
    """
    If name ends with .zip → calibration zip:
      raw_data/<date>_calib.zip
    Else → drive:
      raw_data/<drive>/<drive>_sync.zip
    """
    if name.endswith(".zip"):
        return RAW_BASE + name
    return RAW_BASE + f"{name}/{name}_sync.zip"

def raw_zip_filename(name: str) -> str:
    """Local zip filename for calib/drive."""
    return name if name.endswith(".zip") else f"{name}_sync.zip"

def download_and_extract_raw_subset(drives: List[str], raw_dir: Path, zips_dir: Path) -> None:
    """
    Download calibration zip(s) and the selected drive zip(s),
    then extract them into data/kitti_raw_data.
    """
    ensure_dir(raw_dir)
    ensure_dir(zips_dir)

    dates = required_dates_from_drives(drives)
    calib_zips = [f"{date}_calib.zip" for date in dates]
    to_get = calib_zips + drives

    print("[RAW] Will download & extract:")
    for x in to_get:
        print("  -", raw_zip_filename(x))

    for item in to_get:
        url = raw_zip_url(item)
        zipname = raw_zip_filename(item)
        dst_zip = zips_dir / zipname

        download(url, dst_zip)

        # Extract directly into RAW_DIR
        print(f"[RAW] Extracting: {zipname} → {raw_dir}")
        unzip(dst_zip, raw_dir)

# ==============================================================
# Depth completion download + folder arrangement (same spirit as your bash)
# ==============================================================
def download_and_prepare_depth_completion(dc_dir: Path, zips_dir: Path) -> None:
    """
    Download depth completion zips, extract, and arrange into:
      data/kitti_depth_completion/train_val_split/sparse_depth
      data/kitti_depth_completion/train_val_split/ground_truth
      data/kitti_depth_completion/validation
      data/kitti_depth_completion/testing
    """
    ensure_dir(dc_dir)
    ensure_dir(zips_dir)

    # Download zips
    for zip_name, url in DEPTH_ZIPS.items():
        download(url, zips_dir / zip_name)

    # Prepare folders
    ensure_dir(dc_dir / "train_val_split" / "sparse_depth")
    ensure_dir(dc_dir / "train_val_split" / "ground_truth")
    ensure_dir(dc_dir / "validation")
    ensure_dir(dc_dir / "testing")
    ensure_dir(dc_dir / "tmp")

    # Extract sparse depth
    if not (dc_dir / "train_val_split" / "sparse_depth" / "train").exists():
        unzip(zips_dir / "data_depth_velodyne.zip", dc_dir / "train_val_split" / "sparse_depth")
    else:
        print("[OK] Sparse depth already extracted.")

    # Extract GT depth
    if not (dc_dir / "train_val_split" / "ground_truth" / "train").exists():
        unzip(zips_dir / "data_depth_annotated.zip", dc_dir / "train_val_split" / "ground_truth")
    else:
        print("[OK] GT depth already extracted.")

    # Extract selection to tmp, then move validation/testing like the bash script
    if not (dc_dir / "validation" / "image").exists() and not (dc_dir / "testing" / "image").exists():
        unzip(zips_dir / "data_depth_selection.zip", dc_dir / "tmp")

    tmp_sel = dc_dir / "tmp" / "depth_selection"
    if tmp_sel.exists():
        # Validation
        src_val = tmp_sel / "val_selection_cropped"
        if src_val.exists() and not (dc_dir / "validation" / "image").exists():
            shutil.move(str(src_val / "image"),             str(dc_dir / "validation" / "image"))
            shutil.move(str(src_val / "velodyne_raw"),      str(dc_dir / "validation" / "sparse_depth"))
            shutil.move(str(src_val / "groundtruth_depth"), str(dc_dir / "validation" / "ground_truth"))
            shutil.move(str(src_val / "intrinsics"),        str(dc_dir / "validation" / "intrinsics"))

        # Testing (no GT)
        src_test = tmp_sel / "test_depth_completion_anonymous"
        if src_test.exists() and not (dc_dir / "testing" / "image").exists():
            shutil.move(str(src_test / "image"),        str(dc_dir / "testing" / "image"))
            shutil.move(str(src_test / "velodyne_raw"), str(dc_dir / "testing" / "sparse_depth"))
            shutil.move(str(src_test / "intrinsics"),   str(dc_dir / "testing" / "intrinsics"))

        shutil.rmtree(dc_dir / "tmp", ignore_errors=True)

    print("[OK] Depth completion folders are ready.")

# ==============================================================
# Intrinsics parsing (KITTI raw calib_cam_to_cam.txt)
# ==============================================================
def load_intrinsics(calib_cam_to_cam: Path) -> Dict[str, np.ndarray]:
    """
    Parse calib_cam_to_cam.txt and return camera intrinsics K for:
      - image_02
      - image_03

    Note:
      Intrinsics are constant per camera (they do not change per frame).
      We store one file per (date, camera) and reference it in manifests.
    """
    if not calib_cam_to_cam.exists():
        raise FileNotFoundError(f"Missing calibration file: {calib_cam_to_cam}")

    lines = calib_cam_to_cam.read_text().splitlines()
    K: Dict[str, np.ndarray] = {}

    for line in lines:
        line = line.strip()
        if ":" not in line:
            continue
        key, vals = line.split(":", 1)
        vals = vals.strip()

        if key in ("P_rect_02", "P_rect_03"):
            arr = np.fromstring(vals, sep=" ", dtype=np.float32)
            if arr.size != 12:
                continue
            P = arr.reshape(3, 4)

            if key == "P_rect_02":
                K["image_02"] = P[:3, :3].copy()
            else:
                K["image_03"] = P[:3, :3].copy()

    if "image_02" not in K or "image_03" not in K:
        raise RuntimeError(f"Could not parse P_rect_02/P_rect_03 from {calib_cam_to_cam}")

    return K

# ==============================================================
# Map sparse depth → GT depth and sparse depth → RGB
# ==============================================================
def gt_from_sparse(sp: Path) -> Path:
    """
    Sparse path:
      .../sparse_depth/<split>/<drive>_sync/proj_depth/velodyne_raw/<cam>/<frame>.png
    GT path:
      .../ground_truth/<split>/<drive>_sync/proj_depth/groundtruth/<cam>/<frame>.png
    """
    gt = Path(str(sp).replace(str(SPARSE_ROOT), str(GT_ROOT)))
    gt = Path(str(gt).replace("velodyne_raw", "groundtruth"))
    return gt

def parse_sparse_for_rgb(sp: Path) -> Tuple[str, str, str, Path, Path]:
    """
    From a sparse depth file path, recover:
      date, drive_sync, camera, rgb_path, calib_path

    Expected RGB path:
      data/kitti_raw_data/<date>/<drive_sync>/<cam>/data/<frame>.png
    """
    drive = None
    for part in sp.parts:
        if part.endswith("_sync") and "drive" in part:
            drive = part
            break
    if drive is None:
        raise RuntimeError(f"Cannot parse drive from: {sp}")

    date = drive[:10]
    cam = sp.parent.name
    frame = sp.name

    rgb = RAW_DIR / date / drive / cam / "data" / frame
    calib = RAW_DIR / date / "calib_cam_to_cam.txt"
    return date, drive, cam, rgb, calib

def collect_usable_items(split: str) -> List[Tuple[Path, Path, Path, str, str, Path]]:
    """
    Collect usable supervised samples for a split (train or val):
      (rgb, sparse, gt, date, cam, calib_file)

    We keep only samples where:
      - GT exists
      - RGB exists (means you downloaded that raw drive)
      - calib exists
    """
    sparse_split = SPARSE_ROOT / split
    if not sparse_split.exists():
        raise FileNotFoundError(f"Missing: {sparse_split}")

    sparse_paths = sorted(sparse_split.glob("**/*.png"))
    items: List[Tuple[Path, Path, Path, str, str, Path]] = []

    for sp in sparse_paths:
        gt = gt_from_sparse(sp)
        if not gt.exists():
            continue

        try:
            date, drive, cam, rgb, calib = parse_sparse_for_rgb(sp)
        except Exception:
            continue

        if not rgb.exists():
            continue
        if not calib.exists():
            continue

        items.append((rgb, sp, gt, date, cam, calib))

    return items

# ==============================================================
# Export dataset subset + manifests
# ==============================================================
def write_manifest(path: Path, rows: List[Tuple[Path, Path, Path, Path]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        for rgb, sp, gt, K in rows:
            f.write(f"{rgb} {sp} {gt} {K}\n")

def export_supervised_subset(
    out_dir: Path,
    n_total: int,
    n_val: int,
    seed: int,
    do_copy: bool
) -> None:
    """
    Build a supervised dataset subset:
      - train: n_total samples from train split
      - val  : n_val samples from val split

    Output structure:
      out_dir/train/{rgb,sparse_depth,gt_depth,intrinsics}
      out_dir/val/{rgb,sparse_depth,gt_depth,intrinsics}
      out_dir/manifests/{train.txt,val.txt}
    """
    rng = random.Random(seed)

    for split in ("train", "val"):
        for sub in ("rgb", "sparse_depth", "gt_depth", "intrinsics"):
            ensure_dir(out_dir / split / sub)
    ensure_dir(out_dir / "manifests")

    print("[INFO] Scanning usable TRAIN samples (RGB + sparse + GT)...")
    train_items = collect_usable_items("train")
    print(f"[INFO] Usable TRAIN samples found: {len(train_items)}")

    if len(train_items) == 0:
        raise RuntimeError(
            "No usable train samples found.\n"
            "Most common reason: too few raw drives were downloaded.\n"
            "Solution: add more drives to DEFAULT_DRIVES and rerun."
        )

    rng.shuffle(train_items)
    train_keep = train_items[:min(n_total, len(train_items))]
    if n_total > len(train_items):
        print(f"[WARN] Requested n_total={n_total}, but only {len(train_items)} usable. Keeping {len(train_keep)}.")

    print("[INFO] Scanning usable VAL samples (RGB + sparse + GT)...")
    val_items = collect_usable_items("val")
    print(f"[INFO] Usable VAL samples found: {len(val_items)}")

    rng.shuffle(val_items)
    val_keep = val_items[:min(n_val, len(val_items))]
    if n_val > len(val_items):
        print(f"[WARN] Requested n_val={n_val}, but only {len(val_items)} usable. Keeping {len(val_keep)}.")

    # Cache intrinsics per (date, cam). Intrinsics are constant per camera.
    intr_cache: Dict[Tuple[str, str], Path] = {}

    def get_intrinsics_path(date: str, cam: str, calib_file: Path) -> Path:
        key = (date, cam)
        if key in intr_cache:
            return intr_cache[key]

        K_all = load_intrinsics(calib_file)
        K = K_all[cam]

        K_path = out_dir / "train" / "intrinsics" / f"intr_{date}_{cam}.npy"
        np.save(K_path, K.astype(np.float32))
        intr_cache[key] = K_path
        return K_path

    def export_split(split_name: str, items: List[Tuple[Path, Path, Path, str, str, Path]]) -> None:
        manifest_rows: List[Tuple[Path, Path, Path, Path]] = []
        for i, (rgb, sp, gt, date, cam, calib) in enumerate(items):
            name = f"{i:010d}.png"

            dst_rgb = out_dir / split_name / "rgb" / name
            dst_sp  = out_dir / split_name / "sparse_depth" / name
            dst_gt  = out_dir / split_name / "gt_depth" / name

            copy_or_link(rgb, dst_rgb, do_copy)
            copy_or_link(sp,  dst_sp,  do_copy)
            copy_or_link(gt,  dst_gt,  do_copy)

            K_path = get_intrinsics_path(date, cam, calib)
            manifest_rows.append((dst_rgb, dst_sp, dst_gt, K_path))

            if (i + 1) % 1000 == 0:
                print(f"[INFO] {split_name}: {i+1}/{len(items)} exported")

        write_manifest(out_dir / "manifests" / f"{split_name}.txt", manifest_rows)

    print(f"[INFO] Exporting TRAIN subset: {len(train_keep)} samples")
    export_split("train", train_keep)

    print(f"[INFO] Exporting VAL subset: {len(val_keep)} samples")
    export_split("val", val_keep)

    print("\n✅ Dataset is ready.")
    print(f"Saved to: {out_dir}")
    print("Manifests:")
    print(f"  {out_dir/'manifests'/'train.txt'}")
    print(f"  {out_dir/'manifests'/'val.txt'}")
    print("\nNote: If you requested more than available, the script kept the maximum usable amount.")

# ==============================================================
# Main
# ==============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Default is 3000 (clean + fast for a Master project)
    ap.add_argument("--n_total", type=int, default=3000, help="Number of TRAIN samples to export (default: 3000)")
    ap.add_argument("--n_val", type=int, default=500, help="Number of VAL samples to export (default: 500)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlink (uses more disk)")

    # Output folder is explicitly 3k
    ap.add_argument("--out_dir", type=str, default="data/kitti_supervised_3k", help="Output dataset folder")

    # Where zips are stored (so downloads can resume)
    ap.add_argument("--raw_zips_dir", type=str, default="data/kitti_raw_zips", help="Folder to store raw KITTI zips")
    ap.add_argument("--depth_zips_dir", type=str, default="data/kitti_depth_zips", help="Folder to store depth zips")

    # Optional override for drives (comma-separated)
    ap.add_argument(
        "--drives",
        type=str,
        default="",
        help="Comma-separated drive list. If empty, uses DEFAULT_DRIVES in the script."
    )

    return ap.parse_args()

def main() -> None:
    args = parse_args()

    # Choose drives
    if args.drives.strip():
        drives = [d.strip() for d in args.drives.split(",") if d.strip()]
    else:
        drives = DEFAULT_DRIVES

    print("[INFO] Selected KITTI drives (raw RGB):")
    for d in drives:
        print("  -", d)

    # Download + extract KITTI raw subset (RGB + calib)
    ensure_dir(RAW_DIR)
    raw_zips_dir = Path(args.raw_zips_dir)
    ensure_dir(raw_zips_dir)
    download_and_extract_raw_subset(drives=drives, raw_dir=RAW_DIR, zips_dir=raw_zips_dir)

    # Download + prepare depth completion benchmark (sparse + GT + val/test selection)
    depth_zips_dir = Path(args.depth_zips_dir)
    ensure_dir(depth_zips_dir)
    download_and_prepare_depth_completion(DC_DIR, depth_zips_dir)

    # Export supervised subset (~3k)
    export_supervised_subset(
        out_dir=Path(args.out_dir),
        n_total=args.n_total,
        n_val=args.n_val,
        seed=args.seed,
        do_copy=args.copy
    )

if __name__ == "__main__":
    main()
