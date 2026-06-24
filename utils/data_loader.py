"""
Part 1 — Data loading, validation, and band stacking.
Loads Sentinel-2 B02/B03/B04 bands for two dates, verifies spatial
consistency, and writes stacked GeoTIFFs to data/processed/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import rasterio
from rasterio.transform import Affine

DATA_DIR = Path(__file__).parents[1] / "data"
PROCESSED_DIR = DATA_DIR / "processed"

BAND_FILES = ["B02.tif", "B03.tif", "B04.tif"]  # Blue, Green, Red
DATE_DIRS = {
    "20230812": DATA_DIR / "sentinel2_20230812",
    "20230902": DATA_DIR / "sentinel2_20230902",
}


def _load_single_band(path: Path) -> Tuple[np.ndarray, Dict]:
    """Return (array float32, profile) for a single-band GeoTIFF."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def load_and_stack(date_key: str) -> Tuple[np.ndarray, Dict]:
    """
    Load B02, B03, B04 for a given date key and stack into (H, W, 3).

    Parameters
    ----------
    date_key : str
        One of '20230812' or '20230902'.

    Returns
    -------
    stack : np.ndarray, shape (H, W, 3), float32
        Band order: [Blue(B02), Green(B03), Red(B04)]
    profile : dict
        Rasterio profile of the first band (used for writing outputs).
    """
    date_dir = DATE_DIRS[date_key]
    bands, profile = [], None

    for band_file in BAND_FILES:
        arr, p = _load_single_band(date_dir / band_file)
        bands.append(arr)
        if profile is None:
            profile = p

    stack = np.stack(bands, axis=-1)  # (H, W, 3)
    return stack, profile


def verify_consistency(stack1: np.ndarray, stack2: np.ndarray,
                       profile1: Dict, profile2: Dict) -> None:
    """Raise ValueError if spatial properties differ between dates."""
    if stack1.shape != stack2.shape:
        raise ValueError(
            f"Shape mismatch: {stack1.shape} vs {stack2.shape}"
        )
    if profile1["crs"] != profile2["crs"]:
        raise ValueError(
            f"CRS mismatch: {profile1['crs']} vs {profile2['crs']}"
        )
    if profile1["transform"] != profile2["transform"]:
        raise ValueError("Geotransform mismatch between dates.")
    print(f"  Verified: shape={stack1.shape}, CRS={profile1['crs']}, "
          f"pixel_size={profile1['transform'].a:.1f}m")


def save_stack(stack: np.ndarray, profile: Dict, out_path: Path) -> None:
    """Write (H, W, B) float32 array as a multi-band GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, b = stack.shape
    out_profile = profile.copy()
    out_profile.update(count=b, dtype="float32", driver="GTiff",
                       compress="lzw")
    with rasterio.open(out_path, "w", **out_profile) as dst:
        for i in range(b):
            dst.write(stack[:, :, i], i + 1)


def load_processed_stack(date_key: str) -> Tuple[np.ndarray, Dict]:
    """Load a previously saved processed stack from data/processed/."""
    path = PROCESSED_DIR / f"sentinel2_{date_key}_stack.tif"
    with rasterio.open(path) as src:
        arr = src.read()                  # (B, H, W)
        arr = np.moveaxis(arr, 0, -1)    # (H, W, B)
        profile = src.profile.copy()
    return arr.astype(np.float32), profile


def run_part1() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Execute Part 1 end-to-end.

    Returns
    -------
    img1, img2 : np.ndarray  (H, W, 3) float32
    profile    : dict         shared rasterio profile
    """
    print("[Part 1] Loading and stacking bands...")

    img1, p1 = load_and_stack("20230812")
    img2, p2 = load_and_stack("20230902")
    verify_consistency(img1, img2, p1, p2)

    out1 = PROCESSED_DIR / "sentinel2_20230812_stack.tif"
    out2 = PROCESSED_DIR / "sentinel2_20230902_stack.tif"
    save_stack(img1, p1, out1)
    save_stack(img2, p2, out2)
    print(f"  Saved: {out1.name}, {out2.name}")

    return img1, img2, p1
