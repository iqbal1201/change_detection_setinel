"""
Method 1 — Multi-Spectral Index PCA Fusion.

Computes five visible-band indices (VARI, NGRDI, BI, ExR, ExG) for both
dates, stacks the absolute differences, then fuses them via PCA weighting.
Binary thresholding uses Otsu's method from scikit-image.

Applicable bands: B02 (Blue), B03 (Green), B04 (Red).
"""

from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_closing, binary_opening

EPS = 1e-8


# ── Index calculations ────────────────────────────────────────────────────────

def _split(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Blue, Green, Red) float32 bands from (H, W, 3) array."""
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return b.astype(np.float32), g.astype(np.float32), r.astype(np.float32)


def vari(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Visible Atmospherically Resistant Index — vegetation sensitivity."""
    return (g - r) / (g + r - b + EPS)


def ngrdi(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Normalized Green-Red Difference Index — vegetation vs. bare soil."""
    return (g - r) / (g + r + EPS)


def brightness_index(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Brightness Index — overall surface reflectance (mine expansion)."""
    return np.sqrt((r ** 2 + g ** 2 + b ** 2) / 3.0)


def excess_red(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Excess Red — sensitive to reddish bare soil / excavation."""
    return 1.4 * r - g


def excess_green(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Excess Green — green cover; loss indicates clearing."""
    return 2.0 * g - r - b


def compute_indices(img: np.ndarray) -> Dict[str, np.ndarray]:
    b, g, r = _split(img)
    return {
        "VARI":  vari(b, g, r),
        "NGRDI": ngrdi(g, r),
        "BI":    brightness_index(b, g, r),
        "ExR":   excess_red(g, r),
        "ExG":   excess_green(b, g, r),
    }


# ── PCA fusion ────────────────────────────────────────────────────────────────

def _clip_and_normalize(arr: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Percentile clip then min-max normalise to [0, 1]."""
    lo, hi = np.percentile(arr, low), np.percentile(arr, high)
    return np.clip((arr - lo) / (hi - lo + EPS), 0, 1)


def pca_fusion(diff_stack: np.ndarray) -> np.ndarray:
    """
    Fuse a (H, W, N_indices) difference stack to a single change score via PCA.

    The first principal component (PC1), weighted by explained variance, is
    returned as the composite change map. This is equivalent to a weighted
    linear combination that maximises variance in the change signal.
    """
    h, w, n = diff_stack.shape
    X = diff_stack.reshape(-1, n)

    pca = PCA(n_components=n)
    scores = pca.fit_transform(X)                  # (H*W, n)
    var_ratios = pca.explained_variance_ratio_     # weights

    # Weighted sum of all PCs (PC1 dominates; sign-flip so more change = higher value)
    composite = np.dot(scores, var_ratios)
    composite = composite.reshape(h, w)

    # Ensure positive orientation (large change = high value)
    if composite.mean() < 0:
        composite = -composite

    return composite.astype(np.float32)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    img1: np.ndarray,
    img2: np.ndarray,
    morph_radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the multi-index PCA fusion change detection.

    Parameters
    ----------
    img1, img2 : (H, W, 3) float32  — stacked bands for T1 and T2
    morph_radius : int — kernel size for morphological clean-up

    Returns
    -------
    change_prob   : (H, W) float32, range [0, 1]
    change_binary : (H, W) uint8,   values {0, 1}
    """
    print("  [Method 1] Computing spectral indices...")
    idx1 = compute_indices(img1)
    idx2 = compute_indices(img2)

    # Absolute index differences, normalised individually
    diffs = []
    for key in idx1:
        delta = np.abs(idx2[key] - idx1[key])
        diffs.append(_clip_and_normalize(delta))

    diff_stack = np.stack(diffs, axis=-1)  # (H, W, 5)

    print("  [Method 1] Fusing via PCA...")
    composite = pca_fusion(diff_stack)
    change_prob = _clip_and_normalize(composite)

    print("  [Method 1] Thresholding (Otsu)...")
    thresh = threshold_otsu(change_prob)
    binary = (change_prob > thresh).astype(np.uint8)

    # Morphological clean-up: remove speckle, fill small holes
    struct = np.ones((morph_radius, morph_radius), dtype=bool)
    binary = binary_opening(binary, structure=struct).astype(np.uint8)
    binary = binary_closing(binary, structure=struct).astype(np.uint8)

    n_changed = int(binary.sum())
    pct = 100.0 * n_changed / binary.size
    print(f"  [Method 1] Done — {n_changed:,} changed pixels ({pct:.2f}%)")

    return change_prob, binary
