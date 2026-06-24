"""
Local radiometric normalization — sliding-window z-score.

Removes mosaic seams, inter-scene brightness offsets, and within-scene
illumination gradients by normalizing each pixel relative to its local
neighbourhood statistics.

How it works
------------
For every pixel (r, c) and every band b:

    z(r, c, b) = [img(r, c, b) − μ_local(r, c, b)] / [σ_local(r, c, b) + ε]

where μ_local and σ_local are computed over a square window of side `window_size`
centred on (r, c) using a fast uniform (box) filter.

Why this suppresses artefacts
------------------------------
A mosaic seam is a step-change in absolute reflectance.  After local z-score
normalization both dates independently "forget" their absolute brightness and
express each pixel only relative to its immediate neighbours.  When T1_norm and
T2_norm are subsequently differenced, the seam contribution cancels out because
both sides of the seam are normalized to zero mean / unit variance regardless of
their raw level.  Real structural changes survive because a changed pixel will
have a different z-score than its neighbours in one date but not the other.

Trade-off
---------
If real change covers an area *larger* than `window_size × window_size`, the
local statistics are themselves influenced by the change and the signal is partly
suppressed.  For Sentinel-2 at 10 m resolution the default window of 64 px
(640 m ground) is a pragmatic balance for mining-scale features.  Increase it
if you are looking for large-area change; decrease it for fine-grained targets.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter


def _band_local_zscore(
    band: np.ndarray,
    window: int,
    eps: float,
) -> np.ndarray:
    """
    Compute sliding-window z-score for a single 2D band.

    Local mean and variance are estimated via a box (uniform) filter which
    runs in O(N) time regardless of window size.
    """
    b = band.astype(np.float64)
    local_mean   = uniform_filter(b,      size=window, mode="reflect")
    local_sq     = uniform_filter(b ** 2, size=window, mode="reflect")
    local_var    = np.maximum(local_sq - local_mean ** 2, 0.0)
    local_std    = np.sqrt(local_var)
    return ((b - local_mean) / (local_std + eps)).astype(np.float32)


def local_normalize(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 64,
    clip_sigma: float = 3.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply per-band local z-score normalization to both images independently,
    then rescale to [0, 1].

    Parameters
    ----------
    img1, img2   : (H, W, C) float32 — T1 and T2 multi-band stacks
    window_size  : sliding-window side length in pixels (default 64).
                   At Sentinel-2 10 m resolution → 640 m neighbourhood.
    clip_sigma   : z-scores are clipped to ±clip_sigma before rescaling
                   to [0, 1].  Limits influence of extreme outliers.
    eps          : floor added to local std to avoid division by zero in
                   flat (zero-variance) regions such as water or cloud.

    Returns
    -------
    norm1, norm2 : (H, W, C) float32, values in [0, 1]
    """
    n_bands = img1.shape[2]
    norm1 = np.empty_like(img1, dtype=np.float32)
    norm2 = np.empty_like(img2, dtype=np.float32)

    for b in range(n_bands):
        z1 = _band_local_zscore(img1[:, :, b], window_size, eps)
        z2 = _band_local_zscore(img2[:, :, b], window_size, eps)

        # Clip outliers, then map [-clip_sigma, +clip_sigma] → [0, 1]
        z1 = np.clip(z1, -clip_sigma, clip_sigma)
        z2 = np.clip(z2, -clip_sigma, clip_sigma)

        norm1[:, :, b] = (z1 + clip_sigma) / (2.0 * clip_sigma)
        norm2[:, :, b] = (z2 + clip_sigma) / (2.0 * clip_sigma)

    return norm1, norm2
