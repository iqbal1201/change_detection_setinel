"""
Method 5 — Deep Change Vector Analysis (DCVA).

Reference:
  Saha, S., Bovolo, F. & Bruzzone, L. (2019). Unsupervised Deep Change Vector
  Analysis for Multiple-Change Detection in VHR Images.
  IEEE Transactions on Geoscience and Remote Sensing, 57(6), 3677–3693.

Method:
  1. Feed both dates through a frozen VGG16 (ImageNet-pretrained) backbone and
     extract dense feature maps up to relu-after-conv3_2 (stride ×4, 256 ch).
  2. Compute the per-location feature difference D = F_T2 − F_T1.
  3. CVA change magnitude = ‖D‖₂ across all 256 channels.
  4. Bilinear-upsample the magnitude map to the original image resolution.
  5. Apply Otsu thresholding on the normalised magnitude to produce the binary mask.

Operating in VGG16's mid-level feature space makes the detector sensitive to
structural / land-cover changes while being more robust to atmospheric and
illumination differences than raw-band CVA.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening
from skimage.filters import threshold_otsu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_uint8_rgb(img: np.ndarray, percentile: int = 2) -> np.ndarray:
    """Convert (H, W, 3) float32 [B, G, R] stack to uint8 RGB."""
    rgb = img[:, :, [2, 1, 0]].copy().astype(np.float32)
    lo = np.percentile(rgb, percentile)
    hi = np.percentile(rgb, 100 - percentile)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _build_extractor() -> nn.Module:
    """
    Return a frozen VGG16 sub-network (features[0:15]).

    Layers included: conv1_1, relu, conv1_2, relu, maxpool,
                     conv2_1, relu, conv2_2, relu, maxpool,
                     conv3_1, relu, conv3_2, relu  ← cut here.
    Output stride = 4,  channels = 256.
    """
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(vgg.features.children())[:15])
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor.to(_DEVICE)


def _extract(extractor: nn.Module, img_uint8: np.ndarray) -> torch.Tensor:
    """
    Forward-pass a single (H, W, 3) uint8 RGB image through the extractor.
    Returns a CPU float32 tensor of shape (256, H/4, W/4).
    """
    from torchvision.transforms import functional as TF

    tensor = TF.to_tensor(img_uint8).unsqueeze(0)      # (1, 3, H, W) [0,1]
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = tensor.to(_DEVICE)

    with torch.no_grad():
        feat = extractor(tensor)   # (1, 256, H/4, W/4)
    return feat.squeeze(0).cpu()  # (256, H/4, W/4)


def run(
    img1: np.ndarray,
    img2: np.ndarray,
    sigma_smooth: float = 2.0,
    morph_radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Deep Change Vector Analysis.

    Parameters
    ----------
    img1, img2    : (H, W, 3) float32  [B02, B03, B04]
    sigma_smooth  : Gaussian sigma applied to the upsampled magnitude map
    morph_radius  : structuring element size for morphological clean-up

    Returns
    -------
    change_prob   : (H, W) float32, range [0, 1]
    change_binary : (H, W) uint8,   values {0, 1}
    """
    h, w = img1.shape[:2]

    print(f"  [DCVA] Loading VGG16 feature extractor on {_DEVICE}...")
    extractor = _build_extractor()

    print("  [DCVA] Extracting VGG16 features for T1...")
    feat1 = _extract(extractor, _to_uint8_rgb(img1))  # (256, H/4, W/4)

    print("  [DCVA] Extracting VGG16 features for T2...")
    feat2 = _extract(extractor, _to_uint8_rgb(img2))  # (256, H/4, W/4)

    # CVA magnitude: L2-norm of the feature-space difference vector
    diff      = feat2 - feat1                # (256, H/4, W/4)
    magnitude = diff.norm(dim=0).numpy()     # (H/4, W/4)

    print("  [DCVA] Upsampling magnitude map...")
    mag_t  = torch.tensor(magnitude).unsqueeze(0).unsqueeze(0)   # (1,1,h',w')
    mag_up = F.interpolate(
        mag_t, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().numpy()                      # (H, W)

    # Smooth and normalise to [0, 1]
    change_prob = gaussian_filter(mag_up.astype(np.float32), sigma=sigma_smooth)
    lo, hi = change_prob.min(), change_prob.max()
    change_prob = ((change_prob - lo) / (hi - lo + 1e-8)).astype(np.float32)

    print("  [DCVA] Thresholding (Otsu)...")
    thresh = threshold_otsu(change_prob)
    binary = (change_prob > thresh).astype(np.uint8)

    struct = np.ones((morph_radius, morph_radius), dtype=bool)
    binary = binary_opening(binary, structure=struct).astype(np.uint8)
    binary = binary_closing(binary, structure=struct).astype(np.uint8)

    n_changed = int(binary.sum())
    pct = 100.0 * n_changed / binary.size
    print(f"  [DCVA] Done — {n_changed:,} changed pixels ({pct:.2f}%)")

    return change_prob, binary
