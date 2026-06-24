"""
Method 4 — Geospatial Foundation Models (GFM): DINOv2 + SAM2 / AnyChange.

4a — DINOv2 Zero-Shot Feature Distance
  Reference:
    Oquab et al. (2024). DINOv2: Learning Robust Visual Features without
    Supervision. TMLR 2024.
  Method:
    Extract dense patch-level tokens from both dates using a frozen
    DINOv2-Base (ViT-B/14) backbone. Compute cosine distance between
    corresponding patch tokens, then bilinear-upsample to full resolution.
    No fine-tuning required — purely zero-shot.

4b — SAM2 / AnyChange Automatic Mask Comparison
  Reference:
    Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos.
    FAIR / Meta AI.
    Zheng, C. et al. (2024). AnyChange: Adaptable Zero-Shot Change
    Detection via Structured Semantic Correspondence. NeurIPS 2024.
  Method:
    Run SAM2's automatic mask generator on both images independently.
    Structural change is estimated by comparing mask density maps:
    regions where SAM2 produces fundamentally different segmentation
    structures (mask count, size, centroid displacement) are flagged
    as changed.  This operationalises the AnyChange principle without
    requiring a text prompt.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_closing, binary_opening
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_base_plus.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


# ── Utility ───────────────────────────────────────────────────────────────────

def _to_uint8_rgb(img: np.ndarray, percentile: int = 2) -> np.ndarray:
    """Convert (H,W,3) float [B,G,R] stack to uint8 RGB display image."""
    rgb = img[:, :, [2, 1, 0]].copy().astype(np.float32)
    lo = np.percentile(rgb, percentile)
    hi = np.percentile(rgb, 100 - percentile)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _pad_to_multiple(img: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple]:
    """Pad image height/width to the nearest multiple of `multiple`."""
    h, w = img.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return padded, (h, w)


# ═════════════════════════════════════════════════════════════════════════════
# 4a — DINOv2
# ═════════════════════════════════════════════════════════════════════════════

class DINOv2ChangeDetector:
    """
    Zero-shot change detector using DINOv2 (ViT-B/14) patch tokens.

    Pipeline:
      1. Resize + normalise both images as pseudo-RGB (B4→R, B3→G, B2→B).
      2. Forward-pass through frozen DINOv2 → extract patch tokens.
      3. Compute cosine distance between corresponding patch tokens.
      4. Bilinear upsample to original resolution.
      5. Bilateral-filter smoothing to preserve edges.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    PATCH_SIZE = 14

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        print(f"  [DINOv2] Loading {model_name} on {DEVICE}...")
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print("  [DINOv2] Model ready.")

    @torch.no_grad()
    def _extract_features(self, img_uint8: np.ndarray) -> torch.Tensor:
        """
        Return patch feature tensor of shape (1, n_patches, D) on CPU.
        img_uint8: (H, W, 3) uint8 RGB.
        """
        inputs = self.processor(
            images=img_uint8,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(DEVICE)

        outputs = self.model(pixel_values=pixel_values,
                             output_hidden_states=False)
        # last_hidden_state: (1, 1 + n_patches, D)  — drop [CLS] token
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (1, n_patches, D)
        return patch_tokens.cpu().float()

    def detect(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        sigma_smooth: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run DINOv2 change detection.

        Parameters
        ----------
        img1, img2    : (H, W, 3) float32 (B02, B03, B04)
        sigma_smooth  : Gaussian smoothing on upsampled distance map

        Returns
        -------
        change_prob   : (H, W) float32 [0, 1]
        change_binary : (H, W) uint8 {0, 1}
        """
        h_orig, w_orig = img1.shape[:2]

        rgb1 = _to_uint8_rgb(img1)
        rgb2 = _to_uint8_rgb(img2)

        print("  [DINOv2] Extracting features for T1...")
        feat1 = self._extract_features(rgb1)   # (1, P, D)
        print("  [DINOv2] Extracting features for T2...")
        feat2 = self._extract_features(rgb2)   # (1, P, D)

        # Cosine distance per patch: 1 - cosine_similarity
        f1 = F.normalize(feat1, dim=-1)   # (1, P, D)
        f2 = F.normalize(feat2, dim=-1)
        cosine_sim = (f1 * f2).sum(dim=-1).squeeze(0)  # (P,)
        cosine_dist = (1.0 - cosine_sim).numpy()        # (P,) in [0, 2]

        # Determine spatial layout of patches
        # DINOv2 processes image at 224×224 by default; patches = (16×16)
        n_patches = cosine_dist.shape[0]
        patch_grid = int(np.sqrt(n_patches))
        dist_map = cosine_dist.reshape(patch_grid, patch_grid)

        # Bilinear upsample to original resolution
        zoom_h = h_orig / patch_grid
        zoom_w = w_orig / patch_grid
        change_prob = zoom(dist_map, (zoom_h, zoom_w), order=1)
        change_prob = change_prob[:h_orig, :w_orig]

        change_prob = gaussian_filter(change_prob.astype(np.float32), sigma=sigma_smooth)
        change_prob = _normalise(change_prob)

        thresh = change_prob.mean() + 1.5 * change_prob.std()
        thresh = min(thresh, 0.85)
        binary = (change_prob > thresh).astype(np.uint8)

        struct = np.ones((3, 3), dtype=bool)
        binary = binary_opening(binary, structure=struct).astype(np.uint8)
        binary = binary_closing(binary, structure=struct).astype(np.uint8)

        n_changed = int(binary.sum())
        pct = 100.0 * n_changed / binary.size
        print(f"  [DINOv2] Done — {n_changed:,} changed pixels ({pct:.2f}%)")

        return change_prob, binary


# ═════════════════════════════════════════════════════════════════════════════
# 4b — SAM2 / AnyChange
# ═════════════════════════════════════════════════════════════════════════════

def _download_sam2_checkpoint() -> Path:
    """Download SAM2 checkpoint if not already present."""
    ckpt_path = CHECKPOINT_DIR / "sam2.1_hiera_base_plus.pt"
    if ckpt_path.exists():
        return ckpt_path

    print(f"  [SAM2] Downloading checkpoint → {ckpt_path.name}...")
    import urllib.request
    urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, ckpt_path)
    print("  [SAM2] Checkpoint downloaded.")
    return ckpt_path


class SAM2ChangeDetector:
    """
    AnyChange-style change detector using SAM2 automatic mask generation.

    For each image, SAM2 generates a set of object/region masks.  The
    change map is built by comparing the local mask density (number of
    distinct segments per unit area) between dates:

      - High local segment count at T2, low at T1 → new structure appeared.
      - Low local segment count at T2, high at T1 → structure disappeared.
      - Absolute difference in mask density → change signal.

    This operationalises AnyChange's observation that semantic segment
    shifts reflect real-world surface changes, without requiring text prompts.
    """

    def __init__(self):
        ckpt = _download_sam2_checkpoint()
        print(f"  [SAM2] Loading model on {DEVICE}...")

        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            self.model = build_sam2(
                SAM2_CONFIG,
                str(ckpt),
                device=DEVICE,
                apply_postprocessing=False,
            )
            self.generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=200,
            )
            print("  [SAM2] Model ready.")
            self._available = True

        except Exception as e:
            print(f"  [SAM2] WARNING: Could not load SAM2 ({e}).")
            print("  [SAM2] Falling back to OpenCV-based structural change.")
            self._available = False

    def _generate_masks(self, img_uint8: np.ndarray) -> List[Dict]:
        """Run SAM2 automatic mask generator. Returns list of mask dicts."""
        return self.generator.generate(img_uint8)

    def _mask_density_map(
        self, masks: List[Dict], h: int, w: int, sigma: float = 15.0
    ) -> np.ndarray:
        """
        Build a smooth 'segment density' map: each pixel gets a value
        proportional to how many distinct masks cover it.
        """
        density = np.zeros((h, w), dtype=np.float32)
        for m in masks:
            seg = m["segmentation"].astype(np.float32)
            # Weight smaller masks more — they indicate fine-grained changes
            size_weight = 1.0 / (np.sqrt(seg.sum()) + 1.0)
            density += seg * size_weight
        return gaussian_filter(density, sigma=sigma)

    def _opencv_structural_change(
        self, rgb1: np.ndarray, rgb2: np.ndarray
    ) -> np.ndarray:
        """
        Fallback: use edge density difference (Canny) as structural change signal
        when SAM2 is unavailable.
        """
        import cv2
        gray1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
        edges1 = cv2.Canny(gray1, 50, 150).astype(np.float32)
        edges2 = cv2.Canny(gray2, 50, 150).astype(np.float32)
        diff = gaussian_filter(np.abs(edges2 - edges1), sigma=5.0)
        return diff

    def detect(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        sigma_smooth: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run SAM2 / AnyChange structural change detection.

        Parameters
        ----------
        img1, img2    : (H, W, 3) float32 (B02, B03, B04)
        sigma_smooth  : additional smoothing sigma on the final map

        Returns
        -------
        change_prob   : (H, W) float32 [0, 1]
        change_binary : (H, W) uint8 {0, 1}
        """
        h, w = img1.shape[:2]
        rgb1 = _to_uint8_rgb(img1)
        rgb2 = _to_uint8_rgb(img2)

        if self._available:
            print("  [SAM2] Generating masks for T1...")
            masks1 = self._generate_masks(rgb1)
            print(f"    T1: {len(masks1)} masks")

            print("  [SAM2] Generating masks for T2...")
            masks2 = self._generate_masks(rgb2)
            print(f"    T2: {len(masks2)} masks")

            density1 = self._mask_density_map(masks1, h, w)
            density2 = self._mask_density_map(masks2, h, w)

            # Structural change = absolute difference in mask density
            change_signal = np.abs(density2 - density1)

            # Also incorporate IoU-based appearance/disappearance
            # Build binary coverage maps and compare
            coverage1 = np.zeros((h, w), dtype=np.float32)
            coverage2 = np.zeros((h, w), dtype=np.float32)
            for m in masks1:
                coverage1 = np.maximum(coverage1, m["segmentation"].astype(np.float32))
            for m in masks2:
                coverage2 = np.maximum(coverage2, m["segmentation"].astype(np.float32))

            coverage_diff = np.abs(coverage2 - coverage1)
            coverage_diff = gaussian_filter(coverage_diff, sigma=5.0)

            # Fuse: density difference + coverage difference
            change_signal = 0.6 * _normalise(change_signal) + 0.4 * _normalise(coverage_diff)

        else:
            print("  [SAM2] Using OpenCV structural change fallback...")
            change_signal = self._opencv_structural_change(rgb1, rgb2)

        change_prob = gaussian_filter(
            _normalise(change_signal).astype(np.float32), sigma=sigma_smooth
        )
        change_prob = _normalise(change_prob)

        thresh = change_prob.mean() + 1.5 * change_prob.std()
        thresh = min(thresh, 0.85)
        binary = (change_prob > thresh).astype(np.uint8)

        struct = np.ones((5, 5), dtype=bool)
        binary = binary_opening(binary, structure=struct).astype(np.uint8)
        binary = binary_closing(binary, structure=struct).astype(np.uint8)

        n_changed = int(binary.sum())
        pct = 100.0 * n_changed / binary.size
        print(f"  [SAM2] Done — {n_changed:,} changed pixels ({pct:.2f}%)")

        return change_prob, binary


# ═════════════════════════════════════════════════════════════════════════════
# Public entry points
# ═════════════════════════════════════════════════════════════════════════════

def run_dinov2(
    img1: np.ndarray,
    img2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run DINOv2 zero-shot change detection (Approach 4a)."""
    detector = DINOv2ChangeDetector()
    return detector.detect(img1, img2)


def run_sam2(
    img1: np.ndarray,
    img2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run SAM2 / AnyChange structural change detection (Approach 4b)."""
    detector = SAM2ChangeDetector()
    return detector.detect(img1, img2)
